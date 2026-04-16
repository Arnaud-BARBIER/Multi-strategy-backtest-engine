
import numpy as np

import pandas as pd
import plotly.graph_objects as go

from .Njit_plots import plot_results as _plot_results
from .backtest_bundle import SignalPrep
from .pipeline_config import BacktestConfig, DataPipeline
from .context_engine import TradeContextEngine, build_default_context_df
from .indicators import ema_njit
from .core_engine import (
    backtest_njit,
    compute_metrics_full,
    signals_ema_vs_close_njit,
    signals_ema_cross_njit,
)
from .multi_setup_layer import (
    SetupSpec,
    DecisionConfig,
    aggregate_and_decide,
    _validate_setup_df,
)
from .feature_compiler import CompiledFeatures, to_compiled_features
from .Exit_system import N_EXIT_RT_COLS
from .exit_strategy_system import (N_EXIT_STRAT_RT_COLS, N_STATEFUL_CFG_COLS)
from .regime_policy import RegimePolicy, RegimeContext, build_regime_context

class NJITEngine:

    @staticmethod
    def _resolve_signal_cols(df, signal_col):
        if isinstance(signal_col, str):
            if signal_col not in df.columns:
                raise ValueError(f"'{signal_col}' not found in df columns: {list(df.columns)}")
            return signal_col

        if isinstance(signal_col, (list, tuple)):
            cols = list(signal_col)
            if len(cols) != 2:
                raise ValueError(
                    "signal_col as a list/tuple must contain exactly two columns: "
                    "['long_active', 'short_active']."
                )

            missing = [col for col in cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing signal columns: {missing}. Available: {list(df.columns)}")
            return cols

        raise TypeError("signal_col must be either a string or a 2-item list/tuple of column names.")

    @staticmethod
    def _extract_signal_masks(df, signal_col):
        resolved = NJITEngine._resolve_signal_cols(df, signal_col)

        if isinstance(resolved, str):
            signal = pd.to_numeric(df[resolved], errors="coerce").fillna(0.0)
            long_mask = signal > 0
            short_mask = signal < 0
            return resolved, long_mask, short_mask

        long_col, short_col = resolved
        long_mask = pd.to_numeric(df[long_col], errors="coerce").fillna(0.0) > 0
        short_mask = pd.to_numeric(df[short_col], errors="coerce").fillna(0.0) > 0
        return resolved, long_mask, short_mask

    @staticmethod
    def _ensure_signal_column(df, signal_col, out_col="Signal"):
        resolved, long_mask, short_mask = NJITEngine._extract_signal_masks(df, signal_col)

        if isinstance(resolved, str):
            return df, resolved

        out = df.copy()
        signal = np.zeros(len(out), dtype=np.int8)
        long_only = long_mask & ~short_mask
        short_only = short_mask & ~long_mask
        both_sides = long_mask & short_mask

        signal[long_only.to_numpy(dtype=bool)] = 1
        signal[short_only.to_numpy(dtype=bool)] = -1
        signal[both_sides.to_numpy(dtype=bool)] = 2

        out[out_col] = signal
        return out, out_col

    @staticmethod
    def signal_df_to_setup_df(
        signal_df,
        signal_col="Signal",
        setup_id=0,
        score=1.0,
        score_from_signal=False,
    ):
        """
        Convert a conventional uni-signal DataFrame into a setup-compatible DataFrame.

        Expected convention:
        - long signal  -> positive values
        - short signal -> negative values
        - flat         -> 0

        If signal_col is a pair like ['long_active', 'short_active'], the helper
        also works and simply maps those columns into setup format.
        """
        if not isinstance(signal_df, pd.DataFrame):
            raise TypeError("signal_df must be a pandas DataFrame")

        resolved, long_mask, short_mask = NJITEngine._extract_signal_masks(signal_df, signal_col)

        long_active = long_mask.astype("int8")
        short_active = short_mask.astype("int8")

        if score_from_signal:
            if isinstance(resolved, str):
                raw_signal = pd.to_numeric(signal_df[resolved], errors="coerce").fillna(0.0).abs()
            else:
                raw_signal = pd.Series(
                    np.maximum(
                        pd.to_numeric(signal_df[resolved[0]], errors="coerce").fillna(0.0).abs().to_numpy(dtype=np.float64),
                        pd.to_numeric(signal_df[resolved[1]], errors="coerce").fillna(0.0).abs().to_numpy(dtype=np.float64),
                    ),
                    index=signal_df.index,
                )
            long_score = np.where(long_active.to_numpy(dtype=np.int8) == 1, raw_signal.to_numpy(dtype=np.float64), 0.0)
            short_score = np.where(short_active.to_numpy(dtype=np.int8) == 1, raw_signal.to_numpy(dtype=np.float64), 0.0)
        else:
            long_score = np.where(long_active.to_numpy(dtype=np.int8) == 1, float(score), 0.0)
            short_score = np.where(short_active.to_numpy(dtype=np.int8) == 1, float(score), 0.0)

        return pd.DataFrame(
            {
                "long_score": np.asarray(long_score, dtype=np.float64),
                "short_score": np.asarray(short_score, dtype=np.float64),
                "long_active": long_active.to_numpy(dtype=np.int8),
                "short_active": short_active.to_numpy(dtype=np.int8),
                "setup_id": np.full(len(signal_df), int(setup_id), dtype=np.int32),
            },
            index=signal_df.index,
        )

    @staticmethod
    def wrap_signal_strategy(
        strategy_fn,
        signal_col="Signal",
        score_from_signal=False,
    ):
        """
        Wrap a uni-signal strategy function so it becomes SetupSpec-compatible.

        Example:
            setup_fn = NJITEngine.wrap_signal_strategy(rsi_reentry_df)
            SetupSpec(fn=setup_fn, params=dict(feature=feature, setup_id=0, score=1.0))
        """
        if not callable(strategy_fn):
            raise TypeError("strategy_fn must be callable")

        def _wrapped_setup(df, setup_id=0, score=1.0, **kwargs):
            signal_df = strategy_fn(df, **kwargs)
            return NJITEngine.signal_df_to_setup_df(
                signal_df=signal_df,
                signal_col=signal_col,
                setup_id=setup_id,
                score=score,
                score_from_signal=score_from_signal,
            )

        _wrapped_setup.__name__ = getattr(strategy_fn, "__name__", "wrapped_signal_strategy") + "_setup"
        return _wrapped_setup

    @staticmethod
    def _prepare_main_df(df_like, start=None, end=None) -> pd.DataFrame:
        df = pd.DataFrame(df_like).copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("main_df must have a DatetimeIndex.")

        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()

        if start is not None or end is not None:
            df = df.loc[start:end]

        required = ["Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"main_df is missing required OHLC columns: {missing}"
            )
        return df

    @staticmethod
    def _resolve_atr_period(cfg, atr_period):
        if atr_period is not None:
            return int(atr_period)
        if cfg is not None and hasattr(cfg, "atr_period"):
            return int(cfg.atr_period)
        return 14

    @staticmethod
    def _resolve_atr_array(atr_array, index: pd.Index) -> np.ndarray:
        if isinstance(atr_array, pd.Series):
            arr = atr_array.reindex(index).to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(atr_array, dtype=np.float64)

        if arr.ndim != 1:
            raise ValueError("atr_array must be a 1D array or pandas Series.")
        if len(arr) != len(index):
            raise ValueError(
                f"atr_array length mismatch: got {len(arr)}, expected {len(index)}."
            )
        return arr

    def __init__(self, pipeline=None, ticker=None, start=None, end=None, cfg=None,
                atr_period=None, atr_array=None, MAX_TRADES=50_000, MAX_POS=500, main_df=None):
        self.cfg        = cfg
        self.MAX_TRADES = MAX_TRADES
        self.MAX_POS    = MAX_POS

        if main_df is not None:
            df = self._prepare_main_df(main_df, start=start, end=end)

            if atr_array is not None:
                df["ATR"] = self._resolve_atr_array(atr_array, df.index)
            elif "ATR" in df.columns:
                df["ATR"] = np.asarray(df["ATR"], dtype=np.float64)
            else:
                resolved_atr_period = self._resolve_atr_period(cfg, atr_period)
                df = DataPipeline.compute_atr(df, resolved_atr_period)
        else:
            if pipeline is None:
                raise ValueError("pipeline must be provided when main_df is not supplied.")
            if ticker is None:
                raise ValueError("ticker must be provided when main_df is not supplied.")

            resolved_atr_period = self._resolve_atr_period(cfg, atr_period)
            df = pipeline.prepare_df(
                ticker=ticker,
                start=start,
                end=end,
                timezone_shift=getattr(cfg, "timezone_shift", 0),
                atr_period=resolved_atr_period
                )

        self.bar_index      = df.index
        self.opens          = df["Open"].to_numpy(dtype=np.float64)
        self.highs          = df["High"].to_numpy(dtype=np.float64)
        self.lows           = df["Low"].to_numpy(dtype=np.float64)
        self.closes         = df["Close"].to_numpy(dtype=np.float64)
        self.atrs           = df["ATR"].to_numpy(dtype=np.float64)
        self.minutes_of_day = (df.index.hour * 60 + df.index.minute).to_numpy(dtype=np.int16)
        self.day_index      = ((df.index - df.index[0]).days).to_numpy(dtype=np.int32)
        self.day_of_week = df.index.dayofweek.to_numpy(dtype=np.int8)
        self.last_signal_df  = None
        self.last_signal_col = "Signal"
        self.last_feature_context: CompiledFeatures | None = None

        self._warmup()

    def inspect_signals(
        self,
        strategy_fn=None,
        signal_col="Signal",
        plot=False,
        crypto=False,
        return_df=True,
        start=None,
        end=None,
        setup_specs=None,
        decision_cfg=None,
        include_price_cols=True,
        lock_setup_id=None,
        setup_name_map=None,
        **kwargs,
    ):
        # -------------------------------------------------
        # MODE 1 : stratégie simple
        # -------------------------------------------------
        if strategy_fn is not None:
            base_df = pd.DataFrame({
                "Open": self.opens,
                "High": self.highs,
                "Low": self.lows,
                "Close": self.closes,
                "ATR": self.atrs,
            }, index=self.bar_index)

            df = strategy_fn(base_df.copy(), **kwargs)
            df = self._ensure_price_cols_for_inspection(df, base_df)

            resolved_signal_cols = self._resolve_signal_cols(df, signal_col)
            df, normalized_signal_col = self._ensure_signal_column(df, resolved_signal_cols)

            self.last_signal_df = df.copy()
            self.last_signal_col = normalized_signal_col

            if plot:
                self._plot_signal_df(df, signal_col=resolved_signal_cols, crypto=crypto, start=start, end=end)

            if return_df:
                return df
            return df[normalized_signal_col].to_numpy(dtype=np.int8)

        # -------------------------------------------------
        # MODE 2 : multi-setup
        # -------------------------------------------------
        if setup_specs is not None:
            if decision_cfg is None:
                raise ValueError("decision_cfg must be provided when setup_specs is used")

            multi_out = self.prepare_multi_setup_signals(
                setup_specs=setup_specs,
                decision_cfg=decision_cfg,
                include_price_cols=include_price_cols,
            )

            df = self._base_price_df().copy()
            df["raw_signal"] = np.asarray(multi_out["signals"], dtype=np.int8)
            df["Signal"] = df["raw_signal"].copy()
            df["selected_setup_id"] = np.asarray(multi_out["selected_setup_id"], dtype=np.int32)
            df["selected_score"] = np.asarray(multi_out["selected_score"], dtype=np.float64)

            if lock_setup_id is not None:
                locked = df["selected_setup_id"] == int(lock_setup_id)
                df["Signal"] = np.where(locked, df["raw_signal"], 0)

            if setup_name_map is not None:
                df["selected_setup_name"] = df["selected_setup_id"].map(setup_name_map)
            else:
                local_map = {}
                for i, spec in enumerate(setup_specs):
                    name = spec.name if getattr(spec, "name", None) else f"setup_{i}"
                    local_map[i] = name
                df["selected_setup_name"] = df["selected_setup_id"].map(local_map)

            self.last_signal_df = df.copy()
            self.last_signal_col = "Signal"

            if plot:
                self._plot_signal_df(df, signal_col="Signal", crypto=crypto, start=start, end=end)

            if return_df:
                return df
            return df["Signal"].to_numpy(dtype=np.int8)

        raise ValueError("Provide either strategy_fn or setup_specs")

    def signal_generation_inspection(self, strategy_fn=None, signal_col="Signal", plot=False,
                                     crypto=False, return_df_signals=True, start=None, end=None, **kwargs):
        base_df = pd.DataFrame({
            "Open": self.opens, "High": self.highs,
            "Low":  self.lows,  "Close": self.closes, "ATR": self.atrs,
        }, index=self.bar_index)

        if strategy_fn is not None:
            df = strategy_fn(base_df.copy(), **kwargs)
            df = self._ensure_price_cols_for_inspection(df, base_df)
        else:
            ema1 = ema_njit(self.closes, self.cfg.period_1)
            ema2 = ema_njit(self.closes, self.cfg.period_2)
            _, _, sig = signals_ema_vs_close_njit(
                self.opens, self.closes, self.cfg.period_1, self.cfg.period_2
            )
            print('Automatic Fallback on default ema_close strategy; pass strategy_fn=my_strategy to use yours.')
            df = base_df.copy()
            df["EMA1"] = ema1; df["EMA2"] = ema2; df[signal_col] = sig

        resolved_signal_cols = self._resolve_signal_cols(df, signal_col)
        df, normalized_signal_col = self._ensure_signal_column(df, resolved_signal_cols)

        self.last_signal_df  = df.copy()
        self.last_signal_col = normalized_signal_col

        if plot:
            self._plot_signal_df(df, signal_col=resolved_signal_cols, crypto=crypto, start=start, end=end)

        if return_df_signals:
            return df
        return df[normalized_signal_col].to_numpy(dtype=np.int8)

    def _build_after_run_df(
        self,
        trades_df,
        full_df=False,
        window_before=200,
        window_after=50,
        entry_on_signal_close_price=False,
        source_df=None,
        bar_index=None,
        opens=None,
        highs=None,
        lows=None,
        closes=None,
        atrs=None,
    ):
        if source_df is not None:
            df = source_df.copy()
        elif self.last_signal_df is not None:
            df = self.last_signal_df.copy()
        else:
            df = pd.DataFrame({
                "Open": self.opens if opens is None else opens,
                "High": self.highs if highs is None else highs,
                "Low":  self.lows if lows is None else lows,
                "Close": self.closes if closes is None else closes,
                "ATR": self.atrs if atrs is None else atrs,
            }, index=self.bar_index if bar_index is None else bar_index)

        df["EntryTradeID"] = np.nan
        df["ExitTradeID"]  = np.nan
        long_id = 0
        short_id = 0
        trade_ids = []

        trades_df = trades_df.copy()

        if entry_on_signal_close_price:
            trades_df["plot_entry_idx"] = np.maximum(trades_df["entry_idx"].astype(int) - 1, 0)
        else:
            trades_df["plot_entry_idx"] = trades_df["entry_idx"].astype(int)

        for _, tr in trades_df.iterrows():
            if tr["side"] == 1:
                long_id += 1
                tid = long_id
            else:
                short_id += 1
                tid = -short_id

            trade_ids.append(tid)

            pei = int(tr["plot_entry_idx"])
            exi = int(tr["exit_idx"])

            df.iloc[pei, df.columns.get_loc("EntryTradeID")] = tid
            df.iloc[exi, df.columns.get_loc("ExitTradeID")]  = tid

        trades_df["trade_id"] = trade_ids

        if len(trades_df) == 0 or full_df:
            return df.copy(), trades_df

        first_idx = max(0, int(trades_df["plot_entry_idx"].min()) - window_before)
        last_idx  = min(len(df), int(trades_df["exit_idx"].max()) + window_after + 1)

        return df.iloc[first_idx:last_idx].copy(), trades_df

    @staticmethod
    def _crop_plot_df(df, start=None, end=None):
        plot_df = df.copy().sort_index()

        if start is not None:
            plot_df = plot_df.loc[pd.Timestamp(start):]
        if end is not None:
            plot_df = plot_df.loc[:pd.Timestamp(end)]

        if len(plot_df) == 0:
            raise ValueError("Plot window is empty after applying start/end.")

        return plot_df

    @staticmethod
    def _resolve_index_slice(index, start=None, end=None):
        idx = pd.DatetimeIndex(index)
        left = 0 if start is None else int(idx.searchsorted(pd.Timestamp(start), side="left"))
        right = len(idx) if end is None else int(idx.searchsorted(pd.Timestamp(end), side="right"))
        if left >= right:
            raise ValueError("Backtest window is empty after applying backtest_start/backtest_end.")
        return slice(left, right)

    @staticmethod
    def _ensure_price_cols_for_inspection(df, base_df):
        required_cols = ["Open", "High", "Low", "Close"]
        if all(col in df.columns for col in required_cols):
            return df

        missing_base = [col for col in required_cols if col not in base_df.columns]
        if missing_base:
            raise ValueError(
                "Unable to enrich inspection DataFrame with OHLC columns. "
                f"Missing in base_df: {missing_base}"
            )

        extra_cols = [col for col in df.columns if col not in base_df.columns]
        merged = base_df.copy()
        if extra_cols:
            merged = merged.join(df[extra_cols], how="left")
        return merged

    @staticmethod
    def _plot_signal_df(df, signal_col="Signal", title="Signal preparation", crypto=False, start=None, end=None):
        df = NJITEngine._crop_plot_df(df, start=start, end=end)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
        _, long_mask, short_mask = NJITEngine._extract_signal_masks(df, signal_col)
        long_signals = df[long_mask]
        short_signals = df[short_mask]
        fig.add_trace(go.Scatter(x=long_signals.index,  y=long_signals["Low"]  * 0.999,
            mode="markers", marker=dict(symbol="triangle-up",   size=8, color="lime"), name="Long signal"))
        fig.add_trace(go.Scatter(x=short_signals.index, y=short_signals["High"] * 1.001,
            mode="markers", marker=dict(symbol="triangle-down", size=8, color="red"),  name="Short signal"))
        rangebreaks = [] if crypto else [dict(bounds=["sat", "mon"])]
        fig.update_layout(title=title, template="plotly_dark", xaxis_rangeslider_visible=False,
            xaxis=dict(rangebreaks=rangebreaks), hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0), height=750)
        fig.show()

    @staticmethod
    def _plot_backtest(df, trades, title="Backtest results", crypto=False, start=None, end=None):
        df = NJITEngine._crop_plot_df(df, start=start, end=end)

        plot_trades = trades.copy()
        if start is not None:
            start_ts = pd.Timestamp(start)
            plot_trades = plot_trades[plot_trades["exit_time"] >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            plot_trades = plot_trades[plot_trades["entry_time"] <= end_ts]

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
        long_entries  = plot_trades[plot_trades["side"] == 1]
        short_entries = plot_trades[plot_trades["side"] == -1]
        fig.add_trace(go.Scatter(x=long_entries["entry_time"],  y=long_entries["entry"],
            mode="markers", marker=dict(symbol="triangle-up",   size=10, color="lime"), name="Long entry"))
        fig.add_trace(go.Scatter(x=short_entries["entry_time"], y=short_entries["entry"],
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),  name="Short entry"))
        for _, r in plot_trades.iterrows():
            color = "lime" if r["side"] == 1 else "red"
            fig.add_trace(go.Scatter(x=[r["entry_time"], r["exit_time"]], y=[r["entry"], r["exit"]],
                mode="lines", line=dict(color=color, width=1), opacity=0.5, showlegend=False))
            fig.add_trace(go.Scatter(x=[r["exit_time"]], y=[r["exit"]],
                mode="markers", marker=dict(symbol="x", size=9, color=color), showlegend=False))
        rangebreaks = [] if crypto else [dict(bounds=["sat", "mon"])]
        fig.update_layout(title=title, template="plotly_dark", xaxis_rangeslider_visible=False,
            xaxis=dict(rangebreaks=rangebreaks), hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0), height=700)
        fig.show()

    @staticmethod
    def _parse_session(s):
        if s is None: return -1
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    def _warmup(self):
        n = min(500, len(self.opens))
        sig = np.zeros(n, dtype=np.int8)
        sig[10] = 1
        sig[20] = -1

        selected_setup_id = np.full(n, -1, dtype=np.int32)
        selected_score = np.zeros(n, dtype=np.float64)
        features = np.zeros((n, 0), dtype=np.float64)

        state_per_pos_dummy = np.zeros((1, 1), dtype=np.float64)
        state_global_dummy  = np.zeros((1, 1), dtype=np.float64)
        regime_dummy        = np.zeros(n, dtype=np.int32)
        regime_ep_dummy     = np.full(1, -1, dtype=np.int32)
        regime_es_dummy     = np.full(1, -1, dtype=np.int32)

        use_exit_system = False
        profile_rt_matrix = np.zeros((1, N_EXIT_RT_COLS), dtype=np.float64)
        strategy_rt_matrix = np.zeros((1, N_EXIT_STRAT_RT_COLS), dtype=np.float64)

        setup_to_exit_profile = np.full(1, -1, dtype=np.int32)
        setup_to_exit_strategy = np.full(1, -1, dtype=np.int32)

        strategy_allowed_profiles = np.full((1, 1), -1, dtype=np.int32)
        strategy_allowed_counts = np.zeros(1, dtype=np.int32)

        backtest_njit(
            self.opens[:n], self.highs[:n], self.lows[:n], self.closes[:n], self.atrs[:n],
            sig, selected_setup_id, selected_score, features,
            use_exit_system, profile_rt_matrix, strategy_rt_matrix,
            setup_to_exit_profile, setup_to_exit_strategy,
            strategy_allowed_profiles, strategy_allowed_counts,
            self.minutes_of_day[:n], self.day_index[:n], self.day_of_week[:n],
            1,
            -1, -1, -1, -1, -1, -1,
            0.0, 0.0, False, 0.0, 1.0,
            False, 0.001, 0.01, 0,
            2.0, 1.0, True,
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), False, False, False,
            np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.float64), False, 1,
            0.0, 0.0, 0,
            0.0, 2.0,
            True, False, 0, 0, 0,
            1000, 10, True,
            0, 0, 1,
            0, 0, 1,
            False, False,
            0,
            0, -1,
            0, 0, 0,
            0,    # n_state_per_pos
            0,    # n_state_global
            0,    # n_strategies
            regime_dummy,
            regime_ep_dummy,
            regime_es_dummy,
            False,  # use_regime
            np.full(1, -1.0, dtype=np.float64),  # entry_limit_prices
            5,                                    # limit_expiry_bars
            np.full(1, -1.0, dtype=np.float64),  # tp_price_array
            np.full(1, -1.0, dtype=np.float64),  # sl_price_array
            True,                                 # check_filters_on_fill
            False,                                # has_limit_orders
            np.full((1, 1, 1), -1.0, dtype=np.float64),   # partial_rt_matrix
            np.full((1, 1, 1), -1.0, dtype=np.float64),   # pyramid_rt_matrix
            np.full((1, 1, 1), -1.0, dtype=np.float64),   # averaging_rt_matrix
            np.full((1, 1, 1), -1.0, dtype=np.float64),   # phase_rt_matrix
            np.full((1, 1, 1), -1.0, dtype=np.float64),   # rule_trigger_matrix
            np.full((1, 1, 1, 1), -1.0, dtype=np.float64),# rule_action_matrix
            np.zeros((1, N_STATEFUL_CFG_COLS), dtype=np.float64),  # stateful_cfg_rt
            0, 0, 0, 0, 0, 0,                              # max_*
            False, False, False, False, False, False,False       # has_*
        )

        print("NJITEngine — JIT warmup done ✓")

    def signals_ema(self, span1=None, span2=None, mode="close_vs_ema"):
        s1 = span1 or self.cfg.period_1
        s2 = span2 or self.cfg.period_2
        if mode == "close_vs_ema":
            _, _, sig = signals_ema_vs_close_njit(self.opens, self.closes, s1, s2)
        else:
            _, _, sig = signals_ema_cross_njit(self.closes, s1, s2)
        return sig

    # Helper prevent importing bundle function by hand 
    def run_bundle(self, bundle, **run_kwargs):
        return self.run(bundle=bundle, **run_kwargs)
    
    def prepare_signal_inputs(
        self,
        setup_specs,
        decision_cfg,
        include_price_cols=True,
        build_signal_df=True,
        regime: np.ndarray | None = None,
        regime_policy=None,
        regime_context: RegimeContext | None = None,
    ):
        if regime_context is None and regime is not None and regime_policy is not None:
            regime_context = build_regime_context(
                regime=regime,
                policy=regime_policy,
                setup_specs=setup_specs,
            )

        multi_out = self.prepare_multi_setup_signals(
            setup_specs=setup_specs,
            decision_cfg=decision_cfg,
            include_price_cols=include_price_cols,
            regime_context=regime_context,
        )

        signals = np.asarray(multi_out["signals"], dtype=np.int8)
        selected_setup_id = np.asarray(multi_out["selected_setup_id"], dtype=np.int32)
        selected_score = np.asarray(multi_out["selected_score"], dtype=np.float64)

        if all(k in multi_out for k in ("open", "high", "low", "close")):
            opens = np.asarray(multi_out["open"], dtype=np.float64)
            highs = np.asarray(multi_out["high"], dtype=np.float64)
            lows = np.asarray(multi_out["low"], dtype=np.float64)
            closes = np.asarray(multi_out["close"], dtype=np.float64)
        else:
            opens = np.asarray(self.opens, dtype=np.float64)
            highs = np.asarray(self.highs, dtype=np.float64)
            lows = np.asarray(self.lows, dtype=np.float64)
            closes = np.asarray(self.closes, dtype=np.float64)

        df_signal = None
        if build_signal_df:
            df_signal = self._base_price_df().copy()
            df_signal["Signal"] = signals
            df_signal["selected_setup_id"] = selected_setup_id
            df_signal["selected_score"] = selected_score

        return SignalPrep(
            multi_out=multi_out,
            signals=signals,
            selected_setup_id=selected_setup_id,
            selected_score=selected_score,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            df_signal=df_signal,
        )

    def prepare_bundle(
        self,
        setup_specs,
        decision_cfg,
        cfg=None,
        feature_specs=None,
        exit_profile_specs=None,
        setup_exit_binding=None,
        strategy_profile_binding=None,
        exit_strategy_specs=None,
        strategy_param_names=(),
        include_price_cols=True,
        volumes=None,
        regime: np.ndarray | None = None,
        regime_policy=None,
        regime_context: RegimeContext | None = None,
    ):
        from .backtest_bundle import prepare_backtest_bundle
        return prepare_backtest_bundle(
            engine=self,
            setup_specs=setup_specs,
            decision_cfg=decision_cfg,
            cfg=cfg,
            feature_specs=feature_specs,
            exit_profile_specs=exit_profile_specs,
            setup_exit_binding=setup_exit_binding,
            strategy_profile_binding=strategy_profile_binding,
            exit_strategy_specs=exit_strategy_specs,
            strategy_param_names=strategy_param_names,
            include_price_cols=include_price_cols,
            volumes=volumes,
            regime=regime,
            regime_policy=regime_policy,
            regime_context=regime_context,
        )

    def signals_from_strategy(self, strategy_fn, signal_col="Signal", **kwargs):
        df = pd.DataFrame({
            "Open": self.opens, "High": self.highs,
            "Low":  self.lows,  "Close": self.closes,
        }, index=self.bar_index)
        df_out = strategy_fn(df, **kwargs)
        if signal_col not in df_out.columns:
            raise ValueError(f"strategy_fn must return a DataFrame with '{signal_col}'")
        return df_out[signal_col].to_numpy(dtype=np.int8)

    def run(
        self,
        signals,use_exit_system=None,execution_context=None, features=None,bundle=None,cfg=None,
        tp_pct=None, sl_pct=None,
        use_atr_sl_tp=None, tp_atr_mult=None, sl_atr_mult=None,
        entry_delay=None,
        session_1=None, session_2=None, session_3=None,
        max_gap_signal=None, max_gap_entry=None,
        candle_size_filter=None, min_size_pct=None, max_size_pct=None,
        prev_candle_direction=None,
        allow_exit_on_entry_bar=None,
        exit_ema1=None, exit_ema2=None, use_ema1_tp=None, use_ema2_tp=None,
        use_ema_cross_tp=None,
        exit_signals=None, signal_tags=None,
        use_exit_signal=None, exit_delay=None,
        be_trigger_pct=None, be_offset_pct=None, be_delay_bars=None,
        trailing_trigger_pct=None, runner_trailing_mult=None,
        multi_entry=None, reverse_mode=None,
        me_max=None, me_period=None, me_reset_mode=None,
        cooldown_entries=None, cooldown_bars=None, cooldown_mode=None,
        entry_on_close=None,
        track_mae_mfe=None, hold_minutes=None, bar_duration_min=None,
        commission_pct=None, commission_per_lot_usd=None, contract_size=None,
        spread_pct=None, spread_abs=None, slippage_pct=None,
        alpha=None, period_freq=None,
        return_df_after=None, plot=None, crypto=None, start=None, end=None,
        backtest_start=None, backtest_end=None,
        full_df_after=None, window_before=None, window_after=None,entry_on_signal_close_price=None,
        max_holding_bars=None,
        forced_flat_frequency=None, forced_flat_time=None,
        max_tp=None, tp_period_mode=None, tp_period_bars=None,
        max_sl=None, sl_period_mode=None, sl_period_bars=None,
        selected_setup_id=None, selected_score=None,multi_setup_mode=None,    
        regime: np.ndarray | None = None, regime_policy = None,regime_context = None,setup_specs: list | None = None, entry_limit_prices=None,     
        limit_expiry_bars=5,
        tp_prices=None,
        sl_prices=None,
        check_filters_on_fill=True,sl_tp_be_priority=None, plot_results=None,label="",         
    ):
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError(
                "cfg must be provided either when creating NJITEngine or directly in run()."
            )


        # Initialiser avant le bloc bundle
        n_state_per_pos = 0
        n_state_global  = 0
        n_strategies    = 0
        
        #-- Bundle -- 
        if bundle is not None:
            signals = bundle.signals
            selected_setup_id = bundle.selected_setup_id
            selected_score = bundle.selected_score
            features = bundle.features

            if execution_context is None:
                execution_context = bundle.execution_context

            if use_exit_system is None:
                use_exit_system = bundle.execution_context is not None

            if use_exit_system:
                n_state_per_pos = execution_context.n_state_per_pos
                n_state_global  = execution_context.n_state_global
                n_strategies    = int(execution_context.strategy_rt_matrix.shape[0])

            else:
                n_state_per_pos = 0
                n_state_global  = 0
                n_strategies    = 0
        #-- 
        if signals is None:
            raise ValueError("signals must be provided, either directly or via bundle")
        
        
        tp_pct               = tp_pct               if tp_pct               is not None else cfg.tp_pct
        sl_pct               = sl_pct               if sl_pct               is not None else cfg.sl_pct
        use_atr_sl_tp        = use_atr_sl_tp        if use_atr_sl_tp        is not None else cfg.use_atr_sl_tp
        tp_atr_mult          = tp_atr_mult          if tp_atr_mult          is not None else cfg.tp_atr_mult
        sl_atr_mult          = sl_atr_mult          if sl_atr_mult          is not None else cfg.sl_atr_mult
        entry_delay          = entry_delay          if entry_delay          is not None else cfg.entry_delay
        session_1            = session_1            if session_1            is not None else cfg.session_1
        session_2            = session_2            if session_2            is not None else cfg.session_2
        session_3            = session_3            if session_3            is not None else cfg.session_3
        max_gap_signal       = max_gap_signal       if max_gap_signal       is not None else cfg.max_gap_signal
        max_gap_entry        = max_gap_entry        if max_gap_entry        is not None else cfg.max_gap_entry
        candle_size_filter   = candle_size_filter   if candle_size_filter   is not None else cfg.candle_size_filter
        min_size_pct         = min_size_pct         if min_size_pct         is not None else cfg.min_size_pct
        max_size_pct         = max_size_pct         if max_size_pct         is not None else cfg.max_size_pct
        prev_candle_direction= prev_candle_direction if prev_candle_direction is not None else cfg.prev_candle_direction
        allow_exit_on_entry_bar = allow_exit_on_entry_bar if allow_exit_on_entry_bar is not None else cfg.allow_exit_on_entry_bar
        use_ema1_tp          = use_ema1_tp          if use_ema1_tp          is not None else cfg.use_ema1_tp
        use_ema2_tp          = use_ema2_tp          if use_ema2_tp          is not None else cfg.use_ema2_tp
        use_ema_cross_tp     = use_ema_cross_tp     if use_ema_cross_tp     is not None else cfg.use_ema_cross_tp
        use_exit_signal      = use_exit_signal      if use_exit_signal      is not None else cfg.use_exit_signal
        exit_delay           = exit_delay           if exit_delay           is not None else cfg.exit_delay
        be_trigger_pct       = be_trigger_pct       if be_trigger_pct       is not None else cfg.be_trigger_pct
        be_offset_pct        = be_offset_pct        if be_offset_pct        is not None else cfg.be_offset_pct
        be_delay_bars        = be_delay_bars        if be_delay_bars        is not None else cfg.be_delay_bars
        trailing_trigger_pct = trailing_trigger_pct if trailing_trigger_pct is not None else cfg.trailing_trigger_pct
        runner_trailing_mult = runner_trailing_mult if runner_trailing_mult is not None else cfg.runner_trailing_mult
        multi_entry          = multi_entry          if multi_entry          is not None else cfg.multi_entry
        reverse_mode         = reverse_mode         if reverse_mode         is not None else cfg.reverse_mode
        me_max               = me_max               if me_max               is not None else cfg.me_max
        me_period            = me_period            if me_period            is not None else cfg.me_period
        me_reset_mode        = me_reset_mode        if me_reset_mode        is not None else cfg.me_reset_mode
        cooldown_entries     = cooldown_entries     if cooldown_entries     is not None else cfg.cooldown_entries
        cooldown_bars        = cooldown_bars        if cooldown_bars        is not None else cfg.cooldown_bars
        cooldown_mode        = cooldown_mode        if cooldown_mode        is not None else cfg.cooldown_mode
        entry_on_close       = entry_on_close       if entry_on_close       is not None else cfg.entry_on_close
        track_mae_mfe        = track_mae_mfe        if track_mae_mfe        is not None else cfg.track_mae_mfe
        hold_minutes         = hold_minutes         if hold_minutes         is not None else cfg.hold_minutes
        bar_duration_min     = bar_duration_min     if bar_duration_min     is not None else cfg.bar_duration_min
        commission_pct       = commission_pct       if commission_pct       is not None else cfg.commission_pct
        commission_per_lot_usd = commission_per_lot_usd if commission_per_lot_usd is not None else cfg.commission_per_lot_usd
        contract_size        = contract_size        if contract_size        is not None else cfg.contract_size
        spread_pct           = spread_pct           if spread_pct           is not None else cfg.spread_pct
        spread_abs           = spread_abs           if spread_abs           is not None else cfg.spread_abs
        slippage_pct         = slippage_pct         if slippage_pct         is not None else cfg.slippage_pct
        alpha                = alpha                if alpha                is not None else cfg.alpha
        period_freq          = period_freq          if period_freq          is not None else cfg.period_freq
        return_df_after      = return_df_after      if return_df_after      is not None else cfg.return_df_after
        plot                 = plot                 if plot                 is not None else cfg.plot
        crypto               = crypto               if crypto               is not None else cfg.crypto
        full_df_after        = full_df_after        if full_df_after        is not None else cfg.full_df_after
        window_before        = window_before        if window_before        is not None else cfg.window_before
        window_after         = window_after         if window_after         is not None else cfg.window_after
        entry_on_signal_close_price = entry_on_signal_close_price if entry_on_signal_close_price is not None else cfg.entry_on_signal_close_price
        max_holding_bars            = max_holding_bars if max_holding_bars is not None else cfg.max_holding_bars
        forced_flat_frequency       = forced_flat_frequency if forced_flat_frequency is not None else cfg.forced_flat_frequency
        forced_flat_time            = forced_flat_time if forced_flat_time is not None else cfg.forced_flat_time
        max_tp                      = max_tp if max_tp is not None else cfg.max_tp
        tp_period_mode              = tp_period_mode if tp_period_mode is not None else cfg.tp_period_mode
        tp_period_bars              = tp_period_bars if tp_period_bars is not None else cfg.tp_period_bars
        max_sl                      = max_sl if max_sl is not None else cfg.max_sl
        sl_period_mode              = sl_period_mode if sl_period_mode is not None else cfg.sl_period_mode
        sl_period_bars              = sl_period_bars if sl_period_bars is not None else cfg.sl_period_bars
        sl_tp_be_priority = sl_tp_be_priority if sl_tp_be_priority is not None else cfg.sl_tp_be_priority
        plot_results                = plot_results if plot_results is not None else cfg.plot_results
        plot_results = plot_results if plot_results is not None else cfg.plot_results


        # On Off exit system
        if use_exit_system and execution_context is None:
            raise ValueError("execution_context must be provided when use_exit_system=True")

        if use_exit_system:
            profile_rt_matrix = np.asarray(execution_context.profile_rt_matrix, dtype=np.float64)
            strategy_rt_matrix = np.asarray(execution_context.strategy_rt_matrix, dtype=np.float64)

            setup_to_exit_profile = np.asarray(execution_context.setup_to_exit_profile, dtype=np.int32)
            setup_to_exit_strategy = np.asarray(execution_context.setup_to_exit_strategy, dtype=np.int32)

            strategy_allowed_profiles = np.asarray(execution_context.strategy_allowed_profiles, dtype=np.int32)
            strategy_allowed_counts = np.asarray(execution_context.strategy_allowed_counts, dtype=np.int32)
             
            n_state_per_pos = execution_context.n_state_per_pos
            n_state_global  = execution_context.n_state_global
            n_strategies    = int(execution_context.strategy_rt_matrix.shape[0])

        else:
            profile_rt_matrix = np.zeros((1, N_EXIT_RT_COLS), dtype=np.float64)
            strategy_rt_matrix = np.zeros((1, N_EXIT_STRAT_RT_COLS), dtype=np.float64)

            setup_to_exit_profile = np.full(1, -1, dtype=np.int32)
            setup_to_exit_strategy = np.full(1, -1, dtype=np.int32)

            strategy_allowed_profiles = np.full((1, 1), -1, dtype=np.int32)
            strategy_allowed_counts = np.zeros(1, dtype=np.int32)


        if features is None:
            feature_context = CompiledFeatures(
                matrix=np.zeros((len(signals), 0), dtype=np.float64),
                col_map={},
                names=(),
                index=self.bar_index,
                meta={"source": "empty"},
            )
        else:
            feature_context = to_compiled_features(
                features,
                align_index=self.bar_index,
                meta={"source": type(features).__name__},
            )

        features = np.asarray(feature_context.matrix, dtype=np.float64)

        if len(features) != len(signals):
            raise ValueError("features must have same number of rows as signals")

        self.last_feature_context = feature_context
        
        n = len(signals)
        has_limit_orders = entry_limit_prices is not None

        _entry_limits = np.asarray(entry_limit_prices, dtype=np.float64) if has_limit_orders else np.full(1, -1.0, dtype=np.float64)
        _tp_prices    = np.asarray(tp_prices, dtype=np.float64) if tp_prices is not None else np.full(1, -1.0, dtype=np.float64)
        _sl_prices    = np.asarray(sl_prices, dtype=np.float64) if sl_prices is not None else np.full(1, -1.0, dtype=np.float64)

        # Préparer les matrices AVANT l'appel
        _partial_rt   = np.asarray(execution_context.partial_rt_matrix,   dtype=np.float64) if use_exit_system else np.full((1,1,1), -1.0, dtype=np.float64)
        _pyramid_rt   = np.asarray(execution_context.pyramid_rt_matrix,   dtype=np.float64) if use_exit_system else np.full((1,1,1), -1.0, dtype=np.float64)
        _averaging_rt = np.asarray(execution_context.averaging_rt_matrix, dtype=np.float64) if use_exit_system else np.full((1,1,1), -1.0, dtype=np.float64)
        _phase_rt     = np.asarray(execution_context.phase_rt_matrix,     dtype=np.float64) if use_exit_system else np.full((1,1,1), -1.0, dtype=np.float64)
        _rule_trig    = np.asarray(execution_context.rule_trigger_matrix,  dtype=np.float64) if use_exit_system else np.full((1,1,1), -1.0, dtype=np.float64)
        _rule_act     = np.asarray(execution_context.rule_action_matrix,   dtype=np.float64) if use_exit_system else np.full((1,1,1,1), -1.0, dtype=np.float64)
        _stateful_cfg = np.asarray(execution_context.stateful_cfg_rt,      dtype=np.float64) if use_exit_system else np.zeros((1, N_STATEFUL_CFG_COLS), dtype=np.float64)

        _max_partial   = execution_context.max_partial_levels   if use_exit_system else 0
        _max_pyramid   = execution_context.max_pyramid_levels   if use_exit_system else 0
        _max_avg       = execution_context.max_avg_levels       if use_exit_system else 0
        _max_phases    = execution_context.max_phases           if use_exit_system else 0
        _max_rules     = execution_context.max_rules            if use_exit_system else 0
        _max_actions   = execution_context.max_actions_per_rule if use_exit_system else 0
        _has_partial   = execution_context.has_partial          if use_exit_system else False
        _has_pyramid   = execution_context.has_pyramid          if use_exit_system else False
        _has_averaging = execution_context.has_averaging        if use_exit_system else False
        _has_phases    = execution_context.has_phases           if use_exit_system else False
        _has_rules     = execution_context.has_rules            if use_exit_system else False
        _has_stateful  = execution_context.has_stateful_cfg     if use_exit_system else False

        signals = np.asarray(signals, dtype=np.int8)

         #----- Multi Set-up mode ------
        multi_setup_mode = multi_setup_mode if multi_setup_mode is not None else cfg.multi_setup_mode

        if multi_setup_mode:
            if selected_setup_id is None or selected_score is None:
                raise ValueError("In multi_setup_mode, selected_setup_id and selected_score must be provided")

            selected_setup_id = np.asarray(selected_setup_id, dtype=np.int32)
            selected_score = np.asarray(selected_score, dtype=np.float64)

            if len(selected_setup_id) != len(signals):
                raise ValueError("selected_setup_id must have same length as signals")
            if len(selected_score) != len(signals):
                raise ValueError("selected_score must have same length as signals")
            
        else:
            if selected_setup_id is None:
                selected_setup_id = np.full(len(signals), -1, dtype=np.int32)
            else:
                selected_setup_id = np.asarray(selected_setup_id, dtype=np.int32)

            if selected_score is None:
                selected_score = np.zeros(len(signals), dtype=np.float64)
            else:
                selected_score = np.asarray(selected_score, dtype=np.float64)

        if use_exit_system:
            invalid_mask = (signals != 0) & (selected_setup_id < 0)
            if invalid_mask.any():
                raise ValueError(
                    "use_exit_system=True requires valid selected_setup_id on all non-zero signal bars"
                )

        s1_start = self._parse_session(session_1[0]) if session_1 else -1
        s1_end   = self._parse_session(session_1[1]) if session_1 else -1
        s2_start = self._parse_session(session_2[0]) if session_2 else -1
        s2_end   = self._parse_session(session_2[1]) if session_2 else -1
        s3_start = self._parse_session(session_3[0]) if session_3 else -1
        s3_end   = self._parse_session(session_3[1]) if session_3 else -1

        _e_ema1 = exit_ema1    if exit_ema1    is not None else np.empty(0, dtype=np.float64)
        _e_ema2 = exit_ema2    if exit_ema2    is not None else np.empty(0, dtype=np.float64)
        _e_sig  = exit_signals if exit_signals is not None else np.zeros(0, dtype=np.int8)
        _s_tags = signal_tags  if signal_tags  is not None else np.zeros(0, dtype=np.float64)
        
        if entry_on_close and entry_on_signal_close_price:
            raise ValueError("entry_on_close and entry_on_signal_close_price cannot both be True")
        if forced_flat_frequency is None or forced_flat_time is None:
            forced_flat_mode = 0
            forced_flat_minute = -1
        else:
            if forced_flat_frequency == "day":
                forced_flat_mode = 1
            elif forced_flat_frequency == "weekend":
                forced_flat_mode = 2
            else:
                raise ValueError('forced_flat_frequency must be None, "day" or "weekend"')
            forced_flat_minute = self._parse_session(forced_flat_time)

        # Build regime context
        if regime is not None and regime_policy is not None:
            if setup_specs is None:
                raise ValueError("setup_specs required when regime_policy is provided")
            regime_context = build_regime_context(
                regime=regime,
                policy=regime_policy,
                setup_specs=setup_specs,
            )

        use_regime = regime_context is not None
        n_setups_for_regime = len(setup_specs) if setup_specs else 1

        if use_regime:
            regime_arr                 = regime_context.regime
            regime_multipliers         = regime_context.score_multipliers
            regime_exit_prof_override  = regime_context.exit_profile_override
            regime_exit_strat_override = regime_context.exit_strategy_override
        else:
            regime_arr                 = np.zeros(len(signals), dtype=np.int32)
            regime_multipliers         = np.ones((1, n_setups_for_regime, 2), dtype=np.float64)
            regime_exit_prof_override  = np.full(1, -1, dtype=np.int32)
            regime_exit_strat_override = np.full(1, -1, dtype=np.int32)

        window_slice = self._resolve_index_slice(
            self.bar_index,
            start=backtest_start,
            end=backtest_end,
        )

        bar_index = self.bar_index[window_slice]
        opens = self.opens[window_slice]
        highs = self.highs[window_slice]
        lows = self.lows[window_slice]
        closes = self.closes[window_slice]
        atrs = self.atrs[window_slice]
        minutes_of_day = self.minutes_of_day[window_slice]
        day_index = self.day_index[window_slice]
        day_of_week = self.day_of_week[window_slice]

        signals = signals[window_slice]
        selected_setup_id = selected_setup_id[window_slice]
        selected_score = selected_score[window_slice]
        features = features[window_slice]
        regime_arr = regime_arr[window_slice]

        if has_limit_orders:
            _entry_limits = _entry_limits[window_slice]
        if tp_prices is not None:
            _tp_prices = _tp_prices[window_slice]
        if sl_prices is not None:
            _sl_prices = _sl_prices[window_slice]

        if _e_ema1.size > 0:
            _e_ema1 = np.asarray(_e_ema1, dtype=np.float64)[window_slice]
        if _e_ema2.size > 0:
            _e_ema2 = np.asarray(_e_ema2, dtype=np.float64)[window_slice]
        if _e_sig.size > 0:
            _e_sig = np.asarray(_e_sig, dtype=np.int8)[window_slice]
        if _s_tags.size > 0:
            _s_tags = np.asarray(_s_tags, dtype=np.float64)[window_slice]

        source_signal_df = None
        if self.last_signal_df is not None:
            source_signal_df = self.last_signal_df.iloc[window_slice].copy()

        rets, sides, entry_idx, exit_idx, reasons, exit_prices, mae, mfe, \
        trade_setup_ids, trade_selected_score, trade_exit_profile_ids, \
        trade_exit_strategy_ids, trade_regime_ids, trade_sizes, \
        trade_phase, trade_n_tp_hit, trade_add_count, \
        trade_remaining, trade_avg_entry, trade_bars, \
        trade_group_ids, \
        phase_ev_idx, phase_ev_trade, phase_ev_group, \
        phase_ev_from, phase_ev_to, phase_ev_profile, phase_ev_side = backtest_njit(
            opens, highs, lows, closes, atrs,
            signals, selected_setup_id, selected_score, features,
            use_exit_system, profile_rt_matrix, strategy_rt_matrix,
            setup_to_exit_profile, setup_to_exit_strategy,
            strategy_allowed_profiles, strategy_allowed_counts,
            minutes_of_day, day_index, day_of_week,
            entry_delay, s1_start, s1_end, s2_start, s2_end, s3_start, s3_end,
            max_gap_signal, max_gap_entry, candle_size_filter, min_size_pct, max_size_pct,
            prev_candle_direction, tp_pct, sl_pct, use_atr_sl_tp,
            tp_atr_mult, sl_atr_mult, allow_exit_on_entry_bar,
            _e_ema1, _e_ema2, use_ema1_tp, use_ema2_tp, use_ema_cross_tp,
            _e_sig, _s_tags, use_exit_signal, exit_delay,
            be_trigger_pct, be_offset_pct, be_delay_bars,
            trailing_trigger_pct, runner_trailing_mult,
            multi_entry, reverse_mode,
            me_max, me_period, me_reset_mode,
            self.MAX_TRADES, self.MAX_POS, track_mae_mfe,
            cooldown_entries, cooldown_bars, cooldown_mode,
            entry_on_close, entry_on_signal_close_price,
            max_holding_bars,
            forced_flat_mode, forced_flat_minute,
            max_tp, tp_period_mode, tp_period_bars,
            max_sl, sl_period_mode, sl_period_bars,
            n_state_per_pos, n_state_global, n_strategies,
            regime_arr, regime_exit_prof_override, regime_exit_strat_override, use_regime,
            _entry_limits, limit_expiry_bars, _tp_prices, _sl_prices, check_filters_on_fill, has_limit_orders, _partial_rt,
            _pyramid_rt,
            _averaging_rt,
            _phase_rt,
            _rule_trig,
            _rule_act,
            _stateful_cfg,
            _max_partial,
            _max_pyramid,
            _max_avg,
            _max_phases,
            _max_rules,
            _max_actions,
            _has_partial,
            _has_pyramid,
            _has_averaging,
            _has_phases,
            _has_rules,
            _has_stateful,
            sl_tp_be_priority,
        )

        # Construire phase_events_df
        if len(phase_ev_idx) > 0:
            phase_events_df = pd.DataFrame({
                "event_idx":       phase_ev_idx,
                "entry_idx":       phase_ev_trade,
                "group_id":        phase_ev_group,
                "phase_from":      phase_ev_from,
                "phase_to":        phase_ev_to,
                "exit_profile_id": phase_ev_profile,
                "side":            phase_ev_side,
                "event_time":      bar_index[phase_ev_idx],
                "event_type":      "PHASE_CHANGE",
                "reason":          "PHASE_CHANGE",
                "return":          np.nan,
            })
        else:
            phase_events_df = pd.DataFrame()

        if entry_on_signal_close_price:
            trade_entry_prices = closes[np.maximum(entry_idx - 1, 0)]
        elif entry_on_close:
            trade_entry_prices = closes[entry_idx]
        else:
            trade_entry_prices = opens[entry_idx]

        metrics = compute_metrics_full(
            rets, sides, entry_idx, exit_idx, reasons, exit_prices, mae, mfe,
            bar_index,
            highs=highs, lows=lows,
            hold_minutes=hold_minutes, bar_duration_min=bar_duration_min,
            commission_pct=commission_pct,
            commission_per_lot_usd=commission_per_lot_usd,
            contract_size=contract_size,
            spread_pct=spread_pct, spread_abs=spread_abs,
            slippage_pct=slippage_pct, alpha=alpha, period_freq=period_freq,
            entry_on_signal_close_price=entry_on_signal_close_price,
            trade_entry_prices=trade_entry_prices,
            trade_setup_ids=trade_setup_ids if multi_setup_mode else None,
            trade_selected_score=trade_selected_score if multi_setup_mode else None,
            trade_exit_profile_ids=trade_exit_profile_ids if use_exit_system else None,
            trade_exit_strategy_ids=trade_exit_strategy_ids if use_exit_system else None,
            trade_regime_ids=trade_regime_ids if use_exit_system else None,
            trade_sizes=trade_sizes,
            trade_phase=trade_phase,
            trade_n_tp_hit=trade_n_tp_hit,
            trade_add_count=trade_add_count,
            trade_remaining=trade_remaining,
            trade_avg_entry=trade_avg_entry,
            trade_bars=trade_bars,
            trade_group_ids=trade_group_ids,
            phase_events_df=phase_events_df,
        )
        # -- security if no trades ----
        if metrics is None:
            return rets, {"trades_df": pd.DataFrame()}
        
        if entry_on_signal_close_price:
            metrics["trades_df"]["entry"] = closes[np.maximum(entry_idx - 1, 0)]
        elif entry_on_close:
            metrics["trades_df"]["entry"] = closes[entry_idx]
        else:
            metrics["trades_df"]["entry"] = opens[entry_idx]
        metrics["trades_df"]["exit"]      = exit_prices
        metrics["trades_df"]["entry_idx"] = entry_idx
        metrics["trades_df"]["exit_idx"]  = exit_idx
        metrics["backtest_start"] = bar_index[0]
        metrics["backtest_end"] = bar_index[-1]

        # Il faudrait que trades_df contienne un plot_entry_idx ou que _build_after_run_df() sache quel mode d’entrée a été utilisé.
        if return_df_after:
            df_after, trades_df_annot = self._build_after_run_df(
                metrics["trades_df"],
                full_df=full_df_after,
                window_before=window_before,
                window_after=window_after,
                entry_on_signal_close_price=entry_on_signal_close_price,
                source_df=source_signal_df,
                bar_index=bar_index,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                atrs=atrs,
            )
            metrics["trades_df"] = trades_df_annot
            metrics["df_after"]  = df_after

        if plot:
            df_price = pd.DataFrame({
                "Open": opens, "High": highs,
                "Low":  lows,  "Close": closes,
            }, index=bar_index)
            self._plot_backtest(df_price, metrics["trades_df"], crypto=crypto, start=start, end=end)

        if plot_results:
            _plot_results(metrics, label=label)

        return rets, metrics

    def run_with_inspection(self, signals, signal_df=None, plot_before=False, plot_after=False,
                            signal_col="Signal", crypto=False, start=None, end=None, **run_kwargs):
        if signal_df is not None and plot_before:
            self._plot_signal_df(signal_df, signal_col=signal_col, crypto=crypto, start=start, end=end)

        rets, metrics = self.run(signals, **run_kwargs)
        trades_df = metrics["trades_df"].copy()

        if "entry" not in trades_df.columns:
            raise ValueError("trades_df must contain 'entry' and 'exit' columns for plotting.")

        if plot_after:
            df_price = pd.DataFrame({
                "Open": self.opens, "High": self.highs,
                "Low":  self.lows,  "Close": self.closes,
            }, index=self.bar_index)
            self._plot_backtest(df_price, trades_df, crypto=crypto, start=start, end=end)

        return rets, metrics

    def _base_price_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Open": self.opens,
            "High": self.highs,
            "Low": self.lows,
            "Close": self.closes,
            "ATR": self.atrs,
        }, index=self.bar_index)

    def build_trade_context_engine(
        self,
        trades_df: pd.DataFrame,
        extra_context_df: pd.DataFrame | None = None,
        include_default_context: bool = True,
    ) -> TradeContextEngine:
        price_df = self._base_price_df()
        context_df = None

        if include_default_context:
            context_df = build_default_context_df(
                price_df=price_df,
                extra_context=extra_context_df,
            )
        elif extra_context_df is not None:
            if not extra_context_df.index.equals(price_df.index):
                raise ValueError("extra_context_df.index must exactly match the engine/bar index")
            context_df = extra_context_df.copy()

        return TradeContextEngine(
            trades_df=trades_df,
            price_df=price_df,
            context_df=context_df,
        )

    def prepare_multi_setup_signals(
        self,
        setup_specs: list[SetupSpec],
        decision_cfg: DecisionConfig,
        include_price_cols: bool = True,
        regime_context: RegimeContext | None = None,
    ) -> dict[str, np.ndarray]:
        if len(setup_specs) == 0:
            raise ValueError("setup_specs cannot be empty")

        df = self._base_price_df() if include_price_cols else pd.DataFrame(index=self.bar_index)

        setup_dfs: list[pd.DataFrame] = []
        for k, spec in enumerate(setup_specs):
            setup_name = spec.name if spec.name else f"setup[{k}]"
            sdf = spec.fn(df.copy(), **spec.params)
            _validate_setup_df(sdf, expected_index=self.bar_index, setup_name=setup_name)
            setup_dfs.append(sdf)

        return aggregate_and_decide(
            setup_dfs=setup_dfs,
            decision_cfg=decision_cfg,
            regime_context=regime_context,
        )

    def enrich_trades_df_with_context(
        self,
        metrics: dict | None = None,
        trades_df: pd.DataFrame | None = None,
        extra_context_df: pd.DataFrame | None = None,
        include_default_context: bool = True,
        entry_cols: list[str] | None = None,
        exit_cols: list[str] | None = None,
        path_specs=None,
        feature_specs=None,
        inplace_metrics: bool = False,
    ) -> pd.DataFrame:
        if trades_df is None:
            if metrics is None:
                raise ValueError("Provide either trades_df or metrics containing trades_df")
            if "trades_df" not in metrics:
                raise ValueError("metrics must contain 'trades_df'")
            trades_df = metrics["trades_df"]

        tce = self.build_trade_context_engine(
            trades_df=trades_df,
            extra_context_df=extra_context_df,
            include_default_context=include_default_context,
        )

        enriched = trades_df.copy()

        if entry_cols is not None:
            tce.trades_df = enriched
            enriched = tce.attach_entry_context(cols=entry_cols)

        if exit_cols is not None:
            tce.trades_df = enriched
            enriched = tce.attach_exit_context(cols=exit_cols)

        if path_specs is not None and len(path_specs) > 0:
            tce.trades_df = enriched
            enriched = tce.attach_path_aggregations(path_specs)

        if feature_specs is not None and len(feature_specs) > 0:
            tce.trades_df = enriched
            enriched = tce.compute_trade_features(feature_specs)

        if inplace_metrics and metrics is not None:
            metrics["trades_df"] = enriched

        return enriched

        def grid_search(self, signals,
                        tp_values, sl_values,
                        commission_pct=0.0, commission_per_lot_usd=0.0, contract_size=1.0,
                        spread_pct=0.0, slippage_pct=0.0,
                        n_trades_per_year=None,
                        entry_delay=None, session_1=None, session_2=None, session_3=None,
                        use_session=False, max_gap_size=None,
                        candle_size_filter=None, min_size_pct=None,
                        max_size_pct=None, prev_candle_direction=None,
                        use_atr_sl_tp=0, tp_atr_mult=2.0, sl_atr_mult=1.0,
                        allow_exit_on_entry_bar=None, multi_entry=None,
                        me_max=0, me_period=0, me_reset_mode=0,
                        be_trigger_pct=None, be_offset_pct=None, be_delay_bars=None,
                        trailing_trigger_pct=None, runner_trailing_mult=None,
                        sort_by="sharpe"):
            cfg = self.cfg
            entry_delay           = entry_delay           if entry_delay           is not None else cfg.entry_delay
            max_gap_size          = max_gap_size          if max_gap_size          is not None else 0.0
            candle_size_filter    = candle_size_filter    if candle_size_filter    is not None else cfg.candle_size_filter
            min_size_pct          = min_size_pct          if min_size_pct          is not None else cfg.min_size_pct
            max_size_pct          = max_size_pct          if max_size_pct          is not None else cfg.max_size_pct
            prev_candle_direction = prev_candle_direction if prev_candle_direction is not None else cfg.prev_candle_direction
            allow_exit_on_entry_bar = allow_exit_on_entry_bar if allow_exit_on_entry_bar is not None else cfg.allow_exit_on_entry_bar
            multi_entry           = multi_entry           if multi_entry           is not None else cfg.multi_entry
            be_trigger_pct        = be_trigger_pct        if be_trigger_pct        is not None else cfg.be_trigger_pct
            be_offset_pct         = be_offset_pct         if be_offset_pct         is not None else cfg.be_offset_pct
            be_delay_bars         = be_delay_bars         if be_delay_bars         is not None else cfg.be_delay_bars
            trailing_trigger_pct  = trailing_trigger_pct  if trailing_trigger_pct  is not None else cfg.trailing_trigger_pct
            runner_trailing_mult  = runner_trailing_mult  if runner_trailing_mult  is not None else cfg.runner_trailing_mult

            s1_start = self._parse_session(session_1[0]) if session_1 else -1
            s1_end   = self._parse_session(session_1[1]) if session_1 else -1
            s2_start = self._parse_session(session_2[0]) if session_2 else -1
            s2_end   = self._parse_session(session_2[1]) if session_2 else -1
            s3_start = self._parse_session(session_3[0]) if session_3 else -1
            s3_end   = self._parse_session(session_3[1]) if session_3 else -1

            if commission_per_lot_usd > 0.0:
                ref_price = max(float(np.nanmean(self.closes)), 1e-12)
                cost = commission_per_lot_usd / (contract_size * ref_price) + spread_pct + slippage_pct * 2
            else:
                cost = commission_pct * 2 + spread_pct + slippage_pct * 2

            if n_trades_per_year is None:
                n_days  = (self.bar_index[-1] - self.bar_index[0]).days
                n_years = max(n_days / 365, 0.1)
                n_trades_per_year = float(int((signals != 0).sum())) / n_years

            #results = grid_search_njit(
            #    self.opens, self.highs, self.lows, self.closes, self.atrs,
            #    signals, self.minutes_of_day,
            #    tp_values.astype(np.float64), sl_values.astype(np.float64),
            #    cost, n_trades_per_year,
            #    entry_delay, s1_start, s1_end, s2_start, s2_end, s3_start, s3_end,
            #    max_gap_size, candle_size_filter, min_size_pct, max_size_pct,
            #    prev_candle_direction, use_atr_sl_tp, tp_atr_mult, sl_atr_mult,
            #    allow_exit_on_entry_bar, multi_entry,
            #    be_trigger_pct, be_offset_pct, be_delay_bars,
            #    trailing_trigger_pct, runner_trailing_mult,
            #    me_max, me_period, me_reset_mode, self.MAX_TRADES, self.MAX_POS, False,
            #)

            #df = pd.DataFrame(
            #    results,
            #    columns=["tp", "sl", "sharpe", "win_rate", "mdd", "profit_factor", "cum_return"]
            #)
            #return df.sort_values(sort_by, ascending=False).reset_index(drop=True)
