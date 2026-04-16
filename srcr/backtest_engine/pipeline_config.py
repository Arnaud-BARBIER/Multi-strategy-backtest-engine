from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

import pandas as pd

@dataclass(slots=True)
class BacktestConfig:
    # Multi_setup structure activation
    multi_setup_mode: bool = True

    # Data / preprocessing
    timezone_shift: int = 1
    atr_period: int = 14

    # Default signal params
    period_1: int = 50
    period_2: int = 100

    # Entry / signal execution
    entry_delay: int = 1
    session_1: Optional[Tuple[str, str]] = None
    session_2: Optional[Tuple[str, str]] = None
    session_3: Optional[Tuple[str, str]] = None

    max_gap_signal: float = 0.0
    max_gap_entry: float = 0.0

    candle_size_filter: bool = False
    min_size_pct: float = 0.0
    max_size_pct: float = 1.0
    prev_candle_direction: bool = False

    multi_entry: bool = True
    reverse_mode: bool = False

    sl_tp_be_priority: bool = False # met le sl tp be trailing avant partials, pyramiding... 

    # Cooldown
    cooldown_entries: int = 0   # nb entrées avant déclenchement du cooldown
    cooldown_bars:    int = 0   # durée du cooldown en barres
    cooldown_mode:    int = 1   # 1=indépendant, 2=reset session, 3=reset jour

    # Entry cap logic
    me_max: int = 0
    me_period: int = 0
    me_reset_mode: int = 0   # 0=off, 1=day, 2=session, 3=rolling bars

    # Entry price mode
    entry_on_close: bool = False  # False=open[i] (défaut), True=close[i]
    entry_on_signal_close_price: bool= False #True=close[i-1]

    # Exit — fixed / ATR TP-SL
    tp_pct: float = 0.01
    sl_pct: float = 0.005
    use_atr_sl_tp: int = 0
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    allow_exit_on_entry_bar: bool = True

    # Exit — EMA mode
    use_ema1_tp: bool = False
    use_ema2_tp: bool = False
    use_ema_cross_tp: bool = False

    # Exit — external signal
    use_exit_signal: bool = False
    exit_delay: int = 1

    # Break-even
    be_trigger_pct: float = 0.0
    be_offset_pct: float = 0.0
    be_delay_bars: int = 0

    # Runner trailing
    trailing_trigger_pct: float = 0.0
    runner_trailing_mult: float = 2.0

    # Metrics defaults
    track_mae_mfe: bool = True
    hold_minutes: int = 0
    bar_duration_min: int = 5
    commission_pct: float = 0.0
    commission_per_lot_usd: float = 0.0
    contract_size: float = 1.0
    spread_pct: float = 0.0
    spread_abs: float = 0.0
    slippage_pct: float = 0.0
    alpha: float = 5.0
    period_freq: str = "ME"

    # Inspection / plotting defaults
    return_df_after: bool = False
    plot: bool = False
    plot_results: bool = False
    crypto: bool = False
    full_df_after: bool = False
    window_before: int = 200
    window_after: int = 50

    # Trade lifetime / forced flat / TP quota
    max_holding_bars: int = 0

    forced_flat_frequency: Optional[str] = None   # None, "day", "weekend"
    forced_flat_time: Optional[str] = None        # ex: "21:30"

    max_tp: int = 0
    tp_period_mode: int = 0   # 0=off, 1=day, 2=session, 3=rolling bars
    tp_period_bars: int = 0
    max_sl: int = 0
    sl_period_mode: int = 0   # 0=off, 1=day, 2=session, 3=rolling bars
    sl_period_bars: int = 0

    def __post_init__(self):
        if self.period_1 <= 0:       raise ValueError("period_1 must be > 0")
        if self.period_2 <= 0:       raise ValueError("period_2 must be > 0")
        if self.atr_period <= 0:     raise ValueError("atr_period must be > 0")
        if self.entry_delay <= 0:    raise ValueError("entry_delay below 1 implies look-ahead bias")
        if self.max_gap_signal < 0:  raise ValueError("max_gap_signal must be >= 0")
        if self.max_gap_entry < 0:   raise ValueError("max_gap_entry must be >= 0")
        if self.min_size_pct < 0:    raise ValueError("min_size_pct must be >= 0")
        if self.max_size_pct <= 0:   raise ValueError("max_size_pct must be > 0")
        if self.min_size_pct > self.max_size_pct: raise ValueError("min_size_pct cannot exceed max_size_pct")
        if self.tp_pct < 0:          raise ValueError("tp_pct must be >= 0")
        if self.sl_pct < 0:          raise ValueError("sl_pct must be >= 0")
        if self.use_atr_sl_tp not in (-1, 0, 1, 2): raise ValueError("use_atr_sl_tp must be in {-1,0,1,2}")
        if self.tp_atr_mult < 0:     raise ValueError("tp_atr_mult must be >= 0")
        if self.sl_atr_mult < 0:     raise ValueError("sl_atr_mult must be >= 0")
        if self.me_max < 0:          raise ValueError("me_max must be >= 0")
        if self.me_period < 0:       raise ValueError("me_period must be >= 0")
        if self.me_reset_mode not in (0, 1, 2, 3): raise ValueError("me_reset_mode must be in {0,1,2,3}")
        if self.me_reset_mode == 3 and self.me_period <= 0: raise ValueError("me_period must be > 0 when me_reset_mode == 3")
        if self.cooldown_entries < 0: raise ValueError("cooldown_entries must be >= 0")
        if self.cooldown_bars < 0:    raise ValueError("cooldown_bars must be >= 0")
        if self.cooldown_mode not in (1, 2, 3): raise ValueError("cooldown_mode must be 1, 2 or 3")
        if self.be_trigger_pct < 0:  raise ValueError("be_trigger_pct must be >= 0")
        if self.be_offset_pct < 0:   raise ValueError("be_offset_pct must be >= 0")
        if self.be_delay_bars < 0:   raise ValueError("be_delay_bars must be >= 0")
        if self.trailing_trigger_pct < 0: raise ValueError("trailing_trigger_pct must be >= 0")
        if self.runner_trailing_mult < 0: raise ValueError("runner_trailing_mult must be >= 0")
        if self.commission_pct < 0:  raise ValueError("commission_pct must be >= 0")
        if self.commission_per_lot_usd < 0: raise ValueError("commission_per_lot_usd must be >= 0")
        if self.contract_size <= 0: raise ValueError("contract_size must be > 0")
        if self.spread_pct < 0:      raise ValueError("spread_pct must be >= 0")
        if self.spread_abs < 0:      raise ValueError("spread_abs must be >= 0")
        if self.slippage_pct < 0:    raise ValueError("slippage_pct must be >= 0")
        if not (0 < self.alpha < 100): raise ValueError("alpha must be between 0 and 100")
        if self.bar_duration_min <= 0: raise ValueError("bar_duration_min must be > 0")
        if self.window_before < 0:   raise ValueError("window_before must be >= 0")
        if self.window_after < 0:    raise ValueError("window_after must be >= 0")
        if self.entry_on_close and self.entry_on_signal_close_price: raise ValueError("entry_on_close and entry_on_signal_close_price cannot both be True")
        if self.max_holding_bars < 0: raise ValueError("max_holding_bars must be >= 0")
        if self.forced_flat_frequency not in (None, "day", "weekend"):raise ValueError('forced_flat_frequency must be None, "day" or "weekend"')
        if (self.forced_flat_frequency is None) != (self.forced_flat_time is None):raise ValueError("forced_flat_frequency and forced_flat_time must be both set or both None")
        if self.max_tp < 0:raise ValueError("max_tp must be >= 0")
        if self.tp_period_mode not in (0, 1, 2, 3):raise ValueError("tp_period_mode must be in {0,1,2,3}")
        if self.tp_period_bars < 0:raise ValueError("tp_period_bars must be >= 0")
        if self.tp_period_mode == 3 and self.tp_period_bars <= 0:raise ValueError("tp_period_bars must be > 0 when tp_period_mode == 3")
        if self.max_sl < 0:raise ValueError("max_sl must be >= 0")
        if self.sl_period_mode not in (0, 1, 2, 3):raise ValueError("sl_period_mode must be in {0,1,2,3}")
        if self.sl_period_bars < 0:raise ValueError("sl_period_bars must be >= 0")
        if self.sl_period_mode == 3 and self.sl_period_bars <= 0:raise ValueError("sl_period_bars must be > 0 when sl_period_mode == 3")


class DataPipeline:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def fetchdata(
        self,
        ticker: str,
        start: str,
        end: str,
        timezone_shift: int = 0,
    ) -> pd.DataFrame:
        df = pd.read_csv(
            f"{self.base_path}/{ticker}.csv",
            header=None,
            names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
        )
        df["Datetime"] = pd.to_datetime(df["Datetime"]) + pd.Timedelta(hours=timezone_shift)
        df = df.set_index("Datetime").sort_index()
        return df.loc[start:end]

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)

        out = df.copy()
        out["ATR"] = tr.rolling(period).mean()
        return out

    @staticmethod
    def add_basic_features(
        df: pd.DataFrame,
        prefix: str,
        add_returns: bool = True,
        return_periods: Sequence[int] = (1, 5, 20),
        add_range: bool = True,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        out[f"{prefix}_Open"] = df["Open"]
        out[f"{prefix}_High"] = df["High"]
        out[f"{prefix}_Low"] = df["Low"]
        out[f"{prefix}_Close"] = df["Close"]
        out[f"{prefix}_Volume"] = df["Volume"]

        if "ATR" in df.columns:
            out[f"{prefix}_ATR"] = df["ATR"]

        if add_returns:
            for p in return_periods:
                out[f"{prefix}_ret_{p}"] = df["Close"].pct_change(p)

        if add_range:
            out[f"{prefix}_hl_range_pct"] = (df["High"] - df["Low"]) / df["Close"]

        return out

    def prepare_df(
        self,
        ticker: str,
        start: str,
        end: str,
        timezone_shift: int = 0,
        atr_period: int = 14,
        data=None,
        use_main_df: bool = True,
    ) -> pd.DataFrame:
        if use_main_df and data is not None and data.has_main_df():
            df = data.get_main_df()
            df = df.loc[start:end]
        else:
            df = self.fetchdata(
                ticker=ticker,
                start=start,
                end=end,
                timezone_shift=timezone_shift,
            )

        df = self.compute_atr(df, atr_period)
        return df

    def prepare_surface_inputs(
        self,
        primary_ticker: str,
        start: str,
        end: str,
        timezone_shift: int = 0,
        atr_period: int = 14,
        extra_tickers: Sequence[str] | None = None,
        include_primary_in_context: bool = True,
        add_returns: bool = True,
        return_periods: Sequence[int] = (1, 5, 20),
        add_range: bool = True,
        data=None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        price_df = self.prepare_df(
            ticker=primary_ticker,
            start=start,
            end=end,
            timezone_shift=timezone_shift,
            atr_period=atr_period,
            data=data,
            use_main_df=True,
        )

        context_parts: list[pd.DataFrame] = []

        if include_primary_in_context:
            primary_ctx = pd.DataFrame(index=price_df.index)
            if "ATR" in price_df.columns:
                primary_ctx["ATR"] = price_df["ATR"]
            context_parts.append(primary_ctx)

        for ticker in extra_tickers or []:
            df_extra = self.prepare_df(
                ticker=ticker,
                start=start,
                end=end,
                timezone_shift=timezone_shift,
                atr_period=atr_period,
                data=data,
                use_main_df=False,
            )

            extra_ctx = self.add_basic_features(
                df_extra,
                prefix=ticker,
                add_returns=add_returns,
                return_periods=return_periods,
                add_range=add_range,
            )

            extra_ctx = extra_ctx.reindex(price_df.index)
            context_parts.append(extra_ctx)

        if context_parts:
            context_df = pd.concat(context_parts, axis=1)
            context_df = context_df.reindex(price_df.index)
        else:
            context_df = pd.DataFrame(index=price_df.index)

        return price_df, context_df


@dataclass
class ExtSeries:
    name: str
    values: np.ndarray
    index: np.ndarray | None
    source: str
    asset: str | None = None
    tf: str | None = None
    timezone_shift: int | float = 0
    original_name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def series(self) -> pd.Series:
        if self.index is None:
            raise ValueError(f"ExtSeries '{self.name}' has no index.")
        return pd.Series(self.values, index=pd.DatetimeIndex(self.index), name=self.name)



def agg_open(arr: np.ndarray, start: int, end: int) -> float:
    """Premier open de la fenêtre [start:end)."""
    return arr[start]
def agg_close(arr: np.ndarray, start: int, end: int) -> float:
    """Dernier close de la fenêtre [start:end)."""
    return arr[end - 1]
def agg_high(arr: np.ndarray, start: int, end: int) -> float:
    """High max de la fenêtre [start:end)."""
    return np.max(arr[start:end])
def agg_low(arr: np.ndarray, start: int, end: int) -> float:
    """Low min de la fenêtre [start:end)."""
    return np.min(arr[start:end])
def tf_ratio(src_minutes: int, dst_minutes: int) -> int:
    if dst_minutes % src_minutes != 0:
        raise ValueError("Destination TF must be a multiple of source TF.")
    return dst_minutes // src_minutes

@dataclass
class OHLCVItem:
    name: str
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray | None
    index: np.ndarray
    source: str | None = None
    asset: str | None = None
    tf: str | None = None
    timezone_shift: int | float = 0
    original_columns: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

class Data:
    def __init__(self) -> None:
        self._ext_store: dict[str, ExtSeries] = {}
        self._ohlcv_store: dict[str, OHLCVItem] = {}
        self._ohlcv_matrix_store: dict[str, pd.DataFrame] = {}
        self._ohlcv_matrix_meta: dict[str, dict[str, Any]] = {}

        self._aliases: dict[str, str] = {}
        self._meta_store: dict[str, dict[str, Any]] = {}

        self._main_df: pd.DataFrame | None = None
        self._main_meta: dict[str, Any] | None = None

        self._aligned_store: dict[tuple[str | None, str | None], dict[str, ExtSeries]] = {}


        
        # ex: {("BTC", "M5"): {"gamma_BTC_M5": ExtSeries(...)}}


    # =========================================================
    # IMPORT
    # =========================================================

    def register_csv(
        self,
        path: str | Path,
        col_map: dict[int | str, str] | None = None,
        asset: str | None = None,
        tf: str | None = None,
        kind: str = "ext",
        timezone_shift: int | float = 0,
        sep: str = ",",
        parse_dates: bool = True,
    ) -> "Data":
        path = Path(path)
        df = pd.read_csv(path, sep=sep)

        df = self._apply_col_map(df, col_map)

        time_col = self._detect_time_column(df)
        if parse_dates and time_col is not None:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            if timezone_shift:
                df[time_col] = df[time_col] + pd.to_timedelta(timezone_shift, unit="h")

        kind = (kind or "ext").lower().strip()
        if kind not in {"ext", "ohlcv", "data"}:
            raise ValueError("kind doit être 'ext', 'ohlcv' ou 'data'")

        if kind == "ext":
            self._register_ext_df(
                df=df,
                source=str(path),   # ou source=source dans register_df
                asset=asset,
                tf=tf,
                timezone_shift=timezone_shift,
                time_col=time_col,
            )
        elif kind == "ohlcv":
            self._register_ohlcv_df(
                df=df,
                source=str(path),   # ou source=source dans register_df
                asset=asset,
                tf=tf,
                timezone_shift=timezone_shift,
                time_col=time_col,
            )
        else:
            self._register_main_data_df(
                df=df,
                source=str(path),   # ou source=source dans register_df
                asset=asset,
                tf=tf,
                timezone_shift=timezone_shift,
            )

        return self

    def register_df(
        self,
        df: pd.DataFrame,
        col_map: dict[int | str, str] | None = None,
        asset: str | None = None,
        tf: str | None = None,
        kind: str = "ext",
        timezone_shift: int | float = 0,
        parse_dates: bool = True,
        source: str = "dataframe",
    ) -> "Data":
        df = df.copy()
        df = self._apply_col_map(df, col_map)

        time_col = self._detect_time_column(df)
        if parse_dates and time_col is not None:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            if timezone_shift:
                df[time_col] = df[time_col] + pd.to_timedelta(timezone_shift, unit="h")

        kind = (kind or "ext").lower().strip()
        if kind not in {"ext", "ohlcv", "data"}:
            raise ValueError("kind doit être 'ext', 'ohlcv' ou 'data'")

        if kind == "ext":
            self._register_ext_df(
                df=df,
                source=source,
                asset=asset,
                tf=tf,
                timezone_shift=timezone_shift,
                time_col=time_col,
            )
        elif kind == "ohlcv":
            self._register_ohlcv_df(
                df=df,
                source=source,
                asset=asset,
                tf=tf,
                timezone_shift=timezone_shift,
                time_col=time_col,
            )
        else:
            self._register_main_data_df(
                df=df,
                source=source,
                asset=asset,
                tf=tf,
                timezone_shift=timezone_shift,
            )

        return self

    def load_ohlcv_csv(
        self,
        ticker: str,
        base_path: str | Path,
        timezone_shift: int | float = 0,
        col_map: dict[int | str, str] | None = None,
        sep: str = ",",
        overwrite: bool = False,
    ) -> str:
        """
        Charge un OHLCV depuis un CSV nommé selon la convention ASSET_TF.csv
        puis l'enregistre comme kind='ohlcv'.

        Retourne le nom canonique OHLCV enregistré.
        """
        asset, tf = self._parse_ticker_name(ticker)
        canonical_name = self._build_ohlcv_name(asset, tf)
        if not overwrite and canonical_name in self._ohlcv_store:
            return canonical_name

        path = Path(base_path) / f"{ticker}.csv"

        self.register_csv(
            path=path,
            col_map=col_map or {
                0: "Datetime",
                1: "Open",
                2: "High",
                3: "Low",
                4: "Close",
                5: "Volume",
            },
            asset=asset,
            tf=tf,
            kind="ohlcv",
            timezone_shift=timezone_shift,
            sep=sep,
        )

        return canonical_name

    # Import strait to matrix
    def build_ohlcv_matrix(
        self,
        name: str,
        tf: str,
        base_path: str | Path,
        assets: list[str] | None = None,
        autoload: bool = True,
        align_to: str | OHLCVItem | pd.DataFrame | pd.DatetimeIndex | None = None,
        dropna: bool = False,
        native_names_mode: str = "auto",
        timezone_shift: int | float = 0,
        col_map: dict[int | str, str] | None = None,
        sep: str = ",",
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Build an OHLCV matrix from a list of assets at one timeframe.

        If assets is None:
        - use every already-registered OHLCV matching tf
        - if none are registered, raise an error
        """
        tf_norm = self._normalize_tf(tf)
        base_path = Path(base_path)

        if assets is None:
            # take everything already loaded for this tf
            names = [
                item.name
                for item in self._ohlcv_store.values()
                if item.tf == tf_norm
            ]
            if not names:
                raise ValueError(
                    f"No OHLCV loaded for tf='{tf_norm}'. "
                    f"Provide assets=... or load OHLCV first."
                )

            return self.to_ohlcv_matrix(
                name=name,
                names=names,
                tf=tf_norm,
                align_to=align_to,
                dropna=dropna,
                native_names_mode=native_names_mode,
                overwrite=overwrite,
            )

        assets_norm = [self._normalize_asset(a) for a in assets]
        tickers = [f"{a}_{tf_norm}" for a in assets_norm]

        return self.to_ohlcv_matrix(
            name=name,
            tickers=tickers,
            base_path=base_path,
            autoload=autoload,
            tf=tf_norm,
            align_to=align_to,
            dropna=dropna,
            native_names_mode=native_names_mode,
            timezone_shift=timezone_shift,
            col_map=col_map,
            sep=sep,
            overwrite=overwrite,
        )

    def build_ext_group(
        self,
        base_path: str | Path,
        names: list[str] | None = None,
        asset: str | None = None,
        tf: str | None = None,
        align_to: str | ExtSeries | OHLCVItem | pd.DataFrame | pd.DatetimeIndex | None = None,
        target_tf: str | None = None,
        resample_method: str = "ffill",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        dropna: bool = False,
        col_map: dict[int | str, str] | None = None,
        sep: str = ",",
    ) -> pd.DataFrame:
        """
        Load a group of external series from CSV files, optionally resample them,
        align them, and return a matrix.

        Rules:
        - if names is provided: load these files only
        - if names is None: load every CSV in base_path
        - if asset is None: keep everything found
        - if asset is not None: keep only matching series
        """
        base_path = Path(base_path)
        tf_norm = self._normalize_tf(tf) if tf is not None else None
        asset_norm = self._normalize_asset(asset) if asset is not None else None

        if names is None:
            csv_files = sorted(base_path.glob("*.csv"))
            if not csv_files:
                raise ValueError(f"No CSV files found in '{base_path}'.")
            names_to_load = [p.stem for p in csv_files]
        else:
            names_to_load = list(names)

        loaded_canonicals: list[str] = []

        for raw_name in names_to_load:
            path = base_path / f"{raw_name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")

            before = set(self._ext_store.keys())

            self.register_csv(
                path=path,
                col_map=col_map,
                asset=asset_norm,
                tf=tf_norm,
                kind="ext",
                sep=sep,
                parse_dates=True,
            )

            after = set(self._ext_store.keys())
            created = sorted(after - before)
            loaded_canonicals.extend(created)

        if not loaded_canonicals:
            return pd.DataFrame()

        selected_names = loaded_canonicals

        if asset_norm is not None:
            selected_names = [
                n for n in selected_names
                if self._ext_store[n].asset == asset_norm
            ]

        if tf_norm is not None:
            selected_names = [
                n for n in selected_names
                if self._ext_store[n].tf == tf_norm
            ]

        if target_tf is not None:
            resampled_names: list[str] = []
            for n in selected_names:
                new_name = self.resample_ext(
                    name=n,
                    target_tf=target_tf,
                    method=resample_method,
                    start=start,
                    end=end,
                )
                resampled_names.append(new_name)
            selected_names = resampled_names
            tf_norm = self._normalize_tf(target_tf)

        aligned = self.align_all(
            tf=tf_norm,
            asset=asset_norm,
            names=selected_names,
            align_to=align_to,
            start=start,
            end=end,
            dropna=dropna,
        )

        if not aligned:
            return pd.DataFrame()

        first_item = next(iter(aligned.values()))
        idx = pd.DatetimeIndex(first_item.index)
        df = pd.DataFrame(index=idx)

        for series_name, item in aligned.items():
            df[series_name] = item.values

        df.attrs["asset"] = asset_norm
        df.attrs["tf"] = tf_norm
        df.attrs["source_group"] = str(base_path)
        df.attrs["is_ext_group"] = True

        return df

    # =========================================================
    # MAIN DF manipulation
    # =========================================================
    def has_main_df(self) -> bool:
        return self._main_df is not None

    def get_main_df(self) -> pd.DataFrame:
        if self._main_df is None:
            raise ValueError("Aucun main_df n'est enregistré dans Data.")
        return self._main_df.copy()

    def _ohlcv_item_to_df(self, item: OHLCVItem) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "Open": np.asarray(item.open, dtype=np.float64),
                "High": np.asarray(item.high, dtype=np.float64),
                "Low": np.asarray(item.low, dtype=np.float64),
                "Close": np.asarray(item.close, dtype=np.float64),
            },
            index=pd.DatetimeIndex(item.index),
        )

        if item.volume is not None:
            df["Volume"] = np.asarray(item.volume, dtype=np.float64)

        df.attrs["name"] = item.name
        df.attrs["asset"] = item.asset
        df.attrs["tf"] = item.tf
        df.attrs["source"] = item.source
        df.attrs["timezone_shift"] = item.timezone_shift
        df.attrs["meta"] = item.meta.copy()
        return df

    def _extract_ohlcv_matrix_asset_df(
        self,
        df: pd.DataFrame,
        asset: str | None,
    ) -> pd.DataFrame | None:
        required = ["Open", "High", "Low", "Close"]
        if all(c in df.columns for c in required):
            return None

        if asset is None:
            suffix_candidates = [c for c in df.columns if any(c.startswith(f"{base}_") for base in required)]
            if suffix_candidates:
                raise ValueError(
                    "main_df appears to be a multi-asset OHLCV matrix. "
                    "Pass asset='XAUUSD' (or the desired asset) to extract native OHLC columns."
                )
            return None

        asset_norm = self._normalize_asset(asset)
        suffix = f"_{asset_norm}"
        cols = [c for c in df.columns if c.endswith(suffix)]
        if not cols:
            missing = [f"{base}{suffix}" for base in required if f"{base}{suffix}" not in df.columns]
            raise ValueError(
                f"Unable to extract main_df from the OHLCV matrix for asset='{asset_norm}'. "
                f"Missing expected columns: {missing}"
            )

        out = pd.DataFrame(df[cols].copy())
        rename_map = {c: c[: -len(suffix)] for c in cols}
        out = out.rename(columns=rename_map)
        out.attrs["asset"] = asset_norm
        if "tf" in getattr(df, "attrs", {}):
            out.attrs["tf"] = df.attrs.get("tf")
        out.attrs["is_ohlcv_matrix"] = True
        return out

    def set_main_df(
        self,
        df: pd.DataFrame,
        asset: str | None = None,
        tf: str | None = None,
        source: str = "manual",
        timezone_shift: int | float = 0,
    ) -> None:
        df = pd.DataFrame(df).copy()
        extracted = self._extract_ohlcv_matrix_asset_df(df, asset)
        if extracted is not None:
            df = pd.DataFrame(extracted).copy()

        required = ["Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Invalid main_df: missing columns {missing}. "
                "The main DataFrame must contain at least Open, High, Low, Close."
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            time_col = self._detect_time_column(df)
            if time_col is None:
                raise ValueError(
                    "main_df must have a DatetimeIndex or a detectable time column."
                )
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.set_index(time_col)

        df = df.sort_index()

        self._main_df = df
        self._main_meta = {
            "kind": "data",
            "source": source,
            "asset": self._normalize_asset(asset or df.attrs.get("asset")),
            "tf": self._normalize_tf(tf or df.attrs.get("tf")),
            "timezone_shift": timezone_shift,
            "columns": list(df.columns),
        }

    def _register_main_data_df(
        self,
        df: pd.DataFrame,
        source: str,
        asset: str | None,
        tf: str | None,
        timezone_shift: int | float,
    ) -> None:
        self.set_main_df(
            df=df,
            asset=asset,
            tf=tf,
            source=source,
            timezone_shift=timezone_shift,
        )

    # =========================================================
    # GETTERS
    # =========================================================

    def ext(self, name: str) -> ExtSeries:
        canonical = self._resolve_ext_name(name)
        return self._ext_store[canonical]

    def list_ext(self) -> list[str]:
        return sorted(self._ext_store.keys())

    def list_aliases(self) -> dict[str, str]:
        return dict(sorted(self._aliases.items()))

    def meta(self, name: str) -> dict[str, Any]:
        canonical = self._resolve_ext_name(name)
        return self._meta_store[canonical].copy()

    def has_ext(self, name: str) -> bool:
        return name in self._ext_store or name in self._aliases

    def list_tf(self) -> list[str]:
        tfs = {item.tf for item in self._ext_store.values() if item.tf is not None}
        return sorted(tfs)

    def list_ext_by_tf(self, tf: str) -> list[str]:
        tf = self._normalize_tf(tf)
        return sorted([k for k, v in self._ext_store.items() if v.tf == tf])

    # Resample Manipulation

    def _make_resampled_name(
        self,
        original_name: str,
        original_tf: str | None,
        target_tf: str,
        method: str,
    ) -> str:
        """
        Exemple :
            funding_BTC_H1 -> funding_BTC_M5_from_H1_ffill
        """
        if original_tf is None:
            return f"{original_name}_{target_tf}_{method}"

        suffix_old = f"_{original_tf}"
        if original_name.endswith(suffix_old):
            base = original_name[: -len(suffix_old)]
            return f"{base}_{target_tf}_from_{original_tf}_{method}"

        return f"{original_name}_{target_tf}_from_{original_tf}_{method}"

    def resample_ext(
        self,
        name: str,
        target_tf: str,
        method: str = "ffill",
        new_name: str | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> str:
        """
        Resample explicitement une série externe vers une autre timeframe.

        Exemples :
            data.resample_ext("funding_BTC_H1", target_tf="M5", method="ffill")
            data.resample_ext("delta_BTC_M1", target_tf="M5", method="sum")

        Retourne le nom canonique de la nouvelle série créée.
        """
        canonical = self._resolve_ext_name(name)
        item = self._ext_store[canonical]

        if item.index is None:
            raise ValueError(f"La série '{canonical}' n'a pas d'index temporel")

        target_tf_norm = self._normalize_tf(target_tf)
        method = method.lower().strip()

        if method not in {"ffill", "exact", "last", "mean", "sum", "min", "max"}:
            raise ValueError("method doit être dans {'ffill', 'exact', 'last', 'mean', 'sum','min', 'max'}")

        s = pd.Series(item.values, index=pd.DatetimeIndex(item.index))
        s = s[~s.index.duplicated(keep="last")].sort_index()

        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None

        if start_ts is not None:
            s = s[s.index >= start_ts]
        if end_ts is not None:
            s = s[s.index <= end_ts]

        pandas_freq = self._tf_to_pandas_freq(target_tf_norm)

        # -----------------------------------------------------
        # Cas 1 : projection vers fréquence plus fine / identique
        # ex: H1 -> M5 avec ffill / exact
        # -----------------------------------------------------
        if method in {"ffill", "exact"}:
            if len(s.index) == 0:
                raise ValueError(f"Aucune donnée restante après filtre temporel pour '{canonical}'")

            if start_ts is None:
                start_ts = s.index.min()
            if end_ts is None:
                end_ts = s.index.max()

            target_index = pd.date_range(start=start_ts, end=end_ts, freq=pandas_freq)

            out = s.reindex(target_index)

            if method == "ffill":
                out = out.ffill()

        # -----------------------------------------------------
        # Cas 2 : resample pandas standard
        # utile pour downsample ou agrégation explicite
        # ex: M1 -> M5 avec sum/mean/last
        # -----------------------------------------------------
        else:
            if method == "last":
                out = s.resample(pandas_freq).last()
            elif method == "mean":
                out = s.resample(pandas_freq).mean()
            elif method == "sum":
                out = s.resample(pandas_freq).sum()
            elif method == "min":
                out = s.resample(pandas_freq).min()
            elif method == "max":
                out = s.resample(pandas_freq).max()
            else:
                raise ValueError(f"Méthode non supportée : {method}")

            if start_ts is not None:
                out = out[out.index >= start_ts]
            if end_ts is not None:
                out = out[out.index <= end_ts]

        out_name = new_name or self._make_resampled_name(
            original_name=item.name,
            original_tf=item.tf,
            target_tf=target_tf_norm,
            method=method,
        )

        resampled_item = ExtSeries(
            name=out_name,
            values=out.to_numpy(),
            index=out.index.to_numpy(),
            source=item.source,
            asset=item.asset,
            tf=target_tf_norm,
            timezone_shift=item.timezone_shift,
            original_name=item.original_name,
            meta={
                **item.meta,
                "resampled": True,
                "resample_from": item.name,
                "resample_from_tf": item.tf,
                "resample_to_tf": target_tf_norm,
                "resample_method": method,
            },
        )

        self._ext_store[out_name] = resampled_item
        self._meta_store[out_name] = resampled_item.meta.copy()

        # refresh alias éventuel
        self._refresh_aliases([(resampled_item.original_name or out_name, out_name)])

        return out_name

    def resample_ohlcv_arrays(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray | None,
        src_minutes: int,
        dst_minutes: int,
        drop_incomplete: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        ratio = tf_ratio(src_minutes, dst_minutes)
        n = len(close)

        if volume is not None:
            if not (len(open_) == len(high) == len(low) == len(close) == len(volume)):
                raise ValueError("All OHLCV arrays must have the same length.")
        else:
            if not (len(open_) == len(high) == len(low) == len(close)):
                raise ValueError("All OHLC arrays must have the same length.")

        n_out = n // ratio if drop_incomplete else (n + ratio - 1) // ratio

        out_open = np.full(n_out, np.nan)
        out_high = np.full(n_out, np.nan)
        out_low = np.full(n_out, np.nan)
        out_close = np.full(n_out, np.nan)
        out_volume = np.full(n_out, np.nan) if volume is not None else None

        for i in range(n_out):
            start = i * ratio
            end = min(start + ratio, n)

            if end - start < ratio and drop_incomplete:
                break

            out_open[i] = agg_open(open_, start, end)
            out_high[i] = agg_high(high, start, end)
            out_low[i] = agg_low(low, start, end)
            out_close[i] = agg_close(close, start, end)

            if out_volume is not None:
                out_volume[i] = np.sum(volume[start:end])

        if drop_incomplete and n_out > 0:
            valid = ~np.isnan(out_close)
            out_open = out_open[valid]
            out_high = out_high[valid]
            out_low = out_low[valid]
            out_close = out_close[valid]
            if out_volume is not None:
                out_volume = out_volume[valid]

        return out_open, out_high, out_low, out_close, out_volume

    def _tf_to_minutes(self, tf: str) -> int:
        tf = self._normalize_tf(tf)
        mapping = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
        }
        if tf not in mapping:
            raise ValueError(f"TF non supportée : {tf}")
        return mapping[tf]

    def resample_ohlcv(
        self,
        name: str,
        target_tf: str,
        new_name: str | None = None,
        drop_incomplete: bool = True,
    ) -> pd.DataFrame:
        item = self.ohlcv(name)

        if item.tf is None:
            raise ValueError(f"L'objet OHLCV '{item.name}' n'a pas de tf")

        src_minutes = self._tf_to_minutes(item.tf)
        dst_tf_norm = self._normalize_tf(target_tf)
        dst_minutes = self._tf_to_minutes(dst_tf_norm)

        if dst_minutes < src_minutes:
            raise ValueError("resample_ohlcv gère ici seulement l'agrégation vers une tf supérieure")
        freq = self._tf_to_pandas_freq(dst_tf_norm)
        src_delta = pd.Timedelta(minutes=src_minutes)
        dst_delta = pd.Timedelta(minutes=dst_minutes)

        src_df = self._ohlcv_item_to_df(item)
        agg_map: dict[str, str] = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
        if "Volume" in src_df.columns:
            agg_map["Volume"] = "sum"

        resampled_df = src_df.resample(
            freq,
            label="left",
            closed="left",
        ).agg(agg_map)

        resampled_df = resampled_df.dropna(subset=["Open", "High", "Low", "Close"])

        if drop_incomplete and not resampled_df.empty:
            first_ts = pd.Timestamp(src_df.index[0])
            first_bin_start = first_ts.floor(freq)
            if first_ts != first_bin_start and first_bin_start in resampled_df.index:
                resampled_df = resampled_df.drop(index=first_bin_start)

            if not resampled_df.empty:
                last_ts = pd.Timestamp(src_df.index[-1])
                last_bin_start = last_ts.floor(freq)
                last_bar_end = last_ts + src_delta
                expected_bin_end = last_bin_start + dst_delta
                if last_bar_end != expected_bin_end and last_bin_start in resampled_df.index:
                    resampled_df = resampled_df.drop(index=last_bin_start)

        if "Volume" in resampled_df.columns:
            resampled_df["Volume"] = resampled_df["Volume"].astype(np.float64)

        out_open = resampled_df["Open"].to_numpy(dtype=np.float64)
        out_high = resampled_df["High"].to_numpy(dtype=np.float64)
        out_low = resampled_df["Low"].to_numpy(dtype=np.float64)
        out_close = resampled_df["Close"].to_numpy(dtype=np.float64)
        out_volume = (
            resampled_df["Volume"].to_numpy(dtype=np.float64)
            if "Volume" in resampled_df.columns
            else None
        )
        out_index = pd.DatetimeIndex(resampled_df.index)

        out_name = new_name or f"{item.asset}_{dst_tf_norm}"

        new_item = OHLCVItem(
            name=out_name,
            open=out_open,
            high=out_high,
            low=out_low,
            close=out_close,
            volume=out_volume,
            index=out_index.to_numpy(),
            source=item.source,
            asset=item.asset,
            tf=dst_tf_norm,
            timezone_shift=item.timezone_shift,
            original_columns=item.original_columns.copy(),
            meta={
                **item.meta,
                "resampled": True,
                "resample_from": item.name,
                "resample_from_tf": item.tf,
                "resample_to_tf": dst_tf_norm,
            },
        )

        self._ohlcv_store[out_name] = new_item
        self._meta_store[out_name] = new_item.meta.copy()

        return self._ohlcv_item_to_df(new_item)

    def _tf_to_pandas_freq(self, tf: str) -> str:
        tf = self._normalize_tf(tf)
        mapping = {
            "M1": "1min",
            "M5": "5min",
            "M15": "15min",
            "M30": "30min",
            "H1": "1h",
            "H4": "4h",
            "D1": "1d",
        }
        if tf not in mapping:
            raise ValueError(f"TF non supportée : {tf}")
        return mapping[tf]

    def ohlcv(self, name: str) -> OHLCVItem:
        key = self._resolve_ohlcv_name(name)
        return self._ohlcv_store[key]

    def ohlcv_df(self, name: str) -> pd.DataFrame:
        return self._ohlcv_item_to_df(self.ohlcv(name))

    def list_ohlcv(self) -> list[str]:
        return sorted(self._ohlcv_store.keys())


    # =========================================================
    # CROP
    # ========================================================

    def crop(
        self,
        obj: str | ExtSeries | pd.DataFrame,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> ExtSeries | pd.DataFrame:
        """
        Crop un objet précis, pas tout le système.

        obj peut être :
        - nom d'une série externe
        - ExtSeries
        - DataFrame indexé temporellement
        """
        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None

        if isinstance(obj, str):
            item = self.ext_item(obj)
            return self._crop_extseries(item, start_ts, end_ts)

        if isinstance(obj, ExtSeries):
            return self._crop_extseries(obj, start_ts, end_ts)

        if isinstance(obj, pd.DataFrame):
            return self._crop_dataframe(obj, start_ts, end_ts)

        raise TypeError("obj doit être un nom de série, un ExtSeries ou un DataFrame")

    # =========================================================
    # MASTER INDEX
    # =========================================================

    def new_index(
        self,
        freq: str | None = None,
        from_obj: str | ExtSeries | OHLCVItem | pd.DataFrame | pd.DatetimeIndex | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DatetimeIndex:
        """
        Construit un index cible unique.

        Cas possibles :
        - freq + start + end
        - from_obj = nom ext
        - from_obj = nom ohlcv
        - from_obj = ExtSeries
        - from_obj = OHLCVItem
        - from_obj = DataFrame avec DatetimeIndex
        - from_obj = DatetimeIndex

        Cette fonction ne modifie rien dans le store.
        Elle retourne seulement l'index cible.
        """
        if (freq is None and from_obj is None) or (freq is not None and from_obj is not None):
            raise ValueError("Choisir soit freq, soit from_obj")

        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None

        # -----------------------------------------------------
        # Cas 1 : construire un index régulier
        # -----------------------------------------------------
        if freq is not None:
            if start_ts is None or end_ts is None:
                raise ValueError("Avec freq, il faut fournir start et end")

            pandas_freq = self._tf_to_pandas_freq(freq)
            idx = pd.date_range(start=start_ts, end=end_ts, freq=pandas_freq)
            return pd.DatetimeIndex(idx).sort_values().unique()

        # -----------------------------------------------------
        # Cas 2 : dériver l'index depuis un objet existant
        # -----------------------------------------------------
        idx: pd.DatetimeIndex | None = None

        if isinstance(from_obj, str):
            # ext
            if from_obj in self._ext_store or from_obj in self._aliases:
                item = self.ext_item(from_obj)
                if item.index is None:
                    raise ValueError(f"La série ext '{from_obj}' n'a pas d'index")
                idx = pd.DatetimeIndex(item.index)

            # ohlcv
            elif from_obj in self._ohlcv_store:
                item = self.ohlcv(from_obj)
                idx = pd.DatetimeIndex(item.index)

            else:
                # tentative résolution souple OHLCV
                try:
                    item = self.ohlcv(from_obj)
                    idx = pd.DatetimeIndex(item.index)
                except Exception as e:
                    raise KeyError(f"Objet introuvable pour make_target_index : {from_obj}") from e

        elif isinstance(from_obj, ExtSeries):
            if from_obj.index is None:
                raise ValueError("L'ExtSeries fournie n'a pas d'index")
            idx = pd.DatetimeIndex(from_obj.index)

        elif isinstance(from_obj, OHLCVItem):
            idx = pd.DatetimeIndex(from_obj.index)

        elif isinstance(from_obj, pd.DataFrame):
            if not isinstance(from_obj.index, pd.DatetimeIndex):
                raise ValueError("Le DataFrame fourni doit avoir un DatetimeIndex")
            idx = pd.DatetimeIndex(from_obj.index)

        elif isinstance(from_obj, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(from_obj)

        else:
            raise TypeError("from_obj doit être un nom, ExtSeries, OHLCVItem, DataFrame ou DatetimeIndex")

        if start_ts is not None:
            idx = idx[idx >= start_ts]
        if end_ts is not None:
            idx = idx[idx <= end_ts]

        return idx.sort_values().unique()
    # =========================================================
    # ALIGN
    # =========================================================

    def align_all(
        self,
        tf: str | None = None,
        asset: str | None = None,
        names: list[str] | None = None,
        align_to: str | ExtSeries | pd.DataFrame | pd.DatetimeIndex | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        dropna: bool = False,
    ) -> dict[str, ExtSeries]:
        """
        Aligne toutes les séries candidates d'une même timeframe.

        - si names=None : prend toutes les séries de la tf demandée
        - si align_to=None : choisit l'index le plus long parmi les candidates
        - si dropna=False : garde les trous -> NaN
        - si dropna=True : coupe automatiquement à la zone commune exploitable

        Retourne les séries alignées séparément.
        """
        candidates = self._collect_alignment_candidates(tf=tf, asset=asset, names=names)
        if not candidates:
            return {}

        if align_to is None:
            valid = [x for x in candidates if x.index is not None]
            if not valid:
                raise ValueError("Aucune série candidate ne possède d'index temporel")
            ref = max(valid, key=lambda x: len(x.index))
            ref_index = self.make_target_index(
                from_obj=ref,
                start=start,
                end=end,
            )
        else:
            ref_index = self.make_target_index(
                from_obj=align_to,
                start=start,
                end=end,
            )

        aligned: dict[str, ExtSeries] = {}
        for item in candidates:
            new_values = self._align_series_to_index(item=item, target_index=ref_index)
            aligned[item.name] = ExtSeries(
                name=item.name,
                values=new_values,
                index=ref_index.to_numpy(),
                source=item.source,
                asset=item.asset,
                tf=item.tf,
                timezone_shift=item.timezone_shift,
                original_name=item.original_name,
                meta={**item.meta, "aligned": True},
            )

        if dropna:
            aligned = self._dropna_aligned_dict(aligned)

        asset_norm = self._normalize_asset(asset) if asset is not None else self._infer_single_asset(candidates)
        tf_key = tf or self._infer_single_tf(candidates) or "UNKNOWN"
        self._aligned_store[(asset_norm, tf_key)] = aligned
        return aligned

    def align_ext(
        self,
        name: str,
        align_to: str | ExtSeries | pd.DataFrame | pd.DatetimeIndex,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> ExtSeries:
        """
        Aligne une seule série sur une cible.
        """
        item = self.ext_item(name)
        ref_index = self._resolve_align_target(
            align_to=align_to,
            candidates=[item],
            start=start,
            end=end,
        )

        new_values = self._align_series_to_index(item=item, target_index=ref_index)
        return ExtSeries(
            name=item.name,
            values=new_values,
            index=ref_index.to_numpy(),
            source=item.source,
            asset=item.asset,
            tf=item.tf,
            timezone_shift=item.timezone_shift,
            original_name=item.original_name,
            meta={**item.meta, "aligned": True},
        )

    # =========================================================
    # MATRIX
    # =========================================================

    def to_matrix(
        self,
        tf: str | None = None,
        asset: str | None = None,
        names: list[str] | None = None,
        align_to: str | ExtSeries | OHLCVItem | pd.DataFrame | pd.DatetimeIndex | None = None,
        dropna: bool = False,
        use_aligned: bool = True,
    ) -> pd.DataFrame:
        """
        Fusionne les séries alignées en une matrice finale.

        Règles :
        - un seul DatetimeIndex commun
        - aucune colonne date séparée
        - uniquement les colonnes de valeurs
        """
        tf_norm = self._normalize_tf(tf) if tf is not None else None
        asset_norm = self._normalize_asset(asset) if asset is not None else None
        key = (asset_norm, tf_norm)

        if use_aligned and key in self._aligned_store and self._aligned_store[key]:
            aligned_dict = self._aligned_store[key]

            if names is not None:
                selected = {}
                for n in names:
                    canonical = self._resolve_ext_name(n)
                    if canonical in aligned_dict:
                        selected[canonical] = aligned_dict[canonical]
                aligned_dict = selected

            if not aligned_dict:
                return pd.DataFrame()

        else:
            aligned_dict = self.align_all(
                tf=tf_norm,
                asset=asset_norm,
                names=names,
                align_to=align_to,
                dropna=dropna,
            )

        if not aligned_dict:
            return pd.DataFrame()

        first_item = next(iter(aligned_dict.values()))
        idx = pd.DatetimeIndex(first_item.index)

        df = pd.DataFrame(index=idx)

        for name, item in aligned_dict.items():
            df[name] = item.values

        # sécurité : aucune colonne date parasite
        forbidden_date_cols = {"timestamp", "Timestamp", "date", "Date", "datetime", "Datetime", "time", "Time"}
        cols_to_drop = [c for c in df.columns if c in forbidden_date_cols]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        if dropna:
            df = df.dropna()

        return df

    def to_ohlcv_matrix(
        self,
        name: str | None = None,
        names: list[str] | None = None,
        tickers: list[str] | None = None,
        base_path: str | Path | None = None,
        autoload: bool = False,
        tf: str | None = None,
        asset: str | None = None,
        align_to: str | OHLCVItem | pd.DataFrame | pd.DatetimeIndex | None = None,
        dropna: bool = False,
        native_names_mode: str = "auto",
        timezone_shift: int | float = 0,
        col_map: dict[int | str, str] | None = None,
        sep: str = ",",
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Construit une matrice OHLCV single-asset ou multi-asset.

        Si `name` est fourni :
        - la matrice est stockée dans self._ohlcv_matrix_store[name]
        - elle est aussi retournée

        Deux modes d'entrée :
        - names   : OHLCV déjà enregistrés
        - tickers : chargement auto depuis CSV si autoload=True

        native_names_mode :
        - "auto"   -> noms natifs si un seul asset, sinon suffixés
        - "always" -> toujours noms natifs
        - "never"  -> toujours suffixés
        """
        if native_names_mode not in {"auto", "always", "never"}:
            raise ValueError("native_names_mode doit être 'auto', 'always' ou 'never'")

        # -----------------------------------------------------
        # 1. Autoload éventuel depuis tickers
        # -----------------------------------------------------
        if tickers is not None:
            loaded_names: list[str] = []
            for ticker in tickers:
                if overwrite:
                    if base_path is None:
                        raise ValueError("base_path is required when overwrite=True with tickers")
                    ohlcv_name = self.load_ohlcv_csv(
                        ticker=ticker,
                        base_path=base_path,
                        timezone_shift=timezone_shift,
                        col_map=col_map,
                        sep=sep,
                        overwrite=True,
                    )
                else:
                    try:
                        ohlcv_name = self._resolve_ohlcv_name(ticker)
                    except Exception:
                        if not autoload:
                            raise KeyError(
                                f"OHLCV '{ticker}' not loaded and autoload=False"
                            )
                        if base_path is None:
                            raise ValueError("base_path is required if autoload=True")

                        ohlcv_name = self.load_ohlcv_csv(
                            ticker=ticker,
                            base_path=base_path,
                            timezone_shift=timezone_shift,
                            col_map=col_map,
                            sep=sep,
                            overwrite=False,
                        )

                loaded_names.append(ohlcv_name)

            if names is None:
                names = loaded_names
            else:
                names = list(names) + loaded_names

        # -----------------------------------------------------
        # 2. Collecte des candidats
        # -----------------------------------------------------
        candidates = self._collect_ohlcv_candidates(
            names=names,
            tf=tf,
            asset=asset,
        )

        if not candidates:
            df = pd.DataFrame()
            df.attrs["tf"] = self._normalize_tf(tf) if tf is not None else None
            df.attrs["assets"] = [self._normalize_asset(asset)] if asset is not None else []
            df.attrs["source_matrix"] = name
            df.attrs["is_ohlcv_matrix"] = True

            if name is not None:
                self._ohlcv_matrix_store[name] = df.copy()
                self._ohlcv_matrix_store[name].attrs["tf"] = df.attrs["tf"]
                self._ohlcv_matrix_store[name].attrs["assets"] = df.attrs["assets"]
                self._ohlcv_matrix_store[name].attrs["source_matrix"] = name
                self._ohlcv_matrix_store[name].attrs["is_ohlcv_matrix"] = True

                self._ohlcv_matrix_meta[name] = {
                    "names": names,
                    "tickers": tickers,
                    "tf": self._normalize_tf(tf) if tf is not None else None,
                    "asset": self._normalize_asset(asset) if asset is not None else None,
                    "assets": [self._normalize_asset(asset)] if asset is not None else [],
                    "dropna": dropna,
                    "native_names_mode": native_names_mode,
                    "rows": 0,
                    "cols": 0,
                }
            return df

        # -----------------------------------------------------
        # 3. Index cible commun
        # -----------------------------------------------------
        ref_index = self._resolve_ohlcv_align_target(
            align_to=align_to,
            candidates=candidates,
        )

        # -----------------------------------------------------
        # 4. Alignement OHLCV vers index commun
        # -----------------------------------------------------
        aligned = [self._align_ohlcv_to_index(item, ref_index) for item in candidates]

        if dropna:
            aligned = self._dropna_ohlcv_items(aligned)

        if not aligned:
            df = pd.DataFrame()
            df.attrs["tf"] = self._normalize_tf(tf) if tf is not None else None
            df.attrs["assets"] = [self._normalize_asset(asset)] if asset is not None else []
            df.attrs["source_matrix"] = name
            df.attrs["is_ohlcv_matrix"] = True

            if name is not None:
                self._ohlcv_matrix_store[name] = df.copy()
                self._ohlcv_matrix_store[name].attrs["tf"] = df.attrs["tf"]
                self._ohlcv_matrix_store[name].attrs["assets"] = df.attrs["assets"]
                self._ohlcv_matrix_store[name].attrs["source_matrix"] = name
                self._ohlcv_matrix_store[name].attrs["is_ohlcv_matrix"] = True

                self._ohlcv_matrix_meta[name] = {
                    "names": names,
                    "tickers": tickers,
                    "tf": self._normalize_tf(tf) if tf is not None else None,
                    "asset": self._normalize_asset(asset) if asset is not None else None,
                    "assets": [self._normalize_asset(asset)] if asset is not None else [],
                    "dropna": dropna,
                    "native_names_mode": native_names_mode,
                    "rows": 0,
                    "cols": 0,
                }
            return df

        # -----------------------------------------------------
        # 5. Construction matrice finale
        # -----------------------------------------------------
        first_idx = pd.DatetimeIndex(aligned[0].index)
        df = pd.DataFrame(index=first_idx)

        single_asset = len(aligned) == 1

        if native_names_mode == "auto":
            use_native_names = single_asset
        elif native_names_mode == "always":
            use_native_names = True
        else:  # "never"
            use_native_names = False

        for item in aligned:
            suffix = ""
            if not use_native_names:
                suffix = f"_{item.asset}" if item.asset else f"_{item.name}"

            df[f"Open{suffix}"] = item.open
            df[f"High{suffix}"] = item.high
            df[f"Low{suffix}"] = item.low
            df[f"Close{suffix}"] = item.close

            if item.volume is not None:
                df[f"Volume{suffix}"] = item.volume

        if dropna:
            df = df.dropna()

        # sécurité : aucune colonne date parasite
        forbidden_date_cols = {"timestamp", "Timestamp", "date", "Date", "datetime", "Datetime", "time", "Time"}
        cols_to_drop = [c for c in df.columns if c in forbidden_date_cols]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        df.attrs["tf"] = self._infer_single_ohlcv_tf(aligned)
        df.attrs["assets"] = [x.asset for x in aligned]
        df.attrs["source_matrix"] = name
        df.attrs["is_ohlcv_matrix"] = True
        # -----------------------------------------------------
        # 6. Save éventuel
        # -----------------------------------------------------
        if name is not None:
            self._ohlcv_matrix_store[name] = df.copy()
            self._ohlcv_matrix_store[name].attrs["tf"] = self._infer_single_ohlcv_tf(aligned)
            self._ohlcv_matrix_store[name].attrs["assets"] = [x.asset for x in aligned]
            self._ohlcv_matrix_store[name].attrs["source_matrix"] = name
            self._ohlcv_matrix_store[name].attrs["is_ohlcv_matrix"] = True
            self._ohlcv_matrix_meta[name] = {
                "names": [x.name for x in aligned],
                "tickers": tickers,
                "tf": self._infer_single_ohlcv_tf(aligned),
                "assets": [x.asset for x in aligned],
                "dropna": dropna,
                "native_names_mode": native_names_mode,
                "rows": len(df),
                "cols": len(df.columns),
            }

        return df

    def _collect_ohlcv_candidates(
        self,
        names: list[str] | None = None,
        tf: str | None = None,
        asset: str | None = None,
    ) -> list[OHLCVItem]:
        tf_norm = self._normalize_tf(tf) if tf is not None else None
        asset_norm = self._normalize_asset(asset) if asset is not None else None

        if names is not None:
            items = [self.ohlcv(n) for n in names]
        else:
            items = list(self._ohlcv_store.values())

        if tf_norm is not None:
            items = [x for x in items if x.tf == tf_norm]

        if asset_norm is not None:
            items = [x for x in items if x.asset == asset_norm]

        return items

    def _resolve_ohlcv_align_target(
        self,
        align_to: str | OHLCVItem | pd.DataFrame | pd.DatetimeIndex | None,
        candidates: list[OHLCVItem],
    ) -> pd.DatetimeIndex:
        if align_to is None:
            ref = max(candidates, key=lambda x: len(x.index))
            return pd.DatetimeIndex(ref.index).sort_values().unique()

        if isinstance(align_to, str):
            item = self.ohlcv(align_to)
            return pd.DatetimeIndex(item.index).sort_values().unique()

        if isinstance(align_to, OHLCVItem):
            return pd.DatetimeIndex(align_to.index).sort_values().unique()

        if isinstance(align_to, pd.DataFrame):
            if not isinstance(align_to.index, pd.DatetimeIndex):
                raise ValueError("Le DataFrame donné à align_to doit avoir un DatetimeIndex")
            return pd.DatetimeIndex(align_to.index).sort_values().unique()

        if isinstance(align_to, pd.DatetimeIndex):
            return pd.DatetimeIndex(align_to).sort_values().unique()

        raise TypeError("align_to doit être None, un nom d'OHLCV, OHLCVItem, DataFrame ou DatetimeIndex")

    def _align_ohlcv_to_index(
        self,
        item: OHLCVItem,
        target_index: pd.DatetimeIndex,
    ) -> OHLCVItem:
        def _reindex(arr: np.ndarray) -> np.ndarray:
            s = pd.Series(arr, index=pd.DatetimeIndex(item.index))
            s = s[~s.index.duplicated(keep="last")].sort_index()
            return s.reindex(target_index).to_numpy()

        out_open = _reindex(item.open)
        out_high = _reindex(item.high)
        out_low = _reindex(item.low)
        out_close = _reindex(item.close)
        out_volume = _reindex(item.volume) if item.volume is not None else None

        return OHLCVItem(
            name=item.name,
            open=out_open,
            high=out_high,
            low=out_low,
            close=out_close,
            volume=out_volume,
            index=target_index.to_numpy(),
            source=item.source,
            asset=item.asset,
            tf=item.tf,
            timezone_shift=item.timezone_shift,
            original_columns=item.original_columns.copy(),
            meta={**item.meta, "aligned": True},
        )

    def _dropna_ohlcv_items(
        self,
        items: list[OHLCVItem],
    ) -> list[OHLCVItem]:
        if not items:
            return items

        idx = pd.DatetimeIndex(items[0].index)
        df = pd.DataFrame(index=idx)

        for item in items:
            asset_key = item.asset or item.name
            df[f"Open_{asset_key}"] = item.open
            df[f"High_{asset_key}"] = item.high
            df[f"Low_{asset_key}"] = item.low
            df[f"Close_{asset_key}"] = item.close
            if item.volume is not None:
                df[f"Volume_{asset_key}"] = item.volume

        df = df.dropna()

        out: list[OHLCVItem] = []
        for item in items:
            asset_key = item.asset or item.name
            volume = df[f"Volume_{asset_key}"].to_numpy() if item.volume is not None else None

            out.append(
                OHLCVItem(
                    name=item.name,
                    open=df[f"Open_{asset_key}"].to_numpy(),
                    high=df[f"High_{asset_key}"].to_numpy(),
                    low=df[f"Low_{asset_key}"].to_numpy(),
                    close=df[f"Close_{asset_key}"].to_numpy(),
                    volume=volume,
                    index=df.index.to_numpy(),
                    source=item.source,
                    asset=item.asset,
                    tf=item.tf,
                    timezone_shift=item.timezone_shift,
                    original_columns=item.original_columns.copy(),
                    meta=item.meta.copy(),
                )
            )

        return out

    def get_ohlcv_matrix_asset(
        self,
        name: str,
        asset: str,
        copy: bool = True,
    ) -> pd.DataFrame:
        df = self.ohlcv_matrix(name)
        asset_norm = self._normalize_asset(asset)

        suffix = f"_{asset_norm}"
        cols = [c for c in df.columns if c.endswith(suffix)]

        # single-asset matrix with native column names
        if not cols:
            native = ["Open", "High", "Low", "Close", "Volume"]
            native_cols = [c for c in native if c in df.columns]
            meta = self.ohlcv_matrix_meta(name)
            assets = meta.get("assets", []) or []

            if len(assets) == 1 and assets[0] == asset_norm and native_cols:
                out = df[native_cols].copy() if copy else df[native_cols]
                out.attrs["asset"] = asset_norm
                out.attrs["tf"] = meta.get("tf")
                out.attrs["source_matrix"] = name
                out.attrs["is_ohlcv_matrix"] = True
                return out

            raise KeyError(
                f"No columns found for asset '{asset_norm}' in matrix '{name}'."
            )

        out = df[cols].copy() if copy else df[cols]
        rename_map = {c: c[: -len(suffix)] for c in cols}
        out = out.rename(columns=rename_map)

        meta = self.ohlcv_matrix_meta(name)
        out.attrs["asset"] = asset_norm
        out.attrs["tf"] = meta.get("tf")
        out.attrs["source_matrix"] = name
        out.attrs["is_ohlcv_matrix"] = True
        return out

    def get_ohlcv_matrix_assets(
        self,
        name: str,
        assets: list[str],
        copy: bool = True,
    ) -> pd.DataFrame:
        df = self.ohlcv_matrix(name)
        assets_norm = [self._normalize_asset(a) for a in assets]

        selected_cols: list[str] = []
        for asset in assets_norm:
            suffix = f"_{asset}"
            selected_cols.extend([c for c in df.columns if c.endswith(suffix)])

        if not selected_cols:
            raise KeyError(
                f"No columns found for assets={assets_norm} in matrix '{name}'."
            )

        out = df[selected_cols].copy() if copy else df[selected_cols]

        meta = self.ohlcv_matrix_meta(name)
        out.attrs["assets"] = assets_norm
        out.attrs["tf"] = meta.get("tf")
        out.attrs["source_matrix"] = name
        out.attrs["is_ohlcv_matrix"] = True
        return out

    # getters 
    def ohlcv_matrix(self, name: str) -> pd.DataFrame:
        if name not in self._ohlcv_matrix_store:
            available = ", ".join(sorted(self._ohlcv_matrix_store.keys())[:20])
            raise KeyError(f"Matrice OHLCV introuvable : '{name}'. Disponibles : {available}")
        return self._ohlcv_matrix_store[name].copy()

    def list_ohlcv_matrices(self) -> list[str]:
        return sorted(self._ohlcv_matrix_store.keys())

    def ohlcv_matrix_meta(self, name: str) -> dict[str, Any]:
        if name not in self._ohlcv_matrix_meta:
            available = ", ".join(sorted(self._ohlcv_matrix_meta.keys())[:20])
            raise KeyError(f"Métadonnées de matrice OHLCV introuvables : '{name}'. Disponibles : {available}")
        return self._ohlcv_matrix_meta[name].copy()

    def delete_ohlcv_matrix(self, name: str) -> None:
        self._ohlcv_matrix_store.pop(name, None)
        self._ohlcv_matrix_meta.pop(name, None)
    # =========================================================
    # INTERNAL IMPORT HELPERS
    # =========================================================

    def _apply_col_map(
        self,
        df: pd.DataFrame,
        col_map: dict[int | str, str] | None,
    ) -> pd.DataFrame:
        if not col_map:
            return df

        rename_map: dict[str, str] = {}
        cols = list(df.columns)

        for old_key, new_name in col_map.items():
            if isinstance(old_key, int):
                if old_key < 0 or old_key >= len(cols):
                    raise IndexError(f"Index de colonne invalide dans col_map : {old_key}")
                old_name = cols[old_key]
            else:
                if old_key not in df.columns:
                    raise KeyError(f"Colonne introuvable dans col_map : {old_key}")
                old_name = old_key

            rename_map[old_name] = new_name

        return df.rename(columns=rename_map)

    def _detect_time_column(self, df: pd.DataFrame) -> str | None:
        candidates = (
            "timestamp", "Timestamp",
            "date", "Date",
            "datetime", "Datetime",
            "time", "Time",
        )
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _register_ext_df(
        self,
        df: pd.DataFrame,
        source: str,
        asset: str | None,
        tf: str | None,
        timezone_shift: int | float,
        time_col: str | None,
    ) -> None:
        asset_norm = self._normalize_asset(asset)
        tf_norm = self._normalize_tf(tf)

        index_arr = None
        if time_col is not None:
            index_arr = df[time_col].to_numpy()

        non_value_cols = set()
        if time_col is not None:
            non_value_cols.add(time_col)

        short_candidates: list[tuple[str, str]] = []

        for col in df.columns:
            if col in non_value_cols:
                continue

            series_name = self._build_ext_name(col, asset_norm, tf_norm)
            values = df[col].to_numpy()

            item = ExtSeries(
                name=series_name,
                values=np.asarray(values),
                index=None if index_arr is None else np.asarray(index_arr),
                source=source,
                asset=asset_norm,
                tf=tf_norm,
                timezone_shift=timezone_shift,
                original_name=col,
                meta={
                    "kind": "ext",
                    "source": source,
                    "asset": asset_norm,
                    "tf": tf_norm,
                    "timezone_shift": timezone_shift,
                    "time_col": time_col,
                    "original_name": col,
                },
            )

            self._ext_store[series_name] = item
            self._meta_store[series_name] = item.meta.copy()
            short_candidates.append((col, series_name))

        self._refresh_aliases(short_candidates)

    def _register_ohlcv_df(
        self,
        df: pd.DataFrame,
        source: str,
        asset: str | None,
        tf: str | None,
        timezone_shift: int | float,
        time_col: str | None,
    ) -> None:
        asset_norm = self._normalize_asset(asset)
        tf_norm = self._normalize_tf(tf)

        required = ["Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes OHLC manquantes pour kind='ohlcv' : {missing}")

        if time_col is None:
            raise ValueError("kind='ohlcv' nécessite une colonne temps détectable")

        volume = df["Volume"].to_numpy() if "Volume" in df.columns else None

        base_name = self._build_ohlcv_name(asset_norm, tf_norm)

        item = OHLCVItem(
            name=base_name,
            open=df["Open"].to_numpy(),
            high=df["High"].to_numpy(),
            low=df["Low"].to_numpy(),
            close=df["Close"].to_numpy(),
            volume=volume,
            index=df[time_col].to_numpy(),
            source=source,
            asset=asset_norm,
            tf=tf_norm,
            timezone_shift=timezone_shift,
            original_columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume" if "Volume" in df.columns else "",
            },
            meta={
                "kind": "ohlcv",
                "source": source,
                "asset": asset_norm,
                "tf": tf_norm,
                "timezone_shift": timezone_shift,
                "time_col": time_col,
            },
        )

        self._ohlcv_store[base_name] = item
        self._meta_store[base_name] = item.meta.copy()

    def _normalize_asset(self, asset: str | None) -> str | None:
        if asset is None:
            return None
        asset = str(asset).strip()
        if not asset:
            return None
        return asset.upper()

    def _parse_ticker_name(self, ticker: str) -> tuple[str, str]:
        """
        Exemple :
            XAUUSD_M5 -> (XAUUSD, M5)
        """
        s = str(ticker).strip()
        if "_" not in s:
            raise ValueError(f"Ticker invalide, format attendu ASSET_TF : {ticker}")

        asset, tf = s.rsplit("_", 1)
        asset = self._normalize_asset(asset)
        tf = self._normalize_tf(tf)
        return asset, tf

    def _infer_single_ohlcv_tf(self, items: list[OHLCVItem]) -> str | None:
        tfs = {x.tf for x in items if x.tf is not None}
        if len(tfs) == 1:
            return next(iter(tfs))
        return None

    def _normalize_tf(self, tf: str | None) -> str | None:
        if tf is None:
            return None
        tf = str(tf).strip().upper()
        if not tf:
            return None
        return tf

    def _build_ext_name(self, base_col: str, asset: str | None, tf: str | None) -> str:
        name = str(base_col).strip()
        if asset and tf:
            return f"{name}_{asset}_{tf}"
        if asset:
            return f"{name}_{asset}"
        if tf:
            return f"{name}_{tf}"
        return name

    def _refresh_aliases(self, short_candidates: list[tuple[str, str]]) -> None:
        counts: dict[str, int] = {}
        last_seen: dict[str, str] = {}

        for canonical_name, item in self._ext_store.items():
            short = item.original_name or canonical_name
            counts[short] = counts.get(short, 0) + 1
            last_seen[short] = canonical_name

        self._aliases.clear()
        for short, n in counts.items():
            if n == 1:
                self._aliases[short] = last_seen[short]

    def _resolve_ext_name(self, name: str) -> str:
        if name in self._ext_store:
            return name
        if name in self._aliases:
            return self._aliases[name]

        available = ", ".join(sorted(self._ext_store.keys())[:20])
        raise KeyError(
            f"Série externe introuvable : '{name}'. "
            f"Exemples disponibles : {available}"
        )

    def _resolve_ohlcv_name(self, name: str) -> str:
        if name in self._ohlcv_store:
            return name

        key = str(name).strip().upper()
        for k in self._ohlcv_store:
            if k.upper() == key:
                return k

        available = ", ".join(sorted(self._ohlcv_store.keys())[:20])
        raise KeyError(
            f"OHLCV introuvable : '{name}'. Exemples disponibles : {available}"
        )

    def _build_ohlcv_name(self, asset: str | None, tf: str | None) -> str:
        if asset and tf:
            return f"{asset}_{tf}"
        if asset:
            return asset
        if tf:
            return f"OHLCV_{tf}"
        return "OHLCV"

    # =========================================================
    # INTERNAL CROP HELPERS
    # =========================================================

    def _crop_extseries(
        self,
        item: ExtSeries,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> ExtSeries:
        if item.index is None:
            return item

        idx = pd.DatetimeIndex(item.index)
        mask = np.ones(len(idx), dtype=bool)

        if start is not None:
            mask &= idx >= start
        if end is not None:
            mask &= idx <= end

        return ExtSeries(
            name=item.name,
            values=item.values[mask],
            index=idx[mask].to_numpy(),
            source=item.source,
            asset=item.asset,
            tf=item.tf,
            timezone_shift=item.timezone_shift,
            original_name=item.original_name,
            meta=item.meta.copy(),
        )

    def _crop_dataframe(
        self,
        df: pd.DataFrame,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> pd.DataFrame:
        out = df.copy()

        if not isinstance(out.index, pd.DatetimeIndex):
            time_col = self._detect_time_column(out)
            if time_col is None:
                raise ValueError("Le DataFrame n'a pas d'index DatetimeIndex ni de colonne temps détectable")
            out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
            out = out.set_index(time_col)

        if start is not None:
            out = out[out.index >= start]
        if end is not None:
            out = out[out.index <= end]

        return out

    # =========================================================
    # INTERNAL ALIGN HELPERS
    # =========================================================

    def _collect_alignment_candidates(
        self,
        tf: str | None = None,
        asset: str | None = None,
        names: list[str] | None = None,
    ) -> list[ExtSeries]:
        tf_norm = self._normalize_tf(tf) if tf is not None else None
        asset_norm = self._normalize_asset(asset) if asset is not None else None

        if names is not None:
            items = [self.ext_item(n) for n in names]
            if tf_norm is not None:
                items = [x for x in items if x.tf == tf_norm]
            if asset_norm is not None:
                items = [x for x in items if x.asset == asset_norm]
            return items

        items = list(self._ext_store.values())
        if tf_norm is not None:
            items = [x for x in items if x.tf == tf_norm]
        if asset_norm is not None:
            items = [x for x in items if x.asset == asset_norm]
        return items

    def _resolve_align_target(
        self,
        align_to: str | ExtSeries | pd.DataFrame | pd.DatetimeIndex | None,
        candidates: list[ExtSeries],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DatetimeIndex:
        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None

        if align_to is None:
            # on prend l'index le plus long
            valid = [x for x in candidates if x.index is not None]
            if not valid:
                raise ValueError("Aucune série candidate ne possède d'index temporel")
            ref = max(valid, key=lambda x: len(x.index))
            idx = pd.DatetimeIndex(ref.index)

        elif isinstance(align_to, str):
            item = self.ext_item(align_to)
            if item.index is None:
                raise ValueError(f"La série '{align_to}' n'a pas d'index")
            idx = pd.DatetimeIndex(item.index)

        elif isinstance(align_to, ExtSeries):
            if align_to.index is None:
                raise ValueError("L'ExtSeries fournie n'a pas d'index")
            idx = pd.DatetimeIndex(align_to.index)

        elif isinstance(align_to, pd.DataFrame):
            if not isinstance(align_to.index, pd.DatetimeIndex):
                raise ValueError("Le DataFrame fourni doit avoir un DatetimeIndex")
            idx = align_to.index

        elif isinstance(align_to, pd.DatetimeIndex):
            idx = align_to

        else:
            raise TypeError("align_to doit être None, un nom, ExtSeries, DataFrame ou DatetimeIndex")

        if start_ts is not None:
            idx = idx[idx >= start_ts]
        if end_ts is not None:
            idx = idx[idx <= end_ts]

        return idx.sort_values().unique()

    def _align_series_to_index(
        self,
        item: ExtSeries,
        target_index: pd.DatetimeIndex,
    ) -> np.ndarray:
        if item.index is None:
            return np.full(len(target_index), np.nan)

        s = pd.Series(item.values, index=pd.DatetimeIndex(item.index))
        s = s[~s.index.duplicated(keep="last")]
        s = s.sort_index()

        aligned = s.reindex(target_index)
        return aligned.to_numpy()

    def _dropna_aligned_dict(self, aligned: dict[str, ExtSeries]) -> dict[str, ExtSeries]:
        if not aligned:
            return aligned

        first_item = next(iter(aligned.values()))
        idx = pd.DatetimeIndex(first_item.index)
        df = pd.DataFrame(index=idx)

        for name, item in aligned.items():
            df[name] = item.values

        df = df.dropna()

        out: dict[str, ExtSeries] = {}
        for name, item in aligned.items():
            out[name] = ExtSeries(
                name=item.name,
                values=df[name].to_numpy(),
                index=df.index.to_numpy(),
                source=item.source,
                asset=item.asset,
                tf=item.tf,
                timezone_shift=item.timezone_shift,
                original_name=item.original_name,
                meta=item.meta.copy(),
            )
        return out

    def _infer_single_tf(self, candidates: list[ExtSeries]) -> str | None:
        tfs = {x.tf for x in candidates if x.tf is not None}
        if len(tfs) == 1:
            return next(iter(tfs))
        return None

    def _infer_single_asset(self, candidates: list[ExtSeries]) -> str | None:
        assets = {x.asset for x in candidates if x.asset is not None}
        if len(assets) == 1:
            return next(iter(assets))
        return None
