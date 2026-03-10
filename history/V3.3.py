import pandas as pd 
import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Union
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import stats as scipy_stats

@dataclass(slots=True)
class BacktestConfig:

    # --- Signal Generation and strategy Choice & metrics ---
    strategy: str = "close_vs_ema"
    entry_delay: int=1
    exit_delay: int=1
    use_exit_signal: bool = False
    compute_metrics: bool = False  # True = retourne aussi les métriques
    
    # --- Asset type ---
    crypto: bool = False 

    # --- Data / indicators ---
    timezone_shift: int = 0
    period_1: int = 50
    period_2: int = 100
    max_gap_size: Optional[float] = None
    period_atr: int = 14
    
    # --- Filters ---
    Candle_Size_filter: bool = False
    Previous_Candle_same_direction: bool = True
    min_size_pct: float = 0.001
    max_size_pct: float = 0.01
    Exit_filter_EMA1: int = 50
    Exit_filter_EMA2: int = 100

    # --- TP/SL ---
    tp_pct: float = None
    sl_pct: float = None
    use_atr_sl_tp: int = 0       # 0=fixed, 1=ATR TP only, -1=ATR SL only, 2=both
    tp_atr_mult: float = None
    sl_atr_mult: float = None

    # --- Exit mode toggles ---
    EMA1_TP: bool = False
    EMA2_TP: bool = False
    EMA_CROSS_TP: bool = False
    EMA_SL: bool = False

    # --- Entries cap ---
    MaxEntries4Periods: bool = False
    ME_X: int = 2
    ME_Period_Y: int = 8
    ME_reset_mode: Optional[str] = None  # "day" / "session" / None

    # --- Engine behavior ---
    allow_exit_on_entry_bar: bool = True
    multi_entry: bool = True
    reverse_mode: bool = False # close an opened position if an opposit signal occur

    # --- BE / runner trailing ---
    be_trigger_pct: Optional[float] = None
    be_offset_pct: float = 0.0
    be_delay_bars: int = 0
    trailing_trigger_pct: Optional[float] = None
    runner_trailing_mult: float = 2.0

    # --- Time windows ---
    time_window_1: Optional[str] = None
    time_window_2: Optional[str] = None
    time_window_3: Optional[str] = None

    # --- Observation for MAE-MFE hold ---
    observation_hours: Optional[float] = None
    timeframe_minutes: int = None

    # --- Mae/Mfe calculations ---
    track_mae_mfe: bool = False

    def __post_init__(self):
        if self.period_1 <= 0 or self.period_2 <= 0:
            raise ValueError("EMA periods must be positive")
        if self.period_1 >= self.period_2:
            raise ValueError("period_1 must be less than period_2")
        if self.ME_X <= 0:
            raise ValueError("ME_X must be positive")
        if self.use_atr_sl_tp not in (0, 1, -1, 2):
            raise ValueError("use_atr_sl_tp must be 0=No use, 1=tp*atr & sl pct, -1=sl*atr & tp pct or 2=Both*atr")
        if self.entry_delay <=0:
            print(
                "Warning: For unbiased results, keep the entry_delay argument >= 1. "
                "Otherwise, the engine will look at future candles to enter on the current one. "
                "It is your responsibility if you choose to introduce this look-ahead bias."
                "For information, the engine interpret the signal candle independently from its generation"
                "sig_prev = self.signals[i - cfg.entry_delay]"
            )
        if self.exit_delay <= 0:  # ← ajouter ici
            print("Warning: exit_delay <= 0 introduces look-ahead bias on exit signals, the same way it does with entry_delay")

@dataclass(slots=True)
class SimulationConfig:
    
    # --- Commission ---
    commission_pct: float = 0.0      # % du prix d'entrée/sortie
    commission_fixed: float = 0.0    # montant fixe par trade
    
    # --- Spread ---
    spread_pct: float = 0.0          # spread simulé en % du prix
    
    # --- Slippage ---
    slippage_pct: float = 0.0        # slippage en % sur entrée et sortie
    
    # --- Capital ---
    initial_capital: float = 10000.0
    
    # --- Timeframe pour Sharpe annualisé ---
    timeframe_minutes: int = 5

    # --- Metrics ---
    alpha4Var: float = 5
    period_freq: str = "ME"  # "D" / "W" / "ME" / "YE"

class Strategy_Signal:
        @staticmethod
        def apply(df, cfg):
            if cfg.strategy == "close_vs_ema":
                return Strategy_Signal.ema_cross(df, cfg.period_1, cfg.period_2, cfg.max_gap_size)
            else:
                raise ValueError(f"Unknown strategy: {cfg.strategy}")
            
        @staticmethod
        def ema_cross(df: pd.DataFrame, period_1=50, period_2=100, max_gap_size=None) -> pd.DataFrame:
            d = df.copy()
            d["EMA1"] = d["Close"].ewm(span=period_1, adjust=False).mean()
            d["EMA2"] = d["Close"].ewm(span=period_2, adjust=False).mean()

            if max_gap_size is not None:
                gap_pct = (d["Open"] - d["Close"].shift(1)).abs() / d["Close"].shift(1)
                gap_filter = gap_pct < max_gap_size
            else:
                gap_filter = pd.Series(True, index=d.index)

            d["Signal"] = np.where(
                (d["EMA1"].shift(1) < d["Close"].shift(1)) & (d["EMA1"] > d["Close"]) & gap_filter,
                -1,
                np.where(
                    (d["EMA1"].shift(1) > d["Close"].shift(1)) & (d["EMA1"] < d["Close"]) & gap_filter,
                    1,
                    0,
                ),
            )

            d["Signal2"] = 0
            bull = (d["EMA2"].shift(1) > d["Close"].shift(1)) & (d["EMA2"] < d["Close"]) & gap_filter
            bear = (d["EMA2"].shift(1) < d["Close"].shift(1)) & (d["EMA2"] > d["Close"]) & gap_filter
            d.loc[bull, "Signal2"] = 1
            d.loc[bear, "Signal2"] = -1

            d["EMA_CROSS"] = 0
            down_cross = (d["EMA1"].shift(1) > d["EMA2"].shift(1)) & (d["EMA1"] < d["EMA2"]) & gap_filter
            up_cross = (d["EMA1"].shift(1) < d["EMA2"].shift(1)) & (d["EMA1"] > d["EMA2"]) & gap_filter
            d.loc[up_cross, "EMA_CROSS"] = 1
            d.loc[down_cross, "EMA_CROSS"] = -1

            #d["Entry_Price"] = d["Open"].where(d["Signal"].shift(1) != 0)
            return d
   
class DataPipeline:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def fetchdata(self, ticker: str, start: str, end: str, timezone_shift=1) -> pd.DataFrame:
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

        df = df.copy()
        df["ATR"] = tr.rolling(period).mean()
        return df

    @staticmethod
    def apply_exitfilter_indicators(df, cfg):
        if any([cfg.EMA1_TP, cfg.EMA2_TP, cfg.EMA_CROSS_TP]):
            df["EMA1_exit"] = df["Close"].ewm(span=cfg.Exit_filter_EMA1, adjust=False).mean()
            df["EMA2_exit"] = df["Close"].ewm(span=cfg.Exit_filter_EMA2, adjust=False).mean()
        return df

    def prepare(self, ticker, start, end, cfg, strategy_fn, plot=False, **strategy_kwargs):
        df = self.fetchdata(ticker, start, end, timezone_shift=cfg.timezone_shift)  # ← ajouter
        df = self.compute_atr(df, cfg.period_atr)
        df = strategy_fn(df, **strategy_kwargs)
        df = self.apply_exitfilter_indicators(df, cfg)
        if plot:
            DataPipeline._plot_signals(df, crypto=cfg.crypto)
        return df
    
    @staticmethod
    def _plot_signals(df, title="Signal preparation", crypto=False):
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ))

        long_signals  = df[df["Signal"] == 1]
        short_signals = df[df["Signal"] == -1]

        fig.add_trace(go.Scatter(
            x=long_signals.index,
            y=long_signals["Low"] * 0.999,
            mode="markers",
            marker=dict(symbol="triangle-up", size=8, color="lime"),
            name="Long signal"
        ))

        fig.add_trace(go.Scatter(
            x=short_signals.index,
            y=short_signals["High"] * 1.001,
            mode="markers",
            marker=dict(symbol="triangle-down", size=8, color="red"),
            name="Short signal"
        ))

        # Signaux sortie — affiche le type (1, 2, -2, 3, tag)
        if "ExitSignal" in df.columns:
            exit_bars = df[df["ExitSignal"] != 0]

            # Label lisible selon le mode
            def exit_label(v):
                if v == 1:   return "LIFO"
                if v == 2:   return "ALL_L"   # ferme tous longs
                if v == -2:  return "ALL_S"   # ferme tous shorts
                if v == 3:   return "ALL"
                return str(int(v))            # tag numérique

            labels = exit_bars["ExitSignal"].apply(exit_label)

            fig.add_trace(go.Scatter(
                x=exit_bars.index,
                y=exit_bars["High"] * 1.003,
                mode="markers+text",
                marker=dict(symbol="x", size=9, color="orange"),
                text=labels,
                textposition="top center",
                textfont=dict(color="orange", size=10),
                name="Exit signal"
            ))

        rangebreaks = [] if crypto else [dict(bounds=["sat", "mon"])]

        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            xaxis=dict(rangebreaks=rangebreaks),
            hovermode="x unified",
            height=700
        )

        fig.show()

    def build(self, ticker, start, end, cfg):
        df = self.fetchdata(ticker, start, end, timezone_shift=cfg.timezone_shift)
        df = self.compute_atr(df, cfg.period_atr)
        df = Strategy_Signal.apply(df, cfg)
        df = self.apply_exitfilter_indicators(df, cfg)  # ← self, pas Strategy_Signal
        return df

class BacktestEngine:
    Bar = namedtuple("Bar", ["Open", "High", "Low", "Close", "name"])

    def __init__(self, df: pd.DataFrame, cfg: BacktestConfig):
        self.df = df
        self.cfg = cfg

        # State
        self.positions = []
        self.trades = []
        self.recent_entries = []
        self.last_session_id = None
        self.last_day = None

        # Observation
        self.use_observation = cfg.observation_hours is not None
        self.pending_observations = []
        self.observation_bars = int(cfg.observation_hours * 60 / cfg.timeframe_minutes) if self.use_observation else 0

        # Precompute arrays (exactement comme toi)
        self.opens = df["Open"].to_numpy()
        self.highs = df["High"].to_numpy()
        self.lows = df["Low"].to_numpy()
        self.closes = df["Close"].to_numpy()
        self.signals = df["Signal"].to_numpy()
        self.exit_signals  = df["ExitSignal"].to_numpy()  if "ExitSignal"  in df.columns else None
        self.signal_tags   = df["SignalTag"].to_numpy()   if "SignalTag"   in df.columns else None
        self.atrs = df["ATR"].to_numpy() if "ATR" in df.columns else None
        self.ema1s = df["EMA1_exit"].to_numpy() if "EMA1_exit" in df.columns else None
        self.ema2s = df["EMA2_exit"].to_numpy() if "EMA2_exit" in df.columns else None
        self.index = df.index
        self.has_exit_signal = self.exit_signals is not None
        self.has_signal_tags = self.signal_tags is not None

        self.parsed_windows = [
            self.parse_window(cfg.time_window_1),
            self.parse_window(cfg.time_window_2),
            self.parse_window(cfg.time_window_3),
        ]

        self.exit_mode = "ema" if sum([cfg.EMA1_TP, cfg.EMA2_TP, cfg.EMA_CROSS_TP]) == 1 else "fixed"

    # ---------  Ancienne fonctions du V2, mais en methods) ---------

    @classmethod
    def from_ticker(cls, pipeline: DataPipeline, ticker: str, start: str, end: str, cfg: BacktestConfig):
        df = pipeline.build(ticker, start, end, cfg)
        return cls(df, cfg)

    @classmethod #**kwargs is the mecanism Python used to allow an undefined number of argument. since the engine wont know the number of argument a strategy has
    def from_df(cls, pipeline, ticker, start, end, cfg, strategy_fn, **strategy_kwargs):
        """
        strategy_kwargs : any parameters to pass to strategy_fn
        e.g. rsi_period=20, oversold=25,
        update: ajout de prepare data qui sépare l'engine du calcul des signaux 
        """
        df = pipeline.prepare(ticker, start, end, cfg, strategy_fn, **strategy_kwargs)
        return cls(df, cfg)

    @staticmethod
    def parse_window(w):
        if w is None:
            return None
        s, e = w.split("-")
        return pd.to_datetime(s).time(), pd.to_datetime(e).time()

    @staticmethod
    def time_in_window_fast(ts, parsed_window):
        if parsed_window is None:
            return False
        start, end = parsed_window
        current = ts.time()
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    @staticmethod
    def _update_be_logic(pos, i, bar, be_trigger_pct, be_offset_pct, be_delay_bars):
        position    = pos["side"]
        entry_price = pos["entry_price"]
        base_sl     = pos["sl"]
        entry_index = pos["entry_index"]
        be_armed      = pos["be_armed"]
        pending_be_sl = pos["pending_be_sl"]
        be_arm_index  = pos["be_arm_index"]
        be_active     = pos["be_active"]

        if be_trigger_pct is None or position == 0:
            return base_sl, be_armed, pending_be_sl, be_arm_index, be_active

        delay_ok = (i - entry_index) >= be_delay_bars

        # ===== LONG =====
        if position == 1 and delay_ok and not be_armed:
            trigger_price = entry_price * (1 + be_trigger_pct)
            if bar.High >= trigger_price:
                be_armed      = True
                be_arm_index  = i
                pending_be_sl = entry_price * (1 + be_offset_pct)

        # ===== SHORT =====
        if position == -1 and delay_ok and not be_armed:
            trigger_price = entry_price * (1 - be_trigger_pct)
            if bar.Low <= trigger_price:
                be_armed      = True
                be_arm_index  = i
                pending_be_sl = entry_price * (1 - be_offset_pct)

        # ===== Activation bougie suivante =====
        if be_armed and i > be_arm_index:
            if position == 1:
                base_sl = max(base_sl, pending_be_sl)
            elif position == -1:
                base_sl = min(base_sl, pending_be_sl)
            be_armed      = False
            be_active     = True
            pending_be_sl = None
            be_arm_index  = None

        return base_sl, be_armed, pending_be_sl, be_arm_index, be_active

    @staticmethod
    def trade_analysis(position, exit_price, entry_price, entry_time, ts, exit_reason,
                    mae=None, mfe=None, mae_h=None, mfe_h=None):

        ret = position * (exit_price - entry_price) / entry_price

        trade = {
            "entry_time": entry_time,
            "exit_time": ts,
            "side": position,
            "entry": entry_price,
            "exit": exit_price,
            "return": ret,
            "reason": exit_reason,
        }

        # --- Ajouter MAE / MFE uniquement si présents ---
        if (mae is not None) and (mfe is not None):
            trade["mae"] = mae
            trade["mfe"] = mfe
            trade["capture_ratio"] = ret / mfe if mfe > 0 else None

        # --- Ajouter Hold metrics uniquement si présents ---
        if (mae_h is not None) and (mfe_h is not None):
            trade["hold_mfe"] = mfe_h
            trade["hold_mae"] = mae_h

        return trade

    @staticmethod
    def Tp_Sl_prices(side, entry_price, tp_pct=None, sl_pct=None,
                 use_atr=False, atr_value=None, tp_atr_mult=None, sl_atr_mult=None):

        if use_atr  == 2:
            # Les deux en ATR
            if atr_value is None:
                return None, None
            tp = entry_price + side * atr_value * tp_atr_mult
            sl = entry_price - side * atr_value * sl_atr_mult

        elif use_atr == 1:
            # TP en ATR, SL fixe
            if atr_value is None:
                return None, None
            tp = entry_price + side * atr_value * tp_atr_mult
            sl = entry_price * (1 - side * sl_pct)

        elif use_atr == -1:
            # SL en ATR, TP fixe
            if atr_value is None:
                return None, None
            tp = entry_price * (1 + side * tp_pct)
            sl = entry_price - side * atr_value * sl_atr_mult

        else:
            # Les deux fixe
            tp = entry_price * (1 + side * tp_pct)
            sl = entry_price * (1 - side * sl_pct)

        return tp, sl

    @staticmethod
    def _update_trailing_runner(pos, bar, atr_value, trailing_trigger_pct, trailing_mult):
        side = pos["side"]
        ep   = pos["entry_price"]

        # 1. Armement
        if not pos["runner_active"] and not pos.get("runner_armed", False):
            threshold = ep * (1 + side * trailing_trigger_pct)
            if (side == 1 and bar.High >= threshold) or \
            (side == -1 and bar.Low <= threshold):
                pos["runner_armed"]     = True
                pos["runner_threshold"] = threshold

        # Activation bougie suivante
        if pos.get("runner_armed") and not pos["runner_active"]:
            pos["runner_active"] = True
            pos["runner_armed"]  = False
            initial_sl = bar.Close - side * atr_value * trailing_mult
            if side == 1:
                pos["runner_sl"] = max(initial_sl, ep)
            else:
                pos["runner_sl"] = min(initial_sl, ep)
            return pos

        # Trailing ATR dynamique
        if pos["runner_active"]:
            new_sl = bar.Close - side * atr_value * trailing_mult
            if side == 1:
                pos["runner_sl"] = max(pos["runner_sl"], new_sl, ep)
            else:
                pos["runner_sl"] = min(pos["runner_sl"], new_sl, ep)

        return pos

    @staticmethod
    def _gap_filter(opens, closes, i, max_gap_size):
        """
        Retourne False si le gap entre close[i-1] et open[i] est trop grand.
        max_gap_size = pourcentage max du gap autorisé ex: 0.002 = 0.2%
        """
        if max_gap_size is None:
            return True  # pas de filtre — entrée autorisée
        gap_pct = abs(opens[i] - closes[i-1]) / closes[i-1]
        return gap_pct < max_gap_size

    def _apply_reverse(self, sig_prev, bar_open, ts):
        cfg = self.cfg
        if not cfg.reverse_mode or sig_prev == 0:
            return

        closed = []
        surviving = []
        for pos in self.positions:
            if pos["side"] == -sig_prev:
                if cfg.track_mae_mfe is not True:
                    closed.append(self.trade_analysis(
                        position=pos["side"],
                        exit_price=bar_open,
                        entry_price=pos["entry_price"],
                        entry_time=pos["entry_time"],
                        ts=ts,
                        exit_reason="REVERSE",
                    ))
                else:
                    closed.append(self.trade_analysis(
                        position=pos["side"],
                        exit_price=bar_open,
                        entry_price=pos["entry_price"],
                        entry_time=pos["entry_time"],
                        ts=ts,
                        exit_reason="REVERSE",
                        mae=pos.get("mae", 0.0),
                        mfe=pos.get("mfe", 0.0)
                    ))
            else:
                surviving.append(pos)

        self.positions = surviving
        self.trades.extend(closed)

    def get_active_session(self, ts):
        for idx, w in enumerate(self.parsed_windows):
            if w is not None and self.time_in_window_fast(ts, w):
                return idx
        return None

    def _entry_logic(self, i, ts, bar, sig_prev):

        # Time filter
        cfg = self.cfg
        atr_value = self.atrs[i-cfg.entry_delay] if (cfg.use_atr_sl_tp and self.atrs is not None) else None
        # windows est déjà garanti dans __init__
        if any(w is not None for w in self.parsed_windows):
            in_any_window = any(
                self.time_in_window_fast(ts, w) 
                for w in self.parsed_windows 
                if w is not None
            )
            if not in_any_window:
                return None

        if sig_prev == 0:
            return None

        body_pct = abs(self.opens[i-cfg.entry_delay] - self.closes[i-cfg.entry_delay]) / self.closes[i-cfg.entry_delay]
        size_ok = cfg.min_size_pct < body_pct < cfg.max_size_pct

        direction_ok = (
            (sig_prev == 1 and self.closes[i-cfg.entry_delay] > self.opens[i-cfg.entry_delay]) or
            (sig_prev == -1 and self.closes[i-cfg.entry_delay] < self.opens[i-cfg.entry_delay])
        )

        # Filtre 
        if cfg.Candle_Size_filter:
            if not (size_ok and (not cfg.Previous_Candle_same_direction or direction_ok)):
                return None

        # entrée exécutée si filtre OK ou filtre désactivé
        entry_price = bar.Open

        if cfg.use_atr_sl_tp and atr_value is not None and np.isnan(atr_value):
            return None
        if not self._gap_filter(self.opens, self.closes, i, cfg.max_gap_size):
            return None

        tp, sl = self.Tp_Sl_prices(
            side=sig_prev,
            entry_price=entry_price,
            tp_pct=cfg.tp_pct,
            sl_pct=cfg.sl_pct,
            use_atr=cfg.use_atr_sl_tp,
            tp_atr_mult=cfg.tp_atr_mult,
            sl_atr_mult=cfg.sl_atr_mult,
            atr_value=atr_value
        )

        new_position_dict={
        "side": sig_prev,
        "entry_price": bar.Open,
        "entry_time": ts,
        "entry_index": i,
        "tp": tp,
        "sl": sl,
        "be_armed": False,
        "pending_be_sl": None,
        "be_arm_index": None,
        "be_active": False,
        "runner_active": False,
        "runner_armed":  False,   
        "runner_sl":     None,
        "mae":0.0,
        "mfe":0.0,
        "tag": self.signal_tags[i] if self.signal_tags is not None else None       
        }
        return new_position_dict

    def _max_entries_reset(self,ts):
        cfg = self.cfg
        current_day = ts.date()

        # --- Reset journalier ---
        if cfg.ME_reset_mode == "day":
            if self.last_day is None:
                self.last_day = current_day
            elif current_day != self.last_day:
                self.recent_entries = []
                self.last_day = current_day

        # --- Reset par session ---
        elif cfg.ME_reset_mode == "session":
            current_session = self.get_active_session(ts)
            if self.last_session_id is None:
                self.last_session_id = current_session
            elif current_session != self.last_session_id:
                self.recent_entries = []
                self.last_session_id = current_session

    def _max_entries_filter(self, i, entry):
        cfg = self.cfg
        if not cfg.MaxEntries4Periods:
            return entry

        # Nettoyage fenêtre glissante
        cutoff = i - cfg.ME_Period_Y
        j = 0
        while j < len(self.recent_entries) and self.recent_entries[j] < cutoff:
            j += 1
        self.recent_entries = self.recent_entries[j:]

        # Vérification quota
        if len(self.recent_entries) >= cfg.ME_X:
            return None

        return entry

    def _exit_logic(self, i, bar, pos):
        cfg = self.cfg

        if cfg.use_exit_signal and self.has_exit_signal :
            exit_sig = self.exit_signals[i-cfg.exit_delay]
            
            if exit_sig != 0:  # ← seulement si signal actif
                if exit_sig == 1:
                    pass  # LIFO géré dans _update_positions
                elif exit_sig == 2:
                    if pos["side"] == 1:
                        return {"exit_price": bar.Open, "exit_time": bar.name, "reason": "EXIT_SIGNAL"}
                elif exit_sig == -2:
                    if pos["side"] == -1:
                        return {"exit_price": bar.Open, "exit_time": bar.name, "reason": "EXIT_SIGNAL"}
                elif exit_sig == 3:
                    return {"exit_price": bar.Open, "exit_time": bar.name, "reason": "EXIT_SIGNAL"}
                elif self.has_signal_tags:
                    if pos.get("tag") == exit_sig:
                        return {"exit_price": bar.Open, "exit_time": bar.name, "reason": "EXIT_SIGNAL"}
                return None  # signal actif mais pas de match → bloque SL/TP
            # exit_sig == 0 → continue vers SL/TP normalement
    
        ema1 = self.ema1s[i] if self.ema1s is not None else None
        ema2 = self.ema2s[i] if self.ema2s is not None else None
        
        if not cfg.allow_exit_on_entry_bar and i == pos['entry_index']:
            return None

        side  = pos['side']
        tp    = pos['tp']
        sl    = pos['sl']
        o     = bar.Open
        h     = bar.High
        l     = bar.Low
        close = bar.Close

        # ── RUNNER ACTIF ─────────────────────────────────────────────
        if pos.get("runner_active") and pos["runner_sl"] is not None:
            runner_sl       = pos["runner_sl"]
            threshold_price = pos.get("runner_threshold")
            be_reason       = "BE" if pos.get("be_active", False) else "SL"

            under_threshold = threshold_price is not None and (
                (side == 1 and runner_sl <= threshold_price) or
                (side == -1 and runner_sl >= threshold_price)
            )

            if under_threshold:
                # SL fixe prend le relais
                if side == 1:
                    if o <= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                    elif l <= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                else:
                    if o >= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                    elif h >= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                return None  # TP bloqué

            # runner_sl valide → logique runner
            if side == 1:
                if o <= runner_sl:   return {"exit_price": o,         "exit_time": bar.name, "reason": "RUNNER_SL"}
                elif l <= runner_sl: return {"exit_price": runner_sl, "exit_time": bar.name, "reason": "RUNNER_SL"}
                elif o <= sl:        return {"exit_price": o,         "exit_time": bar.name, "reason": be_reason}
                elif l <= sl:        return {"exit_price": sl,        "exit_time": bar.name, "reason": be_reason}
            else:
                if o >= runner_sl:   return {"exit_price": o,         "exit_time": bar.name, "reason": "RUNNER_SL"}
                elif h >= runner_sl: return {"exit_price": runner_sl, "exit_time": bar.name, "reason": "RUNNER_SL"}
                elif o >= sl:        return {"exit_price": o,         "exit_time": bar.name, "reason": be_reason}
                elif h >= sl:        return {"exit_price": sl,        "exit_time": bar.name, "reason": be_reason}

            return None  # TP bloqué
                
        # =========================
        # FIXED MODE
        # =========================
        if self.exit_mode == "fixed":
            be_reason = "BE" if pos.get("be_active", False) else "SL"

            if side == 1:
                if o <= sl:
                    return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                elif l <= sl:
                    return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                elif h >= tp:                                                      # ← TP enfin accessible
                    return {"exit_price": tp, "exit_time": bar.name, "reason": "TP"}

            elif side == -1:
                if o >= sl:
                    return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                elif h >= sl:
                    return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                elif l <= tp:                                                      # ← TP enfin accessible
                    return {"exit_price": tp, "exit_time": bar.name, "reason": "TP"}

        # =========================
        # EMA MODE
        # =========================
        elif self.exit_mode == "ema":

            entry_price = pos["entry_price"]

            # =========================
            # 1️⃣ SL FIXE PRIORITAIRE (intrabar)
            # =========================

            be_reason = "BE" if pos.get("be_active", False) else "SL"

            if side == 1:
                if o <= sl:
                    return {"exit_price": o, "exit_time": bar.name, "reason": be_reason}
                if l <= sl:
                    return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}

            elif side == -1:
                if o >= sl:
                    return {"exit_price": o, "exit_time": bar.name, "reason": be_reason}
                if h >= sl:
                    return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}

            # =========================
            # 2️⃣ EMA TP UNIQUEMENT SI PROFIT
            # =========================

            trade_positive = (
                (side == 1 and close > entry_price) or
                (side == -1 and close < entry_price)
            )

            if not trade_positive:
                return None   # on bloque toute sortie EMA en perte

            # ----- EMA1 TP -----
            if cfg.EMA1_TP:
                if side == 1 and close < ema1:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA1_TP"}
                if side == -1 and close > ema1:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA1_TP"}

            # ----- EMA2 TP -----
            if cfg.EMA2_TP:
                if side == 1 and close < ema2:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA2_TP"}
                if side == -1 and close > ema2:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA2_TP"}

            # ----- EMA CROSS TP -----
            if cfg.EMA_CROSS_TP:
                if side == 1 and ema1 < ema2:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA_CROSS_TP"}
                if side == -1 and ema1 > ema2:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA_CROSS_TP"}


        return None

    def _update_observations(self, i, bar):
        still_pending = []
        for obs in self.pending_observations:
            if obs["side"] == 1:
                favorable = (bar.High - obs["ep"]) / obs["ep"]
                adverse   = (bar.Low  - obs["ep"]) / obs["ep"]
            else:
                favorable = (obs["ep"] - bar.Low)  / obs["ep"]
                adverse   = (obs["ep"] - bar.High) / obs["ep"]

            t = self.trades[obs["idx"]]
            t["hold_mfe"] = max(t["hold_mfe"], favorable)
            t["hold_mae"] = min(t["hold_mae"], adverse)

            if i < obs["end_i"]:
                still_pending.append(obs)

        self.pending_observations = still_pending

    def _update_positions(self, i, bar):
        cfg = self.cfg

        # LIFO — avant la boucle sur positions
        if cfg.use_exit_signal and self.has_exit_signal:
            if self.exit_signals[i-1] == 1 and len(self.positions) > 0:
                last_pos = self.positions[-1]
                self.trades.append(self.trade_analysis(
                    position=last_pos["side"],
                    exit_price=self.opens[i],
                    entry_price=last_pos["entry_price"],
                    entry_time=last_pos["entry_time"],
                    ts=self.index[i],
                    exit_reason="EXIT_SIGNAL",
                    mae=last_pos["mae"] if cfg.track_mae_mfe else None,
                    mfe=last_pos["mfe"] if cfg.track_mae_mfe else None,
                ))
                self.positions = self.positions[:-1]
        surviving = []
        for pos in self.positions:
            
            # MAE/MFE
            if cfg.track_mae_mfe:
                ep   = pos["entry_price"]
                side = pos["side"]
                pos["mfe"] = max(pos["mfe"], (bar.High - ep) / ep * side)
                pos["mae"] = min(pos["mae"], (bar.Low  - ep) / ep * side)

            # BE
            if cfg.be_trigger_pct is not None:
                pos["sl"], \
                pos["be_armed"], \
                pos["pending_be_sl"], \
                pos["be_arm_index"], \
                pos["be_active"] = self._update_be_logic(
                    pos=pos,
                    i=i,
                    bar=bar,
                    be_trigger_pct=cfg.be_trigger_pct,
                    be_offset_pct=cfg.be_offset_pct,
                    be_delay_bars=cfg.be_delay_bars
                )

            # Runner
            if cfg.trailing_trigger_pct is not None and cfg.trailing_trigger_pct > 0.0 and self.atrs is not None:
                pos = self._update_trailing_runner(
                    pos=pos, bar=bar,
                    atr_value = self.atrs[i-1],# on regarde le close. Anti look ahead
                    trailing_trigger_pct=cfg.trailing_trigger_pct,
                    trailing_mult=cfg.runner_trailing_mult
                )

            # Exit
            exit_event = self._exit_logic(i, bar, pos) # mettre signa_exit et creer une parti qui interprete ce signal

            if exit_event is not None:
                if cfg.track_mae_mfe is not True:
                    self.trades.append(self.trade_analysis(
                        position=pos["side"],
                        exit_price=exit_event["exit_price"],
                        entry_price=pos["entry_price"],
                        entry_time=pos["entry_time"],
                        ts=exit_event["exit_time"],
                        exit_reason=exit_event["reason"],
                    ))
                else:
                    self.trades.append(self.trade_analysis(
                        position=pos["side"],
                        exit_price=exit_event["exit_price"],
                        entry_price=pos["entry_price"],
                        entry_time=pos["entry_time"],
                        ts=exit_event["exit_time"],
                        exit_reason=exit_event["reason"],
                        mae=pos["mae"],
                        mfe=pos["mfe"],
                        mae_h=0.0 if self.use_observation else None,
                        mfe_h=0.0 if self.use_observation else None
                    ))

                    if self.use_observation:
                        self.pending_observations.append({
                            "idx":   len(self.trades) - 1,
                            "ep":    exit_event["exit_price"],   # ← exit price comme référence
                            "side":  pos["side"],
                            "end_i": i + 1 + self.observation_bars
                        })
            else:
                surviving.append(pos)

        self.positions = surviving

    # --------- CORE ---------
    def run(self, return_df=False, plot=False, sim: SimulationConfig = None) -> pd.DataFrame:
        cfg = self.cfg


        for i in range(1, len(self.df)):
            ts = self.index[i]
            bar = self.Bar(self.opens[i], self.highs[i], self.lows[i], self.closes[i], ts)
            sig_prev = self.signals[i - cfg.entry_delay]

            # 1) reverse mode
            self._apply_reverse(sig_prev, bar.Open, ts)

            # 2) entry logic
            entry = self._entry_logic(i, ts, bar, sig_prev)
            self._max_entries_reset(ts)
            entry = self._max_entries_filter(i, entry)

            if entry is not None:
                if cfg.multi_entry or len(self.positions) == 0:
                    self.positions.append(entry)
                if cfg.MaxEntries4Periods:
                    self.recent_entries.append(i)

            # 3) exit logic loop
            self._update_positions(i, bar)

            # 4) post-trade observation
            if (cfg.track_mae_mfe) and self.use_observation and self.pending_observations:
                self._update_observations(i, bar)

        trades = pd.DataFrame(self.trades)
        if self.use_observation and "hold_mfe" in trades.columns and "mfe" in trades.columns:
            trades["capture_ratio_hold"] = trades.apply(
                lambda r: round(r["return"] / r["hold_mfe"], 3) if r["hold_mfe"] > 0 else None,
                axis=1
            )
        if plot:
            BacktestEngine._plot_backtest(self.df, trades, crypto=self.cfg.crypto)
        # Métriques — uniquement si sim fourni ET compute_metrics=True
        metrics = None
        if sim is not None and self.cfg.compute_metrics:
            metrics = BacktestEngine.compute_metrics(trades, sim)
        # Retour
        if return_df and metrics is not None:
            return trades, self.df, metrics
        if return_df:
            return trades, self.df
        if metrics is not None:
            return trades, metrics
        return trades
    
    #--------- After ---------
    @staticmethod
    def _plot_backtest(df, trades, title="Backtest results", crypto=False):
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ))

        long_entries  = trades[trades["side"] == 1]
        short_entries = trades[trades["side"] == -1]

        # Entrées
        fig.add_trace(go.Scatter(
            x=long_entries["entry_time"],
            y=long_entries["entry"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="lime"),
            name="Long entry"
        ))

        fig.add_trace(go.Scatter(
            x=short_entries["entry_time"],
            y=short_entries["entry"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="red"),
            name="Short entry"
        ))

        # Sorties — croix couleur selon le sens + trait reliant entrée/sortie
        for _, r in trades.iterrows():
            color = "lime" if r["side"] == 1 else "red"

            # Trait entrée → sortie
            fig.add_trace(go.Scatter(
                x=[r["entry_time"], r["exit_time"]],
                y=[r["entry"],      r["exit"]],
                mode="lines",
                line=dict(color=color, width=1),
                opacity=0.5,
                showlegend=False
            ))

            # Croix sortie — même couleur que l'entrée
            fig.add_trace(go.Scatter(
                x=[r["exit_time"]],
                y=[r["exit"]],
                mode="markers",
                marker=dict(symbol="x", size=9, color=color),
                showlegend=False
            ))

        rangebreaks = [] if crypto else [dict(bounds=["sat", "mon"])]

        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            xaxis=dict(rangebreaks=rangebreaks),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0),
            height=700
        )

        fig.show()

    @staticmethod
    def compute_metrics(trades, sim: SimulationConfig):
        if len(trades) == 0:
            return None
        
        returns = trades["return"]
        if sim.commission_pct or sim.spread_pct or sim.slippage_pct:
            cost    = sim.commission_pct * 2 + sim.spread_pct + sim.slippage_pct * 2
            returns = returns - cost

        ret_arr  = returns.to_numpy()
        pos_mask = ret_arr > 0
        neg_mask = ret_arr < 0

        # Equity curve & MDD
        cum      = np.cumprod(1 + ret_arr)
        roll_max = np.maximum.accumulate(cum)
        dd_curve = (cum - roll_max) / roll_max
        mdd      = dd_curve.min()

        # Underwater duration (en trades)
        max_uw = current_uw = 0
        for d in dd_curve < 0:
            current_uw = current_uw + 1 if d else 0
            max_uw = max(max_uw, current_uw)

        # Scalaires
        cum_return = cum[-1] - 1
        n_years    = (trades["exit_time"].iloc[-1] - trades["entry_time"].iloc[0]).days / 365
        ann_return = (1 + cum_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan
        std        = ret_arr.std()
        wins_sum   = ret_arr[pos_mask].sum()
        loss_sum   = abs(ret_arr[neg_mask].sum())
        var_t      = -np.percentile(ret_arr, sim.alpha4Var)

        # Tests statistiques
        from scipy import stats
        t_stat, p_value   = stats.ttest_1samp(ret_arr, 0)
        n_wins            = pos_mask.sum()
        p_binom           = stats.binomtest(n_wins, len(ret_arr), 0.5).pvalue

        # Retours par période — empiriques
        t = trades.copy().set_index("exit_time")
        period_ret = t["return"].resample(sim.period_freq).sum()
        period_ret = period_ret[period_ret != 0].to_numpy()

        pr_pos  = period_ret > 0
        pr_neg  = period_ret < 0
        pr_var  = np.percentile(period_ret, sim.alpha4Var) if len(period_ret) > 0 else np.nan
        pr_cvar = period_ret[period_ret <= pr_var].mean() if (period_ret <= pr_var).any() else np.nan

        n_trades_per_year = len(ret_arr) / n_years
        sharpe = round(ret_arr.mean() / std * np.sqrt(n_trades_per_year), 3) if std > 0 else np.nan

        return {
            # Core
            "n_trades"            : len(ret_arr),
            "win_rate"            : round(pos_mask.mean(), 3),
            "total_return_sum"    : round(ret_arr.sum(), 4),
            "cum_return"          : round(cum_return, 4),
            "ann_return"          : round(ann_return, 4) if not np.isnan(ann_return) else np.nan,
            "max_drawdown"        : round(mdd, 4),
            "max_underwater_trades": max_uw,               # ← nb trades consécutifs sous l'eau
            "calmar"              : round(ann_return / abs(mdd), 3) if mdd != 0 and not np.isnan(ann_return) else np.nan,
            "sharpe"              : round(sharpe,4),
            "profit_factor"       : round(wins_sum / loss_sum, 3) if loss_sum != 0 else np.nan,
            "avg_win"             : round(ret_arr[pos_mask].mean(), 4) if pos_mask.any() else np.nan,
            "avg_loss"            : round(ret_arr[neg_mask].mean(), 4) if neg_mask.any() else np.nan,
            "VaR"                 : round(var_t, 4),
            "CVaR"                : round(-ret_arr[ret_arr <= var_t].mean(), 4) if (ret_arr <= var_t).any() else np.nan,

            # Tests statistiques
            "t_stat"              : round(t_stat, 3),
            "p_value"             : round(p_value, 4),      # < 0.05 → edge significatif
            "p_binom"             : round(p_binom, 4),      # < 0.05 → win_rate non-aléatoire

            # Retours par période (freq)
            "period_freq"         : sim.period_freq,
            "n_periods"           : len(period_ret),
            "n_periods_positive"  : int(pr_pos.sum()),
            "n_periods_negative"  : int(pr_neg.sum()),
            "pct_periods_positive": round(pr_pos.mean(), 3) if len(period_ret) > 0 else np.nan,
            "worst_period"        : round(period_ret.min(), 4) if len(period_ret) > 0 else np.nan,
            "best_period"         : round(period_ret.max(), 4) if len(period_ret) > 0 else np.nan,
            "period_cvar"         : round(pr_cvar, 4) if not np.isnan(pr_cvar) else np.nan,
        }
    
pipeline = DataPipeline("/Users/arnaudbarbier/Desktop/Quant reaserch/Metals")

cfg = BacktestConfig(
    strategy='close_vs_ema',
    tp_pct=0.05,
    sl_pct=0.003,
    timezone_shift=1,
    track_mae_mfe=True,
    compute_metrics=True,
    period_1=30,
    period_2=100,
    be_offset_pct=0.04,
    be_delay_bars=5,
    be_trigger_pct=0.03,
    time_window_1="08:00-12:00",
    time_window_2="13:00-17:00",
    Previous_Candle_same_direction=False,
    Candle_Size_filter=True,
    min_size_pct=0.001,
    max_size_pct=0.002,
    allow_exit_on_entry_bar=True,
    #EMA_CROSS_TP=True,
    #Exit_filter_EMA1=10,
    #Exit_filter_EMA2=20,
    #use_atr_sl_tp=2, sl_atr_mult=2 ,tp_atr_mult=2,
    #reverse_mode=True
    #trailing_trigger_pct=0.005,
    #runner_trailing_mult=3,
    entry_delay=1,
    #exit_delay=2,
    ME_Period_Y=30,
    MaxEntries4Periods=True, 
    ME_X=3,
    #observation_hours=2,
    #timeframe_minutes=5,
    ME_reset_mode='session'
)

sim = SimulationConfig(
    commission_pct=0,
    spread_pct=0.0000,
    slippage_pct=0.0000,
    timeframe_minutes=5,
    alpha4Var=5,
)

engine = BacktestEngine.from_ticker(
    pipeline, "XAUUSD_M5", "2023-01-02", "2026-02-16", cfg
)

trades_v1, metrics_v1 = engine.run(sim=sim)
print(metrics_v1)
trades_v1
