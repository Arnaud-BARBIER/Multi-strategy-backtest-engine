import pandas as pd
import numpy as np
from collections import namedtuple
from dataclasses import dataclass


# =============================================================================
# BacktestConfig
# =============================================================================
# Central configuration dataclass — all parameters defined in one place.
# Passed immutably to DataPipeline, Strategy_Signal, and BacktestEngine.
#
# slots=True reduces memory footprint (~200 bytes → ~50 bytes per instance).
# Next step would be to build a high speed engine for grid search. this work
# is preliminary. 
# =============================================================================

'''
This current version allows users to manually plug their strategy
into the engine. The engine expects a DataFrame with a Signal column
produced by any strategy. As the engine is independant from the signal
generation. 
'''

@dataclass(slots=True)
class BacktestConfig:

    # --- Strategy ---
    strategy: str = "ema_cross"         # strategy name dispatched by Strategy_Signal.apply()

    # --- Indicators ---
    period_1: int = 50
    period_2: int = 100
    max_gap_size: float | None = None
    period_atr: int = 14
    Exit_filter_EMA1: int = 50          # EMA period for exit filter (independent of strategy)
    Exit_filter_EMA2: int = 100

    # --- Entry filters ---
    Candle_Size_filter: bool = True
    Previous_Candle_same_direction: bool = True
    min_size_pct: float = 0.001
    max_size_pct: float = 0.02

    # --- TP / SL ---
    tp_pct: float = 0.05
    sl_pct: float = 0.02
    use_atr_sl_tp: bool | int = False   # 0=fixed, 1=ATR TP only, -1=ATR SL only, True=both
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0

    # --- Exit mode toggles ---
    # Only one EMA exit flag should be active at a time.
    # If none are active, engine defaults to fixed TP/SL mode.
    EMA1_TP: bool = False
    EMA2_TP: bool = False
    EMA_CROSS_TP: bool = False
    EMA_SL: bool = False

    # --- Max entries cap ---
    MaxEntries4Periods: bool = True
    ME_X: int = 2                       # max entries allowed within ME_Period_Y bars
    ME_Period_Y: int = 8                # sliding window size in bars
    ME_reset_mode: str | None = None    # "day" / "session" / None

    # --- Engine behavior ---
    allow_exit_on_entry_bar: bool = True
    multi_entry: bool = True            # allow stacking positions on repeated signals
    reverse_mode: bool = False

    # --- Breakeven ---
    be_trigger_pct: float | None = None
    be_offset_pct: float = 0.0
    be_delay_bars: int = 0              # minimum bars before BE trigger is checked

    # --- ATR trailing runner ---
    trailing_trigger_pct: float | None = None
    runner_trailing_mult: float = 2.0

    # --- Session time windows ---
    time_window_1: str | None = None    # e.g. "08:00-12:00"
    time_window_2: str | None = None
    time_window_3: str | None = None

    # --- MAE / MFE hold period observation ---
    observation_hours: float | None = None
    timeframe_minutes: int = 5

    # --- Performance ---
    fast: bool = True                   # if True, skips MAE/MFE tracking (faster grid search)


# =============================================================================
# Strategy_Signal
# =============================================================================
# Responsible for signal generation only.
# Decoupled from the engine — any strategy can be plugged in here.
#
# Contract:
#   - apply() must return a DataFrame with a "Signal" column (1 / -1 / 0)
#   - Do not shift signals manually — the engine reads Signal[i-1] internally
#   - Entries are executed at the open of the bar following the signal
#
# To add a new strategy:
#   1. Add a static method with your signal logic
#   2. Add an elif branch in apply()
#   3. Add any new parameters to BacktestConfig
# =============================================================================

'''
Strategy contract:

    A strategy must return `df` with a `Signal` column:
        1  = Long
       -1  = Short
        0  = Neutral

    Do not shift signals manually.
    The engine handles bar alignment internally to avoid look-ahead bias.
    Signals are generated on bar close — entries execute at next bar open.
'''

class Strategy_Signal:

    @staticmethod
    def apply(df, cfg):
        """Dispatch to the correct strategy based on cfg.strategy."""
        if cfg.strategy == "ema_cross":
            return Strategy_Signal.ema_cross(df, cfg.period_1, cfg.period_2, cfg.max_gap_size)
        else:
            raise ValueError(f"Unknown strategy: {cfg.strategy}")

    @staticmethod
    def ema_cross(df: pd.DataFrame, period_1=50, period_2=100, max_gap_size=None) -> pd.DataFrame:
        
        """
        This is an example of a strategy.
        
        Generate entry signals on EMA1/price crossovers.

        Signal  = 1  when price crosses above EMA1 (long)
        Signal  = -1 when price crosses below EMA1 (short)

        EMA2, Signal2, EMA_CROSS are computed as additional reference columns
        and can be used as exit filters via Exit_filter_EMA1/2 in BacktestConfig.

        gap_filter optionally rejects signals when the open gaps significantly
        from the previous close — useful for filtering news spikes.
        """
        d = df.copy()
        d["EMA1"] = d["Close"].ewm(span=period_1, adjust=False).mean()
        d["EMA2"] = d["Close"].ewm(span=period_2, adjust=False).mean()

        if max_gap_size is not None:
            gap_pct    = (d["Open"] - d["Close"].shift(1)).abs() / d["Close"].shift(1)
            gap_filter = gap_pct < max_gap_size
        else:
            gap_filter = pd.Series(True, index=d.index)

        # Primary signal — EMA1/price cross
        d["Signal"] = np.where(
            (d["EMA1"].shift(1) < d["Close"].shift(1)) & (d["EMA1"] > d["Close"]) & gap_filter, -1,
            np.where(
                (d["EMA1"].shift(1) > d["Close"].shift(1)) & (d["EMA1"] < d["Close"]) & gap_filter, 1,
                0,
            ),
        )

        # EMA2/price cross 
        d["Signal2"] = 0
        bull = (d["EMA2"].shift(1) > d["Close"].shift(1)) & (d["EMA2"] < d["Close"]) & gap_filter
        bear = (d["EMA2"].shift(1) < d["Close"].shift(1)) & (d["EMA2"] > d["Close"]) & gap_filter
        d.loc[bull, "Signal2"] = 1
        d.loc[bear, "Signal2"] = -1

        # EMA1/EMA2 cross 
        d["EMA_CROSS"] = 0
        up_cross   = (d["EMA1"].shift(1) < d["EMA2"].shift(1)) & (d["EMA1"] > d["EMA2"]) & gap_filter
        down_cross = (d["EMA1"].shift(1) > d["EMA2"].shift(1)) & (d["EMA1"] < d["EMA2"]) & gap_filter
        d.loc[up_cross,   "EMA_CROSS"] = 1
        d.loc[down_cross, "EMA_CROSS"] = -1

        # Entry_Price for reference only — engine executes at bar.Open[i+1]
        #d["Entry_Price"] = d["Open"].where(d["Signal"].shift(1) != 0)

        return d


# =============================================================================
# DataPipeline
# =============================================================================
# Responsible for data loading and indicator computation.
# build() is the single entry point — returns a fully prepared DataFrame.
#
# Order matters:
#   1. fetchdata   — load raw OHLCV
#   2. compute_atr — universal volatility indicator (strategy-agnostic)
#   3. Strategy_Signal.apply — signal generation (strategy-specific)
#   4. apply_exitfilter_indicators — EMA exit columns (engine-specific, optional)
# =============================================================================

class DataPipeline:

    def __init__(self, base_path: str):
        self.base_path = base_path

    def fetchdata(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Load OHLCV from CSV. Timestamps shifted +1h for broker timezone alignment.
        Expected filename format: {base_path}/{ticker}.csv
        """
        df = pd.read_csv(
            f"{self.base_path}/{ticker}.csv",
            header=None,
            names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
        )
        df["Datetime"] = pd.to_datetime(df["Datetime"]) + pd.Timedelta(hours=1)
        df = df.set_index("Datetime").sort_index()
        return df.loc[start:end]

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Standard Average True Range — computed once before the loop.
        Used for ATR-based TP/SL and trailing runner stop.
        Kept in DataPipeline (not Strategy_Signal) because it is strategy-agnostic.
        """
        high  = df["High"]
        low   = df["Low"]
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
        
        """
        Compute EMA exit columns independently of the entry strategy.
        Only computed when an EMA exit flag is active — avoids unnecessary work.

        Columns use dedicated names (EMA1_exit / EMA2_exit) to clearly separate
        exit filter logic from strategy signal logic. This allows combining any
        entry strategy with any EMA-based exit filter.
        
        """
        if any([cfg.EMA1_TP, cfg.EMA2_TP, cfg.EMA_CROSS_TP]):
            df["EMA1_exit"] = df["Close"].ewm(span=cfg.Exit_filter_EMA1, adjust=False).mean()
            df["EMA2_exit"] = df["Close"].ewm(span=cfg.Exit_filter_EMA2, adjust=False).mean()
        return df

    def build(self, ticker: str, start: str, end: str, cfg: BacktestConfig) -> pd.DataFrame:
        """Full data preparation pipeline — called by BacktestEngine.from_ticker()."""
        df = self.fetchdata(ticker, start, end)
        df = self.compute_atr(df, cfg.period_atr)
        df = Strategy_Signal.apply(df, cfg)
        df = self.apply_exitfilter_indicators(df, cfg)
        return df


# =============================================================================
# BacktestEngine
# =============================================================================
# Bar-by-bar simulation engine — multi-position, numpy-optimized.
#
# Key design decisions:
#
# NUMPY PRECOMPUTATION
#   All OHLCV columns and indicator arrays are extracted once in __init__
#   as numpy arrays. Accessing arr[i] inside the loop is ~10x faster than
#   df["col"].iloc[i] which triggers pandas overhead on every bar.
#
# SURVIVING LIST PATTERN
#   Closed positions are filtered out via a surviving list rather than
#   positions.copy() + positions.remove(). Avoids O(n) list scan per close.
#
# SLIDING WINDOW POINTER
#   MaxEntries cleanup uses a pointer j instead of rebuilding the list
#   via list comprehension on every bar.
#
# SEPARATION OF CONCERNS
#   cfg  = parameters fixed before the run (never mutated during simulation)
#   self = state that evolves bar by bar (positions, trades, recent_entries)
# =============================================================================

class BacktestEngine:
    Bar = namedtuple("Bar", ["Open", "High", "Low", "Close", "name"])

    def __init__(self, df: pd.DataFrame, cfg: BacktestConfig):
        self.df  = df
        self.cfg = cfg

        # --- Mutable simulation state ---
        self.positions       = []
        self.trades          = []
        self.recent_entries  = []
        self.last_session_id = None
        self.last_day        = None

        # --- MAE/MFE hold period observation ---
        self.use_observation      = cfg.observation_hours is not None
        self.pending_observations = []
        self.observation_bars     = (
            int(cfg.observation_hours * 60 / cfg.timeframe_minutes)
            if self.use_observation else 0
        )

        # --- Numpy arrays precomputed once — avoids iloc overhead in the loop ---
        self.opens   = df["Open"].to_numpy()
        self.highs   = df["High"].to_numpy()
        self.lows    = df["Low"].to_numpy()
        self.closes  = df["Close"].to_numpy()
        self.signals = df["Signal"].to_numpy()  # strategy signal consumed here
        self.atrs    = df["ATR"].to_numpy()       if "ATR"       in df.columns else None
        self.ema1s   = df["EMA1_exit"].to_numpy() if "EMA1_exit" in df.columns else None
        self.ema2s   = df["EMA2_exit"].to_numpy() if "EMA2_exit" in df.columns else None
        self.index   = df.index

        # --- Session windows parsed once in __init__ ---
        self.parsed_windows = [
            self.parse_window(cfg.time_window_1),
            self.parse_window(cfg.time_window_2),
            self.parse_window(cfg.time_window_3),
        ]

        # --- Exit mode determined once before the loop ---
        self.exit_mode = "ema" if sum([cfg.EMA1_TP, cfg.EMA2_TP, cfg.EMA_CROSS_TP]) == 1 else "fixed"

    @classmethod
    def from_ticker(cls, pipeline: DataPipeline, ticker: str, start: str, end: str, cfg: BacktestConfig):
        """Convenience constructor — builds the DataFrame and instantiates the engine."""
        df = pipeline.build(ticker, start, end, cfg)
        return cls(df, cfg)

    @staticmethod
    def parse_window(w):
        if w is None:
            return None
        s, e = w.split("-")
        return pd.to_datetime(s).time(), pd.to_datetime(e).time()

    @staticmethod
    def time_in_window_fast(ts, parsed_window):
        """
        Check session membership using pre-parsed time objects.
        Supports overnight windows (e.g. "22:00-02:00") via wraparound logic.
        Called once per bar per active window — using .time() objects avoids
        repeated string parsing overhead vs the V2 approach.
        """
        if parsed_window is None:
            return False
        start, end = parsed_window
        current    = ts.time()
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end  # overnight session

    @staticmethod
    def _update_be_logic(pos, i, bar, be_trigger_pct, be_offset_pct, be_delay_bars):
        """
        Two-phase breakeven mechanism — lives entirely inside the position dict.

        Phase 1 — ARM:      price reaches trigger → record pending BE stop
        Phase 2 — ACTIVATE: on the following bar, replace SL with BE stop

        Activation is delayed by one bar (i > be_arm_index) to avoid
        same-bar look-ahead. be_delay_bars adds a minimum holding period
        before the trigger is even checked.
        """
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

        if position == 1 and delay_ok and not be_armed:
            if bar.High >= entry_price * (1 + be_trigger_pct):
                be_armed      = True
                be_arm_index  = i
                pending_be_sl = entry_price * (1 + be_offset_pct)

        if position == -1 and delay_ok and not be_armed:
            if bar.Low <= entry_price * (1 - be_trigger_pct):
                be_armed      = True
                be_arm_index  = i
                pending_be_sl = entry_price * (1 - be_offset_pct)

        # Activate on the next bar — never on the arming bar itself
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
        """
        Build a trade record dict on close.
        MAE/MFE and hold period metrics are only added when present,
        keeping the output schema clean in fast mode.
        """
        ret = position * (exit_price - entry_price) / entry_price

        trade = {
            "entry_time": entry_time,
            "exit_time":  ts,
            "side":       position,
            "entry":      entry_price,
            "exit":       exit_price,
            "return":     ret,
            "reason":     exit_reason,
        }

        if (mae is not None) and (mfe is not None):
            trade["mae"]           = mae
            trade["mfe"]           = mfe
            trade["capture_ratio"] = ret / mfe if mfe > 0 else None

        if (mae_h is not None) and (mfe_h is not None):
            trade["hold_mfe"]           = mfe_h
            trade["hold_mae"]           = mae_h
            trade["capture_ratio_hold"] = round(ret / mfe_h, 3) if mfe_h > 0 else None

        return trade

    @staticmethod
    def Tp_Sl_prices(side, entry_price, tp_pct=None, sl_pct=None,
                     use_atr=False, atr_value=None, tp_atr_mult=None, sl_atr_mult=None):
        """
        Compute TP and SL prices from entry.

        use_atr modes:
          True  — both TP and SL use ATR multiples
          1     — TP uses ATR, SL is fixed percentage
         -1     — SL uses ATR, TP is fixed percentage
          False — both fixed percentage

        ATR-based stops adapt to current volatility, making parameters
        more portable across assets with different volatility profiles.
        """
        if use_atr is True:
            if atr_value is None:
                return None, None
            tp = entry_price + side * atr_value * tp_atr_mult
            sl = entry_price - side * atr_value * sl_atr_mult

        elif use_atr == 1:
            if atr_value is None:
                return None, None
            tp = entry_price + side * atr_value * tp_atr_mult
            sl = entry_price * (1 - side * sl_pct)

        elif use_atr == -1:
            if atr_value is None:
                return None, None
            tp = entry_price * (1 + side * tp_pct)
            sl = entry_price - side * atr_value * sl_atr_mult

        else:
            tp = entry_price * (1 + side * tp_pct)
            sl = entry_price * (1 - side * sl_pct)

        return tp, sl

    @staticmethod
    def _update_trailing_runner(pos, bar, atr_value, trailing_trigger_pct, trailing_mult):
        """
        ATR-based trailing stop — activated once price moves in favor by trailing_trigger_pct.

        Two-phase activation mirrors the BE logic:
          Phase 1 — ARM:      price reaches threshold
          Phase 2 — ACTIVATE: trailing stop initialized on the following bar

        Once active, the stop trails bar.Close by atr_value * trailing_mult,
        but never moves against the trade (max/min guards ensure monotonicity).
        atr_value uses atrs[i-1] — known before bar open, no look-ahead.
        """
        side = pos["side"]
        ep   = pos["entry_price"]

        if not pos["runner_active"] and not pos.get("runner_armed", False):
            threshold = ep * (1 + side * trailing_trigger_pct)
            if (side == 1 and bar.High >= threshold) or \
               (side == -1 and bar.Low <= threshold):
                pos["runner_armed"]     = True
                pos["runner_threshold"] = threshold

        if pos.get("runner_armed") and not pos["runner_active"]:
            pos["runner_active"] = True
            pos["runner_armed"]  = False
            initial_sl = bar.Close - side * atr_value * trailing_mult
            pos["runner_sl"] = max(initial_sl, ep) if side == 1 else min(initial_sl, ep)
            return pos

        if pos["runner_active"]:
            new_sl = bar.Close - side * atr_value * trailing_mult
            if side == 1:
                pos["runner_sl"] = max(pos["runner_sl"], new_sl, ep)
            else:
                pos["runner_sl"] = min(pos["runner_sl"], new_sl, ep)

        return pos

    def _apply_reverse(self, sig_prev, bar_open, ts):
        """
        Close all positions opposite to sig_prev at market open.
        Uses the surviving list pattern — avoids list.remove() O(n) scan.
        """
        cfg = self.cfg
        if not cfg.reverse_mode or sig_prev == 0:
            return

        closed    = []
        surviving = []

        for pos in self.positions:
            if pos["side"] == -sig_prev:
                closed.append(self.trade_analysis(
                    position=pos["side"], exit_price=bar_open,
                    entry_price=pos["entry_price"], entry_time=pos["entry_time"],
                    ts=ts, exit_reason="REVERSE",
                ) if cfg.fast else self.trade_analysis(
                    position=pos["side"], exit_price=bar_open,
                    entry_price=pos["entry_price"], entry_time=pos["entry_time"],
                    ts=ts, exit_reason="REVERSE",
                    mae=pos.get("mae", 0.0), mfe=pos.get("mfe", 0.0),
                ))
            else:
                surviving.append(pos)

        self.positions = surviving
        self.trades.extend(closed)

    def get_active_session(self, ts):
        """Return the index of the first active session window, or None."""
        for idx, w in enumerate(self.parsed_windows):
            if w is not None and self.time_in_window_fast(ts, w):
                return idx
        return None

    def _entry_logic(self, i, ts, bar, sig_prev):
        """
        Evaluate entry conditions for bar[i].

        sig_prev carries the signal from Strategy_Signal — the engine does not
        know or care how it was generated. It only reads: 1, -1, or 0.

        Guard clauses exit early at each filter stage:
          1. Session window check
          2. Signal check    ← strategy signal consumed here
          3. Candle filter
          4. ATR NaN guard

        ATR uses atrs[i-1] — value known before bar open, no look-ahead bias.
        """
        cfg       = self.cfg
        atr_value = self.atrs[i-1] if (cfg.use_atr_sl_tp and self.atrs is not None) else None

        if any(w is not None for w in self.parsed_windows):
            in_any_window = any(
                self.time_in_window_fast(ts, w)
                for w in self.parsed_windows if w is not None
            )
            if not in_any_window:
                return None

        if sig_prev == 0:       # no signal from strategy → no entry
            return None

        body_pct     = abs(self.opens[i-1] - self.closes[i-1]) / self.closes[i-1]
        size_ok      = cfg.min_size_pct < body_pct < cfg.max_size_pct
        direction_ok = (
            (sig_prev ==  1 and self.closes[i-1] > self.opens[i-1]) or
            (sig_prev == -1 and self.closes[i-1] < self.opens[i-1])
        )

        if cfg.Candle_Size_filter:
            if not (size_ok and (not cfg.Previous_Candle_same_direction or direction_ok)):
                return None

        entry_price = bar.Open

        if cfg.use_atr_sl_tp and atr_value is not None and np.isnan(atr_value):
            return None

        tp, sl = self.Tp_Sl_prices(
            side=sig_prev, entry_price=entry_price,
            tp_pct=cfg.tp_pct, sl_pct=cfg.sl_pct,
            use_atr=cfg.use_atr_sl_tp,
            tp_atr_mult=cfg.tp_atr_mult, sl_atr_mult=cfg.sl_atr_mult,
            atr_value=atr_value,
        )

        return {
            "side":          sig_prev,
            "entry_price":   bar.Open,
            "entry_time":    ts,
            "entry_index":   i,
            "tp":            tp,
            "sl":            sl,
            "be_armed":      False,
            "pending_be_sl": None,
            "be_arm_index":  None,
            "be_active":     False,
            "runner_active": False,
            "runner_armed":  False,
            "runner_sl":     None,
            "mae":           0.0,
            "mfe":           0.0,
        }

    def _max_entries_reset(self, ts):
        """
        Reset the recent_entries counter on day or session change.
        Prevents overtrading by ensuring the ME_X cap applies per period,
        not cumulatively across the full backtest.
        """
        cfg         = self.cfg
        current_day = ts.date()

        if cfg.ME_reset_mode == "day":
            if self.last_day is None:
                self.last_day = current_day
            elif current_day != self.last_day:
                self.recent_entries = []
                self.last_day       = current_day

        elif cfg.ME_reset_mode == "session":
            current_session = self.get_active_session(ts)
            if self.last_session_id is None:
                self.last_session_id = current_session
            elif current_session != self.last_session_id:
                self.recent_entries  = []
                self.last_session_id = current_session

    def _max_entries_filter(self, i, entry):
        """
        Sliding window cap — blocks entries if ME_X trades were taken
        within the last ME_Period_Y bars.

        Uses a forward pointer j instead of rebuilding the list via
        list comprehension on every bar (V2 bottleneck).
        """
        cfg = self.cfg
        if not cfg.MaxEntries4Periods:
            return entry

        cutoff = i - cfg.ME_Period_Y
        j = 0
        while j < len(self.recent_entries) and self.recent_entries[j] < cutoff:
            j += 1
        self.recent_entries = self.recent_entries[j:]

        if len(self.recent_entries) >= cfg.ME_X:
            return None

        return entry

    def _exit_logic(self, i, bar, pos):
        """
        Evaluate exit conditions for a single open position.

        Priority order:
          1. Runner SL (if trailing runner is active)
          2. Fixed SL / BE SL
          3. TP (fixed) or EMA condition (EMA mode)

        EMA exit only fires if the trade is currently profitable —
        prevents locking in losses at an arbitrary EMA level.

        EMA values read from precomputed numpy arrays (ema1s / ema2s),
        avoiding repeated iloc calls that were a bottleneck in V2.
        """
        cfg  = self.cfg
        ema1 = self.ema1s[i] if self.ema1s is not None else None
        ema2 = self.ema2s[i] if self.ema2s is not None else None

        if not cfg.allow_exit_on_entry_bar and i == pos["entry_index"]:
            return None

        side  = pos["side"]
        tp    = pos["tp"]
        sl    = pos["sl"]
        o, h, l, close = bar.Open, bar.High, bar.Low, bar.Close

        # ── RUNNER ACTIVE ─────────────────────────────────────────────────────
        if pos.get("runner_active") and pos["runner_sl"] is not None:
            runner_sl       = pos["runner_sl"]
            threshold_price = pos.get("runner_threshold")
            be_reason       = "BE" if pos.get("be_active", False) else "SL"

            under_threshold = threshold_price is not None and (
                (side ==  1 and runner_sl <= threshold_price) or
                (side == -1 and runner_sl >= threshold_price)
            )

            if under_threshold:
                # Runner SL fell back below entry threshold — fixed SL takes over
                if side == 1:
                    if o <= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                    elif l <= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                else:
                    if o >= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                    elif h >= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                return None  # TP blocked while runner is under threshold

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

            return None  # TP blocked while runner is active

        # ── FIXED MODE ────────────────────────────────────────────────────────
        if self.exit_mode == "fixed":
            be_reason = "BE" if pos.get("be_active", False) else "SL"

            if side == 1:
                if o <= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                elif l <= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                elif h >= tp: return {"exit_price": tp, "exit_time": bar.name, "reason": "TP"}

            elif side == -1:
                if o >= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                elif h >= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                elif l <= tp: return {"exit_price": tp, "exit_time": bar.name, "reason": "TP"}

        # ── EMA MODE ─────────────────────────────────────────────────────────
        elif self.exit_mode == "ema":
            entry_price = pos["entry_price"]
            be_reason   = "BE" if pos.get("be_active", False) else "SL"

            # SL always checked first — takes priority over EMA exit
            if side == 1:
                if o <= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                if l <= sl:   return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
            elif side == -1:
                if o >= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                if h >= sl:   return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}

            # EMA exit only if trade is currently profitable
            trade_positive = (
                (side ==  1 and close > entry_price) or
                (side == -1 and close < entry_price)
            )
            if not trade_positive:
                return None

            if cfg.EMA1_TP:
                if side ==  1 and close < ema1: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA1_TP"}
                if side == -1 and close > ema1: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA1_TP"}

            if cfg.EMA2_TP:
                if side ==  1 and close < ema2: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA2_TP"}
                if side == -1 and close > ema2: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA2_TP"}

            if cfg.EMA_CROSS_TP:
                if side ==  1 and ema1 < ema2: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA_CROSS_TP"}
                if side == -1 and ema1 > ema2: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA_CROSS_TP"}

        return None

    def _update_observations(self, i, bar):
        """
        Track MAE/MFE for a configurable hold period after trade close.
        Useful for analyzing how price behaves after exit — informs TP/SL calibration.
        """
        still_pending = []
        for obs in self.pending_observations:
            favorable = (bar.High - obs["ep"]) / obs["ep"] * obs["side"]
            adverse   = (bar.Low  - obs["ep"]) / obs["ep"] * obs["side"]

            t = self.trades[obs["idx"]]
            t["hold_mfe"] = max(t["hold_mfe"], favorable)
            t["hold_mae"] = min(t["hold_mae"], adverse)

            if i < obs["end_i"]:
                still_pending.append(obs)

        self.pending_observations = still_pending

    def _update_positions(self, i, bar):
        """
        Process all open positions for bar[i]:
          1. Update MAE/MFE intra-trade (fast=False only)
          2. Update BE stop
          3. Update trailing runner
          4. Check exit conditions

        Uses surviving list pattern — positions that did not close are
        kept in a new list rather than removed from the original.
        """
        cfg       = self.cfg
        surviving = []

        for pos in self.positions:

            if not cfg.fast:
                ep   = pos["entry_price"]
                side = pos["side"]
                pos["mfe"] = max(pos["mfe"], (bar.High - ep) / ep * side)
                pos["mae"] = min(pos["mae"], (bar.Low  - ep) / ep * side)

            if cfg.be_trigger_pct is not None:
                (pos["sl"], pos["be_armed"], pos["pending_be_sl"],
                 pos["be_arm_index"], pos["be_active"]) = self._update_be_logic(
                    pos=pos, i=i, bar=bar,
                    be_trigger_pct=cfg.be_trigger_pct,
                    be_offset_pct=cfg.be_offset_pct,
                    be_delay_bars=cfg.be_delay_bars,
                )

            if cfg.trailing_trigger_pct is not None and self.atrs is not None:
                pos = self._update_trailing_runner(
                    pos=pos, bar=bar,
                    atr_value=self.atrs[i-1],  # i-1: ATR known before bar open
                    trailing_trigger_pct=cfg.trailing_trigger_pct,
                    trailing_mult=cfg.runner_trailing_mult,
                )

            exit_event = self._exit_logic(i, bar, pos)

            if exit_event is not None:
                if cfg.fast:
                    self.trades.append(self.trade_analysis(
                        position=pos["side"], exit_price=exit_event["exit_price"],
                        entry_price=pos["entry_price"], entry_time=pos["entry_time"],
                        ts=exit_event["exit_time"], exit_reason=exit_event["reason"],
                    ))
                else:
                    self.trades.append(self.trade_analysis(
                        position=pos["side"], exit_price=exit_event["exit_price"],
                        entry_price=pos["entry_price"], entry_time=pos["entry_time"],
                        ts=exit_event["exit_time"], exit_reason=exit_event["reason"],
                        mae=pos["mae"], mfe=pos["mfe"],
                        mae_h=pos["mae"] if self.use_observation else None,
                        mfe_h=pos["mfe"] if self.use_observation else None,
                    ))

                    if self.use_observation:
                        self.pending_observations.append({
                            "idx":   len(self.trades) - 1,
                            "ep":    pos["entry_price"],
                            "side":  pos["side"],
                            "end_i": i + self.observation_bars,
                        })
            else:
                surviving.append(pos)

        self.positions = surviving

    # =========================================================================
    # Core loop
    # =========================================================================

    def run(self) -> pd.DataFrame:
        """
        Main simulation loop — iterates bar by bar in strict chronological order.

        Per-bar sequence:
          1. Reverse mode  — close opposite positions at open
          2. Entry logic   — evaluate signal and filters, open new position
          3. MaxEntries    — reset counter on day/session change, apply cap
          4. Exit logic    — update BE/runner, check exit conditions
          5. Observation   — track hold-period MAE/MFE (fast=False only)
        """
        cfg = self.cfg

        for i in range(1, len(self.df)):
            ts       = self.index[i]
            bar      = self.Bar(self.opens[i], self.highs[i], self.lows[i], self.closes[i], ts)
            sig_prev = self.signals[i - 1]  # signal generated at close of bar[i-1]

            # 1) Reverse
            self._apply_reverse(sig_prev, bar.Open, ts)

            # 2) Entry
            entry = self._entry_logic(i, ts, bar, sig_prev)

            # 3) MaxEntries
            self._max_entries_reset(ts)
            entry = self._max_entries_filter(i, entry)

            if entry is not None:
                if cfg.multi_entry or len(self.positions) == 0:
                    self.positions.append(entry)
                if cfg.MaxEntries4Periods:
                    self.recent_entries.append(i)

            # 4) Exit
            self._update_positions(i, bar)

            # 5) Observation
            if (not cfg.fast) and self.use_observation and self.pending_observations:
                self._update_observations(i, bar)

        return pd.DataFrame(self.trades)
