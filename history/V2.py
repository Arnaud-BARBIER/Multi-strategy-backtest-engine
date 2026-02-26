# =============================================================================
# Backtesting Engine — V2.2 Functional
# =============================================================================
# Major evolution from V1:
#   - Multi-position management (list of dicts instead of single position)
#   - Session-based time filtering (3 configurable windows)
#   - Breakeven logic with delay and offset
#   - ATR-based TP/SL
#   - EMA exit filters (EMA1, EMA2, EMA cross)
#   - MaxEntries cap with sliding window + day/session reset
#
# Known performance bottlenecks (addressed in V2.2):
#   - df.iloc[i] inside the loop → repeated full-row extraction at every bar
#   - df["col"].iloc[i] → double indexing, triggers pandas overhead each call
#   - exit_mode recomputed at every bar instead of once before the loop
#   - positions.copy() + positions.remove() → O(n) list scan per trade
#   - recent_entries list comprehension rebuilt entirely at every bar
#   - df passed into exit_logic for EMA lookup → iloc[i] inside a nested call
#
# Superseded by V2.2 (class architecture + numpy array precomputation).
# =============================================================================

import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def fetchdata(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV and slice to the requested date range.
    Timestamps shifted +1h to align with broker timezone.
    """
    df = pd.read_csv(
        f"/Users/arnaudbarbier/Desktop/research/Metals/{ticker}.csv",
        header=None,
        names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
    )
    df["Datetime"] = pd.to_datetime(df["Datetime"]) + pd.Timedelta(hours=1)
    df = df.set_index("Datetime").sort_index()
    return df.loc[start:end]


# -----------------------------------------------------------------------------
# Signal Generation
# -----------------------------------------------------------------------------

def EMA_signal(df: pd.DataFrame, period_1: int = 50, period_2: int = 100,
               max_gap_size: float = None) -> pd.DataFrame:
    """
    Generate entry signals on EMA1/price crossovers.
    EMA2 and EMA_CROSS columns are computed for use as exit filters.

    gap_filter optionally rejects signals on bars with abnormal open gaps,
    useful for filtering out news spikes on metals futures.
    """
    d = df.copy()  # avoid mutating the original df passed by the caller

    d["EMA1"] = d["Close"].ewm(span=period_1, adjust=False).mean()
    d["EMA2"] = d["Close"].ewm(span=period_2, adjust=False).mean()

    # Optional gap filter — rejects signal if open gaps more than max_gap_size
    if max_gap_size is not None:
        gap_pct    = abs(d["Open"] - d["Close"].shift(1)) / d["Close"].shift(1)
        gap_filter = gap_pct < max_gap_size
    else:
        gap_filter = pd.Series(True, index=d.index)

    # Signal 1: EMA1/price cross — primary entry trigger
    d["Signal"] = np.where(
        (d["EMA1"].shift(1) < d["Close"].shift(1)) & (d["EMA1"] > d["Close"]) & gap_filter, -1,
        np.where(
            (d["EMA1"].shift(1) > d["Close"].shift(1)) & (d["EMA1"] < d["Close"]) & gap_filter, 1,
            0,
        ),
    )

    # Signal 2: EMA2/price cross — used as exit filter, not entry
    d["Signal2"] = 0
    bull = (d["EMA2"].shift(1) > d["Close"].shift(1)) & (d["EMA2"] < d["Close"]) & gap_filter
    bear = (d["EMA2"].shift(1) < d["Close"].shift(1)) & (d["EMA2"] > d["Close"]) & gap_filter
    d.loc[bull, "Signal2"] = 1
    d.loc[bear, "Signal2"] = -1

    # EMA cross column — used as exit filter
    d["EMA_CROSS"] = 0
    up_cross   = (d["EMA1"].shift(1) < d["EMA2"].shift(1)) & (d["EMA1"] > d["EMA2"]) & gap_filter
    down_cross = (d["EMA1"].shift(1) > d["EMA2"].shift(1)) & (d["EMA1"] < d["EMA2"]) & gap_filter
    d.loc[up_cross,   "EMA_CROSS"] = 1
    d.loc[down_cross, "EMA_CROSS"] = -1

    # Entry_Price for reference only — engine executes at bar.Open[i+1]
    d["Entry_Price"] = d["Open"].where(d["Signal"].shift(1) != 0)

    return d


# -----------------------------------------------------------------------------
# ATR
# -----------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Standard Average True Range — used for volatility-adjusted TP/SL.
    Computed once before the loop; accessed via iloc[i] in entry_logic.
    Note: in V2.2 this is precomputed as a numpy array to avoid per-bar iloc cost.
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    df["ATR"] = tr.rolling(period).mean()
    return df


# -----------------------------------------------------------------------------
# TP / SL Calculation
# -----------------------------------------------------------------------------

def Tp_Sl_prices(side, entry_price, tp_pct=None, sl_pct=None,
                 use_atr=False, atr_value=None, tp_atr_mult=None, sl_atr_mult=None):
    """
    Compute TP and SL prices — fixed percentage or ATR-based.
    ATR mode adapts risk levels to current volatility regime,
    making parameters more portable across assets (e.g. XAU vs XAG).
    """
    if use_atr:
        if atr_value is None:
            return None, None
        tp = entry_price + side * atr_value * tp_atr_mult
        sl = entry_price - side * atr_value * sl_atr_mult
    else:
        tp = entry_price * (1 + side * tp_pct)
        sl = entry_price * (1 - side * sl_pct)

    return tp, sl


# -----------------------------------------------------------------------------
# Time Window Filter
# -----------------------------------------------------------------------------

def time_in_window(ts, window_str: str) -> bool:
    """
    Check whether timestamp ts falls within a session window.
    Supports overnight windows (e.g. "22:00-02:00") via wraparound logic.
    Called once per bar per active window — three sessions supported.
    """
    if window_str is None:
        return False

    start_str, end_str = window_str.split("-")
    start   = pd.to_datetime(start_str).time()
    end     = pd.to_datetime(end_str).time()
    current = ts.time()

    if start <= end:
        return start <= current <= end
    else:
        # Overnight session — wraps past midnight
        return current >= start or current <= end


def get_active_session(ts, windows: list):
    """
    Return the index of the first active session window at timestamp ts.
    Used to detect session changes for MaxEntries reset.
    """
    for idx, w in enumerate(windows):
        if w is not None and time_in_window(ts, w):
            return idx
    return None


# -----------------------------------------------------------------------------
# Entry Logic
# -----------------------------------------------------------------------------

def entry_logic(df, i, min_size_pct, max_size_pct,
                Previous_Candle_same_direction, Candle_Size_filter,
                ts, tp_pct, sl_pct, bar, sig_prev,
                time_window_1=None, time_window_2=None, time_window_3=None,
                use_atr_sl_tp=False, tp_atr_mult=None, sl_atr_mult=None):
    """
    Evaluate entry conditions for bar[i] and return a position dict or None.

    Early returns (guard clauses) keep the logic flat:
      1. Time window check
      2. Signal check
      3. Candle filter
      4. ATR validity
    If all pass, a new position dict is returned — consumed by the main loop.

    Performance note: df["col"].iloc[i] is called here for prev_open/close
    and ATR. In V2.2 these are precomputed as numpy arrays, eliminating
    repeated pandas indexing overhead (~10x speedup on large datasets).
    """
    windows = [time_window_1, time_window_2, time_window_3]

    # Reject if outside all active session windows
    if any(w is not None for w in windows):
        if not any(time_in_window(ts, w) for w in windows if w is not None):
            return None

    if sig_prev == 0:
        return None

    # Only computed after signal check — avoids unnecessary iloc calls
    prev_open  = df["Open"].iloc[i - 1]
    prev_close = df["Close"].iloc[i - 1]

    body_pct     = abs(prev_open - prev_close) / prev_close
    size_ok      = min_size_pct < body_pct < max_size_pct
    direction_ok = (
        (sig_prev ==  1 and prev_close > prev_open) or
        (sig_prev == -1 and prev_close < prev_open)
    )

    if Candle_Size_filter:
        if not (size_ok and (not Previous_Candle_same_direction or direction_ok)):
            return None

    entry_price = bar.Open

    # ATR fetched via iloc — bottleneck flagged for V2.2 (precomputed array)
    atr_value = df["ATR"].iloc[i] if use_atr_sl_tp else None

    if use_atr_sl_tp and pd.isna(atr_value):
        return None

    tp, sl = Tp_Sl_prices(
        side=sig_prev, entry_price=entry_price,
        tp_pct=tp_pct, sl_pct=sl_pct,
        use_atr=use_atr_sl_tp, atr_value=atr_value,
        tp_atr_mult=tp_atr_mult, sl_atr_mult=sl_atr_mult,
    )

    return {
        "side":         sig_prev,
        "entry_price":  entry_price,
        "entry_time":   ts,
        "entry_index":  i,
        "tp":           tp,
        "sl":           sl,
        "be_armed":     False,
        "pending_be_sl": None,
        "be_arm_index": None,
        "be_active":    False,
    }


# -----------------------------------------------------------------------------
# Reverse Mode
# -----------------------------------------------------------------------------

def apply_reverse(positions: list, sig_prev: int, bar_open: float,
                  ts, reverse_mode: bool):
    """
    Close all positions opposite to sig_prev at market open.
    Reverse mode flips exposure when a new signal fires in the opposite direction.
    Returns updated positions list and list of closed trades.
    """
    if not reverse_mode or sig_prev == 0:
        return positions, []

    closed    = []
    surviving = [pos for pos in positions if pos["side"] != -sig_prev]

    for pos in positions:
        if pos["side"] == -sig_prev:
            closed.append(trade_analysis(
                position=pos["side"], exit_price=bar_open,
                entry_price=pos["entry_price"], entry_time=pos["entry_time"],
                ts=ts, exit_reason="REVERSE",
            ))

    return surviving, closed


# -----------------------------------------------------------------------------
# Breakeven Logic
# -----------------------------------------------------------------------------

def update_be_logic(position, entry_price, base_sl, i, entry_index, bar,
                    be_armed, pending_be_sl, be_arm_index, be_active,
                    be_trigger_pct=None, be_offset_pct=0.0, be_delay_bars=0):
    """
    Two-phase breakeven mechanism:
      Phase 1 — ARM:    price reaches trigger threshold → record pending BE SL
      Phase 2 — ACTIVATE: on the following bar, replace SL with BE SL

    Activation is delayed by one bar (i > be_arm_index) to avoid
    same-bar look-ahead. be_delay_bars adds a minimum holding period
    before the trigger is even checked.
    """
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

    # Activate on next bar — never on the arming bar itself
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


# -----------------------------------------------------------------------------
# Exit Logic
# -----------------------------------------------------------------------------

def exit_logic(pos, bar, i, allow_exit_on_entry_bar, df,
               exit_mode, EMA1_TP, EMA2_TP, EMA_CROSS_TP, EMA_SL):
    """
    Evaluate exit conditions for a single open position.

    Two exit modes:
      fixed — standard TP/SL with gap-open handling
      ema   — SL fixed, TP triggered by EMA condition (only if trade is profitable)

    Performance note: in EMA mode, df["EMA1"].iloc[i] is called here on every
    bar for every open position. In V2.2, EMA values are precomputed as numpy
    arrays and accessed as self.ema1s[i] — eliminating repeated iloc overhead.
    """
    if not allow_exit_on_entry_bar and i == pos["entry_index"]:
        return None

    side  = pos["side"]
    tp    = pos["tp"]
    sl    = pos["sl"]
    o, h, l, close = bar.Open, bar.High, bar.Low, bar.Close
    be_reason = "BE" if pos.get("be_active", False) else "SL"

    # =========================
    # FIXED MODE
    # =========================
    if exit_mode == "fixed":

        if side == 1:
            if o <= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
            elif l <= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
            elif h >= tp: return {"exit_price": tp, "exit_time": bar.name, "reason": "TP"}

        elif side == -1:
            if o >= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
            elif h >= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
            elif l <= tp: return {"exit_price": tp, "exit_time": bar.name, "reason": "TP"}

    # =========================
    # EMA MODE
    # =========================
    elif exit_mode == "ema":

        # EMA values fetched via iloc — bottleneck flagged for V2.2
        ema1 = df["EMA1"].iloc[i]
        ema2 = df["EMA2"].iloc[i]
        entry_price = pos["entry_price"]

        # SL always checked first — takes priority over EMA exit
        if side == 1:
            if o <= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
            if l <= sl:   return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
        elif side == -1:
            if o >= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
            if h >= sl:   return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}

        # EMA exit only triggered if trade is currently profitable
        # — avoids locking in losses at an arbitrary EMA level
        trade_positive = (
            (side ==  1 and close > entry_price) or
            (side == -1 and close < entry_price)
        )
        if not trade_positive:
            return None

        if EMA1_TP:
            if side == 1  and close < ema1: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA1_TP"}
            if side == -1 and close > ema1: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA1_TP"}

        if EMA2_TP:
            if side == 1  and close < ema2: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA2_TP"}
            if side == -1 and close > ema2: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA2_TP"}

        if EMA_CROSS_TP:
            if side == 1  and ema1 < ema2: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA_CROSS_TP"}
            if side == -1 and ema1 > ema2: return {"exit_price": close, "exit_time": bar.name, "reason": "EMA_CROSS_TP"}

    return None


# -----------------------------------------------------------------------------
# Trade Analysis
# -----------------------------------------------------------------------------

def trade_analysis(position, exit_price, entry_price, entry_time, ts, exit_reason):
    """Build a trade record dict — appended to the trades list on each close."""
    ret = position * (exit_price - entry_price) / entry_price
    return {
        "entry_time": entry_time,
        "exit_time":  ts,
        "side":       position,
        "entry":      entry_price,
        "exit":       exit_price,
        "return":     ret,
        "reason":     exit_reason,
    }


# -----------------------------------------------------------------------------
# Main Engine
# -----------------------------------------------------------------------------

def filtre_AND_Exit_StratV2(
    df,
    # --- Filters ---
    Candle_Size_filter: bool = True,
    Previous_Candle_same_direction: bool = True,
    min_size_pct: float = 0.001,
    max_size_pct: float = 0.02,
    # --- TP/SL ---
    tp_pct: float = 0.05,
    sl_pct: float = 0.02,
    use_atr_sl_tp: bool = False,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    # --- Exit mode ---
    EMA1_TP: bool = False,
    EMA2_TP: bool = False,
    EMA_CROSS_TP: bool = False,
    EMA_SL: bool = False,
    # --- Multi-entry cap ---
    MaxEntries4Periods: bool = True,
    ME_X: int = 2,
    ME_Period_Y: int = 8,
    ME_reset_mode: str = None,     # "day" / "session" / None
    # --- Engine behavior ---
    allow_exit_on_entry_bar: bool = True,
    multi_entry: bool = True,
    reverse_mode: bool = False,
    # --- Breakeven ---
    be_trigger_pct: float = None,
    be_offset_pct: float = 0.0,
    be_delay_bars: int = 0,
    # --- Session windows ---
    time_window_1: str = None,
    time_window_2: str = None,
    time_window_3: str = None,
) -> pd.DataFrame:
    """
    Bar-by-bar simulation loop — multi-position architecture.

    Key upgrades vs V1:
    ─────────────────────────────────────────────────────────────
    MULTI-POSITION
      Positions stored as a list of dicts instead of scalar variables.
      Each position carries its own entry_price, tp, sl, BE state... 
      the dict is created in the entry_logic function after one entry 
      has been aproved.
      multi_entry=True allows stacking positions on repeated signals.

    SESSION FILTERING
      Three configurable time windows (e.g. London, Asia, NY).
      Entry is rejected outside active windows.
      MaxEntries counter can reset per session or per day,
      preventing overtrading within a single session.

    MAX ENTRIES CAP
      Sliding window counter — tracks entries over the last ME_Period_Y bars.
      Reset logic: "day" resets at midnight, "session" resets on session change.

    BREAKEVEN
      Two-phase: arm on trigger bar, activate on next bar.
      be_delay_bars prevents premature BE on volatile entries.

    ─────────────────────────────────────────────────────────────
    Performance bottlenecks (addressed in V2.2):
      - exit_mode recomputed at every bar 
      - df.iloc[i] called inside entry_logic and exit_logic → replaced
        by numpy array precomputation in V2.2
      - positions.copy() + positions.remove() → replaced by surviving
        list pattern in V2.2 (avoids O(n) scan per closed trade)
      - recent_entries rebuilt via list comprehension each bar → replaced
        by sliding pointer (j) in V2.2
      - One last notable bottleneck was the session/day/periode candle count cap
        reset. the optimisation done would be explained in the V2.2.  
    ─────────────────────────────────────────────────────────────
    """

    # Multi-position state — list of open position dicts
    positions     = []
    trades        = []
    recent_entries = []
    last_session_id = None
    last_day        = None

    # BE state lives inside each position dict — not shared across positions
    # (be_armed, pending_be_sl initialized per position in entry_logic)

    # exit_mode determined by EMA flags — recomputed each bar here (V2.2 moves this before loop)
    for i in range(1, len(df)):
        ts       = df.index[i]
        bar      = df.iloc[i]   # full row extraction — bottleneck flagged for V2.2
        sig_prev = df["Signal"].iloc[i - 1]

        # exit_mode recomputed every bar — wasteful, moved before loop in V2.2
        exit_mode = "ema" if sum([EMA1_TP, EMA2_TP, EMA_CROSS_TP]) == 1 else "fixed"

        # ── Reverse ──────────────────────────────────────────────────────────
        positions, closed = apply_reverse(positions, sig_prev, bar.Open, ts, reverse_mode)
        trades.extend(closed)

        # ── Entry ─────────────────────────────────────────────────────────────
        entry_event = entry_logic(
            df=df, i=i, min_size_pct=min_size_pct, max_size_pct=max_size_pct,
            Previous_Candle_same_direction=Previous_Candle_same_direction,
            Candle_Size_filter=Candle_Size_filter, ts=ts,
            tp_pct=tp_pct, sl_pct=sl_pct, bar=bar, sig_prev=sig_prev,
            use_atr_sl_tp=use_atr_sl_tp, tp_atr_mult=tp_atr_mult, sl_atr_mult=sl_atr_mult,
            time_window_1=time_window_1, time_window_2=time_window_2, time_window_3=time_window_3,
        )

        # ── MaxEntries reset ──────────────────────────────────────────────────
        current_day = ts.date()

        if ME_reset_mode == "day":
            if last_day is None:
                last_day = current_day
            elif current_day != last_day:
                recent_entries = []
                last_day = current_day

        elif ME_reset_mode == "session":
            windows         = [time_window_1, time_window_2, time_window_3]
            current_session = get_active_session(ts, windows)
            if last_session_id is None:
                last_session_id = current_session
            elif current_session != last_session_id:
                recent_entries  = []
                last_session_id = current_session

        # Sliding window cleanup — list comprehension rebuilt each bar (V2.2 uses pointer)
        if MaxEntries4Periods:
            recent_entries = [idx for idx in recent_entries if i - idx < ME_Period_Y]
            if len(recent_entries) >= ME_X:
                entry_event = None  # quota reached — block entry

        if entry_event is not None:
            if multi_entry or len(positions) == 0:
                positions.append(entry_event)
            if MaxEntries4Periods:
                recent_entries.append(i)

        # ── Exit ──────────────────────────────────────────────────────────────
        # positions.copy() used to iterate safely while removing closed positions.
        # In V2.2 replaced by a surviving list pattern — avoids O(n) remove() scan.
        for pos in positions.copy():

            pos["sl"], pos["be_armed"], pos["pending_be_sl"], \
            pos["be_arm_index"], pos["be_active"] = update_be_logic(
                position=pos["side"], entry_price=pos["entry_price"],
                base_sl=pos["sl"], i=i, entry_index=pos["entry_index"], bar=bar,
                be_armed=pos["be_armed"], pending_be_sl=pos["pending_be_sl"],
                be_arm_index=pos["be_arm_index"], be_active=pos["be_active"],
                be_trigger_pct=be_trigger_pct, be_offset_pct=be_offset_pct,
                be_delay_bars=be_delay_bars,
            )

            exit_event = exit_logic(
                pos=pos, bar=bar, i=i,
                allow_exit_on_entry_bar=allow_exit_on_entry_bar,
                df=df, exit_mode=exit_mode,
                EMA1_TP=EMA1_TP, EMA2_TP=EMA2_TP,
                EMA_CROSS_TP=EMA_CROSS_TP, EMA_SL=EMA_SL,
            )

            if exit_event is not None:
                trades.append(trade_analysis(
                    position=pos["side"], exit_price=exit_event["exit_price"],
                    entry_price=pos["entry_price"], entry_time=pos["entry_time"],
                    ts=exit_event["exit_time"], exit_reason=exit_event["reason"],
                ))
                positions.remove(pos)  # O(n) scan — replaced by surviving list in V2.2

    return pd.DataFrame(trades)


# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------

df       = fetchdata("XAGUSD_M5", "2021-01-01", "2025-01-01")
df       = compute_atr(df, period=14)
df       = EMA_signal(df, period_1=50, period_2=100)

trades = filtre_AND_Exit_StratV2(
    df,
    Candle_Size_filter=True,
    Previous_Candle_same_direction=False,
    min_size_pct=0.0003,
    max_size_pct=0.01,
    tp_pct=0.01,
    sl_pct=0.004,
    MaxEntries4Periods=True,
    ME_X=3,
    ME_Period_Y=50,
    ME_reset_mode="session",
    allow_exit_on_entry_bar=True,
    multi_entry=True,
    be_trigger_pct=0.003,
    be_offset_pct=0.0005,
    be_delay_bars=5,
    time_window_1="08:00-12:00",
    time_window_2="01:00-06:00",
    time_window_3="13:30-18:30",
)
