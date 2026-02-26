# =============================================================================
# Backtesting Engine — V1 Functional
# =============================================================================
# Single-file, function-based implementation.
# Proof of concept — single position, fixed TP/SL, two EMA signals.
# Superseded by V2 (modular) and V3 (class-based architecture).
# =============================================================================

import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def fetchdata(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV and slice to the requested date range.
    Timestamps are shifted +1h to align with broker timezone.
    """
    df = pd.read_csv(
        f"/Users/arnaudbarbier/Desktop/Quant research/Metals/{ticker}.csv",
        header=None,
        names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
    )
    df["Datetime"] = pd.to_datetime(df["Datetime"]) + pd.Timedelta(hours=1)
    df = df.set_index("Datetime").sort_index()
    return df.loc[start:end]


# -----------------------------------------------------------------------------
# Signal Generation
# -----------------------------------------------------------------------------

def EMA_signal(df: pd.DataFrame, period_1: int = 50, period_2: int = 100) -> pd.DataFrame:
    """
    Compute two EMAs and generate independent entry signals on price crossovers.

    Signal  : EMA1/price cross — primary entry signal (faster EMA)
    Signal2 : EMA2/price cross — secondary confirmation signal (slower EMA)

    Two generation approaches demonstrated:
      - np.where vectorization  → Signal  (production-ready, single pass)
      - pandas boolean masking  → Signal2 (more readable, equivalent output)

    Signal convention:
       1 = long  (price crosses above EMA)
      -1 = short (price crosses below EMA)
       0 = no signal

    Both periods are exposed as arguments — the resulting df is passed
    directly into the engine, which reads the Signal column independently
    of how it was generated.
    """
    d = df.copy()

    d["EMA1"] = d["Close"].ewm(span=period_1, adjust=False).mean()
    d["EMA2"] = d["Close"].ewm(span=period_2, adjust=False).mean()

    # --- Signal 1: np.where vectorization (EMA1/price cross) ---
    # Detects the crossing bar by comparing previous vs current bar state.
    # Signal is generated on close — execution happens at next bar open.
    d["Signal"] = np.where(
        (d["EMA1"].shift(1) < d["Close"].shift(1)) & (d["EMA1"] > d["Close"]), -1,
        np.where(
            (d["EMA1"].shift(1) > d["Close"].shift(1)) & (d["EMA1"] < d["Close"]), 1,
            0,
        ),
    )

    # --- Signal 2: pandas boolean masking (EMA2/price cross) ---
    # Initialized to 0 to avoid NaN propagation downstream.
    d["Signal2"] = 0
    bull = (d["EMA2"].shift(1) > d["Close"].shift(1)) & (d["EMA2"] < d["Close"])
    bear = (d["EMA2"].shift(1) < d["Close"].shift(1)) & (d["EMA2"] > d["Close"])
    d.loc[bull, "Signal2"] = 1
    d.loc[bear, "Signal2"] = -1

    # Entry_Price for reference only — engine executes at bar.Open[i+1]
    d["Entry_Price"] = d["Open"].where(d["Signal"].shift(1) != 0)

    return d


# -----------------------------------------------------------------------------
# TP / SL Calculation
# -----------------------------------------------------------------------------

def Tp_Sl_prices(side: int, entry_price: float, tp_pct: float, sl_pct: float):
    """
    Compute fixed percentage TP and SL from entry price.
    First externalized helper — early step toward a modular architecture.

    Parameters
    ----------
    side        : 1 for long, -1 for short
    entry_price : execution price at bar open
    tp_pct      : take profit distance as fraction of entry
    sl_pct      : stop loss distance as fraction of entry
    """
    tp = entry_price * (1 + side * tp_pct)
    sl = entry_price * (1 - side * sl_pct)
    return tp, sl


# -----------------------------------------------------------------------------
# Backtest Engine
# -----------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    # --- Filters ---
    Candle_Size_filter: bool = True,
    Previous_Candle_same_direction: bool = True,
    min_size_pct: float = 0.001,
    max_size_pct: float = 0.02,
    # --- TP/SL ---
    tp_pct: float = 0.05,
    sl_pct: float = 0.02,
    # --- Engine behavior ---
    allow_exit_on_entry_bar: bool = True,
) -> pd.DataFrame:
    """
    Bar-by-bar simulation loop — single position at a time.

    The loop structure is intentional: state-dependent logic (position tracking,
    entry/exit sequencing) cannot be vectorized with np.where without introducing
    look-ahead bias. Each bar is processed in strict chronological order.

    Returns a DataFrame of completed trades with entry/exit metadata.
    """

    # Mutable engine state — reset at simulation start
    position    = 0
    entry_price = None
    entry_time  = None
    entry_index = None
    tp          = None
    sl          = None
    trades      = []

    for i in range(1, len(df)):
        ts       = df.index[i]
        bar      = df.iloc[i]
        sig_prev = df["Signal"].iloc[i - 1]  # signal generated on previous close

        o = bar.Open
        h = bar.High
        l = bar.Low

        # -----------------------------------------------------------------
        # Entry Logic
        # -----------------------------------------------------------------
        # Signal consumed at the open of bar[i] — generated at close of bar[i-1].
        # Being inside the loop allows stateful filtering without vectorization.

        if position == 0 and sig_prev != 0:

            prev_open  = df["Open"].iloc[i - 1]
            prev_close = df["Close"].iloc[i - 1]
            prev_high  = df["High"].iloc[i - 1]
            prev_low   = df["Low"].iloc[i - 1]

            # Body normalized by close for cross-asset comparability
            body_pct = abs(prev_open - prev_close) / prev_close

            # Pre-computed as booleans — applied as filters rather than
            # buried in nested conditionals
            size_ok      = min_size_pct < body_pct < max_size_pct
            direction_ok = (
                (sig_prev ==  1 and prev_close > prev_open) or
                (sig_prev == -1 and prev_close < prev_open)
            )

            if Candle_Size_filter:
                # size_ok is mandatory; direction_ok only enforced if flag is active
                if size_ok and (not Previous_Candle_same_direction or direction_ok):
                    position    = sig_prev
                    entry_price = o
                    entry_time  = ts
                    entry_index = i
                    tp, sl      = Tp_Sl_prices(position, entry_price, tp_pct, sl_pct)

            else:
                # Filter disabled — any signal triggers an entry
                position    = sig_prev
                entry_price = o
                entry_time  = ts
                entry_index = i
                tp, sl      = Tp_Sl_prices(position, entry_price, tp_pct, sl_pct)

        # -----------------------------------------------------------------
        # Exit Logic
        # -----------------------------------------------------------------

        if position != 0:

            # Hold through entry bar to avoid same-bar whipsaws if configured
            if not allow_exit_on_entry_bar and i == entry_index:
                continue

            exit_price  = None
            exit_reason = None

            if position == 1:
                # Gap open below SL → fill at open (SL price unreachable)
                if o <= sl:
                    exit_price, exit_reason = o,  "SL"
                elif l <= sl:
                    exit_price, exit_reason = sl, "SL"
                elif h >= tp:
                    exit_price, exit_reason = tp, "TP"

            elif position == -1:
                # Gap open above SL → fill at open
                if o >= sl:
                    exit_price, exit_reason = o,  "SL"
                elif h >= sl:
                    exit_price, exit_reason = sl, "SL"
                elif l <= tp:
                    exit_price, exit_reason = tp, "TP"

            if exit_price is not None:
                ret = position * (exit_price - entry_price) / entry_price

                trades.append({
                    "entry_time": entry_time,
                    "exit_time":  ts,
                    "side":       position,
                    "entry":      entry_price,
                    "exit":       exit_price,
                    "return":     ret,
                    "reason":     exit_reason,
                })

                # Reset position state for next trade
                position    = 0
                entry_price = None
                entry_time  = None
                tp          = None
                sl          = None

    return pd.DataFrame(trades)


# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------

df = fetchdata("XAUUSD_M5", "2021-01-01", "2026-02-01")
df = EMA_signal(df, period_1=50, period_2=100)

    trades = run_backtest(
        df,
        tp_pct=0.01,
        sl_pct=0.004,
        Candle_Size_filter=True,
    )

    print(f"Trades      : {len(trades)}")
    print(f"Win rate    : {(trades['return'] > 0).mean():.2%}")
    print(f"Total return: {trades['return'].sum():.4f}")
