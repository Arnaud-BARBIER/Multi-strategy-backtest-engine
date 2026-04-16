# from a .ipynb 
%pip install git+https://github.com/Arnaud-BARBIER/Multi-strategy-backtest-engine.git

# Personalised strategy Usecase
import pandas as pd
import numpy as np

from backtest_engine import *


def rsi_strategy(df, rsi_period=14, oversold=30, overbought=70):
    df = df.copy()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Signal"] = 0

    long_cond = (df["RSI"].shift(1) < oversold) & (df["RSI"] >= oversold)
    short_cond = (df["RSI"].shift(1) > overbought) & (df["RSI"] <= overbought)

    df.loc[long_cond, "Signal"] = 1
    df.loc[short_cond, "Signal"] = -1

    return df


pipeline = DataPipeline("Your/path/to/csv/files")

cfg = BacktestConfig(
    timezone_shift=1,
    crypto=False
)

njit_engine = NJITEngine(
    pipeline,
    "XAUUSD_M5",
    "2024-01-02",
    "2024-02-16",
    cfg,
    MAX_TRADES=50_000,
    MAX_POS=600
)

# Generate signals from a user-defined strategy
signal_df = njit_engine.signal_generation_inspection(
    strategy_fn=rsi_strategy,
    signal_col="Signal",
    plot=False,
    crypto=False,
    return_df_signals=True,
    rsi_period=14,
    oversold=30,
    overbought=70
)

signals = signal_df["Signal"].to_numpy(dtype=np.int8)

rets, metrics = njit_engine.run(
    signals,
    tp_pct=0.002,
    sl_pct=0.01,
    be_trigger_pct=0.005,
    be_offset_pct=0.001,
    be_delay_bars=5,
    me_max=100,
    me_reset_mode=4,
    me_period=50,
    session_1=("08:00", "12:00"),
    session_2=("13:00", "17:00"),
    session_3=None,
    track_mae_mfe=True,
    hold_minutes=2 * 60,
    bar_duration_min=5,
    candle_size_filter=True,
    min_size_pct=0.000,
    max_size_pct=0.1,
    entry_delay=1,
    prev_candle_direction=False,
    trailing_trigger_pct=0.005,
    runner_trailing_mult=3,
    plot=False,
    crypto=cfg.crypto,
    return_df_after=False,
    full_df_after=False,
    window_before=200,
    window_after=50,
)

trades_df = metrics_v2["trades_df"]   # metrics contains the trade history df
#df_after  = metrics_v2["df_after"]    # DF from Signal generation layer containing EntryTradeID, ExitTradeID

print(signal_df[["Close", "RSI", "Signal"]].tail(20))
print(trades_df.head())
print(metrics["win_rate"], metrics["sharpe"])
print(df_after[["Close", "RSI", "Signal", "EntryTradeID", "ExitTradeID"]].tail(50))
#print(metrics_v2["win_rate"], metrics_v2["sharpe"])

# Default ema_close strategy usecase
from backtest_engine import BacktestConfig, DataPipeline, NJITEngine


pipeline = DataPipeline("/Users/arnaudbarbier/Desktop/Quant reaserch/Metals")


cfg = BacktestConfig(
    # signal defaults
    period_1=30,
    period_2=100,

    # entry logic
    entry_delay=1,
    prev_candle_direction=False,

    # session filters
    session_1=("08:00", "12:00"),
    session_2=("13:00", "17:00"),
    session_3=None,

    # candle filter
    candle_size_filter=True,
    min_size_pct=0.000,
    max_size_pct=0.1,

    # TP / SL
    tp_pct=0.002,
    sl_pct=0.01,

    # break-even
    be_trigger_pct=0.005,
    be_offset_pct=0.001,
    be_delay_bars=5,

    # runner trailing
    trailing_trigger_pct=0.005,
    runner_trailing_mult=3,

    # entry cap logic
    me_max=3,
    me_period=10,
    me_reset_mode=3,

    # metrics
    track_mae_mfe=True,
    hold_minutes=2 * 60,
    bar_duration_min=5,

    # optional preprocessing
    timezone_shift=1,
)

njit_engine = NJITEngine(
    pipeline,
    "XAUUSD_M5",
    "2023-01-02",
    "2026-02-16",
    cfg,
    MAX_TRADES=50_000,
    MAX_POS=600,
)

# Built-in default EMA signal generation
signals = njit_engine.signals_ema(
    span1=cfg.period_1,
    span2=cfg.period_2,
    mode="close_vs_ema",
)

rets, metrics = njit_engine.run(signals)

print(metrics)
print("Exit reasons:")
print(metrics["trades_df"]["reason"].value_counts())
