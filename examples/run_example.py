# Install the package directly from GitHub.
%pip install --upgrade --force-reinstall --no-cache-dir "git+https://github.com/Arnaud-BARBIER/Multi-strategy-backtest-engine.git@main"


from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine import *

pd.set_option("display.max_columns", 50)

# Keep the `Metals` folder next to this notebook so data loading works out of the box.
metals_path = Path.cwd() / "Metals"
assert metals_path.exists(), f"Missing data folder: {metals_path}"

pipeline = DataPipeline(str(metals_path))

cfg_ema = BacktestConfig(
    timezone_shift=1,
    entry_delay=1,
    candle_size_filter=True,
    tp_pct=0.004,
    sl_pct=0.001,
    be_trigger_pct=0.006,
    be_offset_pct=0.0,
    be_delay_bars=5,
    forced_flat_time="22:00",
    forced_flat_frequency="weekend",
)

engine = NJITEngine(
    pipeline,
    "XAUUSD_M5",
    "2023-06-19",
    "2026-04-15",
    cfg_ema,
    MAX_TRADES=50_000,
    MAX_POS=600,
)

signals_ema = engine.signals_ema(
    span1=30,
    span2=100,
    mode="close_vs_ema",
)

rets_ema, metrics_ema = engine.run(
    signals=signals_ema,
    cfg=cfg_ema,
    plot=False,
    plot_results=True,
)

print(metrics_ema["win_rate"], metrics_ema["sharpe"])
metrics_ema["trades_df"].head()

##-------------------------------
# Personalised strategy: RSI re-entry signal
##-------------------------------

def rsi_reentry_df(
    df,
    rsi_length=14,
    rsi_low=30,
    rsi_high=70,
    min_breakout=5.0,
):
    df = df.copy()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    long_active = np.zeros(len(df), dtype=np.int8)
    short_active = np.zeros(len(df), dtype=np.int8)

    in_oversold = False
    in_overbought = False
    oversold_min = np.nan
    overbought_max = np.nan

    for i in range(len(df)):
        r = rsi.iloc[i]
        if pd.isna(r):
            continue

        if r < rsi_low:
            if not in_oversold:
                in_oversold = True
                oversold_min = r
            else:
                oversold_min = min(oversold_min, r)

        elif in_oversold and r >= rsi_low:
            if oversold_min <= (rsi_low - min_breakout):
                long_active[i] = 1
            in_oversold = False
            oversold_min = np.nan

        if r > rsi_high:
            if not in_overbought:
                in_overbought = True
                overbought_max = r
            else:
                overbought_max = max(overbought_max, r)

        elif in_overbought and r <= rsi_high:
            if overbought_max >= (rsi_high + min_breakout):
                short_active[i] = 1
            in_overbought = False
            overbought_max = np.nan

    df["RSI"] = rsi
    df["Signal"] = np.where(
        long_active == 1,
        1,
        np.where(short_active == 1, -1, 0),
    ).astype("int8")

    return df

##-------------------------------
# Inspect the raw signal
##-------------------------------
signal_df = engine.inspect_signals(
    strategy_fn=rsi_reentry_df,
    signal_col="Signal",
    plot=True,
    start="2023-07-01",
    end="2023-08-01",
    return_df=True,
    rsi_length=14,
    rsi_low=30,
    rsi_high=70,
    min_breakout=5.0,
)

signal_df[["Close", "RSI", "Signal"]].tail(20)

##-------------------------------
# Bridge the same signal into setup mode
##-------------------------------

rsi_signal_to_setup = NJITEngine.wrap_signal_strategy(rsi_reentry_df)

setup_specs = [
    SetupSpec(
        fn=rsi_signal_to_setup,
        params=dict(
            setup_id=0,
            score=1.0,
            rsi_length=14,
            rsi_low=30,
            rsi_high=70,
            min_breakout=5.0,
        ),
        name="rsi_reentry",
    ),
]

profile_rsi = ExitProfileSpec(
    name="rsi_profile",
    tp_pct=0.01,
    sl_pct=0.001,
    be_trigger_pct=0.006,
    be_offset_pct=0.0,
    be_delay_bars=5,
)

execution_context = build_execution_context(
    cfg=cfg_rsi,
    exit_profile_specs=[profile_rsi],
    setup_exit_binding={
        0: {
            "exit_profile_id": 0,
            "exit_strategy_id": -1,
        }
    },
    strategy_profile_binding={},
    n_setups=len(setup_specs),
    exit_strategy_specs=[],
    n_strategies=0,
)

rets_setup, metrics_setup = engine.run(
    signals=prep.signals,
    selected_setup_id=prep.selected_setup_id,
    selected_score=prep.selected_score,
    execution_context=execution_context,
    use_exit_system=True,
    multi_setup_mode=True,
    cfg=cfg_rsi,
    plot=False,
    plot_results=True,
)

print(metrics_setup["win_rate"], metrics_setup["sharpe"])
metrics_setup["trades_df"].head()


