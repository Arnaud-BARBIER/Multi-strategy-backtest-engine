# Metals Futures Backtesting Engine
### A modular, class-based backtesting framework built for systematic research on OHLCV data

<img width="1100" height="699" alt="Backtest Engine Screenshot" src="https://github.com/user-attachments/assets/32a9db9e-3aaa-4d65-8747-d04215617657" />

---

I built this after realizing that iterating on a monolithic backtest script was getting in the way of doing actual research. Every time I wanted to test a different exit condition or add a session filter, I was rewriting the same loop. This project is the result of refactoring that into something I can actually reuse.

The engine is written in Python and designed around one idea: **signal generation and execution logic should be completely independent**. You bring a strategy that produces a `Signal` column. The engine handles everything else — entries, exits, position sizing, breakeven, trailing stops, session filtering, and trade analytics.

---

## Architecture

```
BacktestConfig       — all parameters in one place
      ↓
DataPipeline         — fetch OHLCV, compute ATR, build indicators
      ↓
Strategy_Signal      — generate Signal column (1 / -1 / 0)
      ↓
BacktestEngine       — bar-by-bar simulation, stateful position management
      ↓
trades DataFrame     — entry/exit/return/MAE/MFE per trade
```

The engine never touches signal generation logic. A strategy only needs to return a DataFrame with a `Signal` column — the rest is handled internally.

---

## Features

- Multi-position management with per-position state
- Breakeven logic with configurable trigger, offset, and delay
- ATR-based trailing stop with two-phase activation
- Fixed or ATR-based TP/SL
- EMA exit filters (price/EMA cross, EMA/EMA cross)
- Session-based time filtering (up to 3 windows, overnight-compatible)
- MaxEntries cap with sliding window — resets per day or per session
- Reverse mode
- Gap filter on signal generation
- MAE/MFE intra-trade tracking
- Hold-period observation (post-exit price behavior analysis)
- NumPy array precomputation for loop performance

---

## Quickstart

```python
from src.backtest_engine import BacktestConfig, DataPipeline, BacktestEngine

pipeline = DataPipeline("/path/to/your/data")

cfg = BacktestConfig(
    strategy="ema_cross",
    period_1=50,
    period_2=100,
    tp_pct=0.01,
    sl_pct=0.004,
    be_trigger_pct=0.003,
    time_window_1="08:00-12:00",
    time_window_2="13:30-18:30",
    fast=True,
)

engine = BacktestEngine.from_ticker(
    pipeline=pipeline,
    ticker="XAUUSD_M5",
    start="2021-01-01",
    end="2025-01-01",
    cfg=cfg,
)

trades = engine.run()
print(trades[["entry_time", "exit_time", "return", "reason"]])
```

---

## Data Format

CSV files should be placed in a local directory and named `{TICKER}.csv`.

Expected columns (no header):

```
Datetime, Open, High, Low, Close, Volume
```

Timestamps are shifted +1h internally to align with broker timezone.

---

## Adding a Strategy

A strategy is a static method in `Strategy_Signal` that returns a DataFrame
with a `Signal` column (`1` = long, `-1` = short, `0` = neutral).

```python
@staticmethod
def my_strategy(df, param_1, param_2):
    d = df.copy()
    # your signal logic
    d["Signal"] = ...
    return d

# Register it in apply()
@staticmethod
def apply(df, cfg):
    if cfg.strategy == "my_strategy":
        return Strategy_Signal.my_strategy(df, cfg.param_1, cfg.param_2)
```

Add any new parameters to `BacktestConfig`. The engine requires no changes.

---

## Version History

This project was built incrementally. Each version is kept in `history/` for reference.

| Version | Description |
|---------|-------------|
| V1      | Single-file proof of concept — one position, fixed TP/SL |
| V2      | Modular functions — entry/exit/trade analysis separated |
| V2.2    | Multi-position, ATR, breakeven, session filtering, EMA exits |
| V3      | Class architecture — `BacktestConfig`, `DataPipeline`, `BacktestEngine` |
| V3.3    | Signal/engine separation — plug-and-play strategy layer |

---

## Known Limitations

- No slippage or commission modeling
- No portfolio-level allocation
- No formal walk-forward validation or statistical significance testing yet
- Tested on metals futures M5 data — behavior on other asset classes not validated

---

## What's Next

The immediate next step is grid search and walk-forward validation to assess
whether the edge identified in-sample holds out-of-sample.

Longer term, I expect statistical analysis to show that optimal parameters
vary significantly across market regimes — trending vs ranging conditions
respond very differently to an EMA-based entry. The plan is to address this
with a regime detection layer using XGBoost, classifying market states and
adapting parameters dynamically rather than using a single static configuration.

---

## Contributing

This is a personal research project — not currently open for contributions.
If you have feedback or want to discuss the architecture, feel free to open an issue.
