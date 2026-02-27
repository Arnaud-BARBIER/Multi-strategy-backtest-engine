# Backtesting Engine
### A modular, class-based backtesting framework built for systematic research on OHLCV data

<img width="1101" height="686" alt="Screenshot 2026-02-27 at 20 56 58" src="https://github.com/user-attachments/assets/801b26e6-cee9-4efc-8978-a1a0aac7c00c" />

---

I built this after realizing that rebuilding a bar-by-bar simulation 
loop across notebooks was not a research workflow, it was a 
bottleneck. Every new strategy meant rewriting execution logic, 
risk management, and session filters from scratch. This project 
is the result of decoupling signal generation from the execution 
engine, so strategy logic can be iterated independently from 
the simulation layer.

The engine is written in Python and designed around one idea: **signal generation and execution logic should be completely independent**. The operator bring a strategy that produces a `Signal` column and the engine handles everything else: entries, exits, position sizing, breakeven, atr trailing stops, session filtering, and later trade analytics and hypothesis testing.

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

The engine never touches signal generation logic. A strategy only needs to return a DataFrame with a `Signal` column the rest is handled internally.

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
pip install git+https://github.com/Arnaud-BARBIER/Multi-strategy-backtest-engine.git
from backtest_engine import BacktestConfig, DataPipeline, BacktestEngine #,Strategy_Signal if you want to use a build in strategy

# Plug in your own strategy and define as many parameters as needed.
# The engine only reads the 'Signal' column (1 / -1 / 0).
def My_strategy(df, rsi_period=14, oversold=20, overbought=80):
    d = df.copy()
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(rsi_period).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    d["Signal"] = 0
    d.loc[rsi < oversold,  "Signal"] = 1
    d.loc[rsi > overbought, "Signal"] = -1
    return d


pipeline = DataPipeline("Your/path/")
cfg = BacktestConfig(tp_pct=0.01, sl_pct=0.004)
engine = BacktestEngine.from_df(
    pipeline, "XAUUSD_M5", "2021-01-01", "2026-01-01", cfg,
    strategy_fn=My_strategy, #<- omit this if using a built-in strategy (e.g. strategy='ema_cross' in cfg)
    rsi_period=14,
    oversold=20,
    overbought=80,
)
trades = engine.run()
trades 
```

---

## Data Format

CSV files should be placed in a local directory and named `{TICKER}.csv`.

Expected columns (no header):

```
Datetime, Open, High, Low, Close, Volume
```
<img width="479" height="76" alt="Screenshot 2026-02-27 at 19 35 20" src="https://github.com/user-attachments/assets/5ed1bdbc-77ab-407d-93e0-ee20cee58e47" />

Timestamps can be manually shifted `cfg = BacktestConfig(timezone_shift=1)` to align with your broker timezone. 

---

## Adding a Strategy

##There are 2 ways to test your strategy:

1.
By creating a function which returns a DataFrame 
with a `Signal` column (`1` = long, `-1` = short, `0` = neutral) that you will assign to `df`.
You can add as many setting as you want. Below is a usecase example :

```python
def ma_strategie(df, rsi_period=14, oversold=20, overbought=80):
    d = df.copy()
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(rsi_period).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    d["Signal"] = 0
    d.loc[rsi < oversold,  "Signal"] = 1
    d.loc[rsi > overbought, "Signal"] = -1
    return d

pipeline = DataPipeline("/Users/arnaudbarbier/Desktop/Quant reaserch/Metals")
cfg = BacktestConfig(tp_pct=0.01, sl_pct=0.004, timezone_shift=1)
engine = BacktestEngine.from_df(
    pipeline, "XAUUSD_M5", "2021-01-01", "2026-01-01", cfg,
    strategy_fn=ma_strategie,
    rsi_period=14,
    oversold=20,
    overbought=80,
)
trades = engine.run()
```
2. 
Anther way could be hard coding it as a static method in `Strategy_Signal` that also returns a DataFrame
with a `Signal` column (`1` = long, `-1` = short, `0` = neutral). 

```python
@staticmethod
def my_strategy(df, param_1, param_2):
    d = df.copy()
    # your signal logic
    d["Signal"] = ...
    return d

# You also must Register it in Strategy_Signal
@staticmethod
def apply(df, cfg):
    if cfg.strategy == "my_strategy":
        return Strategy_Signal.my_strategy(df, cfg.param_1, cfg.param_2)

# Then plug it into BacktestConfig

    # --- Signal Generation and strategy Choice ---
    strategy: str = "ema_cross"
    strategy: str= "my_strategy" #<-- 

# In the end, when configurating your backtest settings
# you should be able to call it as a parameter. 

cfg = BacktestConfig( strategy='my_strategy',
```

The engine requires no changes.

---

## Version History

This project was built incrementally. Each version is kept in `history/` for reference. 
The current Version is the V3.3. 

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

---

## What's Next

The immediate next step is grid search and walk-forward validation to assess
whether the edge identified in-sample holds out-of-sample.

Longer term, I expect with the EMA price cross strategy, statistical analysis 
to show that optimal parameters vary significantly across market regimes : 
trending vs ranging conditions respond very differently to an EMA-based entry. 
The plan is to address this with a regime detection layer using XGBoost, classifying 
market states and adapting parameters dynamically rather than using a single 
static configuration.

---

## Contributing

This is a personal research project, not currently open for contributions.
However if you have feedback or want to discuss the architecture, feel free to contact me.
