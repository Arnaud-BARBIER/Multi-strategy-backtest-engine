# Backtesting Engine
### A modular, class-based, Numba-optimized backtesting framework built for systematic research on OHLCV data

<img width="1306" height="732" alt="Screenshot 2026-03-10 at 16 46 45" src="https://github.com/user-attachments/assets/66066bd7-f043-4f7d-89fe-020a5d59f02e" />


---

Rebuilding a bar-by-bar simulation 
loop across notebooks was not a research workflow, but a 
bottleneck. Every new strategy meant rewriting execution logic, 
risk management, and session filters from scratch. This project 
emerged from the need to decouple signal research from trade simulation, 
so hypotheses can be tested rapidly without rebuilding the execution stack each time.

The engine is written in Python and designed around one idea: **signal generation and execution logic should be completely independent**. The operator bring a strategy that produces a `Signal` column in the same DataFrame that will be used by the engine to handle everything else: entries, exits, position sizing, breakeven, atr trailing stops, session filtering, and later trade analytics and hypothesis testing.

---

## Architecture

- Research layer   → signal_df / indicators / visual check
- Execution layer  → backtest_njit
- Analytics layer  → metrics + trades_df + df_after


## Engine Flow
```
DataPipeline          NJITEngine            signal_generation_inspection
fetchdata      ──►  init + JIT warmup ──►   strategy_fn / EMA fallback
                       OHLC / ATR arrays    signal_df + optional plot
                                                        │
                                                        ▼
                                                 signals [np.int8] 
                                                        │
                                                        ▼
                                     run() — param resolution / sessions / filters
                                                        │
                                                        ▼
                                     backtest_njit — entry · TP/SL · BE · trailing
                                                      EMA exit · signals · MAE/MFE
                                                        │
                                                        ▼
                                     compute_metrics_full — returns · Sharpe · VaR
                                                            DD · trades_df
                                                        │
                                              ┌─────────┴─────────┐
                                              ▼                   ▼
                                      trades_df             df_after
                                                       (EntryID / ExitID)
                                                        │
                                                        ▼
                                               Plotting (optional)
                                          _plot_signal_df / _plot_backtest
```
The engine never touches signal generation logic. A strategy only needs to return a DataFrame with a `Signal` column the rest is handled internally.

---

## Main Features

- Multi-position management with per-position state
- Fast Numba execution
- Breakeven logic with configurable trigger, offset, and delay
- ATR-based trailing stop with two-phase activation
- Fixed or ATR-based TP/SL
- EMA exit filters (price/EMA cross, EMA/EMA cross)
- Session-based time filtering (up to 3 windows, overnight-compatible)
- MaxEntries cap with sliding window, resets per day or per session
- Reverse mode
- Gap filter on signal generation
- MAE/MFE intra-trade tracking
- Hold-period observation (post-exit price behavior analysis)
- Trade log + annotated post-run dataframe
- Plotting 

---
# Get started in 5 minutes 

installation

```bash
pip install git+https://github.com/Arnaud-BARBIER/Multi-strategy-backtest-engine.git
```

---

## Core Architecture

### `DataPipeline`

Handles data loading from local CSV files.

**Responsibilities:**
- Load OHLCV data
- Apply timezone shift
- Return the DataFrame used by the engine


**Data Format**

CSV files should be placed in a local directory and named `{TICKER}.csv`.
Expected columns (no header):

```
Datetime, Open, High, Low, Close, Volume
```
<img width="479" height="76" alt="Screenshot 2026-02-27 at 19 35 20" src="https://github.com/user-attachments/assets/5ed1bdbc-77ab-407d-93e0-ee20cee58e47" />

Timestamps can be manually shifted in `cfg = BacktestConfig(timezone_shift=1)` to align with your broker timezone. 


---

### `BacktestConfig` (`cfg`)

Defines the default behavior of the engine.

**Typical parameters:**

| Category | Parameters |
|---|---|
| TP / SL | `tp_pct`, `sl_pct`, `use_atr_sl_tp` |
| Session filters | `time_window_1`, `time_window_2` |
| Candle filters | `Candle_Size_filter`, `min_size_pct`, `max_size_pct` |
| Break-even | `be_trigger_pct`, `be_offset_pct`, `be_delay_bars` |
| Trailing stop | `trailing_trigger_pct`, `runner_trailing_mult` |
| Metrics | `track_mae_mfe`, `compute_metrics` |
| Data | `timezone_shift`, `entry_delay` |

**Best practice:**

```python
# cfg → store default behaviour
# run() → override parameters temporarily

cfg = BacktestConfig(tp_pct=0.05, sl_pct=0.003)
rets, metrics = njit_engine.run(signals, tp_pct=0.02)  # temporary override
```

---

### `NJITEngine`

Builds the Numba-accelerated execution environment.

**Responsibilities:**
- Load data from `pipeline`
- Compute ATR
- Convert data to NumPy arrays
- Prepare and warm up the JIT-compiled backtest engine

---

## Three Ways to Generate Signals

### 1. Built-in strategy

Fastest method when you do not need signal inspection.

```python
signals = njit_engine.signals_ema(
    span1=30,
    span2=100,
    mode="close_vs_ema"
)
```

---

### 2. Custom strategy returning a DataFrame and/or plotting price chart with signals

Useful when you want to inspect indicators and signals before running the engine.
Your strategy must return a DataFrame containing a `"Signal"` column.

```python
signal_df = njit_engine.signal_generation_inspection(
    strategy_fn=my_strategy,
    signal_col="Signal",
    return_df_signals=True,
    my_param_1=10,
    my_param_2=20
)

print(signal_df[["Close", "RSI", "Signal"]].tail(20))

signals = signal_df["Signal"].to_numpy(dtype=np.int8)
```

---

### 3. Custom strategy returning signals directly

Useful when you do not need DataFrame inspection.

```python
signals = signals_rsi(
    njit_engine,
    rsi_period=14,
    oversold=30,
    overbought=70
)
```
---

## Signal Rules

The engine expects a NumPy array of signals thus your strategy must return it :

```python
dtype = np.int8
```

| Value | Meaning |
|---|---|
| `1` | Long signal |
| `-1` | Short signal |
| `0` | No action |

```python
signals = np.array([0, 1, 0, -1, 0], dtype=np.int8)
```

---

## Running the Backtest

```python
rets, metrics = njit_engine.run(signals)
```

**Outputs:**

| Object | Description |
|---|---|
| `rets` | NumPy array of trade returns |
| `metrics` | Dictionary containing statistics and result DataFrames |

---

## Post-Backtest Outputs

**Main objects inside `metrics`:**

```python
metrics["trades_df"]   # full trade history
metrics["df_after"]    # annotated price DataFrame
metrics["win_rate"]    # win rate
metrics["sharpe"]      # Sharpe ratio
```

### Trade history

```python
trades_df = metrics["trades_df"]
print(trades_df.head())
```

Contains:
- Entry / exit price and time
- Trade return
- Exit reason (`SL`, `TP`, `BE`, `RUNNER_SL`, `EMA1_TP`, ...)
- MAE / MFE intra-trade and hold-period statistics

### Annotated price DataFrame

To map executed trades back to the original bars:

```python
rets, metrics = njit_engine.run(
    signals,
    return_df_after=True
)

df_after = metrics["df_after"]

print(
    df_after[
        ["Close", "RSI", "Signal", "EntryTradeID", "ExitTradeID"]
    ].tail(50)
)
```

---

## Classic Use Case

```python
import numpy as np
from backtest_engine import BacktestConfig, DataPipeline, NJITEngine


def my_strategy(df, rsi_period=14, oversold=30, overbought=70):
    d = df.copy()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(rsi_period).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    d["RSI"] = rsi
    d["Signal"] = 0
    d.loc[rsi < oversold, "Signal"] = 1
    d.loc[rsi > overbought, "Signal"] = -1

    return d


pipeline = DataPipeline("Your/path/to/csv/files")

cfg = BacktestConfig(
    timezone_shift=1,
    crypto=False,
    tp_pct=0.002,
    sl_pct=0.01
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

signal_df = njit_engine.signal_generation_inspection(
    strategy_fn=my_strategy,
    signal_col="Signal",
    return_df_signals=True,
    rsi_period=14,
    oversold=30,
    overbought=70
)

signals = signal_df["Signal"].to_numpy(dtype=np.int8)

rets, metrics = njit_engine.run(
    signals,
    return_df_after=True
)

trades_df = metrics["trades_df"]
df_after = metrics["df_after"]

print(trades_df.head())
print(metrics["win_rate"], metrics["sharpe"])
```

### Mental Model

```
Load data → define cfg → generate signals → run engine → inspect results
```
---

## NJIT Engine — Parameters Reference

### Engine initialization

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipeline` | object | Data pipeline object. Must provide `fetchdata()` and `compute_atr()` |
| `ticker` | str | Asset file / symbol passed to the pipeline |
| `start` | str | Start date passed to `pipeline.fetchdata()` |
| `end` | str | End date passed to `pipeline.fetchdata()` |
| `cfg` | object | Config object used as default fallback for most `run()` parameters |
| `atr_period` | int | ATR period used at engine initialization |
| `MAX_TRADES` | int | Maximum number of trades preallocated in the Numba engine |
| `MAX_POS` | int | Maximum number of simultaneous open positions |

---

### Signal generation & pre-engine inspection

#### `signal_generation_inspection(...)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy_fn` | callable | User strategy function returning a DataFrame with a signal column |
| `signal_col` | str | Name of the signal column returned by the strategy. Default=`"Signal"` |
| `plot` | bool | If `True`, plots the generated signals on an interactive chart |
| `crypto` | bool | If `True`, disables weekend rangebreaks in plots |
| `return_df_signals` | bool | If `True`, returns the full DataFrame; otherwise returns the signal array only |
| `**kwargs` | any | Extra parameters passed directly to `strategy_fn` |

**Behavior**
- If `strategy_fn` is omitted, engine falls back to the built-in default EMA-vs-close signal generation.
- The generated DataFrame is cached internally as `last_signal_df` for post-backtest inspection.

---

### Built-in signal helpers

#### `signals_ema(...)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `span1` | int | First EMA period |
| `span2` | int | Second EMA period |
| `mode` | str | `"close_vs_ema"` or `"ema_cross"` |

#### `signals_from_strategy(...)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy_fn` | callable | User strategy returning a DataFrame |
| `signal_col` | str | Column to extract as engine input |
| `**kwargs` | any | Extra parameters passed to the strategy |

---

## Backtest execution

### `run(...)`

### Signal & Entry

| Parameter | Type | Description |
|-----------|------|-------------|
| `signals` | np.ndarray | Signal array sent to the Numba engine |
| `entry_delay` | int | Bars between signal and entry. Default fallback from `cfg.entry_delay` |
| `max_gap_signal` | float | Gap filter applied on the signal candle context |
| `max_gap_entry` | float | Gap filter applied between actual entry bar open and previous close |
| `candle_size_filter` | bool | Enables signal candle body-size filtering |
| `min_size_pct` | float | Minimum allowed body size of the signal candle |
| `max_size_pct` | float | Maximum allowed body size of the signal candle |
| `prev_candle_direction` | bool | If `True`, signal candle must be in signal direction |
| `multi_entry` | bool | If `True`, allows multiple concurrent positions |
| `reverse_mode` | bool | If `True`, opposite signal closes opposite open positions before new entry |

---

### Time windows / session filters

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_1` | tuple[str, str] or None | First active trading window, e.g. `("08:00","12:00")` |
| `session_2` | tuple[str, str] or None | Second active trading window |
| `session_3` | tuple[str, str] or None | Third active trading window |

If all sessions are `None`, entries are allowed at all times.

---

### Entry cap / max entries logic

| Parameter | Type | Description |
|-----------|------|-------------|
| `me_max` | int | Maximum allowed entries in the counting regime |
| `me_period` | int | Rolling lookback period used in sliding entry cap |
| `me_reset_mode` | int | Entry cap mode selector |

**`me_reset_mode` values**

| Value | Behavior |
|-------|----------|
| `0` | Disabled |
| `1` | Day reset |
| `2` | Session reset |
| `3` | Sliding window only |
| `4` | Session reset + sliding window |
| `5` | Day reset + sliding window |

---

### Exit — TP/SL

| Parameter | Type | Description |
|-----------|------|-------------|
| `tp_pct` | float | Fixed take profit as % of entry price |
| `sl_pct` | float | Fixed stop loss as % of entry price |
| `use_atr_sl_tp` | int | `0`=fixed, `1`=ATR TP + fixed SL, `-1`=ATR SL + fixed TP, `2`=ATR TP + ATR SL |
| `tp_atr_mult` | float | ATR multiplier used for take profit |
| `sl_atr_mult` | float | ATR multiplier used for stop loss |
| `allow_exit_on_entry_bar` | bool | If `False`, blocks exits on the entry bar |

---

### Exit — EMA mode

| Parameter | Type | Description |
|-----------|------|-------------|
| `exit_ema1` | np.ndarray | External EMA1 array used for EMA-based exit logic |
| `exit_ema2` | np.ndarray | External EMA2 array used for EMA-based exit logic |
| `use_ema1_tp` | bool | Exit on close vs EMA1 condition |
| `use_ema2_tp` | bool | Exit on close vs EMA2 condition |
| `use_ema_cross_tp` | bool | Exit on EMA1 / EMA2 cross condition |

**EMA exit logic**
- EMA exits are only allowed if the trade is already in profit.
- If any EMA exit mode is active, engine switches to EMA exit mode.
- Fixed SL / BE still have priority over EMA exits.

---

### Exit — External signal

| Parameter | Type | Description |
|-----------|------|-------------|
| `exit_signals` | np.ndarray | External exit signal array |
| `signal_tags` | np.ndarray | Optional tag array used for targeted exits |
| `use_exit_signal` | bool | Enables external exit signal logic |
| `exit_delay` | int | Bars between exit signal and execution |

**`exit_signals` values**

| Value | Behavior |
|-------|----------|
| `0` | No external exit — normal engine logic applies |
| `1` | LIFO exit — closes last opened position |
| `2` | Closes all long positions |
| `-2` | Closes all short positions |
| `3` | Closes all positions |
| `N` | Closes tagged position `N` if `signal_tags` is provided |

---

### Break-Even

| Parameter | Type | Description |
|-----------|------|-------------|
| `be_trigger_pct` | float | Profit threshold required to arm break-even |
| `be_offset_pct` | float | Break-even stop offset relative to entry |
| `be_delay_bars` | int | Minimum bars before BE can arm |

**Convention**
- BE is armed intrabar when threshold is reached.
- BE becomes active only on the next bar.

---

### Runner trailing

| Parameter | Type | Description |
|-----------|------|-------------|
| `trailing_trigger_pct` | float | Profit threshold required to arm the runner trailing logic |
| `runner_trailing_mult` | float | ATR multiplier used to compute runner stop |

> Below `trailing_trigger_pct`, only fixed SL / BE can exit the trade.

**Convention**
- Runner is armed on trigger bar.
- Runner becomes active on the next bar only.
- Once active, trailing stop is updated from `Close - side * ATR * mult`.
- If runner stop is still below activation threshold for long trades, or above for short trades, fixed SL / BE remains the only valid exit.

---

### Metrics & post-trade analytics

| Parameter | Type | Description |
|-----------|------|-------------|
| `track_mae_mfe` | bool | If `True`, tracks intra-trade MAE/MFE |
| `hold_minutes` | int | Post-exit observation window in minutes for hold MAE/MFE |
| `bar_duration_min` | int | Duration of one bar in minutes |
| `commission_pct` | float | Commission per side as % |
| `spread_pct` | float | Spread cost as % |
| `slippage_pct` | float | Slippage per side as % |
| `alpha` | float | Tail quantile for VaR / CVaR |
| `period_freq` | str | Resampling frequency for period-based return analysis, e.g. `"ME"` |

---

### Inspection & plotting after engine execution

| Parameter | Type | Description |
|-----------|------|-------------|
| `return_df_after` | bool | If `True`, returns an annotated post-engine DataFrame in `metrics["df_after"]` |
| `plot` | bool | If `True`, plots entries / exits after backtest |
| `crypto` | bool | If `True`, disables weekend rangebreaks in plots |
| `full_df_after` | bool | If `True`, returns the full post-engine DataFrame |
| `window_before` | int | Number of bars before first trade kept in `df_after` if `full_df_after=False` |
| `window_after` | int | Number of bars after last trade kept in `df_after` if `full_df_after=False` |

---

## Post-engine inspection output

When `return_df_after=True`, engine adds:

### `metrics["df_after"]`
Annotated DataFrame containing:
- original OHLC
- original signal columns if previously cached through `signal_generation_inspection()`
- `EntryTradeID`
- `ExitTradeID`

### `metrics["trades_df"]`
Trade DataFrame containing:
- `entry_time`
- `exit_time`
- `entry_idx`
- `exit_idx`
- `entry`
- `exit`
- `side`
- `return`
- `reason`
- `mae_intra`
- `mfe_intra`
- `capture_ratio_intra`
- `mae_hold`
- `mfe_hold`
- `capture_ratio_hold`
- `trade_id` if post-engine annotation is enabled

---

## Plot helpers

### `_plot_signal_df(...)`
Plots:
- candlesticks
- long signal markers
- short signal markers

### `_plot_backtest(...)`
Plots:
- candlesticks
- long / short entries
- exit markers
- line connecting each entry to its corresponding exit

---

## Metrics returned by `compute_metrics_full`

| Key | Description |
|-----|-------------|
| `n_trades` | Number of closed trades |
| `win_rate` | Share of positive-return trades |
| `total_return_sum` | Sum of net trade returns |
| `cum_return` | Compounded cumulative return |
| `ann_return` | Annualized return |
| `max_drawdown` | Maximum drawdown on compounded equity curve |
| `max_underwater_trades` | Maximum consecutive underwater trades |
| `calmar` | Annual return divided by absolute max drawdown |
| `sharpe` | Trade-based annualized Sharpe |
| `profit_factor` | Sum wins / absolute sum losses |
| `avg_win` | Mean winning trade return |
| `avg_loss` | Mean losing trade return |
| `VaR` | Historical VaR at `alpha` |
| `CVaR` | Historical CVaR at `alpha` |
| `t_stat` | One-sample t-statistic vs 0 |
| `p_value` | P-value of one-sample t-test |
| `p_binom` | P-value of binomial test on win rate |
| `period_freq` | Resampling frequency used |
| `n_periods` | Number of non-zero resampled periods |
| `n_periods_positive` | Number of positive periods |
| `n_periods_negative` | Number of negative periods |
| `pct_periods_positive` | Fraction of positive periods |
| `worst_period` | Worst resampled period return |
| `best_period` | Best resampled period return |
| `period_cvar` | CVaR on period returns |
| `avg_mae_intra` | Mean intra-trade MAE |
| `avg_mfe_intra` | Mean intra-trade MFE |
| `avg_capture_intra` | Mean intra-trade capture ratio |
| `avg_mae_hold` | Mean post-exit MAE |
| `avg_mfe_hold` | Mean post-exit MFE |
| `avg_capture_hold` | Mean post-exit capture ratio |
| `trades_df` | Full trade-level DataFrame |

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
| V4    | Pandas to Numba transition — inspection possibility | 


---

## Known Limitations

- No multiple setups, or exit strategy architecture
- No formal walk-forward validation or statistical significance testing yet
- No portfolio-level allocation

---

## What's Next

The immediate next step is to extend the engine toward a more complete set of primitive strategy structures.

This includes:

- **multi-signal interpretation**, so several entry signals can coexist and be processed differently;
- **entry setup aggregation**, to classify each trade by the technical context in which it was opened;
- **setup-to-exit linkage**, so a trade can inherit a specific exit logic from its entry profile;
- **stateful trade management**, allowing exit behavior to evolve during the life of the trade when new conditions appear;
- **context and regime tagging**, to make trades conditional on volatility regime, session, trend/range environment, or custom research labels;
- **time-based execution primitives**, such as cooldown, max holding period, next open / next close exits, and execution caps by period;
- **richer research instrumentation**, so each trade can later be studied not only by outcome, but also by setup quality, regime, and management path.

The broader objective is to progressively build a research framework where strategy design, execution, and statistical validation remain modular. In that architecture, the user focuses on expressing hypotheses and market logic at each level of the strategy, so the framework can test not just isolated signals, but the full analytical structure that links signals, context, and decision rules.

---

## Contributing

This is a personal research project, not currently open for contributions.
However if you have feedback or want to discuss the architecture, feel free to contact me.






