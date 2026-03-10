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

```markdown
                                                ┌────────────────────┐
                                                │   1. DataPipeline  │
                                                │ fetchdata + ATR    │
                                                └─────────┬──────────┘
                                                          │
                                                          ▼
                                                ┌────────────────────┐
                                                │  2. NJITEngine     │
                                                │ init + arrays np   │
                                                │ OHLC / ATR / time  │
                                                │ JIT warmup         │
                                                └─────────┬──────────┘
                                                          │
                                                          │ pre-engine research
                                                          ▼
                                                ┌──────────────────────────────────────┐
                                                │ 3. signal_generation_inspection(...) │
                                                │ - strategy_fn user or fallback EMA   │
                                                │ - builds signal_df                   │
                                                │ - optional plot signals              │
                                                │ - returns df or signal array         │
                                                │ - caches last_signal_df              │
                                                └─────────┬────────────────────────────┘
                                                          │
                                                          ▼
                                                ┌────────────────────┐
                                                │ 4. signals array   │
                                                │ np.ndarray[int8]   │
                                                └─────────┬──────────┘
                                                          │
                                                          │ core execution
                                                          ▼
                                                ┌────────────────────┐
                                                │ 5. run(...)        │
                                                │ param resolution   │
                                                │ sessions / filters │
                                                │ exit arrays / tags │
                                                └─────────┬──────────┘
                                                          │
                                                          ▼
                                                ┌────────────────────┐
                                                │ 6. backtest_njit   │
                                                │ - entry logic      │
                                                │ - TP/SL / ATR      │
                                                │ - BE / trailing    │
                                                │ - EMA exit         │
                                                │ - exit signals     │
                                                │ - MAE / MFE        │
                                                └─────────┬──────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────────────┐
                                                │ 7. compute_metrics_full  │
                                                │ - returns / DD / Sharpe  │
                                                │ - VaR / CVaR / tests     │
                                                │ - trades_df              │
                                                └─────────┬────────────────┘
                                                          │
                                                          │ optional post-engine layer
                                                          ▼
                                                ┌──────────────────────────┐
                                                │ 8. _build_after_run_df   │
                                                │ - reuse last_signal_df   │
                                                │ - add EntryTradeID       │
                                                │ - add ExitTradeID        │
                                                │ - add trade_id           │
                                                └─────────┬────────────────┘
                                                          │
                                                          ├──────────────► metrics["trades_df"]
                                                          │
                                                          ├──────────────► metrics["df_after"]
                                                          │
                                                          ▼
                                                ┌──────────────────────────┐
                                                │ 9. Plotting              │
                                                │ - _plot_signal_df        │
                                                │ - _plot_backtest         │
                                                └──────────────────────────┘
```

- Research layer   → signal_df / indicators / visual check
- Execution layer  → backtest_njit
- Analytics layer  → metrics + trades_df + df_after

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
    timezone_shift=1 #<- Broker timezone alignment with UTC offset
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

## Running Your Strategy

There are two ways to plug your strategy into the engine.

---

### Option A — External function (recommended)

Write a function that takes a DataFrame and returns it with a `Signal` column
(`1` = long, `-1` = short, `0` = neutral). Pass it directly to `from_df()`.
```python
def my_strategy(df, rsi_period=14, oversold=20, overbought=80):
    d = df.copy()
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(rsi_period).mean()
    loss  = -delta.clip(upper=0).rolling(rsi_period).mean()
    rsi   = 100 - (100 / (1 + gain / loss))
    d["Signal"] = 0
    d.loc[rsi < oversold,   "Signal"] =  1
    d.loc[rsi > overbought, "Signal"] = -1
    return d

pipeline = DataPipeline("your/data/path")
cfg      = BacktestConfig(tp_pct=0.01, sl_pct=0.004)

engine = BacktestEngine.from_df(
    pipeline, "XAUUSD_M5", "2021-01-01", "2026-01-01", cfg,
    strategy_fn=my_strategy,
    rsi_period=14,
    oversold=20,
    overbought=80,
)
trades = engine.run()
```

Strategy parameters are optional, if omitted, defaults defined in your function will be used.

---

### Option B — Built-in strategy

Add a static method to `Strategy_Signal`, register it in `apply()` (located in the same class),
and declare any new parameters in `BacktestConfig`.
The engine requires no changes.
```python
# 1. Add your strategy to Strategy_Signal
@staticmethod
def my_strategy(df, param_1, param_2):
    d = df.copy()
    d["Signal"] = ...  # your logic here
    return d

# 2. Register it in apply()
@staticmethod
def apply(df, cfg):
    if cfg.strategy == "my_strategy":
        return Strategy_Signal.my_strategy(df, cfg.param_1, cfg.param_2)

# 3. Add parameters to BacktestConfig
param_1: float = ...
param_2: float = ...

# 4. Call it
cfg = BacktestConfig(strategy="my_strategy", param_1=..., param_2=...)
engine = BacktestEngine.from_ticker(pipeline, "XAUUSD_M5", "2021-01-01", "2026-01-01", cfg)
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






