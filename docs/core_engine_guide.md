# Core Engine Guide

This guide documents the low-level framework directly from the source code.

Scope:
- included: `BacktestConfig`, `DataPipeline`, `Data`, `Feature`, `FeatureRuntime`, `FeatureResult`, `SetupSpec`, `DecisionConfig`, `RegimePolicy`, `RegimeContext`, `ExitProfileSpec`, `ExitStrategySpec`, `ExecutionContext`, `TradeContextEngine`, `NJITEngine`, and their related helpers
- excluded on purpose: `StrategyEngine`, `BacktestBundle`, and higher-level convenience wrappers

The goal is to explain the framework as a coherent research workflow:

1. data ingestion and normalization
2. feature construction
3. signal generation
4. setup aggregation
5. regime conditioning
6. execution binding
7. backtest execution
8. post-trade contextual analysis

This is the right mental model for the framework: the engine is not only a backtester, it is a structured environment in which entry logic, execution logic, context, and trade analytics remain separable.

---

## 1. Global mental model

The framework is built around one design choice:

- signal generation should remain independent from execution

In practice, that means:

- a raw strategy can simply return a `Signal` column
- a setup-based strategy can return a setup DataFrame
- execution, exits, regime overrides, and analytics are applied later by the engine

At the low-level API, the main building blocks are:

- `BacktestConfig`: default runtime behaviour
- `DataPipeline`: simple CSV-driven data loader for the engine
- `Data`: advanced research container for external series, OHLCV stores, matrices, and aligned research inputs
- `Feature`: feature registry and execution layer
- `SetupSpec`: declarative setup definition
- `RegimePolicy`: setup- and direction-level regime conditioning
- `ExitProfileSpec`: declarative exit behaviour
- `ExitStrategySpec`: runtime exit logic driven by compiled feature matrices
- `ExecutionContext`: compiled runtime matrices linking setups, profiles, and strategies
- `NJITEngine`: the main bar-based engine
- `TradeContextEngine`: post-trade context enrichment layer

The low-level path is therefore:

`data -> features -> signals or setups -> regime -> execution context -> run() -> metrics/trades_df -> context enrichment`

---

## 2. Configuration Layer

### `BacktestConfig`

`BacktestConfig` is the global default configuration object used by `NJITEngine.run()`.

Important principle:
- `cfg` stores defaults
- `run()` can override most fields temporarily
- `ExitProfileSpec` can also override a subset of exit-related fields

This makes `cfg` the global fallback layer, not the only place where behaviour is defined.

### Main field groups

#### Multi-setup activation

- `multi_setup_mode`: if `True`, `run()` expects setup-aware inputs when provided and validates them accordingly

#### Data and preprocessing

- `timezone_shift`: shifts timestamps at load time
- `atr_period`: ATR period used when ATR is auto-computed

#### Default signal parameters

- `period_1`
- `period_2`

These exist mainly for built-in EMA helpers and for convenience defaults.

#### Entry and signal execution

- `entry_delay`
- `session_1`, `session_2`, `session_3`
- `max_gap_signal`
- `max_gap_entry`
- `candle_size_filter`
- `min_size_pct`
- `max_size_pct`
- `prev_candle_direction`
- `multi_entry`
- `reverse_mode`
- `sl_tp_be_priority`

Important fields:

- `entry_delay`: critical anti-look-ahead control; the config rejects values below `1`
- `session_*`: defines up to three trading windows; entries outside those windows are blocked
- `multi_entry`: allows multiple concurrent positions
- `reverse_mode`: allows immediate flip into the opposite direction when a new opposite signal arrives
- `sl_tp_be_priority`: changes the order in which standard exits versus advanced position-management rules are evaluated

#### Cooldown logic

- `cooldown_entries`
- `cooldown_bars`
- `cooldown_mode`

This is global entry throttling logic, separate from `me_max`.

#### Entry-cap logic

- `me_max`
- `me_period`
- `me_reset_mode`

These control how many entries can be taken inside a day, session, or rolling bar window depending on mode.

#### Entry price mode

- `entry_on_close`
- `entry_on_signal_close_price`

Only one can be active.

Use these carefully:
- `entry_on_close=False` means default next-bar open execution
- `entry_on_close=True` uses current bar close as entry
- `entry_on_signal_close_price=True` uses previous signal bar close

#### Exit: TP / SL

- `tp_pct`
- `sl_pct`
- `use_atr_sl_tp`
- `tp_atr_mult`
- `sl_atr_mult`
- `allow_exit_on_entry_bar`

`use_atr_sl_tp` modes:
- `0`: fixed TP and fixed SL
- `1`: ATR TP + fixed SL
- `-1`: fixed TP + ATR SL
- `2`: ATR TP + ATR SL

#### Exit: EMA mode

- `use_ema1_tp`
- `use_ema2_tp`
- `use_ema_cross_tp`

These are mutually compatible EMA-based exit triggers that only act on trades already in profit.

#### Exit: external signal

- `use_exit_signal`
- `exit_delay`

These enable an external exit stream independent from the entry stream.

#### Break-even

- `be_trigger_pct`
- `be_offset_pct`
- `be_delay_bars`

Important convention:
- BE is armed when the trigger is reached intrabar
- BE becomes active from the next bar

#### Runner trailing

- `trailing_trigger_pct`
- `runner_trailing_mult`

This is the trailing runner logic activated after a profit threshold.

#### Metrics and execution costs

- `track_mae_mfe`
- `hold_minutes`
- `bar_duration_min`
- `commission_pct`
- `commission_per_lot_usd`
- `contract_size`
- `spread_pct`
- `spread_abs`
- `slippage_pct`
- `alpha`
- `period_freq`

Important fields:

- `hold_minutes`: activates post-exit hold analysis (`mae_hold`, `mfe_hold`, `capture_ratio_hold`)
- `commission_per_lot_usd`: interpreted by the engine as a round-trip cost, then converted into return space through `contract_size * entry_price`
- `alpha`: VaR / CVaR tail quantile
- `period_freq`: resampling frequency for period-based return statistics, e.g. `"ME"`

#### Inspection and plotting

- `return_df_after`
- `plot`
- `plot_results`
- `crypto`
- `full_df_after`
- `window_before`
- `window_after`

#### Trade lifetime and quotas

- `max_holding_bars`
- `forced_flat_frequency`
- `forced_flat_time`
- `max_tp`
- `tp_period_mode`
- `tp_period_bars`
- `max_sl`
- `sl_period_mode`
- `sl_period_bars`

Important fields:

- `forced_flat_frequency` and `forced_flat_time` must be used together
- forced-flat is global at run level
- TP / SL quotas can be tracked by day, session, or rolling window

### Why `BacktestConfig` matters

`BacktestConfig` is more than a parameter bag:

- it enforces anti-look-ahead defaults
- it defines the global execution assumptions
- it is also the fallback source when exit profiles leave fields unset

This means the same strategy logic can be tested under different execution environments without rewriting the strategy itself.

---

## 3. Data Layer

There are two data entry styles in the framework.

### 3.1 `DataPipeline`: simple engine-facing loader

Use `DataPipeline` when you mainly want:

- local OHLCV CSV loading
- ATR computation
- a simple way to feed `NJITEngine`

#### Main methods

##### `fetchdata(ticker, start, end, timezone_shift=0)`

Reads a CSV named `{ticker}.csv` with columns:

- `Datetime`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

Returns a time-indexed DataFrame sliced to `[start:end]`.

##### `compute_atr(df, period=14)`

Adds an `ATR` column using a rolling mean of true range.

##### `add_basic_features(df, prefix, add_returns=True, return_periods=(1,5,20), add_range=True)`

Creates a context DataFrame from OHLCV:

- prefixed OHLCV columns
- ATR if present
- return features
- high-low range percentage

Useful for cross-asset context construction.

##### `prepare_df(...)`

Standard engine-facing preparation:

- fetch data or reuse `Data.main_df`
- slice
- compute ATR

##### `prepare_surface_inputs(...)`

Builds:

- `price_df`: primary instrument OHLCV + ATR
- `context_df`: optional context from extra tickers

This is useful when a strategy trades one instrument but wants other assets as contextual drivers.

### 3.2 `Data`: advanced research container

Use `Data` when you want a richer research store than `DataPipeline`.

It manages:

- external series
- OHLCV objects
- OHLCV matrices
- one optional `main_df`
- alignment, cropping, resampling, and group loading

It is the right object for feature-heavy or multi-asset research.

### Core containers

#### `ExtSeries`

Represents one aligned or raw external series:

- `name`
- `values`
- `index`
- `source`
- `asset`
- `tf`
- `timezone_shift`
- `original_name`
- `meta`

Main helper:
- `.series()`

#### `OHLCVItem`

Represents one OHLCV object with arrays:

- `open`
- `high`
- `low`
- `close`
- `volume`
- `index`
- `asset`
- `tf`
- `source`

This is the object stored internally before conversion into DataFrames or matrices.

### Public import methods

#### `register_csv(...)`

Loads a CSV into the `Data` store.

Key arguments:
- `path`
- `col_map`
- `asset`
- `tf`
- `kind`: `"ext"`, `"ohlcv"`, or `"data"`
- `timezone_shift`

Meaning of `kind`:
- `"ext"`: external research series
- `"ohlcv"`: OHLCV object
- `"data"`: sets the main DataFrame

#### `register_df(...)`

Same idea as `register_csv`, but from an existing DataFrame instead of a file.

#### `load_ohlcv_csv(ticker, base_path, ...)`

Convenience loader for OHLCV CSVs following the naming convention:

- `ASSET_TF.csv`

Example:
- `XAUUSD_M5.csv`

This method normalizes and stores the series under the canonical name `ASSET_TF`.

### Main OHLCV matrix methods

#### `build_ohlcv_matrix(...)`

High-level helper to build a matrix of OHLCV columns from one timeframe.

Important arguments:
- `name`: store name
- `tf`
- `base_path`
- `assets`
- `autoload`
- `align_to`
- `dropna`
- `native_names_mode`
- `timezone_shift`
- `overwrite`

Key behaviours:

- if `assets` is given, it can autoload the matching CSVs
- if `assets` is omitted, it uses already-registered OHLCV series
- columns are either native (`Open`, `High`, ...) or suffixed (`Open_XAUUSD`, ...) depending on `native_names_mode`

`native_names_mode`:
- `"auto"`: native names if single asset, suffixed names if multi-asset
- `"always"`: always native names
- `"never"`: always suffix names

This method is central for multi-asset feature workflows.

#### `to_ohlcv_matrix(...)`

Lower-level matrix builder used internally and directly if you want finer control.

#### `get_ohlcv_matrix_asset(name, asset)`

Extracts one asset back out of an OHLCV matrix as a native OHLCV DataFrame.

#### `get_ohlcv_matrix_assets(name, assets)`

Extracts a submatrix for a selected subset of assets.

### External-series group loading

#### `build_ext_group(...)`

Loads a folder of external series CSV files, optionally resamples and aligns them, then returns one matrix.

This is useful for:

- macro inputs
- funding curves
- cross-asset context
- custom features stored outside OHLCV files

### Main DataFrame methods

#### `set_main_df(df, asset=None, tf=None, source="manual", timezone_shift=0)`

Stores a main engine DataFrame.

Requirements:
- must contain at least `Open`, `High`, `Low`, `Close`
- must have a `DatetimeIndex` or a detectable time column

If the input is a single-asset OHLCV matrix, the method can also extract native OHLC columns.

#### `has_main_df()`
#### `get_main_df()`

These expose the stored main DataFrame.

### External-series access and metadata

#### `ext(name)`

Returns an `ExtSeries`.

#### `list_ext()`
#### `list_aliases()`
#### `meta(name)`
#### `has_ext(name)`
#### `list_tf()`
#### `list_ext_by_tf(tf)`

Important naming rule:

- canonical external names are built from `original_column`, `asset`, and `tf`
- aliases are only created when the original short name is unique

Example:
- `funding_BTC_H1`
- `gamma_XAUUSD_M5`

### Resampling methods

#### `resample_ext(name, target_tf, method="ffill", ...)`

Resamples an external series.

Typical methods:
- `ffill`
- `exact`
- `last`
- `mean`
- `sum`
- `min`
- `max`

Important:
- this is for arbitrary external series
- not for OHLCV aggregation logic

#### `resample_ohlcv_arrays(...)`

Pure-array OHLCV aggregation helper.

#### `resample_ohlcv(name, target_tf, ...)`

OHLCV-aware resampling using proper OHLC aggregation semantics:

- open = first
- high = max
- low = min
- close = last
- volume = sum

### Alignment and matrix construction for ext series

#### `crop(obj, start=None, end=None)`

Crops one named series, `ExtSeries`, or time-indexed DataFrame.

#### `new_index(freq=None, from_obj=None, start=None, end=None)`

Builds a target `DatetimeIndex` either:

- from a frequency and time bounds
- or from an existing object

#### `align_all(...)`

Aligns all candidate ext series to a common target index.

#### `align_ext(name, align_to, ...)`

Aligns one ext series to a target.

#### `to_matrix(...)`

Converts aligned external series into one DataFrame matrix.

### Why the `Data` object matters

`Data` is the real research container of the framework.

It lets you move beyond a single OHLCV input and manage:

- multiple assets
- external context
- aligned research matrices
- resampled inputs at multiple timeframes

This is what enables feature-rich and regime-aware workflows without hardcoding everything inside one notebook.

---

## 4. Feature Layer

The `Feature` abstraction is the framework’s user-facing indicator and feature engine.

It is designed to let you:

- register reusable calculations
- run them on a DataFrame, an OHLCV matrix, or the stored `main_df`
- save named results
- convert results into compiled matrices usable by exit strategies

### Main objects

#### `FeatureOutput`

Represents one 1D output:

- `name`
- `values`
- `index`

Helpers:
- `.array()`
- `.series()`

#### `FeatureResult`

Represents one executed feature run.

Contains:

- `feature_name`
- `run_name`
- `outputs`
- `params`
- `source_name`
- `asset`
- `tf`
- `meta`

Helpers:
- `.output(output_name="main")`
- `.array(...)`
- `.series(...)`
- `.dataframe(...)`
- `.compiled(...)`

Important:
- a feature can return multiple outputs through a dict
- each output is wrapped as a `FeatureOutput`

#### `FeatureRuntime`

This is the object passed into feature functions.

It gives the feature access to:

- the source data
- previously saved feature results
- external series in the attached `Data` object
- convenience column accessors

Key methods:

- `process(name, **kwargs)`: run a registered process
- `array(name)`, `series(name)`, `result(name)`: access saved feature runs
- `index(name=None)`: retrieve source or saved-result index
- `ext(name)`: access external series from `Data`
- `assets()`: list assets attached to the current matrix source
- `asset_col(field, asset)`: access `Open_XAUUSD`-style columns from a matrix
- `col(name)`: access a native or named column from the current source

### `Feature`

The main feature registry and execution engine.

#### Registering logic

- `add_process(fn, name=None)`
- `add(fn, name=None)`

Conceptually:
- `processes` are helper computations
- `features` are the main reusable outputs you want to execute and save

#### Listing and cleanup

- `list_processes()`
- `list_features()`
- `list_results()`
- `has_process(name)`
- `has_feature(name)`
- `delete_process(name)`
- `delete_feature(name)`
- `delete_result(name)`
- `clear_results()`

#### Executing a feature

Main entry point:

`feature(name, on=None, on_ohlcv_matrix=None, matrix_asset=None, matrix_assets=None, save=False, export=False, save_as=None, **kwargs)`

Key input modes:

- `on=df`: run on an explicit DataFrame or Series
- `on_ohlcv_matrix="metals_m5"`: run on a named OHLCV matrix stored in `Data`
- `matrix_asset="XAUUSD"`: extract one asset from that matrix
- `matrix_assets=[...]`: extract a subset of assets
- if nothing is given and `data.has_main_df()`, it runs on `main_df`

Important rule:
- only one source mode should be specified at once

#### Result persistence

- `save=True` or `export=True` stores the run
- `save_as="custom_name"` stores it under an explicit run name

Two internal stores exist:

- `_last_runs`: last run by logical feature name
- `_named_runs`: explicit or auto-generated saved runs

#### Retrieving saved results

- `result(name)`
- `array(name)`
- `series(name)`
- `index(name)`

Result references may use:
- `run_name`
- or `run_name:output_name` for multi-output features

#### Converting to compiled features

- `to_compiled(...)`

This is the bridge from research-layer features to engine-layer matrices.

### `feature_compiler.py`

This module is simpler than `Feature` but equally important for exit strategies.

Main objects:

- `FeatureSpec`: declarative feature definition from OHLC arrays
- `CompiledFeatures`: 2D feature matrix + column mapping

Main functions:

- `compile_features(...)`
- `to_compiled_features(...)`

Use `CompiledFeatures` when you need:

- a stable `col_map`
- aligned numeric feature arrays
- a feature matrix consumable by exit strategies

### Why the feature layer matters

The feature layer is the bridge between:

- exploratory research in pandas
- and execution-time feature references used by exit logic

It is what lets the framework stay modular instead of mixing indicator code directly into the execution loop.

---

## 5. Signal and Setup Layer

The framework supports two entry paradigms:

- raw uni-signal mode
- setup mode

### 5.1 Uni-signal mode

The simplest convention is:

- a strategy returns a DataFrame with a `Signal` column

Signal values:
- `1`: long
- `-1`: short
- `0`: flat

This mode is ideal for:

- prototyping
- chart inspection
- validating whether raw signal logic has edge

### 5.2 Setup mode

Setup mode makes entry logic setup-aware and compatible with:

- regime routing
- exit binding
- multi-setup arbitration

#### `SetupSpec`

A `SetupSpec` contains:

- `fn`: callable returning a setup DataFrame
- `params`
- `name`

#### Required setup DataFrame schema

Every setup function must return:

- `long_score`
- `short_score`
- `long_active`
- `short_active`
- `setup_id`

Validation rules:

- `setup_id` must be constant over the whole DataFrame
- scores must be float-like
- activity columns must be integer-like and restricted to `0/1`
- all setups in one aggregation must share the same index

### `DecisionConfig`

Controls how multiple setups are arbitrated.

Fields:

- `min_score`
- `allow_long`
- `allow_short`
- `tie_policy`

`tie_policy`:
- `0`: no trade on equal best scores
- `1`: prefer long
- `2`: prefer short

### Aggregation flow

Main functions:

- `_build_setup_matrices(...)`
- `aggregate_and_decide(...)`
- `aggregate_and_decide_njit(...)`

Logic:

1. all setup DataFrames are validated
2. long and short score matrices are built
3. activity masks are applied
4. regime multipliers are applied if a regime context exists
5. the best long and best short candidate are compared bar by bar
6. one final `Signal`, one `selected_setup_id`, and one `selected_score` are produced

This is the key point:

- setups do not simply add trades together
- they are routed through one final arbitration stream

### Bridging raw strategies into setup mode

`NJITEngine` provides two helpers:

#### `signal_df_to_setup_df(...)`

Converts a raw signal DataFrame into setup format.

Useful when:
- your strategy already returns `Signal`
- you want to reuse it in multi-setup workflows

#### `wrap_signal_strategy(strategy_fn, signal_col="Signal", score_from_signal=False)`

Wraps a raw uni-signal function into a setup-compatible callable.

This is the standard bridge from:

- research-friendly signal DataFrame
- to setup-friendly multi-setup integration

### Important setup-layer implication

A strong workflow in this framework is:

1. build the signal in uni-signal mode
2. inspect and validate it visually
3. wrap it into setup mode only when needed

That keeps strategy logic simple while still allowing multi-setup routing later.

---

## 6. Regime Layer

The regime layer modifies entry permissions and exit overrides based on a market state array.

### `RegimePolicy`

Fields:

- `n_regimes`
- `score_multiplier`
- `exit_profile_override`
- `exit_strategy_override`

#### `score_multiplier`

Structure:

`{regime_id: {setup_name: {"long": x, "short": y}}}`

Use cases:

- disable a setup in one regime
- allow only long or only short
- downweight or upweight one setup

Example:

- regime 0: RSI only
- regime 1: EMA long only
- regime 2: EMA short only

#### `exit_profile_override`

Structure:

`{regime_id: profile_id}`

Important limitation:
- this override is global by regime
- not setup-specific

If profile `1` is forced in regime `0`, that affects all setups currently routed through the exit system.

#### `exit_strategy_override`

Same principle, but for exit strategies.

### `RegimeContext`

Compiled runtime object containing:

- `regime`
- `score_multipliers`
- `exit_profile_override`
- `exit_strategy_override`
- `n_regimes`
- `n_setups`

### `build_regime_context(regime, policy, setup_specs)`

Compiles:

- setup-name-to-index resolution
- score multiplier matrix
- exit-override arrays
- validation of the regime array

### `lag_regime_array(...)`

Important helper to prevent look-ahead bias when regime values are derived from the same bar information as execution.

Use it whenever regime classification itself is computed from market data observed on or near the execution bar.

### Why the regime layer matters

The regime layer is not only directional filtering.

It allows the framework to answer:

- which setup should be active now
- which side is allowed
- whether exit behaviour should change in this market state

This is what makes the framework genuinely regime-aware rather than only signal-aware.

---

## 7. Exit Profile Layer

The exit-profile layer is the main declarative trade-management system.

### `ExitProfileSpec`

Fields:

#### Core TP / SL

- `tp_pct`
- `sl_pct`
- `use_atr_sl_tp`
- `tp_atr_mult`
- `sl_atr_mult`

#### Break-even

- `be_trigger_pct`
- `be_offset_pct`
- `be_delay_bars`

#### Runner trailing

- `trailing_trigger_pct`
- `runner_trailing_mult`

#### Trade lifetime

- `max_holding_bars`

#### Advanced management

- `partial_config`
- `pyramid_config`
- `averaging_config`
- `phases`

### Important fallback rule

When a field is `None` in `ExitProfileSpec`, `compile_exit_profiles(...)` falls back to `BacktestConfig`.

That is one of the most important design choices in the framework:

- `cfg` defines global defaults
- profiles define setup-specific exit variations only where needed

### `CompiledExitProfile`

This is the flattened runtime representation consumed by the Numba engine.

### Runtime matrices built from profiles

The exit-profile system compiles several runtime matrices:

- `profile_rt_matrix`
- `partial_rt_matrix`
- `pyramid_rt_matrix`
- `averaging_rt_matrix`
- `phase_rt_matrix`
- `rule_trigger_matrix`
- `rule_action_matrix`

These are the engine-facing numerical representations of high-level configs.

### `DistributionFn`

Shared mathematical description for:

- sizing schedules
- spacing schedules
- or both

Supported modes:

- `linear`
- `expo`
- `log`
- `sqrt`
- `equal`
- `custom_points`
- `callable`

Supported distance references:

- `rr`
- `mfe_pct`
- `mae_pct`
- `atr`
- `index`

This object is used heavily by:

- partial exits
- pyramiding
- averaging

### `PartialConfig`

Defines automatic partial exits.

Fields:

- `n_levels`
- `spacing`
- `sizing`
- `move_sl_to_be_after_first`
- `ref`

Key idea:
- `spacing` defines when each partial triggers
- `sizing` defines how much is removed at each level
- `ref` tells whether fractions apply to the remaining size or the original size

### `PyramidConfig`

Defines automatic add-to-winner behaviour.

Fields:

- `n_levels`
- `trigger`
- `sizing`
- `move_sl_to_be`
- `sl_mode`
- `sl_feature`
- `sl_atr_mult`
- `group_sl_mode`
- `size_scales_with_mfe`

This is for reinforcing a trade in its own direction.

### `AveragingConfig`

Defines automatic add-on-adverse-move behaviour.

Fields:

- `n_levels`
- `trigger`
- `sizing`
- `sl_mode`
- `tp_mode`
- `max_avg_down_pct`
- `size_scales_with_mae`

This is for mean-reversion or recovery-style management, and should be used carefully.

### `PhaseSpec`

Defines a trade phase that can override:

- `tp_pct`
- `sl_pct`
- `be_trigger_pct`
- `trailing_trigger_pct`
- `max_holding_bars`
- `rules`

The engine tracks phase changes and can modify trade behaviour dynamically through `SetPhase`.

### `PositionRule`

One rule equals:

- one trigger
- one or more actions
- optional phase filtering
- optional max trigger count

Trigger classes:

- `OnRR`
- `OnMFEPct`
- `OnMAEPct`
- `OnATRMult`
- `OnBars`
- `OnBarsAfterLastTP`
- `OnFeature`
- `OnPhase`
- `OnAll`
- `OnAny`

Action classes:

- `ExitPartial`
- `MoveSLtoBE`
- `MoveSLto`
- `SetTP`
- `AddPosition`
- `SetPhase`
- `Invalidate`

This is the expressive logic layer inside `ExitProfileSpec`.

### Why exit profiles matter

Exit profiles are the main place where the framework becomes more than a simple signal backtester.

They allow:

- multiple trade-management styles to coexist
- per-setup trade management
- dynamic trade evolution without rewriting engine code

---

## 8. Exit Strategy Layer

Exit strategies are different from exit profiles.

### Difference in spirit

- `ExitProfileSpec` is declarative and parameter-driven
- `ExitStrategySpec` is strategy-like runtime logic driven by compiled features and state

Use profiles when the behaviour can be expressed through parameters and rules.

Use exit strategies when you need:

- explicit feature-driven runtime decisions
- strategy-level state
- more custom logic than profiles can express

### `ExitStrategySpec`

Fields:

- `strategy_id`
- `name`
- `strategy_type`
- `backend`
- `fn`
- `params`
- `feature_names`
- `window_bars`
- `state_per_pos_custom`
- `state_global_custom`
- `stateful_config`

### Strategy types

- `EXIT_STRAT_INSTANT`
- `EXIT_STRAT_WINDOWED`
- `EXIT_STRAT_STATEFUL`
- `EXIT_STRAT_INSTANT_PY`
- `EXIT_STRAT_WINDOW_PY`

### Backends

- `STRAT_BACKEND_NUMBA`
- `STRAT_BACKEND_PYTHON`

### Important arguments

#### `feature_names`

These must exist in the compiled feature matrix.

Exit strategies do not work directly from pandas feature names. They work from:

- `CompiledFeatures`
- and the column map inside it

#### `params`

Numeric parameters read through the runtime matrix.

#### `window_bars`

Required for windowed strategies.

#### `state_per_pos_custom`
#### `state_global_custom`

Reserve additional state slots beyond the engine defaults.

#### `stateful_config`

Attaches global stateful safety or throttling rules to the strategy.

### `StatefulConfig`

Fields:

- `max_consec_sl`
- `cooldown_bars_after_consec_sl`
- `max_simultaneous_positions`
- `invalidate_on_regime_change`
- `min_rolling_winrate`
- `cooldown_bars_if_low_winrate`

This layer acts at the strategy scope, not the individual trade scope.

### Strategy compilation

Main functions:

- `compile_exit_strategies(...)`
- `build_exit_strategy_rt_matrix(...)`
- `build_stateful_cfg_rt_matrix(...)`

Important dependency:
- if exit strategies are used, `build_execution_context(...)` requires `compiled_features`

### `rule_compiler.py`

This module is a higher-level code generator that can build user exit strategies from declarative rules and emit a Numba-compatible Python file.

It defines:

- condition objects
- action factories
- `ExitRuleSpec`
- `compile_exit_rules(...)`

Use it when you want:

- to generate explicit user exit strategy code
- to preserve performance
- to keep business logic declarative

It is part of the exit-strategy ecosystem, not of the simpler profile system.

---

## 9. Execution Binding Layer

The execution-binding layer turns all high-level declarations into runtime matrices.

### `ExecutionContext`

Contains the matrices and metadata consumed by `backtest_njit`.

Main contents:

- `profile_rt_matrix`
- `strategy_rt_matrix`
- `setup_to_exit_profile`
- `setup_to_exit_strategy`
- `strategy_to_default_profile`
- `strategy_allowed_profiles`
- `strategy_allowed_counts`
- advanced matrices for partial / pyramid / averaging / phases / rules
- `stateful_cfg_rt`
- feature name mapping
- state-size information
- boolean capability flags

### `build_execution_context(...)`

This is the central compiler for the execution layer.

Inputs:

- `cfg`
- `exit_profile_specs`
- `setup_exit_binding`
- `strategy_profile_binding`
- `n_setups`
- `exit_strategy_specs`
- `n_strategies`
- `compiled_features`
- `strategy_param_names`

### `setup_exit_binding`

Maps setup IDs to:

- default exit profile
- default exit strategy

Structure:

`{setup_id: {"exit_profile_id": N, "exit_strategy_id": M}}`

This is the main setup-to-exit bridge.

### `strategy_profile_binding`

Controls profile switching permissions for exit strategies.

Structure:

`{strategy_id: {"default_profile_id": N, "allowed_profile_ids": [...]}}`

This matters when an exit strategy wants to switch profiles at runtime.

### Why binding matters

Without `ExecutionContext`, the engine can still run in simple signal mode.

But once you want:

- setup-aware exits
- exit strategies
- regime-linked overrides
- advanced per-trade management

you need the compiled runtime matrices from `build_execution_context(...)`.

---

## 10. Engine Layer

`NJITEngine` is the main runtime object.

It holds:

- OHLC arrays
- ATR array
- bar index
- session / day arrays
- the last signal DataFrame for inspection
- the last feature context

### Initialization

Constructor:

`NJITEngine(pipeline=None, ticker=None, start=None, end=None, cfg=None, atr_period=None, atr_array=None, MAX_TRADES=50_000, MAX_POS=500, main_df=None)`

Two input modes exist:

- pipeline mode: `pipeline + ticker + start + end`
- direct DataFrame mode: `main_df`

Important:

- in `main_df` mode, ATR is reused if already present, or computed automatically otherwise
- in pipeline mode, `prepare_df()` is used

### Built-in signal helpers

#### `signals_ema(span1=None, span2=None, mode="close_vs_ema")`

Simple built-in entry helper.

Modes:
- `close_vs_ema`
- `ema_cross`

### Signal inspection helpers

#### `inspect_signals(...)`

Preferred inspection API.

Supports:

- `strategy_fn=...` with `signal_col="Signal"`
- `strategy_fn=...` with `signal_col=["long_active", "short_active"]`
- multi-setup inspection through `setup_specs`

Key behaviours:

- automatically restores OHLC columns if a strategy returned setup-only columns
- automatically creates a synthetic `Signal` when using separate long/short columns
- caches the inspection DataFrame in `last_signal_df`

#### `signal_generation_inspection(...)`

Legacy/simple wrapper around the same signal preparation idea.

If no strategy is provided, it falls back to the built-in EMA-vs-close signal generation.

### Signal conversion helpers

#### `signals_from_strategy(...)`

Runs a simple strategy function and extracts one signal column into a NumPy signal array.

#### `signal_df_to_setup_df(...)`
#### `wrap_signal_strategy(...)`

These are the main bridges from uni-signal to setup mode.

### Setup-aware preparation

#### `prepare_multi_setup_signals(...)`

Runs multiple setup functions, validates them, and aggregates them into a final decision stream.

#### `prepare_signal_inputs(...)`

Higher-level helper returning a `SignalPrep` object with:

- final `signals`
- `selected_setup_id`
- `selected_score`
- optional `df_signal`
- and resolved price arrays

This is the preferred preparation step before a multi-setup `run()`.

### `run(...)`

`run()` is the core execution entry point.

It is a large function because it resolves:

- `cfg` fallbacks
- execution context
- feature matrices
- setup mode
- regime mode
- limit order arrays
- slicing
- metrics

The important way to read it is by category.

#### Core inputs

- `signals`
- `cfg`
- `use_exit_system`
- `execution_context`
- `features`

#### TP / SL and classic exit behaviour

- `tp_pct`
- `sl_pct`
- `use_atr_sl_tp`
- `tp_atr_mult`
- `sl_atr_mult`
- `allow_exit_on_entry_bar`
- `be_trigger_pct`
- `be_offset_pct`
- `be_delay_bars`
- `trailing_trigger_pct`
- `runner_trailing_mult`
- `max_holding_bars`

These mirror `BacktestConfig` and can be overridden per run.

#### Entry and filtering

- `entry_delay`
- `session_1`, `session_2`, `session_3`
- `max_gap_signal`
- `max_gap_entry`
- `candle_size_filter`
- `min_size_pct`
- `max_size_pct`
- `prev_candle_direction`
- `multi_entry`
- `reverse_mode`
- `me_max`
- `me_period`
- `me_reset_mode`
- `cooldown_entries`
- `cooldown_bars`
- `cooldown_mode`
- `entry_on_close`
- `entry_on_signal_close_price`

#### External exit inputs

- `exit_ema1`
- `exit_ema2`
- `use_ema1_tp`
- `use_ema2_tp`
- `use_ema_cross_tp`
- `exit_signals`
- `signal_tags`
- `use_exit_signal`
- `exit_delay`

#### Cost and metrics

- `track_mae_mfe`
- `hold_minutes`
- `bar_duration_min`
- `commission_pct`
- `commission_per_lot_usd`
- `contract_size`
- `spread_pct`
- `spread_abs`
- `slippage_pct`
- `alpha`
- `period_freq`

#### Plotting and after-run inspection

- `return_df_after`
- `plot`
- `plot_results`
- `crypto`
- `start`
- `end`
- `backtest_start`
- `backtest_end`
- `full_df_after`
- `window_before`
- `window_after`
- `label`

#### Forced-flat and quotas

- `forced_flat_frequency`
- `forced_flat_time`
- `max_tp`
- `tp_period_mode`
- `tp_period_bars`
- `max_sl`
- `sl_period_mode`
- `sl_period_bars`

#### Multi-setup and regime inputs

- `selected_setup_id`
- `selected_score`
- `multi_setup_mode`
- `regime`
- `regime_policy`
- `regime_context`
- `setup_specs`

Important rules:

- if `multi_setup_mode=True`, `selected_setup_id` and `selected_score` must be provided
- if `use_exit_system=True`, all non-zero signals must have a valid `selected_setup_id`
- if `regime_policy` is provided to `run()`, `setup_specs` must also be provided

#### Limit-order and price-array inputs

- `entry_limit_prices`
- `limit_expiry_bars`
- `tp_prices`
- `sl_prices`
- `check_filters_on_fill`

These are the current low-level hooks for pending limit-entry logic and custom price-driven exits.

### Output structure

`run()` returns:

- `rets`
- `metrics`

Important contents of `metrics`:

- `trades_df`
- `df_after` if requested
- performance statistics
- period statistics
- grouped analysis fields
- phase events if relevant

### Important design implication of `run()`

`run()` is the place where everything meets:

- global defaults
- setup routing
- regime routing
- features
- exit matrices
- costs
- metrics

This is why the framework stays modular upstream: the complexity is intentionally centralized here instead of being repeated in user strategies.

---

## 11. Context and Post-Analysis Layer

The context layer exists to answer:

- where did edge come from
- in what context were trades opened
- how did price behave during the life of the trade

### `TradeContextEngine`

Core object for trade-level context enrichment.

Inputs:

- `trades_df`
- `price_df`
- optional `context_df`

Validation:
- `trades_df` must contain basic trade columns
- `price_df` must contain OHLC
- `context_df` must match `price_df.index` exactly

### Main methods

#### `attach_entry_context(cols=None, prefix="ctx_entry_")`

Joins selected context columns from the entry bar onto each trade.

#### `attach_exit_context(cols=None, prefix="ctx_exit_")`

Same for the exit bar.

#### `attach_path_aggregations(specs, include_exit_bar=True)`

Computes aggregated values over the context path during the trade.

This is how you get features like:

- mean bar range during the trade
- max volatility during the trade
- median contextual value over the trade path

#### `compute_trade_features(specs)`

Runs user-defined feature functions on `TradeAnalysisContext`.

This is the highest-level business-logic hook of the post-analysis layer.

#### `get_trade_context(trade_row_or_idx)`

Returns a `TradeAnalysisContext` object for one trade.

### `TradeAnalysisContext`

This object exposes semantic slices of trade history:

- `before_entry(n_bars)`
- `after_entry(n_bars)`
- `during_trade()`
- `after_exit(n_bars)`
- `entry_context()`
- `exit_context()`
- context windows around entry

### `PriceSegment`

Convenience object exposing:

- OHLC arrays
- bar count
- max/min excursions
- first-bar colour tests

### `build_default_context_df(price_df, extra_context=None)`

Builds a default context DataFrame with:

- `bar_return`
- `bar_range_pct`
- `bar_body_pct`
- `bar_direction`
- `minute_of_day`
- `day_of_week`

Then merges optional `extra_context`.

### `PathAggSpec`

Declarative spec for path aggregation:

- `col`
- `aggs`
- `prefix`

### `TradeFeatureSpec`

Declarative wrapper for one user-defined trade-level feature function.

Fields:

- `name`
- `fn`
- `expand_dict`

### `NJITEngine.enrich_trades_df_with_context(...)`

Convenience wrapper combining:

- `build_trade_context_engine(...)`
- entry context attachment
- exit context attachment
- path aggregation
- user-defined trade features

This is the easiest entry point for notebook-style post-analysis.

### Why the context layer matters

The context layer is what turns the framework from a pure backtester into a research platform.

It gives you a way to analyze not only:

- whether a strategy wins or loses

but also:

- under what setup
- under what regime
- under what local context
- with what trade path behaviour

---

## 12. Important design implications and current limitations

### Global vs local behaviour

Some behaviour is still global at `run()` / `cfg` level, including fields such as:

- sessions
- forced-flat logic
- cooldown and entry caps
- some execution assumptions

This means not every runtime behaviour is currently setup-specific.

### Regime exit overrides are global

`RegimePolicy.exit_profile_override` and `exit_strategy_override` are:

- global by regime
- not setup-specific

This is an important current limitation.

### Setup-specific exits already work

Even with the previous limitation, setup-specific exit behaviour already works through:

- `setup_exit_binding`
- `ExecutionContext`

This is why the framework can already handle heterogeneous management logic across setups.

### `use_exit_system=True` requires proper bindings

If the exit system is enabled:

- `execution_context` must be provided
- valid `selected_setup_id` must exist on non-zero signal bars

### Feature alignment matters

Any feature matrix used by exit strategies must be aligned to the engine bar index.

This is why:

- `CompiledFeatures`
- `align_index`
- and the `Data` alignment helpers

are structurally important.

### The framework is bar-based

The engine is not tick-level.

It is intended for:

- systematic bar-based research
- not order-book or microstructure modelling

### Strategy development recommendation

The cleanest workflow is usually:

1. build raw signals first
2. inspect them
3. convert them to setup mode only when needed
4. add regime routing
5. add exit binding
6. only then enrich and analyze `trades_df`

That is the natural order of complexity in this framework.

---

## 13. Minimal workflow summary

### Simple signal workflow

1. load data through `DataPipeline` or `main_df`
2. create a strategy returning `Signal`
3. inspect it with `inspect_signals(...)`
4. run it with `NJITEngine.run(signals=...)`

### Setup workflow

1. define one or more `SetupSpec`
2. use `prepare_signal_inputs(...)`
3. create exit profiles and optional exit strategies
4. compile `ExecutionContext`
5. run with `selected_setup_id`, `selected_score`, `execution_context`, and `multi_setup_mode=True`

### Regime-aware setup workflow

1. build the regime array
2. define `RegimePolicy`
3. pass regime inputs to `prepare_signal_inputs(...)`
4. pass regime inputs again to `run(...)` if you want regime metadata recorded in `trades_df`

### Post-analysis workflow

1. run the backtest
2. build `extra_context_df` if needed
3. call `enrich_trades_df_with_context(...)`
4. group by `setup_id`, `regime_id`, `reason`, or custom contextual fields

---

## Final takeaway

The low-level framework is best understood as a modular research stack.

It is not only:

- a signal runner

It is a system that separates:

- data management
- feature computation
- setup routing
- regime conditioning
- trade management
- execution assumptions
- post-trade analysis

That separation is the main value of the architecture: it lets you iterate on research logic without rewriting the execution infrastructure every time.
