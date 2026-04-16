# AI Assistant Context

Use this document as a high-level context prompt for an AI assistant working on this framework.

The goal is not to replace the source code, but to let the assistant understand the architecture, conventions, and preferred workflow quickly enough to guide a user who does not want to learn every internal convention first.

This document is intentionally practical:

- it explains how the framework is meant to be used
- it clarifies the most important conventions
- it highlights the current limitations
- it tells the AI what path to prefer when helping the user

This document is written for the low-level engine only.

Out of scope for this context:

- `StrategyEngine`
- `BacktestBundle`
- higher-level convenience orchestration layers

If the repository contains those objects, do not use them unless the user explicitly asks for them.

---

## 1. Project purpose

This project is a modular Python backtesting and research framework for bar-based systematic strategy research on OHLCV data.

Its purpose is not only to run signals, but to provide a reusable research environment where:

- signal generation
- setup routing
- regime conditioning
- trade management
- execution assumptions
- and post-trade analysis

remain separate.

The core architectural principle is:

- a strategy should express market logic
- the engine should handle execution logic

In other words, the user should not have to rebuild a backtest loop every time a new idea is tested.

---

## 2. How the AI should think about the framework

The AI should treat the framework as a layered research stack.

### Main layers

1. data layer
2. feature layer
3. raw signal layer
4. setup layer
5. regime layer
6. exit / execution binding layer
7. engine execution layer
8. context and post-analysis layer

### Important design rule

Do not mix those layers unless necessary.

For example:

- a raw RSI signal should not directly implement execution behaviour
- a regime should not be hardcoded inside a signal unless there is a very good reason
- a trade-management policy should not be embedded in the DataFrame returned by a raw signal strategy

The AI should preserve separation of concerns whenever possible.

---

## 3. Recommended workflow for helping a user

When the user wants to build a strategy, the preferred workflow is:

1. create a raw strategy returning a DataFrame with a `Signal` column
2. inspect and validate the signal first
3. only then bridge it into setup mode if needed
4. add regime conditioning if the strategy is context-dependent
5. attach exit profiles or exit strategies through bindings
6. run the engine with setup metadata preserved
7. enrich `trades_df` for post-trade analysis if needed

This is the main recommended path because it keeps complexity under control.

### Preferred order of complexity

#### First

- build raw entry logic
- verify that the signal is syntactically valid
- inspect the chart
- run a simple baseline backtest

#### Then

- convert the signal into a setup
- combine multiple setups
- use `RegimePolicy` to route them conditionally

#### Finally

- attach setup-specific exit profiles
- use exit strategies if needed
- enrich the trade history with context

The AI should not jump to the most complex abstraction first unless the user explicitly asks for it.

---

## 4. Main modules and what they are for

The low-level engine is mainly structured around these modules:

- `pipeline_config.py`
- `feature.py`
- `feature_compiler.py`
- `multi_setup_layer.py`
- `regime_policy.py`
- `Exit_system.py`
- `partial_config.py`
- `position_rules.py`
- `exit_strategy_system.py`
- `stateful_config.py`
- `execution_binding.py`
- `context_engine.py`
- `NJITEngine.py`

### Reading priority for the AI

If the AI must understand the framework quickly, it should read in this order:

1. `pipeline_config.py`
2. `feature.py`
3. `multi_setup_layer.py`
4. `regime_policy.py`
5. `Exit_system.py`
6. `exit_strategy_system.py`
7. `execution_binding.py`
8. `NJITEngine.py`
9. `context_engine.py`

That order reflects the practical user flow.

---

## 5. Data conventions

The framework supports two main data entry styles:

- `DataPipeline`
- `Data`

### `DataPipeline`

Use `DataPipeline` for:

- simple CSV loading
- engine-ready OHLCV preparation
- quick backtests

Expected CSV format:

- no header
- columns interpreted as:
  - `Datetime`
  - `Open`
  - `High`
  - `Low`
  - `Close`
  - `Volume`

Typical usage:

```python
pipeline = DataPipeline("/path/to/csv/folder")
```

The AI should use `DataPipeline` when the user wants:

- a simple entry point
- one instrument
- one timeframe
- no complex alignment or external context store

### `Data`

Use `Data` when the user needs:

- multiple OHLCV inputs
- multiple assets
- external time series
- resampling
- alignment
- OHLCV matrices
- feature-friendly research storage

`Data` is the advanced research container.

### Important `Data` capabilities

- register external series
- register OHLCV data
- store a `main_df`
- build OHLCV matrices
- align multiple series
- resample ext series
- resample OHLCV
- extract one asset out of a multi-asset matrix

### Important naming conventions in `Data`

- OHLCV names are canonically built like `ASSET_TF`
- external series names often include:
  - original column name
  - asset
  - timeframe

Examples:

- `XAUUSD_M5`
- `funding_BTC_H1`
- `gamma_XAUUSD_M5`

### AI guidance rule

If the user seems confused, the AI should default to:

- `DataPipeline` for simple single-asset examples
- `Data` only when multi-asset or feature/context work is actually needed

---

## 6. Main DataFrame conventions

Many parts of the engine expect a DataFrame with at least:

- `Open`
- `High`
- `Low`
- `Close`

If `ATR` is absent, the engine can often compute it automatically.

If a strategy uses a custom DataFrame via `main_df`, it must:

- have a `DatetimeIndex`
- contain OHLC columns

The AI should never assume arbitrary column names for price data.

---

## 7. Feature layer conventions

The feature system is designed to support both exploratory and compiled use cases.

### Core objects

- `Feature`
- `FeatureRuntime`
- `FeatureResult`
- `FeatureOutput`
- `CompiledFeatures`

### Recommended usage

The user registers features on a `Feature` object, then runs them on:

- a DataFrame
- a stored `main_df`
- an OHLCV matrix

### Important conventions inside feature functions

The feature callable receives a runtime object, usually named `f` or similar.

Inside the feature, the standard accessors are:

- `f.col("Close")`
- `f.asset_col("Close", "XAUUSD")`
- `f.ext("name")`
- `f.process("helper_name")`

### Key recommendation for the AI

If the user wants a simple strategy and does not explicitly need the full feature engine, the AI can compute indicators directly inside the raw strategy DataFrame first.

The AI should introduce `Feature` when:

- the logic should be reusable
- the features are shared across multiple strategies
- compiled features are needed for exit strategies
- multi-asset or matrix inputs are needed

### Saved feature runs

The feature engine can save outputs:

- by logical feature name
- or by explicit run name

This is useful for:

- comparing feature variants
- converting saved outputs into compiled feature matrices later

### `CompiledFeatures`

`CompiledFeatures` is the engine-facing numeric feature matrix.

The AI should use it when:

- exit strategies need feature columns
- feature names must be resolved into column indices
- the engine must consume features as aligned arrays

### AI guidance rule

Use the feature layer when it adds structure and reuse.

Do not force the user through the feature layer if a simple raw strategy is enough.

---

## 8. Raw signal conventions

The simplest strategy convention is:

- return a DataFrame with a `Signal` column

Signal values:

- `1` = long
- `-1` = short
- `0` = flat

This is the preferred first step for most new strategies.

### Why the AI should prefer raw signal mode first

Because it is:

- easier to debug
- easier to plot
- easier to validate
- easier for the user to understand

The AI should only move to setup mode when there is a clear reason.

---

## 9. Signal inspection conventions

`NJITEngine.inspect_signals(...)` is the preferred inspection method.

It supports:

- raw signal strategies using `signal_col="Signal"`
- setup-style strategies using `signal_col=["long_active", "short_active"]`

Important helper behaviour:

- if the strategy returns only setup columns, price columns are automatically restored for inspection
- if long and short activity columns are passed, the engine creates a synthetic `Signal`

The AI should use `inspect_signals(...)` whenever the user wants:

- chart validation
- quick signal debugging
- visual confirmation before backtesting

---

## 10. Setup conventions

The setup layer exists for:

- multiple entry logics
- setup routing
- regime-aware activation
- per-setup exit binding

### Core objects

- `SetupSpec`
- `DecisionConfig`

### Required setup DataFrame schema

A valid setup function must return a DataFrame containing:

- `long_score`
- `short_score`
- `long_active`
- `short_active`
- `setup_id`

Rules:

- `setup_id` must be constant over the whole DataFrame
- activity columns must only contain `0/1`
- score columns must be float-like

### `DecisionConfig`

Controls arbitration between setups.

Key fields:

- `min_score`
- `allow_long`
- `allow_short`
- `tie_policy`

### Important conceptual point

Multiple setups do not simply “sum”.

They are:

- validated
- converted into matrices
- arbitrated bar by bar

The result is one final signal stream plus metadata:

- `signals`
- `selected_setup_id`
- `selected_score`

### AI guidance rule

When the user wants multiple strategies in one run, the AI should:

1. express each one as a `SetupSpec`
2. use `prepare_signal_inputs(...)`
3. inspect the selected setup flow if needed

---

## 11. Bridging raw strategies into setup mode

This is one of the most important conventions.

The engine provides:

- `signal_df_to_setup_df(...)`
- `wrap_signal_strategy(...)`

### When to use them

If the user already has a raw strategy returning `Signal`, and now wants:

- setup routing
- regime-aware activation
- per-setup exit binding

the AI should not rewrite the strategy from scratch.

Instead, the AI should use:

```python
wrapped = NJITEngine.wrap_signal_strategy(my_signal_strategy)
```

This lets the user keep:

- a research-friendly signal function
- and a setup-compatible version of it

### Recommended pattern

- raw function for visual inspection
- wrapped function for `SetupSpec`

This is a major best practice in this framework.

---

## 12. Regime conventions

The regime layer controls:

- setup activation
- directional permissions
- optional exit overrides

### Core objects

- `RegimePolicy`
- `RegimeContext`

### `RegimePolicy`

Fields:

- `n_regimes`
- `score_multiplier`
- `exit_profile_override`
- `exit_strategy_override`

### `score_multiplier`

This is the most important regime mechanism.

It is keyed by:

- regime id
- setup name
- direction (`long` / `short`)

This allows the regime to:

- disable one setup
- allow only long or only short
- prioritize one setup over another

### Important current limitation

`exit_profile_override` and `exit_strategy_override` are:

- global by regime
- not setup-specific

This means:

- if regime 0 overrides the profile to `1`, that override applies to all setups in that regime

The AI must keep this limitation in mind.

If the user wants:

- setup-specific regime exit overrides

the AI should explain that this is a current architectural limitation and suggest a workaround if needed.

### `lag_regime_array(...)`

Use this when regime information must be lagged to avoid look-ahead bias.

The AI should suggest lagging whenever the regime is derived from market information close to the execution bar.

---

## 13. Exit profile conventions

The exit-profile layer is the main declarative trade-management layer.

### Core object

- `ExitProfileSpec`

### What it controls

- TP / SL
- ATR TP / SL
- break-even
- trailing runner
- max holding bars
- partial exits
- pyramiding
- averaging
- phases
- rule-driven trade management

### Important design rule

Any field left as `None` in an `ExitProfileSpec` falls back to `BacktestConfig`.

That means:

- the profile only overrides what it needs to override

### Advanced subconfigs

- `PartialConfig`
- `PyramidConfig`
- `AveragingConfig`
- `PhaseSpec`

### `DistributionFn`

This is the mathematical distribution object used by advanced trade-management configs.

It defines:

- fractions
- spacing
- or both

Supported modes:

- linear
- expo
- log
- sqrt
- equal
- custom_points
- callable

### Position-rule system

The profile layer also supports:

- `PositionRule`
- trigger classes (`OnRR`, `OnFeature`, `OnMAEPct`, etc.)
- action classes (`ExitPartial`, `MoveSLtoBE`, `AddPosition`, `SetPhase`, etc.)

This is the main advanced declarative control layer for trade management.

### AI guidance rule

The AI should prefer `ExitProfileSpec` when:

- the desired behaviour is declarative
- the user wants setup-specific management
- the user does not need full custom feature-driven runtime logic

Exit profiles are usually the first thing to use before exit strategies.

---

## 14. Exit strategy conventions

Exit strategies are more custom than exit profiles.

### Core object

- `ExitStrategySpec`

### Use cases

Use an exit strategy when the user needs:

- explicit runtime feature logic
- window-based exit logic
- custom Python or Numba behaviour
- strategy-level state

### Important arguments

- `strategy_id`
- `strategy_type`
- `backend`
- `feature_names`
- `params`
- `window_bars`
- `state_per_pos_custom`
- `state_global_custom`
- `stateful_config`

### `StatefulConfig`

This controls global stateful behaviour per exit strategy.

It can enforce:

- pause after consecutive stop losses
- max simultaneous positions
- invalidation on regime change
- cooldown after low rolling win rate

### AI guidance rule

The AI should not jump to exit strategies unless:

- declarative profiles are not enough
- or the user explicitly wants more custom runtime logic

For most users, `ExitProfileSpec` is the better first abstraction.

---

## 15. Execution binding conventions

The exit system is activated through compiled bindings.

### Core object

- `ExecutionContext`

### Main builder

- `build_execution_context(...)`

### Key input objects

- `exit_profile_specs`
- `setup_exit_binding`
- `strategy_profile_binding`
- `exit_strategy_specs`
- `compiled_features`

### `setup_exit_binding`

This maps each setup id to:

- one exit profile
- one exit strategy

Structure:

```python
{
    0: {"exit_profile_id": 0, "exit_strategy_id": -1},
    1: {"exit_profile_id": 1, "exit_strategy_id": -1},
}
```

This is the main per-setup exit binding mechanism.

### `strategy_profile_binding`

This controls which profiles an exit strategy is allowed to switch to.

This is more advanced and should only be used when exit strategies actually need dynamic profile switching.

### AI guidance rule

If the user wants setup-specific exits:

1. define profiles
2. define `setup_exit_binding`
3. build `ExecutionContext`
4. run with `use_exit_system=True`

That is the standard path.

---

## 16. `NJITEngine.run()` conventions

`run()` is the central execution method.

It resolves:

- config fallbacks
- setup metadata
- regime metadata
- execution context
- features
- limits and price arrays
- metrics and trade history

### Minimal simple-signal usage

```python
rets, metrics = engine.run(signals=signals, cfg=cfg)
```

### Setup-aware usage

When running setup mode properly, the AI should pass:

- `signals`
- `selected_setup_id`
- `selected_score`
- `execution_context`
- `use_exit_system=True`
- `multi_setup_mode=True`

### Regime-aware setup usage

To preserve regime metadata in `trades_df`, the AI should also pass:

- `regime=regime_arr`
- `regime_policy=regime_policy`
- `setup_specs=setup_specs`

This is important.

If regime routing is only used during signal preparation and not passed again to `run()`, trade-level regime tagging may be lost or incomplete.

### Important cost convention

`commission_per_lot_usd` is interpreted as:

- a round-trip cost

The engine converts it into relative return space using:

- `contract_size`
- entry price

The AI should not assume that `commission_cost` in `trades_df` is stored in raw USD.

### Important hold-analysis convention

`mae_hold` / `mfe_hold` are only meaningful if:

- `hold_minutes > 0`

If the user wants post-exit hold analysis, the AI should set:

- `hold_minutes`
- and make sure `bar_duration_min` matches the timeframe

### Limit-order support

The engine already has low-level support for:

- `entry_limit_prices`
- `limit_expiry_bars`
- `tp_prices`
- `sl_prices`

But this is not yet the most ergonomic high-level user API.

The AI should treat this layer as powerful but lower-level than the signal/setup abstractions.

---

## 17. Context enrichment conventions

The post-analysis layer is built around:

- `TradeContextEngine`
- `build_default_context_df(...)`
- `NJITEngine.enrich_trades_df_with_context(...)`

### Default context columns

The default context builder provides:

- `bar_return`
- `bar_range_pct`
- `bar_body_pct`
- `bar_direction`
- `minute_of_day`
- `day_of_week`

If the user wants additional contextual variables such as:

- `hour`
- `gap_pct`
- `vwap`
- `dist_to_vwap_pct`

the AI should build an `extra_context_df` with the same index as the engine bar index.

### Important index rule

`extra_context_df.index` must exactly match the engine price index.

### Recommended post-analysis workflow

1. run the backtest
2. create `extra_context_df` if needed
3. call `engine.enrich_trades_df_with_context(...)`
4. analyze by:
   - `setup_id`
   - `regime_id`
   - `reason`
   - custom entry/exit context

---

## 18. What the AI should recommend by default

If the user is lazy and wants the AI to “just build it properly”, the AI should default to these behaviours:

### For a new idea

- start with a raw pandas strategy returning `Signal`
- inspect it
- run a baseline backtest

### If the user wants regime filtering

- keep the raw signal strategy
- wrap it into setup mode with `wrap_signal_strategy(...)`
- use `RegimePolicy.score_multiplier`

### If the user wants multiple strategies in one run

- define each one as a setup
- use `prepare_signal_inputs(...)`
- bind exits per setup

### If the user wants setup-specific management

- prefer `ExitProfileSpec` first
- only use `ExitStrategySpec` when profiles are not enough

### If the user wants analysis

- start with `metrics["trades_df"]`
- then add `enrich_trades_df_with_context(...)`

### If the user wants advanced order logic

The AI should explain that:

- the engine already contains pieces of pending-order logic
- but the clean high-level order-management layer is still a future architecture extension

---

## 19. Current limitations the AI must know

These are important because they affect what the AI should promise.

### Regime overrides are not setup-specific yet

`exit_profile_override` and `exit_strategy_override` act globally by regime.

### Some runtime controls are still global

Settings such as:

- sessions
- forced-flat
- some entry-cap rules
- some execution assumptions

are still mostly global at `cfg` / `run()` level rather than fully setup-specific.

### Advanced averaging / partial / pyramiding should be presented carefully

The engine supports them, but:

- plotting does not fully visualize them
- cost/accounting for all advanced path-dependent cases should be treated carefully

The AI should avoid overselling those features unless the user explicitly wants to test them and understands the current maturity level.

### The engine is bar-based

It is not a tick engine, not an order book simulator, and not a production OMS.

The AI should frame it as:

- a research and validation engine

not as a live execution platform.

---

## 20. How the AI should answer the user

When helping the user, the AI should:

- stay practical
- prefer concrete working code
- avoid over-explaining internal implementation unless asked
- preserve the framework conventions
- avoid inventing higher-level shortcuts that do not exist

### Preferred answer style

- give the shortest working path first
- escalate to setup/regime/binding only when needed
- mention important hidden conventions explicitly

Examples of important hidden conventions:

- `entry_delay` already handles shifting, so the strategy should not manually shift signals for that purpose
- `wrap_signal_strategy(...)` is the preferred bridge from raw signal to setup
- `multi_setup_mode=True` matters when preserving setup metadata in `trades_df`
- `regime`, `regime_policy`, and `setup_specs` should be passed to `run()` if the user wants regime-tagged trades
- `extra_context_df` must share the exact engine index

### What the AI should avoid

- forcing the user into the most complex abstraction immediately
- mixing local notebook hacks with proper framework usage
- using `StrategyEngine` or `BacktestBundle` unless explicitly requested
- pretending that setup-specific regime exit overrides already exist

---

## 21. Minimal working templates the AI can safely start from

### Raw signal strategy

```python
def my_strategy(df):
    df = df.copy()
    df["Signal"] = 0
    return df
```

### Setup from wrapped signal

```python
wrapped_setup = NJITEngine.wrap_signal_strategy(my_strategy)

setup_specs = [
    SetupSpec(
        fn=wrapped_setup,
        params=dict(setup_id=0, score=1.0),
        name="my_setup",
    )
]
```

### Simple regime policy

```python
regime_policy = RegimePolicy(
    n_regimes=3,
    score_multiplier={
        0: {"my_setup": {"long": 1.0, "short": 1.0}},
        1: {"my_setup": {"long": 1.0, "short": 0.0}},
        2: {"my_setup": {"long": 0.0, "short": 1.0}},
    },
)
```

### Setup-specific exit profile binding

```python
profile = ExitProfileSpec(name="default_profile", tp_pct=0.01, sl_pct=0.005)

execution_context = build_execution_context(
    cfg=cfg,
    exit_profile_specs=[profile],
    setup_exit_binding={
        0: {"exit_profile_id": 0, "exit_strategy_id": -1},
    },
    strategy_profile_binding={},
    n_setups=len(setup_specs),
    exit_strategy_specs=[],
    n_strategies=0,
)
```

### Multi-setup run with metadata preserved

```python
prep = engine.prepare_signal_inputs(
    setup_specs=setup_specs,
    decision_cfg=DecisionConfig(min_score=0.0001, allow_long=True, allow_short=True, tie_policy=0),
    regime=regime_arr,
    regime_policy=regime_policy,
)

rets, metrics = engine.run(
    signals=prep.signals,
    selected_setup_id=prep.selected_setup_id,
    selected_score=prep.selected_score,
    execution_context=execution_context,
    use_exit_system=True,
    multi_setup_mode=True,
    cfg=cfg,
    regime=regime_arr,
    regime_policy=regime_policy,
    setup_specs=setup_specs,
)
```

---

## 22. Final assistant prompt summary

If needed, the following short prompt can be used as a compact handoff:

> This repository is a low-level modular backtesting framework for bar-based systematic research. Work only with the core engine and ignore StrategyEngine and BacktestBundle unless explicitly asked. Preserve the separation between data, features, raw signals, setups, regime conditioning, execution binding, engine execution, and post-trade analysis. Prefer building raw `Signal` strategies first, inspect them, then bridge them into setup mode with `wrap_signal_strategy(...)` only when setup routing, regime logic, or exit binding is needed. Use `RegimePolicy.score_multiplier` for setup- and direction-level activation. Use `ExitProfileSpec` first for setup-specific trade management, and only use `ExitStrategySpec` when custom runtime feature logic is necessary. When running setup mode, pass `selected_setup_id`, `selected_score`, `execution_context`, `use_exit_system=True`, and `multi_setup_mode=True`. If regime metadata must be preserved in `trades_df`, also pass `regime`, `regime_policy`, and `setup_specs` to `run()`. For post-analysis, use `enrich_trades_df_with_context(...)` and make sure any `extra_context_df` has the exact same index as the engine price data.`

