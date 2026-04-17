# Architecture Tree Centered on `NJITEngine`

This document gives a more explicit map of the framework around its real center:

- `NJITEngine.py` is the orchestration hub
- `core_engine.py` is the execution trunk
- upstream modules feed a run
- downstream modules explain what happened after a run

This view intentionally ignores:

- `backtest_bundle.py`
- `StrategyEngine.py`

because they are not part of the current public core scope.

---

## High-level tree

```text
Data / external inputs
├── pipeline_config.py
│   ├── BacktestConfig
│   ├── DataPipeline
│   ├── Data
│   ├── ExtSeries
│   ├── OHLCVItem
│   ├── tf_ratio / agg_open / agg_high / agg_low / agg_close
│   └── Role:
│       load OHLCV, register extra series, align data, prepare engine inputs
│
├── feature.py
│   ├── Feature
│   ├── FeatureRuntime
│   ├── FeatureResult
│   └── Role:
│       define reusable feature logic
│
├── feature_compiler.py
│   ├── FeatureSpec
│   ├── CompiledFeatures
│   ├── compile_features
│   └── Role:
│       convert features into aligned runtime matrices
│
├── indicators.py
│   ├── ema_njit / atr_wilder_njit
│   ├── ema_feature / rsi_feature
│   └── Role:
│       provide primitive indicators and simple reusable signal ingredients
│
├── multi_setup_layer.py
│   ├── SetupSpec
│   ├── DecisionConfig
│   ├── aggregate_and_decide
│   └── Role:
│       turn multiple setup outputs into one final signal stream
│
├── regime_policy.py
│   ├── RegimePolicy
│   ├── RegimeContext
│   ├── build_regime_context
│   └── Role:
│       condition setup activation, directional permissions, and regime overrides
│
├── Exit_system.py
│   ├── ExitProfileSpec
│   ├── compile_exit_profiles
│   ├── compile_setup_exit_binding
│   ├── build_exit_profile_rt_matrix
│   └── Role:
│       define profile-based exit behavior and setup-to-profile mapping
│
├── partial_config.py
│   ├── DistributionFn
│   ├── PartialConfig
│   ├── PyramidConfig
│   ├── AveragingConfig
│   ├── PhaseSpec
│   └── Role:
│       configure advanced position management blocks used by exit profiles
│
├── position_rules.py
│   ├── trigger objects:
│   │   OnRR / OnMFEPct / OnMAEPct / OnATRMult / OnBars / OnFeature / ...
│   ├── action objects:
│   │   ExitPartial / MoveSLtoBE / MoveSLto / SetTP / AddPosition / SetPhase / Invalidate
│   └── Role:
│       define declarative trade-management rules attached to phases and profiles
│
├── exit_strategy_system.py
│   ├── ExitStrategySpec
│   ├── compile_exit_strategies
│   ├── build_exit_strategy_rt_matrix
│   └── Role:
│       define strategy-style exits beyond simple profiles
│
├── stateful_config.py
│   ├── StatefulConfig
│   └── Role:
│       configure strategy-level state limits and adaptive exit behavior
│
├── rule_compiler.py
│   ├── conditions:
│   │   FeatGtParam / FeatLtFeat / CrossOver / SideIs / Slope / Mean / ...
│   ├── actions:
│   │   FORCE_EXIT / SWITCH_PROFILE / OVERWRITE_TP_SL
│   └── Role:
│       compile higher-level exit rules into machine-usable structures
│
├── execution_binding.py
│   ├── ExecutionContext
│   ├── build_execution_context
│   └── Role:
│       gather compiled exit and setup routing into one runtime object
│
└── NJITEngine.py
    ├── Receives:
    │   config, data, features, setup outputs, regime context, execution context
    ├── Provides:
    │   inspect_signals
    │   signal_generation_inspection
    │   signals_ema
    │   prepare_signal_inputs
    │   run
    │   enrich_trades_df_with_context
    └── Role:
        main orchestration layer and practical entry point for users


                                  │
                                  │
                                  ▼

Trunk execution layer
└── core_engine.py
    ├── backtest_njit
    ├── compute_metrics_full
    ├── signal helpers:
    │   signals_ema_vs_close_njit / signals_ema_cross_njit
    ├── internal mechanics:
    │   TP/SL, BE, runner trailing, exit dispatch, trade recording, trigger evaluation
    └── Role:
        actual bar-by-bar engine and metrics computation core


                                  │
                                  │
                                  ▼

Downstream analysis and extension
├── context_engine.py
│   ├── TradeContextEngine
│   ├── build_default_context_df
│   ├── TradeAnalysisContext / TradeFeatureSpec / PathAggSpec
│   └── Role:
│       enrich trades with contextual data and path-based analysis
│
├── event_log.py
│   ├── build_event_log
│   └── Role:
│       turn trade and event outputs into a readable log
│
├── Njit_plots.py
│   ├── plot_edge / plot_equity / plot_returns_dist / plot_by_reason / plot_mae_mfe
│   ├── plot_period_returns / print_summary / plot_results
│   └── Role:
│       presentation layer for post-run outputs
│
├── regime_plot.py
│   ├── plot_price_with_regime
│   └── Role:
│       visualize the regime itself
│
├── exit_context.py
│   ├── PosCtx / BarCtx / FeatCtx / ParamsCtx
│   ├── no_action / switch_profile / overwrite_tp_sl / force_exit
│   └── Role:
│       shared context and action format for exit strategies
│
├── user_exit_strategies.py
│   ├── run_exit_strategy_instant_user
│   ├── run_exit_strategy_window_user
│   ├── run_exit_strategy_stateful_user
│   └── Role:
│       default user-editable entry points for custom exit strategies
│
├── strategy_module_loader.py
│   ├── use_user_exit_strategies
│   ├── reset_user_exit_strategies
│   └── Role:
│       load and refresh custom exit strategy modules
│
├── active_user_exit_strategies.py
│   └── Role:
│       currently active user strategy module mirror / binding target
│
├── strategy_recipes.py
│   ├── low_vol_bb_rsi_reversion_df
│   ├── low_vol_bb_rsi_reversion_setup
│   ├── make_low_vol_bb_reversion_profile
│   └── Role:
│       higher-level ready-made recipes showing how components fit together
│
└── adaptive_engine.py
    ├── AdaptiveEngine
    ├── AdaptiveResults
    └── Role:
        meta-research layer for adaptive or iterative testing around the core engine
```

---

## The real center: `NJITEngine.py`

If one file had to be considered the operational heart of the framework, it would be `NJITEngine.py`.

Why:

- it is the main user-facing API
- it receives almost every major input layer
- it prepares runtime arrays and aligned context
- it delegates actual execution to `core_engine.backtest_njit`
- it returns the metrics and hooks post-analysis back into the workflow

Conceptually, `NJITEngine` sits between:

- the research description upstream
- the simulation core downstream

That is why it makes sense to center the diagram on it rather than on a config file or utility module.

---

## Conceptual placement of each file

### Upstream: files that feed a run

These modules exist to create or compile what `NJITEngine.run(...)` needs.

- `pipeline_config.py`
- `feature.py`
- `feature_compiler.py`
- `indicators.py`
- `multi_setup_layer.py`
- `regime_policy.py`
- `Exit_system.py`
- `partial_config.py`
- `position_rules.py`
- `exit_strategy_system.py`
- `stateful_config.py`
- `rule_compiler.py`
- `execution_binding.py`

### Center: orchestration

- `NJITEngine.py`

### Trunk: actual execution

- `core_engine.py`

### Downstream: explain or extend what happened after execution

- `context_engine.py`
- `event_log.py`
- `Njit_plots.py`
- `regime_plot.py`
- `exit_context.py`
- `user_exit_strategies.py`
- `strategy_module_loader.py`
- `active_user_exit_strategies.py`
- `strategy_recipes.py`
- `adaptive_engine.py`

### Public surface

- `__init__.py`

Role:

- expose the public API
- decide what the user sees first
- strongly influence perceived complexity

---

## Does any file still miss a conceptual place?

Yes, a few files are easy to forget if the map only shows “data -> signal -> setup -> run -> metrics”.

The main conceptual places that must be added are:

### 1. Public API surface

Without a place for `__init__.py`, the map explains how the framework works internally, but not how a user actually encounters it.

So `__init__.py` deserves its own conceptual role:

- public export layer
- package entry surface

### 2. Signal primitives / low-level computation helpers

If you only show `Feature` and `SetupSpec`, then `indicators.py` and `rolling_engine.py` feel orphaned.

They belong to a conceptual bucket like:

- signal primitives
- reusable analytical kernels

`rolling_engine.py` in particular is not central to the main execution tree, but it still has a valid conceptual place as:

- rolling computation helper layer

### 3. Exit strategy runtime support

If you only map `ExitStrategySpec`, then these files can feel disconnected:

- `exit_context.py`
- `strategy_module_loader.py`
- `user_exit_strategies.py`
- `active_user_exit_strategies.py`

They belong together as:

- custom exit strategy runtime bridge

### 4. Recipe / demonstration layer

`strategy_recipes.py` is not part of the minimal engine path, but it is also not random.

Its conceptual place is:

- recipe layer
- reusable high-level examples built on the core components

### 5. Visualization layer

If you do not explicitly carve out a plotting layer, then:

- `Njit_plots.py`
- `regime_plot.py`

float around the map without a home.

They should be grouped as:

- visualization / presentation layer

### 6. Meta-research layer

`adaptive_engine.py` is also conceptually outside the core trunk.

Its place is:

- adaptive research layer
- meta-loop around the main engine

---

## Conclusion

If you ignore `backtest_bundle.py` and `StrategyEngine.py`, the architecture is still conceptually complete, but only if your map includes the following buckets:

- public API layer
- data layer
- feature layer
- signal layer
- setup layer
- regime layer
- execution binding layer
- orchestration layer (`NJITEngine`)
- execution trunk (`core_engine`)
- post-analysis layer
- visualization layer
- custom exit strategy runtime bridge
- recipe layer
- adaptive research layer

Without these extra buckets, some files will look “left over” even if they are not actually misplaced.
