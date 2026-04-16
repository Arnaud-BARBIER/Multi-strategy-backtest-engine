from .pipeline_config import (
    DataPipeline,
    Data,
    BacktestConfig,
    ExtSeries,
    OHLCVItem,
    tf_ratio,
    agg_open,
    agg_high,
    agg_low,
    agg_close,
)

from .feature import (
    Feature,
    FeatureRuntime,
    FeatureResult,
)


from .NJITEngine import NJITEngine

from .Njit_plots import (
    plot_results,
    plot_edge,
    plot_equity,
    plot_returns_dist,
    plot_by_reason,
    plot_mae_mfe,
    plot_period_returns,
    print_summary,
)

from .indicators import ema_njit, atr_wilder_njit

from .core_engine import (
    backtest_njit,
    compute_metrics_full,
    signals_ema_vs_close_njit,
    signals_ema_cross_njit,
)

from .strategy_module_loader import use_user_exit_strategies, reset_user_exit_strategies

from .multi_setup_layer import (
    DecisionConfig,
    SetupSpec,
    aggregate_and_decide,
)

from .feature_compiler import (
    FeatureSpec,
    CompiledFeatures,
    compile_features,
    to_compiled_features,
)

from .Exit_system import (
    ExitProfileSpec,
    CompiledExitProfile,
    compile_exit_profiles,
    compile_setup_exit_binding,
    build_exit_profile_rt_matrix,
    build_position_rule_matrices,
    PART_TRIGGER_TYPE, PART_FRACTION, PART_MOVE_BE,
    PART_DIST_MODE, PART_DIST_PARAM1, PART_DIST_PARAM2,
    N_PARTIAL_COLS,
    PYR_TRIGGER_TYPE, PYR_SIZE_FRACTION, PYR_SL_MODE,
    PYR_GROUP_SL_MODE, N_PYRAMID_COLS,
    AVG_TRIGGER_TYPE, AVG_SIZE_FRACTION, AVG_MAX_DOWN,
    N_AVG_COLS,
    PHASE_ID, PHASE_TP_PCT, PHASE_SL_PCT, N_PHASE_COLS,
    RT_RULE_PHASE_FILTER, RT_RULE_MAX_TIMES,
    RT_RULE_TRIGGER_TYPE, N_RULE_TRIGGER_COLS,
    RA_ACTION_TYPE, RA_PARAM1, N_RULE_ACTION_COLS,
    TRIGGER_TYPE_RR, TRIGGER_TYPE_MFE_PCT, TRIGGER_TYPE_MAE_PCT,
    TRIGGER_TYPE_ATR_MULT, TRIGGER_TYPE_BARS, TRIGGER_TYPE_BARS_ATP,
    TRIGGER_TYPE_FEATURE, TRIGGER_TYPE_PHASE,
    TRIGGER_TYPE_ALL, TRIGGER_TYPE_ANY,
    ACTION_TYPE_EXIT_PARTIAL, ACTION_TYPE_MOVE_SL_BE,
    ACTION_TYPE_MOVE_SL_FEAT, ACTION_TYPE_SET_TP,
    ACTION_TYPE_ADD_POSITION, ACTION_TYPE_SET_PHASE,
    ACTION_TYPE_INVALIDATE,
    DIST_MODE_LINEAR, DIST_MODE_EXPO, DIST_MODE_LOG,
    DIST_MODE_SQRT, DIST_MODE_EQUAL,
    DIST_MODE_CUSTOM_PTS, DIST_MODE_CALLABLE,
)

from .exit_strategy_system import (
    ExitStrategySpec,
    CompiledExitStrategy,
    compile_exit_strategies,
    build_exit_strategy_rt_matrix,
)

from .execution_binding import (
    ExecutionContext,
    build_execution_context,
)

from .backtest_bundle import (
    BacktestBundle,
    SignalPrep,
    prepare_backtest_bundle,
    bundle_signal_df,
    bundle_feature_df,
    print_bundle_summary,
)


from .indicators import (
    ema_njit,
    atr_wilder_njit,
    ema_feature,
    rsi_feature,
    consecutive_candle_signal_strict,
)

from .core_engine import (
    POS_SIDE,
    POS_ENTRY_PRICE,
    POS_TP,
    POS_SL,
    POS_ENTRY_IDX,
    POS_BE_ARMED,
    POS_BE_ACTIVE,
    POS_BE_ARM_IDX,
    POS_RUNNER_ARMED,
    POS_RUNNER_ACTIVE,
    POS_RUNNER_SL,
    POS_TAG,
    POS_PENDING_BE_SL,
    POS_RUNNER_THRESHOLD,
    POS_MAE,
    POS_MFE,
    POS_SETUP_ID,
    POS_SELECTED_SCORE,
    POS_N_COLS,
    REASON_SL,
    REASON_TP,
    REASON_BE,
    REASON_EMA1_TP,
    REASON_EMA2_TP,
    REASON_EMACROSS_TP,
    REASON_RUNNER_SL,
    REASON_EXIT_SIG,
    REASON_REVERSE,
    REASON_FORCED_FLAT,
    REASON_EXIT_STRAT_FORCE,
    REASON_MAX_HOLD,
    REASON_LABELS,
    TR_SIDE,
    TR_ENTRY_PRICE,
    TR_TP,
    TR_SL,
    TR_ENTRY_IDX,
    TR_MAE,
    TR_MFE,
    TR_SETUP_ID,
    TR_SELECTED_SCORE,
    TR_EXIT_PROFILE_ID,
    TR_EXIT_STRATEGY_ID,
    TR_BE_ACTIVE,
    TR_RUNNER_ACTIVE,
    TR_RUNNER_SL,
    TR_BARS_IN_TRADE,
    N_TRADE_CTX_COLS,
    PEND_SIDE, PEND_LIMIT_PRICE, PEND_EXPIRY_BAR,
    PEND_SIGNAL_BAR, PEND_SETUP_ID, PEND_SCORE,
    PEND_TP_PRICE, PEND_SL_PRICE,
    PEND_EXIT_PROF, PEND_EXIT_STRAT,
    N_PENDING_COLS,
)

from .exit_strategy_system import (
    STRAT_ID,
    STRAT_TYPE,
    STRAT_BACKEND,
    STRAT_WINDOW_BARS,
    STRAT_FEAT_COL_1,
    STRAT_FEAT_COL_2,
    STRAT_FEAT_COL_3,
    STRAT_FEAT_COL_4,
    STRAT_FEAT_COL_5,
    STRAT_FEAT_COL_6,
    STRAT_FEAT_COL_7,
    STRAT_FEAT_COL_8,
    STRAT_FEAT_COL_9,
    STRAT_FEAT_COL_10,
    STRAT_PARAM_1,
    STRAT_PARAM_2,
    STRAT_PARAM_3,
    STRAT_PARAM_4,
    STRAT_PARAM_5,
    STRAT_PARAM_6,
    N_EXIT_STRAT_RT_COLS,
    ACT_TYPE,
    ACT_TARGET_PROFILE_ID,
    ACT_NEW_TP_PRICE,
    ACT_NEW_SL_PRICE,
    ACT_FORCE_EXIT_FLAG,
    ACT_FORCE_EXIT_REASON,
    N_EXIT_ACT_COLS,
    STRAT_BACKEND_NUMBA,
    STRAT_BACKEND_PYTHON,
    EXIT_STRAT_NONE,
    EXIT_STRAT_INSTANT,
    EXIT_STRAT_WINDOWED,
    EXIT_STRAT_STATEFUL,
    EXIT_ACT_NONE,
    EXIT_ACT_SWITCH_PROFILE,
    EXIT_ACT_OVERWRITE_PRICE,
    EXIT_ACT_FORCE_EXIT,
    SP_PHASE, SP_N_TP_HIT, SP_REMAINING_SIZE,
    SP_LAST_HIGH, SP_BARS_SINCE_ENTRY, SP_BARS_SINCE_TP,
    SP_ADD_COUNT, SP_AVG_ENTRY, SP_ENTRY_VALID,
    SP_REGIME_AT_ENTRY, N_SP_DEFAULT, SP_CUSTOM_OFFSET,
    SG_CONSEC_SL, SG_CONSEC_TP, SG_ROLLING_WINRATE,
    SG_TOTAL_EXPOSURE, SG_LAST_TRADE_RETURN,
    SG_DAILY_TRADE_COUNT, SG_SESSION_TRADE_COUNT,
    SG_COOLDOWN_UNTIL, SG_CURRENT_REGIME,
    N_SG_DEFAULT, SG_CUSTOM_OFFSET,
    SCFG_MAX_CONSEC_SL,
    SCFG_COOLDOWN_BARS,
    SCFG_MAX_POSITIONS,
    SCFG_INVALIDATE_ON_REGIME,
    SCFG_MIN_WINRATE,
    SCFG_WINRATE_COOLDOWN,
    N_STATEFUL_CFG_COLS,
    build_stateful_cfg_rt_matrix,
    EXIT_ACT_PARTIAL_EXIT,
    EXIT_ACT_ADD_POSITION,

)

from .adaptive_engine import AdaptiveEngine, AdaptiveResults

from .rule_compiler import (
    compile_exit_rules,
    ExitRuleSpec,
    IF,
    FeatGtParam, FeatLtParam,
    FeatGtFeat, FeatLtFeat,
    FeatGtVal, FeatLtVal,
    CrossOver, CrossUnder,
    PosGtParam, PosLtParam,
    PosGtVal, PosLtVal,
    SideIs,
    BarGtFeat, BarLtFeat,
    AND, OR,
    FORCE_EXIT, SWITCH_PROFILE, OVERWRITE_TP_SL,
    Slope, SlopeGtParam, SlopeLtParam,
    Mean, StdDev, AboveMA, BelowMA,feat,param,
)

from .regime_policy import (
    RegimePolicy,
    RegimeContext,
    build_regime_context,
    lag_regime_array,
    make_regime_exit_policy,
)

from .partial_config import (
    DistributionFn,
    PartialConfig,
    PyramidConfig,
    AveragingConfig,
    PhaseSpec,
)

from .position_rules import (
    OnRR, OnMFEPct, OnMAEPct, OnATRMult,
    OnBars, OnBarsAfterLastTP,
    OnFeature, OnPhase, OnAll, OnAny,
    ExitPartial, MoveSLtoBE, MoveSLto,
    SetTP, AddPosition, SetPhase, Invalidate,
    PositionRule,
)

from .stateful_config import StatefulConfig
from .StrategyEngine import StrategyEngine

from .context_engine import (
    TradeContextEngine,
    build_default_context_df,
)
from .event_log import build_event_log
from .rolling_engine import RollingIndicator, rolling_apply, rolling_apply_2d
from .regime_plot import plot_price_with_regime
from .strategy_recipes import (
    low_vol_bb_rsi_reversion_df,
    low_vol_bb_rsi_reversion_setup,
    make_low_vol_bb_reversion_profile,
)

import sys as _sys
__all__ = [
    name for name in dir(_sys.modules[__name__])
    if not name.startswith('_')
]
