from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, Callable

from .stateful_config import StatefulConfig

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY RUNTIME MATRIX COLS
# ══════════════════════════════════════════════════════════════════════════════

STRAT_ID             = 0
STRAT_TYPE           = 1
STRAT_BACKEND        = 3
STRAT_WINDOW_BARS    = 4

STRAT_FEAT_COL_1     = 5
STRAT_FEAT_COL_2     = 6
STRAT_FEAT_COL_3     = 7
STRAT_FEAT_COL_4     = 8
STRAT_FEAT_COL_5     = 9
STRAT_FEAT_COL_6     = 10
STRAT_FEAT_COL_7     = 11
STRAT_FEAT_COL_8     = 12
STRAT_FEAT_COL_9     = 13
STRAT_FEAT_COL_10    = 14

STRAT_PARAM_1        = 15
STRAT_PARAM_2        = 16
STRAT_PARAM_3        = 17
STRAT_PARAM_4        = 18
STRAT_PARAM_5        = 19
STRAT_PARAM_6        = 20

N_EXIT_STRAT_RT_COLS = 21

# ══════════════════════════════════════════════════════════════════════════════
# EXIT ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

EXIT_ACT_NONE            = 0
EXIT_ACT_SWITCH_PROFILE  = 1
EXIT_ACT_OVERWRITE_PRICE = 2
EXIT_ACT_FORCE_EXIT      = 3
EXIT_ACT_PARTIAL_EXIT    = 4
EXIT_ACT_ADD_POSITION    = 5

ACT_TYPE              = 0
ACT_TARGET_PROFILE_ID = 1
ACT_NEW_TP_PRICE      = 2
ACT_NEW_SL_PRICE      = 3
ACT_FORCE_EXIT_FLAG   = 4
ACT_FORCE_EXIT_REASON = 5
ACT_PARTIAL_FRACTION  = 6
ACT_PARTIAL_PRICE     = 7
ACT_ADD_SIZE_FRACTION = 8
ACT_ADD_SL_PRICE      = 9
ACT_ADD_TP_PRICE      = 10
ACT_ADD_GROUP_SL_MODE = 11
N_EXIT_ACT_COLS       = 12

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION KINDS / BACKEND
# ══════════════════════════════════════════════════════════════════════════════

EXIT_STRAT_NONE       = -1
EXIT_STRAT_INSTANT    = 0
EXIT_STRAT_WINDOWED   = 1
EXIT_STRAT_STATEFUL   = 2
EXIT_STRAT_INSTANT_PY = 3
EXIT_STRAT_WINDOW_PY  = 4

STRAT_BACKEND_NUMBA  = 0
STRAT_BACKEND_PYTHON = 1

# ══════════════════════════════════════════════════════════════════════════════
# STATE PER POSITION — colonnes prédéfinies
# ══════════════════════════════════════════════════════════════════════════════

SP_PHASE              = 0
SP_N_TP_HIT           = 1
SP_REMAINING_SIZE     = 2
SP_LAST_HIGH          = 3
SP_BARS_SINCE_ENTRY   = 4
SP_BARS_SINCE_TP      = 5
SP_ADD_COUNT          = 6
SP_AVG_ENTRY          = 7
SP_ENTRY_VALID        = 8
SP_REGIME_AT_ENTRY    = 9
N_SP_DEFAULT          = 10

# ══════════════════════════════════════════════════════════════════════════════
# STATE GLOBAL — colonnes prédéfinies
# ══════════════════════════════════════════════════════════════════════════════

SG_CONSEC_SL           = 0
SG_CONSEC_TP           = 1
SG_ROLLING_WINRATE     = 2
SG_TOTAL_EXPOSURE      = 3
SG_LAST_TRADE_RETURN   = 4
SG_DAILY_TRADE_COUNT   = 5
SG_SESSION_TRADE_COUNT = 6
SG_COOLDOWN_UNTIL      = 7
SG_CURRENT_REGIME      = 8
N_SG_DEFAULT           = 9

SP_CUSTOM_OFFSET = N_SP_DEFAULT
SG_CUSTOM_OFFSET = N_SG_DEFAULT

# ══════════════════════════════════════════════════════════════════════════════
# STATEFUL CONFIG RUNTIME COLS
# ══════════════════════════════════════════════════════════════════════════════

SCFG_MAX_CONSEC_SL        = 0
SCFG_COOLDOWN_BARS        = 1
SCFG_MAX_POSITIONS        = 2
SCFG_INVALIDATE_ON_REGIME = 3
SCFG_MIN_WINRATE          = 4
SCFG_WINRATE_COOLDOWN     = 5
N_STATEFUL_CFG_COLS       = 6


# ══════════════════════════════════════════════════════════════════════════════
# EXITSTRATEGYSPEC
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ExitStrategySpec:
    strategy_id: int
    name: str = ""
    strategy_type: int = EXIT_STRAT_INSTANT
    backend: int = STRAT_BACKEND_NUMBA
    fn: Optional[Callable] = None
    params: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)
    window_bars: int = 0
    state_per_pos_custom: dict = field(default_factory=dict)
    state_global_custom: dict = field(default_factory=dict)

    # ── Nouveau ───────────────────────────────────────────────────────────────
    stateful_config: Optional[StatefulConfig] = None

    def __post_init__(self):
        if self.strategy_type not in (
            EXIT_STRAT_INSTANT, EXIT_STRAT_WINDOWED, EXIT_STRAT_STATEFUL,
            EXIT_STRAT_INSTANT_PY, EXIT_STRAT_WINDOW_PY,
        ):
            raise ValueError("invalid strategy_type")
        if self.backend not in (STRAT_BACKEND_NUMBA, STRAT_BACKEND_PYTHON):
            raise ValueError("invalid backend")
        if self.strategy_id < 0:
            raise ValueError("strategy_id must be >= 0")
        if self.strategy_type in (EXIT_STRAT_WINDOWED, EXIT_STRAT_WINDOW_PY) and self.window_bars <= 0:
            raise ValueError("window_bars must be > 0 for windowed strategy")
        if self.strategy_type in (EXIT_STRAT_INSTANT_PY, EXIT_STRAT_WINDOW_PY) and self.fn is None:
            raise ValueError("fn is required for Python strategies")
        if self.fn is not None and not callable(self.fn):
            raise TypeError("fn must be callable")
        if len(self.feature_names) > 10:
            raise ValueError("A strategy can reference at most 10 features")
        for name, idx in self.state_per_pos_custom.items():
            if idx < 0:
                raise ValueError(f"state_per_pos_custom['{name}'] index must be >= 0")
        for name, idx in self.state_global_custom.items():
            if idx < 0:
                raise ValueError(f"state_global_custom['{name}'] index must be >= 0")


# ══════════════════════════════════════════════════════════════════════════════
# COMPILEDEXITSTRATEGY — inchangé
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class CompiledExitStrategy:
    strategy_id: int
    name: str
    strategy_type: int
    backend: int
    window_bars: int
    feature_cols: tuple
    param_values: tuple
    n_state_per_pos_custom: int = 0
    n_state_global_custom: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE COMPILATION
# ══════════════════════════════════════════════════════════════════════════════

def compile_exit_strategies(
    strategy_specs: list[ExitStrategySpec],
    compiled_features,
    param_names: tuple = (),
) -> list[CompiledExitStrategy]:
    out = []
    for spec in strategy_specs:
        feature_cols = tuple(compiled_features.col(name) for name in spec.feature_names)
        param_values = tuple(float(spec.params.get(p, -1.0)) for p in param_names)
        out.append(
            CompiledExitStrategy(
                strategy_id=spec.strategy_id,
                name=spec.name or f"exit_strategy_{spec.strategy_id}",
                strategy_type=spec.strategy_type,
                backend=spec.backend,
                window_bars=spec.window_bars,
                feature_cols=feature_cols,
                param_values=param_values,
                n_state_per_pos_custom=len(spec.state_per_pos_custom),
                n_state_global_custom=len(spec.state_global_custom),
            )
        )
    return out


def build_exit_strategy_rt_matrix(
    compiled_strategies: list[CompiledExitStrategy],
) -> np.ndarray:
    if len(compiled_strategies) == 0:
        return np.zeros((0, N_EXIT_STRAT_RT_COLS), dtype=np.float64)

    max_id = max(s.strategy_id for s in compiled_strategies)
    rt = np.full((max_id + 1, N_EXIT_STRAT_RT_COLS), -1.0, dtype=np.float64)

    for s in compiled_strategies:
        sid = s.strategy_id
        rt[sid, STRAT_ID]          = s.strategy_id
        rt[sid, STRAT_TYPE]        = s.strategy_type
        rt[sid, STRAT_BACKEND]     = s.backend
        rt[sid, STRAT_WINDOW_BARS] = s.window_bars

        fc = s.feature_cols
        for i, col_name in enumerate([
            STRAT_FEAT_COL_1, STRAT_FEAT_COL_2, STRAT_FEAT_COL_3,
            STRAT_FEAT_COL_4, STRAT_FEAT_COL_5, STRAT_FEAT_COL_6,
            STRAT_FEAT_COL_7, STRAT_FEAT_COL_8, STRAT_FEAT_COL_9,
            STRAT_FEAT_COL_10,
        ]):
            if i < len(fc):
                rt[sid, col_name] = fc[i]

        pv = s.param_values
        for i, col_name in enumerate([
            STRAT_PARAM_1, STRAT_PARAM_2, STRAT_PARAM_3,
            STRAT_PARAM_4, STRAT_PARAM_5, STRAT_PARAM_6,
        ]):
            if i < len(pv):
                rt[sid, col_name] = pv[i]

    return rt


def build_stateful_cfg_rt_matrix(
    strategy_specs: list[ExitStrategySpec],
) -> np.ndarray:
    """
    Compiler StatefulConfig de toutes les strats en matrice runtime.
    shape: (n_strategies, N_STATEFUL_CFG_COLS)
    """
    if not strategy_specs:
        return np.zeros((1, N_STATEFUL_CFG_COLS), dtype=np.float64)

    max_id = max(s.strategy_id for s in strategy_specs)
    rt = np.zeros((max_id + 1, N_STATEFUL_CFG_COLS), dtype=np.float64)

    for spec in strategy_specs:
        sid = spec.strategy_id
        if spec.stateful_config is not None:
            vals = spec.stateful_config.to_rt_array()
            for col, val in enumerate(vals):
                rt[sid, col] = val

    return rt