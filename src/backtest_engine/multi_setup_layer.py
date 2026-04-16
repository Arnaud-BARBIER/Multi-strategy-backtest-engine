from dataclasses import dataclass, field
from typing import Callable, Any, Optional

import numpy as np
import pandas as pd
from numba import njit
from .regime_policy import RegimePolicy, RegimeContext, build_regime_context
# ══════════════════════════════════════════════════════════════════
# 1. DECISION CONFIG
# ══════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class DecisionConfig:
    min_score: float = 0.0
    allow_long: bool = True
    allow_short: bool = True
    tie_policy: int = 0   # 0=no trade, 1=prefer long, 2=prefer short

    def __post_init__(self):
        if self.min_score < 0:
            raise ValueError("min_score must be >= 0")
        if self.tie_policy not in (0, 1, 2):
            raise ValueError("tie_policy must be in {0,1,2}")


# ══════════════════════════════════════════════════════════════════
# 2. SETUP SPEC
# ══════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SetupSpec:
    fn: Callable[..., pd.DataFrame]
    params: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self):
        if not callable(self.fn):
            raise TypeError("SetupSpec.fn must be callable")
        if not isinstance(self.params, dict):
            raise TypeError("SetupSpec.params must be a dict")


# ══════════════════════════════════════════════════════════════════
# 3. VALIDATION SETUP DF
# ══════════════════════════════════════════════════════════════════

_REQUIRED_SETUP_COLS = [
    "long_score",
    "short_score",
    "long_active",
    "short_active",
    "setup_id",
]


def _validate_setup_df(
    setup_df: pd.DataFrame,
    expected_index: Optional[pd.Index] = None,
    setup_name: str = "setup",
) -> None:
    if not isinstance(setup_df, pd.DataFrame):
        raise TypeError(f"{setup_name}: setup_df must be a pandas DataFrame")

    missing = [c for c in _REQUIRED_SETUP_COLS if c not in setup_df.columns]
    if missing:
        raise ValueError(f"{setup_name}: missing required columns: {missing}")

    if expected_index is not None and not setup_df.index.equals(expected_index):
        raise ValueError(f"{setup_name}: index mismatch with other setups")

    if setup_df["setup_id"].nunique() != 1:
        raise ValueError(f"{setup_name}: setup_id must be constant on the whole DataFrame")

    if not np.issubdtype(setup_df["long_score"].dtype, np.floating):
        raise TypeError(f"{setup_name}: long_score must be float-like")
    if not np.issubdtype(setup_df["short_score"].dtype, np.floating):
        raise TypeError(f"{setup_name}: short_score must be float-like")

    if not np.issubdtype(setup_df["long_active"].dtype, np.integer):
        raise TypeError(f"{setup_name}: long_active must be int-like")
    if not np.issubdtype(setup_df["short_active"].dtype, np.integer):
        raise TypeError(f"{setup_name}: short_active must be int-like")

    la = setup_df["long_active"].to_numpy()
    sa = setup_df["short_active"].to_numpy()
    if not np.isin(la, [0, 1]).all():
        raise ValueError(f"{setup_name}: long_active must contain only 0/1")
    if not np.isin(sa, [0, 1]).all():
        raise ValueError(f"{setup_name}: short_active must contain only 0/1")


# ══════════════════════════════════════════════════════════════════
# 4. BUILD MATRICES
# ══════════════════════════════════════════════════════════════════

def _build_setup_matrices(setup_dfs: list[pd.DataFrame]) -> dict[str, Any]:
    if len(setup_dfs) == 0:
        raise ValueError("setup_dfs cannot be empty")

    ref_index = setup_dfs[0].index
    n_bars = len(ref_index)
    n_setups = len(setup_dfs)

    long_scores_matrix  = np.empty((n_bars, n_setups), dtype=np.float64)
    short_scores_matrix = np.empty((n_bars, n_setups), dtype=np.float64)
    long_active_matrix  = np.empty((n_bars, n_setups), dtype=np.int8)
    short_active_matrix = np.empty((n_bars, n_setups), dtype=np.int8)
    setup_ids           = np.empty(n_setups, dtype=np.int32)

    for j, sdf in enumerate(setup_dfs):
        _validate_setup_df(sdf, expected_index=ref_index, setup_name=f"setup[{j}]")

        long_scores_matrix[:, j]  = sdf["long_score"].to_numpy(dtype=np.float64)
        short_scores_matrix[:, j] = sdf["short_score"].to_numpy(dtype=np.float64)
        long_active_matrix[:, j]  = sdf["long_active"].to_numpy(dtype=np.int8)
        short_active_matrix[:, j] = sdf["short_active"].to_numpy(dtype=np.int8)
        setup_ids[j]              = np.int32(sdf["setup_id"].iloc[0])

    return {
        "index": ref_index,
        "long_scores_matrix": long_scores_matrix,
        "short_scores_matrix": short_scores_matrix,
        "long_active_matrix": long_active_matrix,
        "short_active_matrix": short_active_matrix,
        "setup_ids": setup_ids,
    }


# ══════════════════════════════════════════════════════════════════
# 5. NJIT AGGREGATION + DECISION
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def aggregate_and_decide_njit(
    long_scores_matrix,
    short_scores_matrix,
    long_active_matrix,
    short_active_matrix,
    setup_ids,
    min_score,
    allow_long,
    allow_short,
    tie_policy,
    regime,                   # ← nouveau (n_bars,) int32
    regime_score_multipliers, # ← nouveau (n_regimes, n_setups, 2) float64
    use_regime,               # ← nouveau bool
):
    n_bars, n_setups = long_scores_matrix.shape

    signals             = np.zeros(n_bars, dtype=np.int8)
    selected_setup_id   = np.full(n_bars, -1, dtype=np.int32)
    selected_score      = np.zeros(n_bars, dtype=np.float64)
    best_long_score     = np.zeros(n_bars, dtype=np.float64)
    best_short_score    = np.zeros(n_bars, dtype=np.float64)
    best_long_setup_id  = np.full(n_bars, -1, dtype=np.int32)
    best_short_setup_id = np.full(n_bars, -1, dtype=np.int32)

    for i in range(n_bars):
        r = int(regime[i]) if use_regime else 0

        bls = 0.0; bss = 0.0
        bl_id = -1; bs_id = -1

        if allow_long:
            for j in range(n_setups):
                score = long_scores_matrix[i, j] * long_active_matrix[i, j] # if no regime, only manual activation via *_active_matrix
                if use_regime:
                    score *= regime_score_multipliers[r, j, 0]
                if score > bls:
                    bls = score
                    bl_id = setup_ids[j]

        if allow_short:
            for j in range(n_setups):
                score = short_scores_matrix[i, j] * short_active_matrix[i, j]
                if use_regime:
                    score *= regime_score_multipliers[r, j, 1]
                if score > bss:
                    bss = score
                    bs_id = setup_ids[j]

        best_long_score[i]     = bls
        best_short_score[i]    = bss
        best_long_setup_id[i]  = bl_id
        best_short_setup_id[i] = bs_id

        if bls >= min_score and bls > bss:
            signals[i] = 1
            selected_setup_id[i] = bl_id
            selected_score[i] = bls
        elif bss >= min_score and bss > bls:
            signals[i] = -1
            selected_setup_id[i] = bs_id
            selected_score[i] = bss
        elif bls >= min_score and bss >= min_score and bls == bss:
            if tie_policy == 1:
                signals[i] = 1
                selected_setup_id[i] = bl_id
                selected_score[i] = bls
            elif tie_policy == 2:
                signals[i] = -1
                selected_setup_id[i] = bs_id
                selected_score[i] = bss

    return (
        signals, selected_setup_id, selected_score,
        best_long_score, best_short_score,
        best_long_setup_id, best_short_setup_id,
    )

# ══════════════════════════════════════════════════════════════════
# 6. PYTHON WRAPPER
# ══════════════════════════════════════════════════════════════════
def aggregate_and_decide(
    setup_dfs: list,
    decision_cfg: DecisionConfig,
    regime_context: RegimeContext | None = None,
) -> dict:
    mats = _build_setup_matrices(setup_dfs)
    n_bars = len(setup_dfs[0])
    n_setups = len(setup_dfs)

    if regime_context is not None:
        regime = regime_context.regime
        regime_multipliers = regime_context.score_multipliers
        use_regime = True
    else:
        regime = np.zeros(n_bars, dtype=np.int32)
        regime_multipliers = np.ones((1, n_setups, 2), dtype=np.float64)
        use_regime = False

    (signals, selected_setup_id, selected_score,
     best_long_score, best_short_score,
     best_long_setup_id, best_short_setup_id) = aggregate_and_decide_njit(
        mats["long_scores_matrix"],
        mats["short_scores_matrix"],
        mats["long_active_matrix"],
        mats["short_active_matrix"],
        mats["setup_ids"],
        float(decision_cfg.min_score),
        decision_cfg.allow_long,
        decision_cfg.allow_short,
        decision_cfg.tie_policy,
        regime,
        regime_multipliers,
        use_regime,
    )

    return {
        "signals": signals,
        "selected_setup_id": selected_setup_id,
        "selected_score": selected_score,
        "best_long_score": best_long_score,
        "best_short_score": best_short_score,
        "best_long_setup_id": best_long_setup_id,
        "best_short_setup_id": best_short_setup_id,
    }
