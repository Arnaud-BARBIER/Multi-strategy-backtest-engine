# Backtest_Framework/regime_policy.py
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RegimePolicy:
    n_regimes: int
    score_multiplier: dict[int, dict[str, dict[str, float]]] = field(default_factory=dict)
    exit_profile_override: dict[int, int] = field(default_factory=dict)
    exit_strategy_override: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.n_regimes <= 0:
            raise ValueError("n_regimes must be > 0")
        for r, setup_dict in self.score_multiplier.items():
            if not isinstance(r, int) or r < 0 or r >= self.n_regimes:
                raise ValueError(f"regime_id {r} out of bounds for n_regimes={self.n_regimes}")
            for setup_name, dir_dict in setup_dict.items():
                for d in dir_dict:
                    if d not in ("long", "short"):
                        raise ValueError(f"Direction must be 'long' or 'short', got '{d}'")
                    if dir_dict[d] < 0:
                        raise ValueError(f"Multiplier must be >= 0, got {dir_dict[d]}")
        for r in self.exit_profile_override:
            if r < 0 or r >= self.n_regimes:
                raise ValueError(f"regime_id {r} out of bounds in exit_profile_override")
        for r in self.exit_strategy_override:
            if r < 0 or r >= self.n_regimes:
                raise ValueError(f"regime_id {r} out of bounds in exit_strategy_override")


@dataclass(slots=True)
class RegimeContext:
    regime: np.ndarray                    # (n_bars,) int32
    score_multipliers: np.ndarray         # (n_regimes, n_setups, 2) float64
    exit_profile_override: np.ndarray     # (n_regimes,) int32
    exit_strategy_override: np.ndarray    # (n_regimes,) int32
    n_regimes: int
    n_setups: int


def build_regime_context(
    regime: np.ndarray,
    policy: RegimePolicy,
    setup_specs: list,
) -> RegimeContext:
    n_setups = len(setup_specs)
    n_regimes = policy.n_regimes

    # Résolution noms → indices
    setup_name_to_idx = {spec.name: i for i, spec in enumerate(setup_specs)}

    # Matrice multipliers — défaut 1.0 partout
    score_multipliers = np.ones((n_regimes, n_setups, 2), dtype=np.float64)

    for regime_id, setup_dict in policy.score_multiplier.items():
        for setup_name, dir_dict in setup_dict.items():
            if setup_name not in setup_name_to_idx:
                raise ValueError(
                    f"Unknown setup name: '{setup_name}'. "
                    f"Available: {list(setup_name_to_idx.keys())}"
                )
            j = setup_name_to_idx[setup_name]
            score_multipliers[regime_id, j, 0] = dir_dict.get("long",  1.0)
            score_multipliers[regime_id, j, 1] = dir_dict.get("short", 1.0)

    # Exit overrides
    exit_profile_override = np.full(n_regimes, -1, dtype=np.int32)
    for regime_id, profile_id in policy.exit_profile_override.items():
        exit_profile_override[regime_id] = profile_id

    exit_strategy_override = np.full(n_regimes, -1, dtype=np.int32)
    for regime_id, strategy_id in policy.exit_strategy_override.items():
        exit_strategy_override[regime_id] = strategy_id

    # Validation regime array
    regime_arr = np.asarray(regime, dtype=np.int32)
    if regime_arr.ndim != 1:
        raise ValueError("regime must be a 1D array")
    invalid = (regime_arr < 0) | (regime_arr >= n_regimes)
    if invalid.any():
        raise ValueError(
            f"regime contains values outside [0, {n_regimes-1}]: "
            f"{np.unique(regime_arr[invalid])}"
        )

    return RegimeContext(
        regime=regime_arr,
        score_multipliers=score_multipliers,
        exit_profile_override=exit_profile_override,
        exit_strategy_override=exit_strategy_override,
        n_regimes=n_regimes,
        n_setups=n_setups,
    )
