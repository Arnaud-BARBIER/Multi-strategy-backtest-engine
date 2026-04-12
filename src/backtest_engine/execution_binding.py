from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .Exit_system import (
    compile_exit_profiles,
    build_exit_profile_rt_matrix,
    build_position_rule_matrices,
    compile_setup_exit_binding,
)
from .exit_strategy_system import (
    compile_exit_strategies,
    build_exit_strategy_rt_matrix,
    build_stateful_cfg_rt_matrix,
    N_SP_DEFAULT, N_SG_DEFAULT, N_EXIT_STRAT_RT_COLS,
    N_STATEFUL_CFG_COLS,
)


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ExecutionContext:
    # ── Matrices existantes ───────────────────────────────────────────────────
    profile_rt_matrix: np.ndarray
    strategy_rt_matrix: np.ndarray

    setup_to_exit_profile: np.ndarray
    setup_to_exit_strategy: np.ndarray

    strategy_to_default_profile: np.ndarray
    strategy_allowed_profiles: np.ndarray
    strategy_allowed_counts: np.ndarray

    n_state_per_pos: int = N_SP_DEFAULT
    n_state_global:  int = N_SG_DEFAULT

    # ── Nouvelles matrices ────────────────────────────────────────────────────
    partial_rt_matrix:    np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float64))
    pyramid_rt_matrix:    np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float64))
    averaging_rt_matrix:  np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float64))
    phase_rt_matrix:      np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float64))
    rule_trigger_matrix:  np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float64))
    rule_action_matrix:   np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1, 1), dtype=np.float64))
    stateful_cfg_rt:      np.ndarray = field(default_factory=lambda: np.zeros((1, N_STATEFUL_CFG_COLS), dtype=np.float64))

    # ── Métadonnées ───────────────────────────────────────────────────────────
    feature_name_to_idx:  dict = field(default_factory=dict)
    max_partial_levels:   int = 0
    max_pyramid_levels:   int = 0
    max_avg_levels:       int = 0
    max_phases:           int = 0
    max_rules:            int = 0
    max_actions_per_rule: int = 0

    # ── Flags d'optimisation ──────────────────────────────────────────────────
    has_partial:      bool = False
    has_pyramid:      bool = False
    has_averaging:    bool = False
    has_phases:       bool = False
    has_rules:        bool = False
    has_stateful_cfg: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# BUILD EXECUTION CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

def build_execution_context(
    cfg,
    exit_profile_specs,
    setup_exit_binding,
    strategy_profile_binding,
    n_setups,
    exit_strategy_specs=None,
    n_strategies=None,
    compiled_features=None,
    strategy_param_names=(),
):
    """
    Compiler toutes les specs en matrices runtime prêtes pour backtest_njit.

    Paramètres :
        cfg                     : BacktestConfig — valeurs par défaut
        exit_profile_specs      : list[ExitProfileSpec]
        setup_exit_binding      : dict {setup_id: {"exit_profile_id": N, "exit_strategy_id": M}}
        strategy_profile_binding: dict {strategy_id: {"allowed_profile_ids": [...]}}
        n_setups                : int
        exit_strategy_specs     : list[ExitStrategySpec] (optionnel)
        n_strategies            : int (optionnel, inféré si None)
        compiled_features       : CompiledFeatures (requis si exit_strategy_specs fourni)
        strategy_param_names    : tuple[str, ...] — noms des params numériques

    Retourne :
        ExecutionContext — contient toutes les matrices runtime
    """

    # ── 1. Profils de sortie ──────────────────────────────────────────────────
    compiled_profiles  = compile_exit_profiles(exit_profile_specs, cfg)
    profile_rt_matrix  = build_exit_profile_rt_matrix(compiled_profiles)

    # ── 2. Stratégies de sortie ───────────────────────────────────────────────
    if exit_strategy_specs is None:
        exit_strategy_specs = []

    if len(exit_strategy_specs) > 0:
        if compiled_features is None:
            raise ValueError("compiled_features requis quand exit_strategy_specs est fourni")
        compiled_strategies = compile_exit_strategies(
            strategy_specs=exit_strategy_specs,
            compiled_features=compiled_features,
            param_names=strategy_param_names,
        )
        strategy_rt_matrix = build_exit_strategy_rt_matrix(compiled_strategies)
    else:
        strategy_rt_matrix = np.zeros((0, N_EXIT_STRAT_RT_COLS), dtype=np.float64)

    # ── 3. Bindings setup → profil / strat ───────────────────────────────────
    binding = compile_setup_exit_binding(
        setup_exit_binding=setup_exit_binding,
        strategy_profile_binding=strategy_profile_binding,
        n_setups=n_setups,
        n_strategies=n_strategies,
    )

    # ── 4. Tailles state ──────────────────────────────────────────────────────
    n_state_per_pos = N_SP_DEFAULT
    n_state_global  = N_SG_DEFAULT

    if exit_strategy_specs:
        max_sp = max((len(s.state_per_pos_custom) for s in exit_strategy_specs), default=0)
        max_sg = max((len(s.state_global_custom)  for s in exit_strategy_specs), default=0)
        n_state_per_pos = N_SP_DEFAULT + max_sp
        n_state_global  = N_SG_DEFAULT + max_sg

    # ── 5. Feature name → index ───────────────────────────────────────────────
    feature_name_to_idx = {}
    if compiled_features is not None:
        feature_name_to_idx = dict(compiled_features.col_map)

    # ── 6. Nouvelles matrices — partial / pyramid / averaging / phases / rules ─
    has_partial   = any(s.partial_config   is not None for s in exit_profile_specs)
    has_pyramid   = any(s.pyramid_config   is not None for s in exit_profile_specs)
    has_averaging = any(s.averaging_config is not None for s in exit_profile_specs)
    has_phases    = any(len(s.phases) > 0              for s in exit_profile_specs)
    has_rules     = any(len(s.get_all_rules()) > 0     for s in exit_profile_specs)

    if has_partial or has_pyramid or has_averaging or has_phases or has_rules:
        rule_matrices = build_position_rule_matrices(
            profile_specs=exit_profile_specs,
            feature_name_to_idx=feature_name_to_idx,
        )
    else:
        rule_matrices = {
            "partial_rt_matrix":    np.full((len(exit_profile_specs), 1, 1), -1.0, dtype=np.float64),
            "pyramid_rt_matrix":    np.full((len(exit_profile_specs), 1, 1), -1.0, dtype=np.float64),
            "averaging_rt_matrix":  np.full((len(exit_profile_specs), 1, 1), -1.0, dtype=np.float64),
            "phase_rt_matrix":      np.full((len(exit_profile_specs), 1, 1), -1.0, dtype=np.float64),
            "rule_trigger_matrix":  np.full((len(exit_profile_specs), 1, 1), -1.0, dtype=np.float64),
            "rule_action_matrix":   np.full((len(exit_profile_specs), 1, 1, 1), -1.0, dtype=np.float64),
            "max_partial_levels":   0,
            "max_pyramid_levels":   0,
            "max_avg_levels":       0,
            "max_phases":           0,
            "max_rules":            0,
            "max_actions_per_rule": 0,
        }

    # ── 7. StatefulConfig ─────────────────────────────────────────────────────
    has_stateful_cfg = any(
        s.stateful_config is not None and s.stateful_config.is_active
        for s in exit_strategy_specs
    ) if exit_strategy_specs else False

    stateful_cfg_rt = build_stateful_cfg_rt_matrix(exit_strategy_specs)

    return ExecutionContext(
        # Existant
        profile_rt_matrix=profile_rt_matrix,
        strategy_rt_matrix=strategy_rt_matrix,
        setup_to_exit_profile=binding["setup_to_exit_profile"],
        setup_to_exit_strategy=binding["setup_to_exit_strategy"],
        strategy_to_default_profile=binding["strategy_to_default_profile"],
        strategy_allowed_profiles=binding["strategy_allowed_profiles"],
        strategy_allowed_counts=binding["strategy_allowed_counts"],
        n_state_per_pos=n_state_per_pos,
        n_state_global=n_state_global,
        # Nouveau
        partial_rt_matrix=rule_matrices["partial_rt_matrix"],
        pyramid_rt_matrix=rule_matrices["pyramid_rt_matrix"],
        averaging_rt_matrix=rule_matrices["averaging_rt_matrix"],
        phase_rt_matrix=rule_matrices["phase_rt_matrix"],
        rule_trigger_matrix=rule_matrices["rule_trigger_matrix"],
        rule_action_matrix=rule_matrices["rule_action_matrix"],
        stateful_cfg_rt=stateful_cfg_rt,
        feature_name_to_idx=feature_name_to_idx,
        max_partial_levels=rule_matrices["max_partial_levels"],
        max_pyramid_levels=rule_matrices["max_pyramid_levels"],
        max_avg_levels=rule_matrices["max_avg_levels"],
        max_phases=rule_matrices["max_phases"],
        max_rules=rule_matrices["max_rules"],
        max_actions_per_rule=rule_matrices["max_actions_per_rule"],
        has_partial=has_partial,
        has_pyramid=has_pyramid,
        has_averaging=has_averaging,
        has_phases=has_phases,
        has_rules=has_rules,
        has_stateful_cfg=has_stateful_cfg,
    )
