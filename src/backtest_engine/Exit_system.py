from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import numpy as np

from .pipeline_config import BacktestConfig
from .partial_config import (
    DistributionFn, PartialConfig, PyramidConfig,
    AveragingConfig, PhaseSpec,
)
from .position_rules import (
    OnRR, OnMFEPct, OnMAEPct, OnATRMult,
    OnBars, OnBarsAfterLastTP,
    OnFeature, OnPhase, OnAll, OnAny,
    ExitPartial, MoveSLtoBE, MoveSLto,
    SetTP, AddPosition, SetPhase, Invalidate,
    PositionRule,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES MATRICES RUNTIME — PROFIL DE SORTIE
# ══════════════════════════════════════════════════════════════════════════════

RT_EXIT_PROFILE_ID        = 0
RT_EXIT_STRATEGY_ID       = 1
RT_TP_PCT                 = 2
RT_SL_PCT                 = 3
RT_USE_ATR_SL_TP          = 4
RT_TP_ATR_MULT            = 5
RT_SL_ATR_MULT            = 6
RT_BE_TRIGGER_PCT         = 7
RT_BE_OFFSET_PCT          = 8
RT_BE_DELAY_BARS          = 9
RT_TRAILING_TRIGGER_PCT   = 10
RT_RUNNER_TRAILING_MULT   = 11
RT_MAX_HOLDING_BARS       = 12
N_EXIT_RT_COLS            = 13

# ── Partial runtime matrix cols ───────────────────────────────────────────────
PART_TRIGGER_TYPE   = 0
PART_TRIGGER_VALUE  = 1
PART_TRIGGER_FEAT   = 2   # index feature compilée (-1 si pas de feature)
PART_TRIGGER_OP     = 3   # operator encodé (0=gt,1=lt,2=gte,3=lte,4=cross_above,5=cross_below)
PART_FRACTION       = 4   # fraction fixe
PART_REF            = 5   # 0=remaining, 1=original
PART_PRICE_MODE     = 6   # 0=market, 1=tp, 2=feature
PART_MOVE_BE        = 7   # 0 ou 1
PART_DIST_MODE      = 8   # mode DistributionFn encodé
PART_DIST_PARAM1    = 9   # ratio (expo) ou slope (linear)
PART_DIST_PARAM2    = 10  # start
N_PARTIAL_COLS      = 11

# ── Pyramid runtime matrix cols ───────────────────────────────────────────────
PYR_TRIGGER_TYPE    = 0
PYR_TRIGGER_VALUE   = 1
PYR_TRIGGER_FEAT    = 2
PYR_TRIGGER_OP      = 3
PYR_SIZE_FRACTION   = 4
PYR_SIZE_REF        = 5   # 0=original, 1=remaining
PYR_SL_MODE         = 6   # 0=breakeven, 1=original, 2=feature, 3=atr_mult
PYR_SL_FEAT         = 7   # index feature SL (-1 si pas feature)
PYR_SL_ATR_MULT     = 8
PYR_GROUP_SL_MODE   = 9
PYR_DIST_MODE       = 10
PYR_DIST_PARAM1     = 11
PYR_DIST_PARAM2     = 12
N_PYRAMID_COLS      = 13

# ── Averaging runtime matrix cols ─────────────────────────────────────────────
AVG_TRIGGER_TYPE    = 0
AVG_TRIGGER_VALUE   = 1
AVG_TRIGGER_FEAT    = 2
AVG_TRIGGER_OP      = 3
AVG_SIZE_FRACTION   = 4
AVG_SL_MODE         = 5   # 0=original, 1=breakeven
AVG_TP_MODE         = 6   # 0=same
AVG_MAX_DOWN        = 7   # max_avg_down_pct (négatif)
AVG_DIST_MODE       = 8
AVG_DIST_PARAM1     = 9
AVG_DIST_PARAM2     = 10
N_AVG_COLS          = 11

# ── Phase runtime matrix cols ─────────────────────────────────────────────────
PHASE_ID            = 0
PHASE_TP_PCT        = 1   # -1 = pas de redéfinition
PHASE_SL_PCT        = 2
PHASE_BE_TRIGGER    = 3
PHASE_TRAILING      = 4
PHASE_MAX_HOLD      = 5
N_PHASE_COLS        = 6

# ── Rule trigger matrix cols ──────────────────────────────────────────────────
RT_RULE_PHASE_FILTER   = 0
RT_RULE_MAX_TIMES      = 1
RT_RULE_TRIGGER_TYPE   = 2
RT_RULE_TRIGGER_VALUE  = 3
RT_RULE_TRIGGER_FEAT1  = 4
RT_RULE_TRIGGER_FEAT2  = 5
RT_RULE_TRIGGER_OP     = 6
RT_RULE_N_ACTIONS      = 7
N_RULE_TRIGGER_COLS    = 8

# ── Rule action matrix cols ───────────────────────────────────────────────────
RA_ACTION_TYPE      = 0
RA_PARAM1           = 1
RA_PARAM2           = 2
RA_PARAM3           = 3
RA_FEAT_IDX         = 4
N_RULE_ACTION_COLS  = 5

# ── Trigger type encoding ─────────────────────────────────────────────────────
TRIGGER_TYPE_RR        = 0
TRIGGER_TYPE_MFE_PCT   = 1
TRIGGER_TYPE_MAE_PCT   = 2
TRIGGER_TYPE_ATR_MULT  = 3
TRIGGER_TYPE_BARS      = 4
TRIGGER_TYPE_BARS_ATP  = 5
TRIGGER_TYPE_FEATURE   = 6
TRIGGER_TYPE_PHASE     = 7
TRIGGER_TYPE_ALL       = 8
TRIGGER_TYPE_ANY       = 9
TRIGGER_TYPE_NONE      = -1

# ── Operator encoding ─────────────────────────────────────────────────────────
OP_GT           = 0
OP_LT           = 1
OP_GTE          = 2
OP_LTE          = 3
OP_CROSS_ABOVE  = 4
OP_CROSS_BELOW  = 5

_OP_MAP = {
    "gt": OP_GT, "lt": OP_LT, "gte": OP_GTE, "lte": OP_LTE,
    "cross_above": OP_CROSS_ABOVE, "cross_below": OP_CROSS_BELOW,
}

# ── Action type encoding ──────────────────────────────────────────────────────
ACTION_TYPE_EXIT_PARTIAL = 0
ACTION_TYPE_MOVE_SL_BE   = 1
ACTION_TYPE_MOVE_SL_FEAT = 2
ACTION_TYPE_SET_TP       = 3
ACTION_TYPE_ADD_POSITION = 4
ACTION_TYPE_SET_PHASE    = 5
ACTION_TYPE_INVALIDATE   = 6

# ── Distribution mode encoding ────────────────────────────────────────────────
DIST_MODE_LINEAR        = 0
DIST_MODE_EXPO          = 1
DIST_MODE_LOG           = 2
DIST_MODE_SQRT          = 3
DIST_MODE_EQUAL         = 4
DIST_MODE_CUSTOM_PTS    = 5
DIST_MODE_CALLABLE      = 6


# ══════════════════════════════════════════════════════════════════════════════
# EXITPROFILESPEC
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExitProfileSpec:
    name: str = ""

    # ── TP/SL ─────────────────────────────────────────────────────────────────
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    use_atr_sl_tp: Optional[int] = None      # {-1, 0, 1, 2}
    tp_atr_mult: Optional[float] = None
    sl_atr_mult: Optional[float] = None

    # ── Break-even ────────────────────────────────────────────────────────────
    be_trigger_pct: Optional[float] = None
    be_offset_pct: Optional[float] = None
    be_delay_bars: Optional[int] = None

    # ── Runner trailing ───────────────────────────────────────────────────────
    trailing_trigger_pct: Optional[float] = None
    runner_trailing_mult: Optional[float] = None

    # ── Max holding ───────────────────────────────────────────────────────────
    max_holding_bars: Optional[int] = None

    # ── Nouveau — Partial / Pyramid / Averaging / Phases ──────────────────────
    partial_config:   Optional[PartialConfig]   = None
    pyramid_config:   Optional[PyramidConfig]   = None
    averaging_config: Optional[AveragingConfig] = None
    phases:           list                      = field(default_factory=list)

    def __post_init__(self):
        if self.use_atr_sl_tp is not None and self.use_atr_sl_tp not in (-1, 0, 1, 2):
            raise ValueError("use_atr_sl_tp must be in {-1,0,1,2}")
        for attr in (
            "tp_pct", "sl_pct", "tp_atr_mult", "sl_atr_mult",
            "be_trigger_pct", "be_offset_pct",
            "trailing_trigger_pct", "runner_trailing_mult"
        ):
            val = getattr(self, attr)
            if val is not None and val < 0:
                raise ValueError(f"{attr} must be >= 0")
        if self.be_delay_bars is not None and self.be_delay_bars < 0:
            raise ValueError("be_delay_bars must be >= 0")
        if self.max_holding_bars is not None and self.max_holding_bars < 0:
            raise ValueError("max_holding_bars must be >= 0")

    @property
    def has_partial(self) -> bool:
        return self.partial_config is not None

    @property
    def has_pyramid(self) -> bool:
        return self.pyramid_config is not None

    @property
    def has_averaging(self) -> bool:
        return self.averaging_config is not None

    @property
    def has_phases(self) -> bool:
        return len(self.phases) > 0

    def get_all_rules(self) -> list:
        """Retourner toutes les PositionRule de toutes les phases."""
        rules = []
        for phase in self.phases:
            rules.extend(phase.rules)
        return rules


# ══════════════════════════════════════════════════════════════════════════════
# COMPILEDEXITPROFILE — inchangé
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class CompiledExitProfile:
    exit_profile_id: int
    name: str
    tp_pct: float
    sl_pct: float
    use_atr_sl_tp: int
    tp_atr_mult: float
    sl_atr_mult: float
    be_trigger_pct: float
    be_offset_pct: float
    be_delay_bars: int
    trailing_trigger_pct: float
    runner_trailing_mult: float
    max_holding_bars: int


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE COMPILATION
# ══════════════════════════════════════════════════════════════════════════════

def _encode_trigger(trigger, feature_name_to_idx: dict) -> tuple:
    """
    Encoder un trigger en (trigger_type, trigger_value, feat_idx, op_int).
    Retourne (-1, -1, -1, -1) si trigger est None.
    """
    if trigger is None:
        return TRIGGER_TYPE_NONE, -1.0, -1, -1

    if isinstance(trigger, OnRR):
        return TRIGGER_TYPE_RR, float(trigger.value), -1, -1

    if isinstance(trigger, OnMFEPct):
        return TRIGGER_TYPE_MFE_PCT, float(trigger.value), -1, -1

    if isinstance(trigger, OnMAEPct):
        return TRIGGER_TYPE_MAE_PCT, float(trigger.value), -1, -1

    if isinstance(trigger, OnATRMult):
        return TRIGGER_TYPE_ATR_MULT, float(trigger.value), -1, -1

    if isinstance(trigger, OnBars):
        return TRIGGER_TYPE_BARS, float(trigger.value), -1, -1

    if isinstance(trigger, OnBarsAfterLastTP):
        return TRIGGER_TYPE_BARS_ATP, float(trigger.value), -1, -1

    if isinstance(trigger, OnFeature):
        feat_idx = feature_name_to_idx.get(trigger.name, -1)
        # Si value est un str → c'est un nom de feature (feat2)
        if isinstance(trigger.value, str):
            feat2_idx = feature_name_to_idx.get(trigger.value, -1)
        else:
            feat2_idx = -1
        op_int = _OP_MAP.get(trigger.operator, OP_GT)
        # feat_idx → feat1, feat2_idx → stocké dans RT_RULE_TRIGGER_FEAT2 pour les rules
        trig_val = float(trigger.value) if isinstance(trigger.value, (int, float)) else float(feat2_idx)
        return TRIGGER_TYPE_FEATURE, trig_val, feat_idx, op_int

    if isinstance(trigger, OnPhase):
        return TRIGGER_TYPE_PHASE, float(trigger.value), -1, -1

    if isinstance(trigger, OnAll):
        # Pour OnAll/OnAny dans les configs simples → encoder le premier trigger
        # Les rules multi-triggers sont gérées différemment dans _encode_rule
        return TRIGGER_TYPE_ALL, -1.0, -1, -1

    if isinstance(trigger, OnAny):
        return TRIGGER_TYPE_ANY, -1.0, -1, -1

    return TRIGGER_TYPE_NONE, -1.0, -1, -1


def _encode_dist_fn(dist_fn: Optional[DistributionFn]) -> tuple:
    """Encoder DistributionFn en (mode_int, param1, param2)."""
    if dist_fn is None:
        return DIST_MODE_EQUAL, 0.5, 1.0
    mode_int, p1, p2 = dist_fn.to_rt_encoding()
    return mode_int, p1, p2


def _compile_partial_rt(
    specs: list[ExitProfileSpec],
    feature_name_to_idx: dict,
    max_levels: int,
) -> np.ndarray:
    """
    Compiler partial_config de tous les profils en matrice runtime.
    shape: (n_profiles, max_levels, N_PARTIAL_COLS)
    """
    n = len(specs)
    rt = np.full((n, max_levels, N_PARTIAL_COLS), -1.0, dtype=np.float64)

    for pid, spec in enumerate(specs):
        if spec.partial_config is None:
            continue
        pc = spec.partial_config
        fractions = pc.get_fractions()
        triggers = pc.get_spacings()

        for lvl in range(min(pc.n_levels, max_levels)):
            trig = triggers[lvl] if lvl < len(triggers) else None
            tt, tv, fi, op = _encode_trigger(trig, feature_name_to_idx)
            fraction = fractions[lvl] if lvl < len(fractions) else 0.5
            dist_mode, dp1, dp2 = _encode_dist_fn(pc.sizing)

            rt[pid, lvl, PART_TRIGGER_TYPE]  = float(tt)
            rt[pid, lvl, PART_TRIGGER_VALUE] = float(tv)
            rt[pid, lvl, PART_TRIGGER_FEAT]  = float(fi)
            rt[pid, lvl, PART_TRIGGER_OP]    = float(op)
            rt[pid, lvl, PART_FRACTION]      = float(fraction)
            rt[pid, lvl, PART_REF]           = 0.0 if pc.ref == "remaining" else 1.0
            rt[pid, lvl, PART_PRICE_MODE]    = 0.0  # market par défaut
            rt[pid, lvl, PART_MOVE_BE]       = 1.0 if pc.move_sl_to_be_after_first and lvl == 0 else 0.0
            rt[pid, lvl, PART_DIST_MODE]     = float(dist_mode)
            rt[pid, lvl, PART_DIST_PARAM1]   = float(dp1)
            rt[pid, lvl, PART_DIST_PARAM2]   = float(dp2)

    return rt


def _compile_pyramid_rt(
    specs: list[ExitProfileSpec],
    feature_name_to_idx: dict,
    max_levels: int,
) -> np.ndarray:
    """
    Compiler pyramid_config de tous les profils en matrice runtime.
    shape: (n_profiles, max_levels, N_PYRAMID_COLS)
    """
    n = len(specs)
    rt = np.full((n, max_levels, N_PYRAMID_COLS), -1.0, dtype=np.float64)

    sl_mode_map = {"breakeven": 0, "original": 1, "feature": 2, "atr_mult": 3}

    for pid, spec in enumerate(specs):
        if spec.pyramid_config is None:
            continue
        pc = spec.pyramid_config
        fractions = pc.get_fractions()
        triggers = pc.get_triggers()

        for lvl in range(min(pc.n_levels, max_levels)):
            trig = triggers[lvl] if lvl < len(triggers) else None
            tt, tv, fi, op = _encode_trigger(trig, feature_name_to_idx)
            fraction = fractions[lvl] if lvl < len(fractions) else 0.5
            dist_mode, dp1, dp2 = _encode_dist_fn(pc.sizing)
            sl_feat_idx = feature_name_to_idx.get(pc.sl_feature, -1) if pc.sl_feature else -1

            rt[pid, lvl, PYR_TRIGGER_TYPE]  = float(tt)
            rt[pid, lvl, PYR_TRIGGER_VALUE] = float(tv)
            rt[pid, lvl, PYR_TRIGGER_FEAT]  = float(fi)
            rt[pid, lvl, PYR_TRIGGER_OP]    = float(op)
            rt[pid, lvl, PYR_SIZE_FRACTION] = float(fraction)
            rt[pid, lvl, PYR_SIZE_REF]      = 0.0  # original
            rt[pid, lvl, PYR_SL_MODE]       = float(sl_mode_map.get(pc.sl_mode, 0))
            rt[pid, lvl, PYR_SL_FEAT]       = float(sl_feat_idx)
            rt[pid, lvl, PYR_SL_ATR_MULT]   = float(pc.sl_atr_mult)
            rt[pid, lvl, PYR_GROUP_SL_MODE] = float(pc.group_sl_mode)
            rt[pid, lvl, PYR_DIST_MODE]     = float(dist_mode)
            rt[pid, lvl, PYR_DIST_PARAM1]   = float(dp1)
            rt[pid, lvl, PYR_DIST_PARAM2]   = float(dp2)

    return rt


def _compile_averaging_rt(
    specs: list[ExitProfileSpec],
    feature_name_to_idx: dict,
    max_levels: int,
) -> np.ndarray:
    """
    Compiler averaging_config de tous les profils en matrice runtime.
    shape: (n_profiles, max_levels, N_AVG_COLS)
    """
    n = len(specs)
    rt = np.full((n, max_levels, N_AVG_COLS), -1.0, dtype=np.float64)

    sl_mode_map = {"original": 0, "breakeven": 1}
    tp_mode_map = {"same": 0}

    for pid, spec in enumerate(specs):
        if spec.averaging_config is None:
            continue
        ac = spec.averaging_config
        fractions = ac.get_fractions()
        triggers = ac.get_triggers()

        for lvl in range(min(ac.n_levels, max_levels)):
            trig = triggers[lvl] if lvl < len(triggers) else None
            tt, tv, fi, op = _encode_trigger(trig, feature_name_to_idx)
            fraction = fractions[lvl] if lvl < len(fractions) else 0.5
            dist_mode, dp1, dp2 = _encode_dist_fn(ac.sizing)

            rt[pid, lvl, AVG_TRIGGER_TYPE]  = float(tt)
            rt[pid, lvl, AVG_TRIGGER_VALUE] = float(tv)
            rt[pid, lvl, AVG_TRIGGER_FEAT]  = float(fi)
            rt[pid, lvl, AVG_TRIGGER_OP]    = float(op)
            rt[pid, lvl, AVG_SIZE_FRACTION] = float(fraction)
            rt[pid, lvl, AVG_SL_MODE]       = float(sl_mode_map.get(ac.sl_mode, 0))
            rt[pid, lvl, AVG_TP_MODE]       = float(tp_mode_map.get(ac.tp_mode, 0))
            rt[pid, lvl, AVG_MAX_DOWN]      = float(ac.max_avg_down_pct)
            rt[pid, lvl, AVG_DIST_MODE]     = float(dist_mode)
            rt[pid, lvl, AVG_DIST_PARAM1]   = float(dp1)
            rt[pid, lvl, AVG_DIST_PARAM2]   = float(dp2)

    return rt


def _compile_phase_rt(
    specs: list[ExitProfileSpec],
    max_phases: int,
) -> np.ndarray:
    """
    Compiler les PhaseSpec de tous les profils en matrice runtime.
    shape: (n_profiles, max_phases, N_PHASE_COLS)
    """
    n = len(specs)
    rt = np.full((n, max_phases, N_PHASE_COLS), -1.0, dtype=np.float64)

    for pid, spec in enumerate(specs):
        for phase_spec in spec.phases:
            ph = phase_spec.phase
            if ph >= max_phases:
                continue
            rt[pid, ph, PHASE_ID]       = float(ph)
            rt[pid, ph, PHASE_TP_PCT]   = float(phase_spec.tp_pct) if phase_spec.tp_pct is not None else -1.0
            rt[pid, ph, PHASE_SL_PCT]   = float(phase_spec.sl_pct) if phase_spec.sl_pct is not None else -1.0
            rt[pid, ph, PHASE_BE_TRIGGER]  = float(phase_spec.be_trigger_pct) if phase_spec.be_trigger_pct is not None else -1.0
            rt[pid, ph, PHASE_TRAILING] = float(phase_spec.trailing_trigger_pct) if phase_spec.trailing_trigger_pct is not None else -1.0
            rt[pid, ph, PHASE_MAX_HOLD] = float(phase_spec.max_holding_bars) if phase_spec.max_holding_bars is not None else -1.0

    return rt


def _compile_rules_rt(
    specs: list[ExitProfileSpec],
    feature_name_to_idx: dict,
    max_rules: int,
    max_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compiler les PositionRule de tous les profils en matrices runtime.
    Returns: (rule_trigger_matrix, rule_action_matrix)
        rule_trigger_matrix: (n_profiles, max_rules, N_RULE_TRIGGER_COLS)
        rule_action_matrix:  (n_profiles, max_rules, max_actions, N_RULE_ACTION_COLS)
    """
    n = len(specs)
    trig_rt   = np.full((n, max_rules, N_RULE_TRIGGER_COLS), -1.0, dtype=np.float64)
    action_rt = np.full((n, max_rules, max_actions, N_RULE_ACTION_COLS), -1.0, dtype=np.float64)

    sl_mode_map = {"breakeven": 0, "original": 1, "feature": 2, "atr_mult": 3}
    tp_mode_map = {"same": 0, "feature": 1, "rr": 2}

    for pid, spec in enumerate(specs):
        rules = spec.get_all_rules()
        for rid, rule in enumerate(rules[:max_rules]):
            # ── Trigger ──────────────────────────────────────────────────────
            trig_rt[pid, rid, RT_RULE_PHASE_FILTER] = float(rule.phase_filter)
            trig_rt[pid, rid, RT_RULE_MAX_TIMES]    = float(rule.max_times)

            tt, tv, fi, op = _encode_trigger(rule.trigger, feature_name_to_idx)
            trig_rt[pid, rid, RT_RULE_TRIGGER_TYPE]  = float(tt)
            trig_rt[pid, rid, RT_RULE_TRIGGER_VALUE] = float(tv)
            trig_rt[pid, rid, RT_RULE_TRIGGER_FEAT1] = float(fi)
            trig_rt[pid, rid, RT_RULE_TRIGGER_OP]    = float(op)

            # Pour OnFeature avec cross — feat2 dans FEAT2
            if isinstance(rule.trigger, OnFeature) and isinstance(rule.trigger.value, str):
                feat2 = feature_name_to_idx.get(rule.trigger.value, -1)
                trig_rt[pid, rid, RT_RULE_TRIGGER_FEAT2] = float(feat2)

            n_actions = min(len(rule.actions), max_actions)
            trig_rt[pid, rid, RT_RULE_N_ACTIONS] = float(n_actions)

            # ── Actions ───────────────────────────────────────────────────────
            for aid, action in enumerate(rule.actions[:max_actions]):
                if isinstance(action, ExitPartial):
                    action_rt[pid, rid, aid, RA_ACTION_TYPE] = float(ACTION_TYPE_EXIT_PARTIAL)
                    action_rt[pid, rid, aid, RA_PARAM1]      = float(action.fraction)
                    action_rt[pid, rid, aid, RA_PARAM2]      = 0.0 if action.ref == "remaining" else 1.0
                    action_rt[pid, rid, aid, RA_PARAM3]      = 0.0  # market

                elif isinstance(action, MoveSLtoBE):
                    action_rt[pid, rid, aid, RA_ACTION_TYPE] = float(ACTION_TYPE_MOVE_SL_BE)
                    action_rt[pid, rid, aid, RA_PARAM1]      = float(action.offset_pct)

                elif isinstance(action, MoveSLto):
                    feat_idx = feature_name_to_idx.get(action.feature, -1)
                    action_rt[pid, rid, aid, RA_ACTION_TYPE] = float(ACTION_TYPE_MOVE_SL_FEAT)
                    action_rt[pid, rid, aid, RA_PARAM1]      = float(action.offset_pct)
                    action_rt[pid, rid, aid, RA_FEAT_IDX]    = float(feat_idx)

                elif isinstance(action, SetTP):
                    feat_idx = feature_name_to_idx.get(action.feature, -1) if action.feature else -1
                    action_rt[pid, rid, aid, RA_ACTION_TYPE] = float(ACTION_TYPE_SET_TP)
                    action_rt[pid, rid, aid, RA_PARAM1]      = float(action.rr)
                    action_rt[pid, rid, aid, RA_PARAM2]      = float(action.atr_mult)
                    action_rt[pid, rid, aid, RA_FEAT_IDX]    = float(feat_idx)

                elif isinstance(action, AddPosition):
                    feat_sl = feature_name_to_idx.get(action.sl_feature, -1) if action.sl_feature else -1
                    feat_tp = feature_name_to_idx.get(action.tp_feature, -1) if action.tp_feature else -1
                    action_rt[pid, rid, aid, RA_ACTION_TYPE] = float(ACTION_TYPE_ADD_POSITION)
                    action_rt[pid, rid, aid, RA_PARAM1]      = float(action.size_fraction)
                    action_rt[pid, rid, aid, RA_PARAM2]      = float(sl_mode_map.get(action.sl, 0))
                    action_rt[pid, rid, aid, RA_PARAM3]      = float(action.group_sl_mode)
                    action_rt[pid, rid, aid, RA_FEAT_IDX]    = float(feat_sl)

                elif isinstance(action, SetPhase):
                    action_rt[pid, rid, aid, RA_ACTION_TYPE] = float(ACTION_TYPE_SET_PHASE)
                    action_rt[pid, rid, aid, RA_PARAM1]      = float(action.value)

                elif isinstance(action, Invalidate):
                    action_rt[pid, rid, aid, RA_ACTION_TYPE] = float(ACTION_TYPE_INVALIDATE)

    return trig_rt, action_rt


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE COMPILATION PRINCIPALES — inchangées + nouvelles
# ══════════════════════════════════════════════════════════════════════════════

def compile_exit_profiles(
    profile_specs: list[ExitProfileSpec],
    cfg: BacktestConfig,
) -> list[CompiledExitProfile]:
    compiled = []
    for pid, spec in enumerate(profile_specs):
        compiled.append(
            CompiledExitProfile(
                exit_profile_id=pid,
                name=spec.name or f"exit_profile_{pid}",
                tp_pct=cfg.tp_pct if spec.tp_pct is None else spec.tp_pct,
                sl_pct=cfg.sl_pct if spec.sl_pct is None else spec.sl_pct,
                use_atr_sl_tp=cfg.use_atr_sl_tp if spec.use_atr_sl_tp is None else spec.use_atr_sl_tp,
                tp_atr_mult=cfg.tp_atr_mult if spec.tp_atr_mult is None else spec.tp_atr_mult,
                sl_atr_mult=cfg.sl_atr_mult if spec.sl_atr_mult is None else spec.sl_atr_mult,
                be_trigger_pct=cfg.be_trigger_pct if spec.be_trigger_pct is None else spec.be_trigger_pct,
                be_offset_pct=cfg.be_offset_pct if spec.be_offset_pct is None else spec.be_offset_pct,
                be_delay_bars=cfg.be_delay_bars if spec.be_delay_bars is None else spec.be_delay_bars,
                trailing_trigger_pct=cfg.trailing_trigger_pct if spec.trailing_trigger_pct is None else spec.trailing_trigger_pct,
                runner_trailing_mult=cfg.runner_trailing_mult if spec.runner_trailing_mult is None else spec.runner_trailing_mult,
                max_holding_bars=cfg.max_holding_bars if spec.max_holding_bars is None else spec.max_holding_bars,
            )
        )
    return compiled


def compile_setup_exit_binding(
    setup_exit_binding,
    strategy_profile_binding,
    n_setups,
    n_strategies=None,
):
    # ── inchangé ──────────────────────────────────────────────────────────────
    setup_to_exit_profile  = np.full(n_setups, -1, dtype=np.int32)
    setup_to_exit_strategy = np.full(n_setups, -1, dtype=np.int32)

    if setup_exit_binding is not None:
        for setup_id, cfg in setup_exit_binding.items():
            if not isinstance(setup_id, int):
                raise TypeError(f"setup_id must be int, got {type(setup_id)}")
            if setup_id < 0 or setup_id >= n_setups:
                raise ValueError(f"setup_id {setup_id} out of bounds for n_setups={n_setups}")
            setup_to_exit_profile[setup_id]  = int(cfg.get("exit_profile_id", -1))
            setup_to_exit_strategy[setup_id] = int(cfg.get("exit_strategy_id", -1))

    if strategy_profile_binding is None:
        strategy_profile_binding = {}

    normalized = {}
    for sid, cfg in strategy_profile_binding.items():
        if isinstance(cfg, dict):
            normalized[sid] = cfg
        elif isinstance(cfg, (list, tuple)):
            allowed = [int(x) for x in cfg]
            normalized[sid] = {
                "default_profile_id": allowed[0] if allowed else -1,
                "allowed_profile_ids": allowed,
            }
        else:
            raise TypeError(f"Binding for strategy_id={sid} must be dict/list/tuple")

    if n_strategies is None:
        n_strategies = (max(normalized.keys()) + 1) if normalized else 0

    strategy_to_default_profile = np.full(n_strategies, -1, dtype=np.int32)
    max_allowed = max((len(v.get("allowed_profile_ids", [])) for v in normalized.values()), default=0)

    if n_strategies > 0 and max_allowed > 0:
        strategy_allowed_profiles = np.full((n_strategies, max_allowed), -1, dtype=np.int32)
        strategy_allowed_counts   = np.zeros(n_strategies, dtype=np.int32)
    else:
        strategy_allowed_profiles = np.full((max(n_strategies, 1), max(max_allowed, 1)), -1, dtype=np.int32)
        strategy_allowed_counts   = np.zeros(max(n_strategies, 1), dtype=np.int32)

    for sid, cfg in normalized.items():
        if sid >= n_strategies:
            raise ValueError(f"strategy_id {sid} out of bounds")
        strategy_to_default_profile[sid] = int(cfg.get("default_profile_id", -1))
        allowed = cfg.get("allowed_profile_ids", [])
        strategy_allowed_counts[sid] = len(allowed)
        for j, pid in enumerate(allowed):
            strategy_allowed_profiles[sid, j] = int(pid)

    return {
        "setup_to_exit_profile":      setup_to_exit_profile,
        "setup_to_exit_strategy":     setup_to_exit_strategy,
        "strategy_to_default_profile": strategy_to_default_profile,
        "strategy_allowed_profiles":  strategy_allowed_profiles,
        "strategy_allowed_counts":    strategy_allowed_counts,
    }


def build_exit_profile_rt_matrix(compiled_profiles: list[CompiledExitProfile]) -> np.ndarray:
    rt = np.zeros((len(compiled_profiles), N_EXIT_RT_COLS), dtype=np.float64)
    for p in compiled_profiles:
        pid = p.exit_profile_id
        rt[pid, RT_EXIT_PROFILE_ID]      = p.exit_profile_id
        rt[pid, RT_EXIT_STRATEGY_ID]     = -1
        rt[pid, RT_TP_PCT]               = p.tp_pct
        rt[pid, RT_SL_PCT]               = p.sl_pct
        rt[pid, RT_USE_ATR_SL_TP]        = p.use_atr_sl_tp
        rt[pid, RT_TP_ATR_MULT]          = p.tp_atr_mult
        rt[pid, RT_SL_ATR_MULT]          = p.sl_atr_mult
        rt[pid, RT_BE_TRIGGER_PCT]       = p.be_trigger_pct
        rt[pid, RT_BE_OFFSET_PCT]        = p.be_offset_pct
        rt[pid, RT_BE_DELAY_BARS]        = p.be_delay_bars
        rt[pid, RT_TRAILING_TRIGGER_PCT] = p.trailing_trigger_pct
        rt[pid, RT_RUNNER_TRAILING_MULT] = p.runner_trailing_mult
        rt[pid, RT_MAX_HOLDING_BARS]     = p.max_holding_bars
    return rt


def build_position_rule_matrices(
    profile_specs: list[ExitProfileSpec],
    feature_name_to_idx: dict,
) -> dict:
    """
    Compiler toutes les matrices runtime pour partial, pyramid, averaging,
    phases et rules de tous les profils.

    Retourne un dict avec :
        partial_rt_matrix, pyramid_rt_matrix, averaging_rt_matrix,
        phase_rt_matrix, rule_trigger_matrix, rule_action_matrix,
        max_partial_levels, max_pyramid_levels, max_avg_levels,
        max_phases, max_rules, max_actions_per_rule
    """
    # Calculer les maxima
    max_partial  = max((s.partial_config.n_levels  if s.partial_config  else 0 for s in profile_specs), default=0)
    max_pyramid  = max((s.pyramid_config.n_levels  if s.pyramid_config  else 0 for s in profile_specs), default=0)
    max_avg      = max((s.averaging_config.n_levels if s.averaging_config else 0 for s in profile_specs), default=0)
    max_phases   = max((max((p.phase for p in s.phases), default=-1) + 1 for s in profile_specs), default=0)
    max_rules    = max((len(s.get_all_rules()) for s in profile_specs), default=0)
    max_actions  = max(
        (max((len(r.actions) for r in s.get_all_rules()), default=0) for s in profile_specs),
        default=0
    )

    # Garantir au moins 1 pour éviter les matrices vides
    max_partial = max(max_partial, 1)
    max_pyramid = max(max_pyramid, 1)
    max_avg     = max(max_avg, 1)
    max_phases  = max(max_phases, 1)
    max_rules   = max(max_rules, 1)
    max_actions = max(max_actions, 1)

    partial_rt  = _compile_partial_rt(profile_specs, feature_name_to_idx, max_partial)
    pyramid_rt  = _compile_pyramid_rt(profile_specs, feature_name_to_idx, max_pyramid)
    avg_rt      = _compile_averaging_rt(profile_specs, feature_name_to_idx, max_avg)
    phase_rt    = _compile_phase_rt(profile_specs, max_phases)
    trig_rt, act_rt = _compile_rules_rt(profile_specs, feature_name_to_idx, max_rules, max_actions)

    return {
        "partial_rt_matrix":    partial_rt,
        "pyramid_rt_matrix":    pyramid_rt,
        "averaging_rt_matrix":  avg_rt,
        "phase_rt_matrix":      phase_rt,
        "rule_trigger_matrix":  trig_rt,
        "rule_action_matrix":   act_rt,
        "max_partial_levels":   max_partial,
        "max_pyramid_levels":   max_pyramid,
        "max_avg_levels":       max_avg,
        "max_phases":           max_phases,
        "max_rules":            max_rules,
        "max_actions_per_rule": max_actions,
    }