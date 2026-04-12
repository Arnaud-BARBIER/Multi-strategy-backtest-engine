# rule_compiler.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path


# ==========================================================
# CONDITIONS
# ==========================================================

@dataclass
class _Condition:
    pass

@dataclass
class FeatGtParam(_Condition):
    feat: str
    param: str

@dataclass
class FeatLtParam(_Condition):
    feat: str
    param: str

@dataclass
class FeatGtFeat(_Condition):
    feat1: str
    feat2: str

@dataclass
class FeatLtFeat(_Condition):
    feat1: str
    feat2: str

@dataclass
class FeatGtVal(_Condition):
    feat: str
    val: float

@dataclass
class FeatLtVal(_Condition):
    feat: str
    val: float

@dataclass
class CrossOver(_Condition):
    """feat1 croise au-dessus de feat2"""
    feat1: str
    feat2: str

@dataclass
class CrossUnder(_Condition):
    """feat1 croise en-dessous de feat2"""
    feat1: str
    feat2: str

@dataclass
class PosGtParam(_Condition):
    """Condition sur la position : side, bars_in_trade, mae, mfe"""
    field: str   # "bars_in_trade", "mae", "mfe"
    param: str

@dataclass
class PosLtParam(_Condition):
    field: str
    param: str

@dataclass
class PosGtVal(_Condition):
    field: str
    val: float

@dataclass
class PosLtVal(_Condition):
    field: str
    val: float

@dataclass
class SideIs(_Condition):
    """side == 1.0 (long) ou -1.0 (short)"""
    side: float

@dataclass
class BarGtFeat(_Condition):
    """close/high/low > feat"""
    bar_field: str   # "close", "high", "low", "open"
    feat: str

@dataclass
class BarLtFeat(_Condition):
    bar_field: str
    feat: str

@dataclass
class AND(_Condition):
    conditions: list[_Condition]

@dataclass
class OR(_Condition):
    conditions: list[_Condition]

# ==========================================================
# CONDITIONS CALCULÉES (window required)
# ==========================================================

@dataclass
class Slope(_Condition):
    """Pente de feat sur N barres > 0 (montante) ou < 0 (descendante)"""
    feat: str
    bars: int
    direction: str = "up"   # "up" ou "down"

@dataclass
class Mean(_Condition):
    """feat courante > ou < sa moyenne sur N barres"""
    feat: str
    bars: int
    operator: str = "gt"    # "gt" ou "lt"

@dataclass
class StdDev(_Condition):
    """feat courante > ou < N écarts-types de sa moyenne"""
    feat: str
    bars: int
    n_std: float = 1.0
    operator: str = "gt"

@dataclass
class AboveMA(_Condition):
    """feat courante > sa propre moyenne mobile sur N barres"""
    feat: str
    bars: int

@dataclass
class BelowMA(_Condition):
    """feat courante < sa propre moyenne mobile sur N barres"""
    feat: str
    bars: int

@dataclass
class SlopeGtParam(_Condition):
    """pente de feat sur N barres > param"""
    feat: str
    bars: int
    param: str

@dataclass
class SlopeLtParam(_Condition):
    """pente de feat sur N barres < param"""
    feat: str
    bars: int
    param: str

# ==========================================================
# HELPERS PUBLICS POUR ECRIRE LES CONDITIONS
# ==========================================================

def feat(name: str):
    """Référence à une feature par nom."""
    return name

def param(name: str):
    """Référence à un param par nom."""
    return name

def IF(condition: _Condition, action) -> tuple:
    return (condition, action)


# ==========================================================
# ACTIONS
# ==========================================================

def FORCE_EXIT(reason_param: str = "p2") -> dict:
    return {"type": "force_exit", "reason_param": reason_param}

def SWITCH_PROFILE(profile_param: str) -> dict:
    return {"type": "switch_profile", "profile_param": profile_param}

def OVERWRITE_TP_SL(tp_param: str | None = None, sl_param: str | None = None) -> dict:
    return {"type": "overwrite", "tp_param": tp_param, "sl_param": sl_param}


# ==========================================================
# SPEC DE REGLE
# ==========================================================
def _get_max_bars_needed(spec: ExitRuleSpec) -> int:
    max_bars = 0
    def _scan(cond):
        nonlocal max_bars
        if hasattr(cond, "bars"):
            max_bars = max(max_bars, cond.bars)
        if isinstance(cond, (AND, OR)):
            for c in cond.conditions:
                _scan(c)
        if isinstance(cond, (CrossOver, CrossUnder)):
            max_bars = max(max_bars, 2)
    for rule in spec.rules:
        _scan(rule[0])
    return max_bars

@dataclass
class ExitRuleSpec:
    strategy_id: int
    name: str
    rules: list[tuple]           # liste de IF(condition, action)
    feature_names: list[str]
    params: dict[str, Any]
    window_bars: int = 0         # 0 = instant, >0 = window
    side_aware: bool = False     # True = logique miroir long/short

# ------------ Validateur ------------------------------------

def _get_max_bars_needed(spec: ExitRuleSpec) -> int:
    max_bars = 0
    def _scan(cond):
        nonlocal max_bars
        if hasattr(cond, "bars"):
            max_bars = max(max_bars, cond.bars)
        if isinstance(cond, (AND, OR)):
            for c in cond.conditions:
                _scan(c)
        if isinstance(cond, (CrossOver, CrossUnder)):
            max_bars = max(max_bars, 2)
    for rule in spec.rules:
        _scan(rule[0])
    return max_bars

# ==========================================================
# GENERATEUR DE CODE NUMBA
# ==========================================================

class _CodeGen:
    def __init__(self):
        self._lines: list[str] = []
        self._indent = 0

    def indent(self): self._indent += 1
    def dedent(self): self._indent -= 1

    def line(self, s: str = ""):
        if s:
            self._lines.append("    " * self._indent + s)
        else:
            self._lines.append("")

    def render(self) -> str:
        return "\n".join(self._lines)


def _feat_col_const(idx: int) -> str:
    return f"STRAT_FEAT_COL_{idx + 1}"

def _param_const(idx: int) -> str:
    return f"STRAT_PARAM_{idx + 1}"

def _resolve_feat_var(feat_name: str, feat_names: list[str], g: _CodeGen,
                      declared: set, i_var: str = "i") -> str:
    idx = feat_names.index(feat_name)
    var = f"_f{idx}"
    col_var = f"_col{idx}"
    if var not in declared:
        g.line(f"{col_var} = int(strategy_rt_matrix[strategy_id, {_feat_col_const(idx)}])")
        g.line(f"if {col_var} < 0: return action")
        g.line(f"{var} = features[{i_var}, {col_var}]")
        g.line(f"if np.isnan({var}): return action")
        declared.add(var)
    return var

def _resolve_feat_prev_var(feat_name: str, feat_names: list[str], g: _CodeGen,
                           declared: set) -> str:
    idx = feat_names.index(feat_name)
    var = f"_f{idx}_prev"
    col_var = f"_col{idx}"
    if var not in declared:
        if f"_f{idx}" not in declared:
            g.line(f"{col_var} = int(strategy_rt_matrix[strategy_id, {_feat_col_const(idx)}])")
            g.line(f"if {col_var} < 0: return action")
            declared.add(f"_f{idx}")
        g.line(f"{var} = features[i - 1, {col_var}]")
        g.line(f"if np.isnan({var}): return action")
        declared.add(var)
    return var

def _resolve_param_var(param_name: str, param_names: list[str],
                       g: _CodeGen, declared: set) -> str:
    idx = param_names.index(param_name)
    var = f"_p{idx + 1}"
    if var not in declared:
        g.line(f"{var} = strategy_rt_matrix[strategy_id, {_param_const(idx)}]")
        declared.add(var)
    return var

def _resolve_pos_field(field: str) -> str:
    mapping = {
        "bars_in_trade": "(i - int(pos[k, POS_ENTRY_IDX]))",
        "mae":           "pos[k, POS_MAE]",
        "mfe":           "pos[k, POS_MFE]",
        "side":          "pos[k, POS_SIDE]",
    }
    if field not in mapping:
        raise ValueError(f"Unknown pos field: {field}. Use: {list(mapping.keys())}")
    return mapping[field]

def _resolve_bar_field(field: str) -> str:
    mapping = {
        "open":  "opens[i]",
        "high":  "highs[i]",
        "low":   "lows[i]",
        "close": "closes[i]",
    }
    if field not in mapping:
        raise ValueError(f"Unknown bar field: {field}. Use: {list(mapping.keys())}")
    return mapping[field]

def _gen_condition(cond: _Condition, feat_names: list[str], param_names: list[str],
                   g: _CodeGen, declared: set) -> str:
    """Génère le code de la condition et retourne l'expression booléenne."""

    if isinstance(cond, FeatGtParam):
        fv = _resolve_feat_var(cond.feat, feat_names, g, declared)
        pv = _resolve_param_var(cond.param, param_names, g, declared)
        return f"{fv} > {pv}"

    elif isinstance(cond, FeatLtParam):
        fv = _resolve_feat_var(cond.feat, feat_names, g, declared)
        pv = _resolve_param_var(cond.param, param_names, g, declared)
        return f"{fv} < {pv}"

    elif isinstance(cond, FeatGtFeat):
        fv1 = _resolve_feat_var(cond.feat1, feat_names, g, declared)
        fv2 = _resolve_feat_var(cond.feat2, feat_names, g, declared)
        return f"{fv1} > {fv2}"

    elif isinstance(cond, FeatLtFeat):
        fv1 = _resolve_feat_var(cond.feat1, feat_names, g, declared)
        fv2 = _resolve_feat_var(cond.feat2, feat_names, g, declared)
        return f"{fv1} < {fv2}"

    elif isinstance(cond, FeatGtVal):
        fv = _resolve_feat_var(cond.feat, feat_names, g, declared)
        return f"{fv} > {cond.val}"

    elif isinstance(cond, FeatLtVal):
        fv = _resolve_feat_var(cond.feat, feat_names, g, declared)
        return f"{fv} < {cond.val}"

    elif isinstance(cond, CrossOver):
        fv1      = _resolve_feat_var(cond.feat1, feat_names, g, declared)
        fv2      = _resolve_feat_var(cond.feat2, feat_names, g, declared)
        fv1_prev = _resolve_feat_prev_var(cond.feat1, feat_names, g, declared)
        fv2_prev = _resolve_feat_prev_var(cond.feat2, feat_names, g, declared)
        return f"({fv1_prev} < {fv2_prev} and {fv1} > {fv2})"

    elif isinstance(cond, CrossUnder):
        fv1      = _resolve_feat_var(cond.feat1, feat_names, g, declared)
        fv2      = _resolve_feat_var(cond.feat2, feat_names, g, declared)
        fv1_prev = _resolve_feat_prev_var(cond.feat1, feat_names, g, declared)
        fv2_prev = _resolve_feat_prev_var(cond.feat2, feat_names, g, declared)
        return f"({fv1_prev} > {fv2_prev} and {fv1} < {fv2})"

    elif isinstance(cond, PosGtParam):
        pf = _resolve_pos_field(cond.field)
        pv = _resolve_param_var(cond.param, param_names, g, declared)
        return f"{pf} > {pv}"

    elif isinstance(cond, PosLtParam):
        pf = _resolve_pos_field(cond.field)
        pv = _resolve_param_var(cond.param, param_names, g, declared)
        return f"{pf} < {pv}"

    elif isinstance(cond, PosGtVal):
        pf = _resolve_pos_field(cond.field)
        return f"{pf} > {cond.val}"

    elif isinstance(cond, PosLtVal):
        pf = _resolve_pos_field(cond.field)
        return f"{pf} < {cond.val}"

    elif isinstance(cond, SideIs):
        return f"pos[k, POS_SIDE] == {cond.side}"

    elif isinstance(cond, BarGtFeat):
        bf = _resolve_bar_field(cond.bar_field)
        fv = _resolve_feat_var(cond.feat, feat_names, g, declared)
        return f"{bf} > {fv}"

    elif isinstance(cond, BarLtFeat):
        bf = _resolve_bar_field(cond.bar_field)
        fv = _resolve_feat_var(cond.feat, feat_names, g, declared)
        return f"{bf} < {fv}"

    elif isinstance(cond, AND):
        parts = [_gen_condition(c, feat_names, param_names, g, declared) for c in cond.conditions]
        return "(" + " and ".join(parts) + ")"

    elif isinstance(cond, OR):
        parts = [_gen_condition(c, feat_names, param_names, g, declared) for c in cond.conditions]
        return "(" + " or ".join(parts) + ")"
    

    elif isinstance(cond, Slope):
        idx = feat_names.index(cond.feat)
        col_var = f"_col{idx}"
        var = f"_slope_{idx}_{cond.bars}"
        if var not in declared:
            if col_var not in declared:
                g.line(f"{col_var} = int(strategy_rt_matrix[strategy_id, {_feat_col_const(idx)}])")
                g.line(f"if {col_var} < 0: return action")
                declared.add(col_var)
            g.line(f"_slope_sum_{idx} = 0.0")
            g.line(f"for _si in range(1, {cond.bars}):")
            g.indent()
            g.line(f"_slope_sum_{idx} += features[i - _si + 1, {col_var}] - features[i - _si, {col_var}]")
            g.dedent()
            g.line(f"{var} = _slope_sum_{idx} / {cond.bars - 1}.0")
            declared.add(var)
        if cond.direction == "up":
            return f"{var} > 0.0"
        else:
            return f"{var} < 0.0"

    elif isinstance(cond, SlopeGtParam):
        idx = feat_names.index(cond.feat)
        col_var = f"_col{idx}"
        var = f"_slope_{idx}_{cond.bars}"
        pv = _resolve_param_var(cond.param, param_names, g, declared)
        if var not in declared:
            if col_var not in declared:
                g.line(f"{col_var} = int(strategy_rt_matrix[strategy_id, {_feat_col_const(idx)}])")
                g.line(f"if {col_var} < 0: return action")
                declared.add(col_var)
            g.line(f"_slope_sum_{idx} = 0.0")
            g.line(f"for _si in range(1, {cond.bars}):")
            g.indent()
            g.line(f"_slope_sum_{idx} += features[i - _si + 1, {col_var}] - features[i - _si, {col_var}]")
            g.dedent()
            g.line(f"{var} = _slope_sum_{idx} / {cond.bars - 1}.0")
            declared.add(var)
        return f"{var} > {pv}"

    elif isinstance(cond, SlopeLtParam):
        idx = feat_names.index(cond.feat)
        col_var = f"_col{idx}"
        var = f"_slope_{idx}_{cond.bars}"
        pv = _resolve_param_var(cond.param, param_names, g, declared)
        if var not in declared:
            if col_var not in declared:
                g.line(f"{col_var} = int(strategy_rt_matrix[strategy_id, {_feat_col_const(idx)}])")
                g.line(f"if {col_var} < 0: return action")
                declared.add(col_var)
            g.line(f"_slope_sum_{idx} = 0.0")
            g.line(f"for _si in range(1, {cond.bars}):")
            g.indent()
            g.line(f"_slope_sum_{idx} += features[i - _si + 1, {col_var}] - features[i - _si, {col_var}]")
            g.dedent()
            g.line(f"{var} = _slope_sum_{idx} / {cond.bars - 1}.0")
            declared.add(var)
        return f"{var} < {pv}"

    elif isinstance(cond, Mean):
        idx = feat_names.index(cond.feat)
        col_var = f"_col{idx}"
        var_now = f"_f{idx}"
        var_mean = f"_mean_{idx}_{cond.bars}"
        if var_mean not in declared:
            if col_var not in declared:
                g.line(f"{col_var} = int(strategy_rt_matrix[strategy_id, {_feat_col_const(idx)}])")
                g.line(f"if {col_var} < 0: return action")
                declared.add(col_var)
            if var_now not in declared:
                g.line(f"{var_now} = features[i, {col_var}]")
                g.line(f"if np.isnan({var_now}): return action")
                declared.add(var_now)
            g.line(f"_mean_sum_{idx} = 0.0")
            g.line(f"for _mi in range({cond.bars}):")
            g.indent()
            g.line(f"_mean_sum_{idx} += features[i - _mi, {col_var}]")
            g.dedent()
            g.line(f"{var_mean} = _mean_sum_{idx} / {cond.bars}.0")
            declared.add(var_mean)
        if cond.operator == "gt":
            return f"{var_now} > {var_mean}"
        else:
            return f"{var_now} < {var_mean}"

    elif isinstance(cond, AboveMA):
        idx = feat_names.index(cond.feat)
        col_var = f"_col{idx}"
        var_now = f"_f{idx}"
        var_mean = f"_mean_{idx}_{cond.bars}"
        if var_mean not in declared:
            if col_var not in declared:
                g.line(f"{col_var} = int(strategy_rt_matrix[strategy_id, {_feat_col_const(idx)}])")
                g.line(f"if {col_var} < 0: return action")
                declared.add(col_var)
            if var_now not in declared:
                g.line(f"{var_now} = features[i, {col_var}]")
                g.line(f"if np.isnan({var_now}): return action")
                declared.add(var_now)
            g.line(f"_mean_sum_{idx} = 0.0")
            g.line(f"for _mi in range({cond.bars}):")
            g.indent()
            g.line(f"_mean_sum_{idx} += features[i - _mi, {col_var}]")
            g.dedent()
            g.line(f"{var_mean} = _mean_sum_{idx} / {cond.bars}.0")
            declared.add(var_mean)
        return f"{var_now} > {var_mean}"

    elif isinstance(cond, BelowMA):
        idx = feat_names.index(cond.feat)
        col_var = f"_col{idx}"
        var_now = f"_f{idx}"
        var_mean = f"_mean_{idx}_{cond.bars}"
        if var_mean not in declared:
            if col_var not in declared:
                g.line(f"{col_var} = int(strategy_rt_matrix[strategy_id, {_feat_col_const(idx)}])")
                g.line(f"if {col_var} < 0: return action")
                declared.add(col_var)
            if var_now not in declared:
                g.line(f"{var_now} = features[i, {col_var}]")
                g.line(f"if np.isnan({var_now}): return action")
                declared.add(var_now)
            g.line(f"_mean_sum_{idx} = 0.0")
            g.line(f"for _mi in range({cond.bars}):")
            g.indent()
            g.line(f"_mean_sum_{idx} += features[i - _mi, {col_var}]")
            g.dedent()
            g.line(f"{var_mean} = _mean_sum_{idx} / {cond.bars}.0")
            declared.add(var_mean)
        return f"{var_now} < {var_mean}"

    elif isinstance(cond, StdDev):
        idx = feat_names.index(cond.feat)
        col_var = f"_col{idx}"
        var_now = f"_f{idx}"
        var_mean = f"_mean_{idx}_{cond.bars}"
        var_std = f"_std_{idx}_{cond.bars}"
        if var_std not in declared:
            if col_var not in declared:
                g.line(f"{col_var} = int(strategy_rt_matrix[strategy_id, {_feat_col_const(idx)}])")
                g.line(f"if {col_var} < 0: return action")
                declared.add(col_var)
            if var_now not in declared:
                g.line(f"{var_now} = features[i, {col_var}]")
                g.line(f"if np.isnan({var_now}): return action")
                declared.add(var_now)
            if var_mean not in declared:
                g.line(f"_mean_sum_{idx} = 0.0")
                g.line(f"for _mi in range({cond.bars}):")
                g.indent()
                g.line(f"_mean_sum_{idx} += features[i - _mi, {col_var}]")
                g.dedent()
                g.line(f"{var_mean} = _mean_sum_{idx} / {cond.bars}.0")
                declared.add(var_mean)
            g.line(f"_var_sum_{idx} = 0.0")
            g.line(f"for _vi in range({cond.bars}):")
            g.indent()
            g.line(f"_diff_{idx} = features[i - _vi, {col_var}] - {var_mean}")
            g.line(f"_var_sum_{idx} += _diff_{idx} * _diff_{idx}")
            g.dedent()
            g.line(f"{var_std} = (_var_sum_{idx} / {cond.bars}.0) ** 0.5")
            declared.add(var_std)
        if cond.operator == "gt":
            return f"{var_now} > {var_mean} + {cond.n_std} * {var_std}"
        else:
            return f"{var_now} < {var_mean} - {cond.n_std} * {var_std}"

    else:
        raise ValueError(f"Unknown condition type: {type(cond)}")


def _gen_action(action: dict, param_names: list[str], g: _CodeGen, declared: set):
    t = action["type"]

    if t == "force_exit":
        rp = action.get("reason_param", "p2")
        if rp in param_names:
            rv = _resolve_param_var(rp, param_names, g, declared)
            g.line(f"action[ACT_TYPE] = EXIT_ACT_FORCE_EXIT")
            g.line(f"action[ACT_FORCE_EXIT_REASON] = {rv}")
        else:
            g.line(f"action[ACT_TYPE] = EXIT_ACT_FORCE_EXIT")
            g.line(f"action[ACT_FORCE_EXIT_REASON] = 11.0")
        g.line("return action")

    elif t == "switch_profile":
        pp = action["profile_param"]
        pv = _resolve_param_var(pp, param_names, g, declared)
        g.line(f"action[ACT_TYPE] = EXIT_ACT_SWITCH_PROFILE")
        g.line(f"action[ACT_TARGET_PROFILE_ID] = {pv}")
        g.line("return action")

    elif t == "overwrite":
        g.line(f"action[ACT_TYPE] = EXIT_ACT_OVERWRITE_PRICE")
        if action.get("tp_param"):
            pv = _resolve_param_var(action["tp_param"], param_names, g, declared)
            g.line(f"action[ACT_NEW_TP_PRICE] = {pv}")
        if action.get("sl_param"):
            pv = _resolve_param_var(action["sl_param"], param_names, g, declared)
            g.line(f"action[ACT_NEW_SL_PRICE] = {pv}")
        g.line("return action")


def _gen_strat_function(spec: ExitRuleSpec, g: _CodeGen):
    param_names = list(spec.params.keys())
    feat_names  = list(spec.feature_names)
    fn_name     = f"_strat_{spec.strategy_id}"

    g.line(f"@njit(cache=True)")
    g.line(f"def {fn_name}(")
    g.indent()
    g.line("strategy_id,")
    g.line("strategy_rt_matrix,")
    g.line("i, k,")
    g.line("opens, highs, lows, closes,")
    g.line("features,")
    g.line("pos, exit_rt,")
    g.dedent()
    g.line("):")
    g.indent()

    if spec.name:
        g.line(f'"""')
        g.line(f"STRAT {spec.strategy_id} : {spec.name}")
        for j, pn in enumerate(param_names):
            g.line(f"p{j+1} = {pn} = {spec.params[pn]}")
        for j, fn in enumerate(feat_names):
            g.line(f"feature {j+1} = {fn}")
        g.line(f'"""')

    g.line("action = np.zeros(N_EXIT_ACT_COLS, dtype=np.float64)")
    g.line("action[ACT_TYPE] = EXIT_ACT_NONE")
    g.line("")

    # protection window
    if spec.window_bars > 0:
        g.line(f"if i < {spec.window_bars}:")
        g.indent()
        g.line("return action")
        g.dedent()
    else:
        g.line("if i < 1:")
        g.indent()
        g.line("return action")
        g.dedent()
    g.line("")

    declared: set = set()

    for rule in spec.rules:
        cond, act = rule
        expr = _gen_condition(cond, feat_names, param_names, g, declared)
        g.line(f"if {expr}:")
        g.indent()
        _gen_action(act, param_names, g, declared)
        g.dedent()
        g.line("")

    g.line("return action")
    g.dedent()
    g.line("")


def _gen_dispatcher(specs: list[ExitRuleSpec], g: _CodeGen):
    g.line("@njit(cache=True)")
    g.line("def run_exit_strategy_instant_user(")
    g.indent()
    g.line("strategy_id,")
    g.line("strategy_rt_matrix,")
    g.line("i, k,")
    g.line("opens, highs, lows, closes,")
    g.line("features,")
    g.line("pos, exit_rt,")
    g.dedent()
    g.line("):")
    g.indent()

    for spec in specs:
        fn_name = f"_strat_{spec.strategy_id}"
        g.line(f"if strategy_id == {spec.strategy_id}:")
        g.indent()
        g.line(f"return {fn_name}(")
        g.indent()
        g.line("strategy_id, strategy_rt_matrix, i, k,")
        g.line("opens, highs, lows, closes,")
        g.line("features, pos, exit_rt,")
        g.dedent()
        g.line(")")
        g.dedent()
        g.line("")

    g.line("action = np.zeros(N_EXIT_ACT_COLS, dtype=np.float64)")
    g.line("action[ACT_TYPE] = EXIT_ACT_NONE")
    g.line("return action")
    g.dedent()
    g.line("")

    # stubs window et stateful
    for stub in ("run_exit_strategy_window_user", "run_exit_strategy_stateful_user"):
        g.line("@njit(cache=True)")
        g.line(f"def {stub}(")
        g.indent()
        g.line("strategy_id,")
        g.line("strategy_rt_matrix,")
        g.line("i, k,")
        g.line("opens, highs, lows, closes,")
        g.line("features,")
        g.line("pos, exit_rt,")
        g.dedent()
        g.line("):")
        g.indent()
        g.line("action = np.zeros(N_EXIT_ACT_COLS, dtype=np.float64)")
        g.line("action[ACT_TYPE] = EXIT_ACT_NONE")
        g.line("return action")
        g.dedent()
        g.line("")


def compile_exit_rules(
    specs: list[ExitRuleSpec],
    output_path: str,
    package_import: str = "Backtest_Framework",
) -> Path:
    for spec in specs:
        max_bars_needed = _get_max_bars_needed(spec)
        if max_bars_needed > 0 and spec.window_bars < max_bars_needed:
            raise ValueError(
                f"Strategy '{spec.name}' needs window_bars >= {max_bars_needed}, "
                f"got {spec.window_bars}"
            )
    """
    Compile une liste d'ExitRuleSpec en code Numba et l'écrit dans output_path.

    Parameters
    ----------
    specs : list[ExitRuleSpec]
        Les règles de sortie à compiler.
    output_path : str
        Chemin du fichier .py à générer (fichier user, pas dans le package).
    package_import : str
        Nom du package à importer dans le fichier généré.

    Returns
    -------
    Path : chemin du fichier généré
    """
    g = _CodeGen()

    # Header
    g.line("# =============================================================")
    g.line("# AUTO-GENERATED BY rule_compiler.py — NE PAS MODIFIER")
    g.line(f"# Strategies: {[s.name for s in specs]}")
    g.line("# =============================================================")
    g.line("")
    g.line("import numpy as np")
    g.line("from numba import njit")
    g.line("")
    g.line(f"from {package_import}.exit_strategy_system import (")
    g.indent()
    g.line("N_EXIT_ACT_COLS,")
    g.line("ACT_TYPE,")
    g.line("ACT_TARGET_PROFILE_ID,")
    g.line("ACT_NEW_TP_PRICE,")
    g.line("ACT_NEW_SL_PRICE,")
    g.line("ACT_FORCE_EXIT_REASON,")
    g.line("EXIT_ACT_NONE,")
    g.line("EXIT_ACT_SWITCH_PROFILE,")
    g.line("EXIT_ACT_OVERWRITE_PRICE,")
    g.line("EXIT_ACT_FORCE_EXIT,")
    for i in range(1, 11):
        g.line(f"STRAT_FEAT_COL_{i},")
    for i in range(1, 7):
        g.line(f"STRAT_PARAM_{i},")
    g.dedent()
    g.line(")")
    g.line("")
    g.line(f"from {package_import}.core_engine import (")
    g.indent()
    g.line("POS_SIDE,")
    g.line("POS_ENTRY_IDX,")
    g.line("POS_MAE,")
    g.line("POS_MFE,")
    g.line("POS_BE_ACTIVE,")
    g.line("POS_RUNNER_ACTIVE,")
    g.dedent()
    g.line(")")
    g.line("")

    # Fonctions strat individuelles
    for spec in specs:
        _gen_strat_function(spec, g)

    # Dispatcher principal
    _gen_dispatcher(specs, g)

    # Écriture fichier
    out = Path(output_path).expanduser().resolve()
    out.write_text(g.render(), encoding="utf-8")

    print(f"✓ Exit rules compiled → {out}")
    return out
