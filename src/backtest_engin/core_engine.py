import os
import psutil

os.environ["NUMBA_NUM_THREADS"] = str(psutil.cpu_count(logical=False))
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Callable, Any
from collections import namedtuple
import numpy as np
from numba import njit, prange
from scipy import stats as scipy_stats
import pandas as pd
from .indicators import ema_njit
from .Exit_system import (
    RT_EXIT_PROFILE_ID,
    RT_EXIT_STRATEGY_ID,
    RT_TP_PCT,
    RT_SL_PCT,
    RT_USE_ATR_SL_TP,
    RT_TP_ATR_MULT,
    RT_SL_ATR_MULT,
    RT_BE_TRIGGER_PCT,
    RT_BE_OFFSET_PCT,
    RT_BE_DELAY_BARS,
    RT_TRAILING_TRIGGER_PCT,
    RT_RUNNER_TRAILING_MULT,
    RT_MAX_HOLDING_BARS,
    N_EXIT_RT_COLS,    TRIGGER_TYPE_RR, TRIGGER_TYPE_MFE_PCT, TRIGGER_TYPE_MAE_PCT,
    TRIGGER_TYPE_ATR_MULT, TRIGGER_TYPE_BARS, TRIGGER_TYPE_BARS_ATP,
    TRIGGER_TYPE_FEATURE, TRIGGER_TYPE_PHASE,
    TRIGGER_TYPE_ALL, TRIGGER_TYPE_ANY, TRIGGER_TYPE_NONE,
    ACTION_TYPE_EXIT_PARTIAL, ACTION_TYPE_MOVE_SL_BE,
    ACTION_TYPE_MOVE_SL_FEAT, ACTION_TYPE_SET_TP,
    ACTION_TYPE_ADD_POSITION, ACTION_TYPE_SET_PHASE,
    ACTION_TYPE_INVALIDATE,
    DIST_MODE_LINEAR, DIST_MODE_EXPO, DIST_MODE_LOG,
    DIST_MODE_SQRT, DIST_MODE_EQUAL,
    DIST_MODE_CUSTOM_PTS, DIST_MODE_CALLABLE,
    PART_TRIGGER_TYPE, PART_TRIGGER_VALUE, PART_TRIGGER_FEAT,
    PART_TRIGGER_OP, PART_FRACTION, PART_REF,
    PART_PRICE_MODE, PART_MOVE_BE,
    PART_DIST_MODE, PART_DIST_PARAM1, PART_DIST_PARAM2,
    N_PARTIAL_COLS,
    PYR_TRIGGER_TYPE, PYR_TRIGGER_VALUE, PYR_TRIGGER_FEAT,
    PYR_TRIGGER_OP, PYR_SIZE_FRACTION, PYR_SIZE_REF,
    PYR_SL_MODE, PYR_SL_FEAT, PYR_SL_ATR_MULT,
    PYR_GROUP_SL_MODE, PYR_DIST_MODE, PYR_DIST_PARAM1, PYR_DIST_PARAM2,
    N_PYRAMID_COLS,
    AVG_TRIGGER_TYPE, AVG_TRIGGER_VALUE, AVG_TRIGGER_FEAT,
    AVG_TRIGGER_OP, AVG_SIZE_FRACTION, AVG_SL_MODE,
    AVG_TP_MODE, AVG_MAX_DOWN, AVG_DIST_MODE,
    AVG_DIST_PARAM1, AVG_DIST_PARAM2, N_AVG_COLS,
    PHASE_ID, PHASE_TP_PCT, PHASE_SL_PCT,
    PHASE_BE_TRIGGER, PHASE_TRAILING, PHASE_MAX_HOLD, N_PHASE_COLS,
    RT_RULE_PHASE_FILTER, RT_RULE_MAX_TIMES,
    RT_RULE_TRIGGER_TYPE, RT_RULE_TRIGGER_VALUE,
    RT_RULE_TRIGGER_FEAT1, RT_RULE_TRIGGER_FEAT2,
    RT_RULE_TRIGGER_OP, RT_RULE_N_ACTIONS, N_RULE_TRIGGER_COLS,
    RA_ACTION_TYPE, RA_PARAM1, RA_PARAM2, RA_PARAM3,
    RA_FEAT_IDX, N_RULE_ACTION_COLS,
    OP_GT, OP_LT, OP_GTE, OP_LTE, OP_CROSS_ABOVE, OP_CROSS_BELOW,
)
from .exit_strategy_system import *
from .exit_strategy_system import (
    SCFG_MAX_CONSEC_SL, SCFG_COOLDOWN_BARS,
    SCFG_MAX_POSITIONS, SCFG_INVALIDATE_ON_REGIME,
    SCFG_MIN_WINRATE, SCFG_WINRATE_COOLDOWN,
    N_STATEFUL_CFG_COLS,
)

from .active_user_exit_strategies import (
    run_exit_strategy_instant_user,
    run_exit_strategy_window_user,
    run_exit_strategy_stateful_user,
)

"""
╔══════════════════════════════════════════════════════════════════╗
║           NJIT ENGINE CORE — moteur + métriques                  ║
║                                                                  ║
║  Contient :                                                      ║
║    - Constantes                                                  ║
║    - Indicateurs (EMA, ATR)                                      ║
║    - Générateurs de signaux                                      ║
║    - Helpers exit/entry                                          ║
║    - backtest_njit  ← moteur principal                           ║
║    - compute_metrics_full                                        ║
║    - NJITEngine  (sans grid_search)                              ║
║                                                                  ║
║  NE PAS TOUCHER : njit_grid.py                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════
# 0. CONSTANTES
# ══════════════════════════════════════════════════════════════════

POS_SIDE             = 0
POS_ENTRY_PRICE      = 1
POS_TP               = 2
POS_SL               = 3
POS_ENTRY_IDX        = 4
POS_BE_ARMED         = 5
POS_BE_ACTIVE        = 6
POS_BE_ARM_IDX       = 7
POS_RUNNER_ARMED     = 8
POS_RUNNER_ACTIVE    = 9
POS_RUNNER_SL        = 10
POS_TAG              = 11
POS_PENDING_BE_SL    = 12
POS_RUNNER_THRESHOLD = 13
POS_MAE              = 14
POS_MFE              = 15
POS_SETUP_ID         = 16
POS_SELECTED_SCORE   = 17
POS_REMAINING_SIZE   = 18    # fraction restante (1.0 = pleine)
POS_ORIGINAL_SIZE    = 19  # taille originale
POS_GROUP_ID         = 20   # id du groupe
POS_GROUP_SL_MODE    = 21   # 0=indép, 1=partagé, 2=user
POS_N_COLS           = 22   # ← incrémenter

# Colonnes pending orders
PEND_SIDE        = 0
PEND_LIMIT_PRICE = 1
PEND_EXPIRY_BAR  = 2
PEND_SIGNAL_BAR  = 3
PEND_SETUP_ID    = 4
PEND_SCORE       = 5
PEND_TP_PRICE    = 6
PEND_SL_PRICE    = 7
PEND_EXIT_PROF   = 8   # exit_profile_id au moment du signal
PEND_EXIT_STRAT  = 9   # exit_strategy_id au moment du signal
N_PENDING_COLS   = 10

REASON_SL               = 1
REASON_TP               = 2
REASON_BE               = 3
REASON_EMA1_TP          = 4
REASON_EMA2_TP          = 5
REASON_EMACROSS_TP      = 6
REASON_RUNNER_SL        = 7
REASON_EXIT_SIG         = 8
REASON_REVERSE          = 9
REASON_FORCED_FLAT      = 10
REASON_EXIT_STRAT_FORCE = 11
REASON_MAX_HOLD         = 12   
REASON_PARTIAL_TP       = 13   
REASON_GROUP_SL         = 14   
REASON_INVALIDATED      = 15   

REASON_LABELS = {
    1:  "SL",
    2:  "TP",
    3:  "BE",
    4:  "EMA1_TP",
    5:  "EMA2_TP",
    6:  "EMACROSS_TP",
    7:  "RUNNER_SL",
    8:  "EXIT_SIG",
    9:  "REVERSE",
    10: "FORCED_FLAT",
    11: "REASON_EXIT_STRAT_FORCE",
    12: "MAX_HOLD",
    13: "PARTIAL_TP",
    14: "GROUP_SL",
    15: "INVALIDATED",
}

# ==========
# Bar context cols
# ==========
BAR_OPEN           = 0
BAR_HIGH           = 1
BAR_LOW            = 2
BAR_CLOSE          = 3
BAR_MINUTE_OF_DAY  = 4
BAR_DAY_INDEX      = 5
BAR_DAY_OF_WEEK    = 6
N_BAR_CTX_COLS     = 7

# ==========
# Trade read indices (logical only)
# ==========
TR_SIDE                 = 0
TR_ENTRY_PRICE          = 1
TR_TP                   = 2
TR_SL                   = 3
TR_ENTRY_IDX            = 4
TR_MAE                  = 5
TR_MFE                  = 6
TR_SETUP_ID             = 7
TR_SELECTED_SCORE       = 8
TR_EXIT_PROFILE_ID      = 9
TR_EXIT_STRATEGY_ID     = 10
TR_BE_ACTIVE            = 11
TR_RUNNER_ACTIVE        = 12
TR_RUNNER_SL            = 13
TR_BARS_IN_TRADE        = 14
N_TRADE_CTX_COLS        = 15



# ==========
# Strategy execution backend
# ==========
STRAT_BACKEND_NUMBA  = 0
STRAT_BACKEND_PYTHON = 1

# ══════════════════════════════════════════════════════════════════
# 2. GÉNÉRATEURS DE SIGNAUX
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def signals_ema_vs_close_njit(opens, closes, span1, span2):
    n      = closes.shape[0]
    ema1   = ema_njit(closes, span1)
    ema2   = ema_njit(closes, span2)
    signal = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if closes[i - 1] < ema1[i - 1] and closes[i] > ema1[i]:
            signal[i] = 1
        elif closes[i - 1] > ema1[i - 1] and closes[i] < ema1[i]:
            signal[i] = -1
    return ema1, ema2, signal


@njit(cache=True)
def signals_ema_cross_njit(closes, span1, span2):
    n      = closes.shape[0]
    ema1   = ema_njit(closes, span1)
    ema2   = ema_njit(closes, span2)
    signal = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if ema1[i - 1] < ema2[i - 1] and ema1[i] > ema2[i]:
            signal[i] = 1
        elif ema1[i - 1] > ema2[i - 1] and ema1[i] < ema2[i]:
            signal[i] = -1
    return ema1, ema2, signal


# ══════════════════════════════════════════════════════════════════
# 3. HELPERS EXIT / ENTRY
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _compute_tp_sl(side, entry_price, tp_pct, sl_pct,
                   use_atr_sl_tp, tp_atr_mult, sl_atr_mult, atr_value):
    if use_atr_sl_tp == 2:
        tp = entry_price + side * atr_value * tp_atr_mult
        sl = entry_price - side * atr_value * sl_atr_mult
    elif use_atr_sl_tp == 1:
        tp = entry_price + side * atr_value * tp_atr_mult
        sl = entry_price * (1.0 - side * sl_pct)
    elif use_atr_sl_tp == -1:
        tp = entry_price * (1.0 + side * tp_pct)
        sl = entry_price - side * atr_value * sl_atr_mult
    else:
        tp = entry_price * (1.0 + side * tp_pct)
        sl = entry_price * (1.0 - side * sl_pct)
    return tp, sl


@njit(cache=True)
def _check_exit_fixed(side, o, h, l, tp, sl, be_active):
    reason = REASON_BE if be_active else REASON_SL
    if side == 1:
        if o <= sl: return o,  reason
        if l <= sl: return sl, reason
        if h >= tp: return tp, REASON_TP
    else:
        if o >= sl: return o,  reason
        if h >= sl: return sl, reason
        if l <= tp: return tp, REASON_TP
    return -1.0, 0


@njit(cache=True)
def _check_exit_ema(side, o, h, l, close, tp, sl, be_active, entry_price,
                    ema1, ema2, use_ema1_tp, use_ema2_tp, use_ema_cross_tp):
    be_reason = REASON_BE if be_active else REASON_SL
    if side == 1:
        if o <= sl: return o,  be_reason
        if l <= sl: return sl, be_reason
    else:
        if o >= sl: return o,  be_reason
        if h >= sl: return sl, be_reason
    in_profit = (side == 1 and close > entry_price) or \
                (side == -1 and close < entry_price)
    if not in_profit:
        return -1.0, 0
    if use_ema1_tp:
        if side == 1  and close < ema1: return close, REASON_EMA1_TP
        if side == -1 and close > ema1: return close, REASON_EMA1_TP
    if use_ema2_tp:
        if side == 1  and close < ema2: return close, REASON_EMA2_TP
        if side == -1 and close > ema2: return close, REASON_EMA2_TP
    if use_ema_cross_tp:
        if side == 1  and ema1 < ema2:  return close, REASON_EMACROSS_TP
        if side == -1 and ema1 > ema2:  return close, REASON_EMACROSS_TP
    return -1.0, 0


@njit(cache=True)
def _update_be(side, ep, sl, i, entry_idx, be_armed, be_active,
               be_arm_idx, pending_be_sl, h, l,
               be_trigger_pct, be_offset_pct, be_delay_bars):
    if be_trigger_pct <= 0.0:
        return sl, be_armed, be_active, be_arm_idx, pending_be_sl
    delay_ok = (i - entry_idx) >= be_delay_bars
    if not be_armed and delay_ok:
        if side == 1:
            if h >= ep * (1.0 + be_trigger_pct):
                be_armed = 1.0; be_arm_idx = float(i)
                pending_be_sl = ep * (1.0 + be_offset_pct)
        else:
            if l <= ep * (1.0 - be_trigger_pct):
                be_armed = 1.0; be_arm_idx = float(i)
                pending_be_sl = ep * (1.0 - be_offset_pct)
    if be_armed and i > int(be_arm_idx):
        if side == 1: sl = max(sl, pending_be_sl)
        else:         sl = min(sl, pending_be_sl)
        be_armed = 0.0; be_active = 1.0
        pending_be_sl = 0.0; be_arm_idx = -1.0
    return sl, be_armed, be_active, be_arm_idx, pending_be_sl

#———————————————————————#
# Exit strategy helpers #
#———————————————————————#

@njit(cache=True)
def _is_profile_allowed(strategy_id, target_profile_id, strategy_allowed_profiles, strategy_allowed_counts):
    if strategy_id < 0 or strategy_id >= strategy_allowed_counts.shape[0]:
        return False

    n_allowed = int(strategy_allowed_counts[strategy_id])
    for j in range(n_allowed):
        if strategy_allowed_profiles[strategy_id, j] == target_profile_id:
            return True
    return False

@njit(cache=True)
def _run_exit_strategy_instant(
    strategy_id,
    strategy_rt_matrix,
    i, k,
    opens, highs, lows, closes,
    features,
    pos, exit_rt,
    state_per_pos,    # ← nouveau
    state_global,     # ← nouveau
):
    return run_exit_strategy_instant_user(
        strategy_id, strategy_rt_matrix,
        i, k,
        opens, highs, lows, closes,
        features, pos, exit_rt,
        state_per_pos,
        state_global,
    )

@njit(cache=True)
def _run_exit_strategy_windowed(
    strategy_id,
    strategy_rt_matrix,
    i, k,
    opens, highs, lows, closes,
    features,
    pos, exit_rt,
    state_per_pos,
    state_global,
):
    window_bars = int(strategy_rt_matrix[strategy_id, STRAT_WINDOW_BARS])
    if i < window_bars:
        action = np.zeros(N_EXIT_ACT_COLS, dtype=np.float64)
        action[ACT_TYPE] = EXIT_ACT_NONE
        return action
    return run_exit_strategy_window_user(
        strategy_id, strategy_rt_matrix,
        i, k,
        opens, highs, lows, closes,
        features, pos, exit_rt,
        state_per_pos,
        state_global,
    )


@njit(cache=True)
def _run_exit_strategy_stateful(
    strategy_id,
    strategy_rt_matrix,
    i, k,
    opens, highs, lows, closes,
    features,
    pos, exit_rt,
    state_per_pos,
    state_global,
):
    return run_exit_strategy_stateful_user(
        strategy_id, strategy_rt_matrix,
        i, k,
        opens, highs, lows, closes,
        features, pos, exit_rt,
        state_per_pos,
        state_global,
    )

# strat de sorti dynamique python
def _run_exit_strategy_python(
    strategy_spec,          # ExitStrategySpec
    i: int,
    k: int,
    opens, highs, lows, closes, atrs,
    features,               # np.ndarray (n_bars, n_features)
    feature_names: tuple,
    pos_arr,                # pos[k] row
    exit_rt_row,            # exit_rt[k] row
    entry_idx: int,
):
    from .exit_context import (
        PosCtx, BarCtx, FeatCtx, ParamsCtx,
        EXIT_ACT_NONE, EXIT_ACT_SWITCH_PROFILE,
        EXIT_ACT_OVERWRITE_PRICE, EXIT_ACT_FORCE_EXIT,
    )
    import numpy as np

    # --- PosCtx ---
    pos = PosCtx(
        side             = float(pos_arr[POS_SIDE]),
        entry_price      = float(pos_arr[POS_ENTRY_PRICE]),
        tp               = float(pos_arr[POS_TP]),
        sl               = float(pos_arr[POS_SL]),
        bars_in_trade    = i - int(pos_arr[POS_ENTRY_IDX]),
        mae              = float(pos_arr[POS_MAE]),
        mfe              = float(pos_arr[POS_MFE]),
        be_active        = bool(pos_arr[POS_BE_ACTIVE]),
        runner_active    = bool(pos_arr[POS_RUNNER_ACTIVE]),
        setup_id         = int(pos_arr[POS_SETUP_ID]),
        selected_score   = float(pos_arr[POS_SELECTED_SCORE]),
        exit_profile_id  = int(exit_rt_row[RT_EXIT_PROFILE_ID]),
        exit_strategy_id = int(exit_rt_row[RT_EXIT_STRATEGY_ID]),
    )

    # --- BarCtx ---
    has_atr = len(atrs) > i and not np.isnan(atrs[i])
    bar = BarCtx(
        i     = i,
        open  = float(opens[i]),
        high  = float(highs[i]),
        low   = float(lows[i]),
        close = float(closes[i]),
        atr   = float(atrs[i]) if has_atr else float("nan"),
    )

    # --- FeatCtx ---
    strat_type = strategy_spec.strategy_type

    if strat_type == EXIT_STRAT_INSTANT_PY:
        # vecteur 1D : valeurs courantes de chaque feature
        feat_row = features[i]
        feat = FeatCtx(feat_row, feature_names, window_mode=False)

    else:  # EXIT_STRAT_WINDOW_PY
        w = strategy_spec.window_bars
        start = max(0, i - w + 1)
        # slice (window_bars, n_features) — index 0 = le plus ancien, -1 = courant
        window_slice = features[start:i + 1]
        # pad si pas assez de barres au début
        if window_slice.shape[0] < w:
            pad = np.full((w - window_slice.shape[0], features.shape[1]), np.nan)
            window_slice = np.vstack([pad, window_slice])
        feat = FeatCtx(window_slice, feature_names, window_mode=True)

    # --- ParamsCtx ---
    params = ParamsCtx(strategy_spec.params)

    # --- Appel fonction user ---
    result = strategy_spec.fn(pos, bar, feat, params)

    # --- Conversion résultat → action array ---
    action = np.zeros(N_EXIT_ACT_COLS, dtype=np.float64)
    action[ACT_TYPE] = EXIT_ACT_NONE

    if result is None or result.get("type", 0) == 0:
        return action

    t = result["type"]

    if t == 1:  # switch_profile
        action[ACT_TYPE] = EXIT_ACT_SWITCH_PROFILE
        action[ACT_TARGET_PROFILE_ID] = float(result["target_profile_id"])

    elif t == 2:  # overwrite_tp_sl
        action[ACT_TYPE] = EXIT_ACT_OVERWRITE_PRICE
        action[ACT_NEW_TP_PRICE] = result["tp"]
        action[ACT_NEW_SL_PRICE] = result["sl"]

    elif t == 3:  # force_exit
        action[ACT_TYPE] = EXIT_ACT_FORCE_EXIT
        action[ACT_FORCE_EXIT_REASON] = float(result["reason"])

    return action

@njit(cache=True)
def _run_exit_strategy_dispatch(
    strategy_id,
    strategy_rt_matrix,
    i, k,
    opens, highs, lows, closes,
    features,
    pos, exit_rt,
    state_per_pos,   
    state_global,    
):
    action = np.zeros(N_EXIT_ACT_COLS, dtype=np.float64)
    action[ACT_TYPE] = EXIT_ACT_NONE

    if strategy_id < 0:
        return action
    if strategy_id >= strategy_rt_matrix.shape[0]:
        return action

    backend    = int(strategy_rt_matrix[strategy_id, STRAT_BACKEND])
    strat_type = int(strategy_rt_matrix[strategy_id, STRAT_TYPE])

    if backend != STRAT_BACKEND_NUMBA:
        return action

    if strat_type == EXIT_STRAT_INSTANT:
        if i < 1:
            return action
        return _run_exit_strategy_instant(
            strategy_id, strategy_rt_matrix,
            i, k,
            opens, highs, lows, closes,
            features, pos, exit_rt,
            state_per_pos, state_global,
        )

    elif strat_type == EXIT_STRAT_WINDOWED:
        return _run_exit_strategy_windowed(
            strategy_id, strategy_rt_matrix,
            i, k,
            opens, highs, lows, closes,
            features, pos, exit_rt,
            state_per_pos, state_global,
        )

    elif strat_type == EXIT_STRAT_STATEFUL:
        return _run_exit_strategy_stateful(
            strategy_id, strategy_rt_matrix,
            i, k,
            opens, highs, lows, closes,
            features, pos, exit_rt,
            state_per_pos, state_global,
        )

    return action

@njit(cache=True)
def _record_trade(
    trade_returns, trade_sides, trade_entry_idx, trade_exit_idx, trade_reasons,
    trade_exit_prices, trade_mae, trade_mfe,
    trade_setup_ids, trade_selected_score,
    trade_exit_profile_ids, trade_exit_strategy_ids,
    trade_regime_ids,
    # ── Nouveau ───────────────────────────────────────────
    trade_phase, trade_n_tp_hit, trade_add_count,
    trade_remaining, trade_avg_entry, trade_bars,
    trade_group_ids,
    # ─────────────────────────────────────────────────────
    n_trades, side, exit_price, ep, entry_idx, exit_idx, reason, mae, mfe,
    setup_id, sel_score, exit_profile_id, exit_strategy_id, regime_id,
    state_global, n_state_global, use_exit_system,
    # ── Nouveau ───────────────────────────────────────────
    state_per_pos, k, n_state_per_pos, group_id,
):
    ret = side * (exit_price - ep) / ep
    trade_returns[n_trades]           = ret
    trade_sides[n_trades]             = int(side)
    trade_entry_idx[n_trades]         = entry_idx
    trade_exit_idx[n_trades]          = exit_idx
    trade_reasons[n_trades]           = reason
    trade_exit_prices[n_trades]       = exit_price
    trade_mae[n_trades]               = mae
    trade_mfe[n_trades]               = mfe
    trade_setup_ids[n_trades]         = setup_id
    trade_selected_score[n_trades]    = sel_score
    trade_exit_profile_ids[n_trades]  = exit_profile_id
    trade_exit_strategy_ids[n_trades] = exit_strategy_id
    trade_regime_ids[n_trades]        = regime_id
    trade_group_ids[n_trades]         = group_id    # ← nouveau

    # ── Nouveau — état de la position à la sortie ──────────
    trade_bars[n_trades] = exit_idx - entry_idx
    if use_exit_system and n_state_per_pos >= N_SP_DEFAULT:
        trade_phase[n_trades]     = int(state_per_pos[k, SP_PHASE])
        trade_n_tp_hit[n_trades]  = int(state_per_pos[k, SP_N_TP_HIT])
        trade_add_count[n_trades] = int(state_per_pos[k, SP_ADD_COUNT])
        trade_remaining[n_trades] = state_per_pos[k, SP_REMAINING_SIZE]
        trade_avg_entry[n_trades] = state_per_pos[k, SP_AVG_ENTRY]
    else:
        trade_phase[n_trades]     = 0
        trade_n_tp_hit[n_trades]  = 0
        trade_add_count[n_trades] = 0
        trade_remaining[n_trades] = 1.0
        trade_avg_entry[n_trades] = ep

    # ── Stateful global update ────────────────────────────
    if use_exit_system and exit_strategy_id >= 0 and n_state_global >= N_SG_DEFAULT:
        state_global[exit_strategy_id, SG_LAST_TRADE_RETURN] = ret
        state_global[exit_strategy_id, SG_TOTAL_EXPOSURE]   -= 1.0
        if state_global[exit_strategy_id, SG_TOTAL_EXPOSURE] < 0.0:
            state_global[exit_strategy_id, SG_TOTAL_EXPOSURE] = 0.0

        if reason == REASON_SL or reason == REASON_BE:
            state_global[exit_strategy_id, SG_CONSEC_SL] += 1.0
            state_global[exit_strategy_id, SG_CONSEC_TP]  = 0.0
        elif reason == REASON_TP:
            state_global[exit_strategy_id, SG_CONSEC_TP] += 1.0
            state_global[exit_strategy_id, SG_CONSEC_SL]  = 0.0
        else:
            state_global[exit_strategy_id, SG_CONSEC_SL] = 0.0
            state_global[exit_strategy_id, SG_CONSEC_TP] = 0.0

        alpha = 0.05
        win   = 1.0 if ret > 0.0 else 0.0
        old_wr = state_global[exit_strategy_id, SG_ROLLING_WINRATE]
        if old_wr == 0.0:
            state_global[exit_strategy_id, SG_ROLLING_WINRATE] = win
        else:
            state_global[exit_strategy_id, SG_ROLLING_WINRATE] = (
                (1.0 - alpha) * old_wr + alpha * win
            )

        state_global[exit_strategy_id, SG_DAILY_TRADE_COUNT]   += 1.0
        state_global[exit_strategy_id, SG_SESSION_TRADE_COUNT] += 1.0

    return n_trades + 1


@njit(cache=True)
def _update_runner(side, ep, runner_armed, runner_active, runner_sl,
                   runner_threshold, h, l, close, atr_value,
                   trailing_trigger_pct, runner_trailing_mult):
    if trailing_trigger_pct <= 0.0:
        return runner_armed, runner_active, runner_sl, runner_threshold

    if not runner_armed and not runner_active:
        threshold = ep * (1.0 + side * trailing_trigger_pct)
        if (side == 1 and h >= threshold) or (side == -1 and l <= threshold):
            runner_armed     = 1.0
            runner_threshold = threshold
            return runner_armed, runner_active, runner_sl, runner_threshold

    if runner_armed and not runner_active:
        runner_active = 1.0; runner_armed = 0.0
        init_sl   = close - side * atr_value * runner_trailing_mult
        runner_sl = max(init_sl, ep) if side == 1 else min(init_sl, ep)
        return runner_armed, runner_active, runner_sl, runner_threshold

    if runner_active:
        new_sl = close - side * atr_value * runner_trailing_mult
        if side == 1: runner_sl = max(runner_sl, new_sl, ep)
        else:         runner_sl = min(runner_sl, new_sl, ep)

    return runner_armed, runner_active, runner_sl, runner_threshold

# ══════════════════════════════════════════════════════════════════
# 3.5 HELPERS POSITION RULES — natifs
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _eval_dist_fn(dist_mode, param1, param2, x):
    """
    dist_mode : int encodé (DIST_MODE_*)
    param1    : ratio (expo) ou slope (linear)
    param2    : start
    x         : distance (niveau courant)
    """
    if dist_mode == DIST_MODE_EXPO:      # expo : start * ratio^x
        return param2 * (param1 ** x)

    elif dist_mode == DIST_MODE_LINEAR:  # linear : start - slope * x
        val = param2 - param1 * x
        return val if val > 0.0 else 0.0

    elif dist_mode == DIST_MODE_LOG:     # log approx : start / (x + e)
        # e ≈ 2.71828 — pas besoin de math.e
        return param2 / (x + 2.71828182845904)

    elif dist_mode == DIST_MODE_SQRT:    # sqrt : start / sqrt(x+1)
        return param2 / ((x + 1.0) ** 0.5)

    elif dist_mode == DIST_MODE_EQUAL:   # constant
        return param2

    else:                                # fallback expo
        return param2 * (param1 ** x)


@njit(cache=True)
def _eval_trigger_simple(
    trigger_type, trigger_value, trigger_feat, trigger_op,
    i, k,
    pos, features, state_per_pos, atrs, has_atr,
    ep, side,
):
    """
    Évaluer un trigger simple (pas OnAll/OnAny).
    Retourne True si le trigger est déclenché.
    """
    if trigger_type < 0:
        return False

    if trigger_type == TRIGGER_TYPE_RR:
        sl_dist = abs(ep - pos[k, POS_SL])
        if sl_dist <= 0.0:
            return False
        mfe = pos[k, POS_MFE]
        return mfe >= trigger_value * sl_dist / ep

    elif trigger_type == TRIGGER_TYPE_MFE_PCT:
        return pos[k, POS_MFE] >= trigger_value

    elif trigger_type == TRIGGER_TYPE_MAE_PCT:
        return pos[k, POS_MAE] <= trigger_value  # trigger_value est négatif

    elif trigger_type == TRIGGER_TYPE_ATR_MULT:
        if not has_atr or i < 1:
            return False
        atr_val = atrs[i - 1]
        mfe_abs = pos[k, POS_MFE] * ep
        return mfe_abs >= trigger_value * atr_val

    elif trigger_type == TRIGGER_TYPE_BARS:
        return state_per_pos[k, SP_BARS_SINCE_ENTRY] >= trigger_value

    elif trigger_type == TRIGGER_TYPE_BARS_ATP:
        return state_per_pos[k, SP_BARS_SINCE_TP] >= trigger_value

    elif trigger_type == TRIGGER_TYPE_PHASE:
        return state_per_pos[k, SP_PHASE] == trigger_value

    elif trigger_type == TRIGGER_TYPE_FEATURE:
        feat_idx = int(trigger_feat)
        if feat_idx < 0 or feat_idx >= features.shape[1]:
            return False
        feat_val = features[i, feat_idx]
        op = int(trigger_op)
        # trigger_value = float val ou feat2 idx si cross
        if op == OP_GT:
            return feat_val > trigger_value
        elif op == OP_LT:
            return feat_val < trigger_value
        elif op == OP_GTE:
            return feat_val >= trigger_value
        elif op == OP_LTE:
            return feat_val <= trigger_value
        elif op == OP_CROSS_ABOVE:
            feat2_idx = int(trigger_value)
            if feat2_idx < 0 or feat2_idx >= features.shape[1] or i < 1:
                return False
            return (features[i, feat_idx] > features[i, feat2_idx] and
                    features[i - 1, feat_idx] <= features[i - 1, feat2_idx])
        elif op == OP_CROSS_BELOW:
            feat2_idx = int(trigger_value)
            if feat2_idx < 0 or feat2_idx >= features.shape[1] or i < 1:
                return False
            return (features[i, feat_idx] < features[i, feat2_idx] and
                    features[i - 1, feat_idx] >= features[i - 1, feat2_idx])

    return False

# ══════════════════════════════════════════════════════════════════
# 4. MOTEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def backtest_njit(
    opens, highs, lows, closes, atrs,
    signals, selected_setup_id, selected_score,features,

    use_exit_system, profile_rt_matrix, strategy_rt_matrix,
    setup_to_exit_profile, setup_to_exit_strategy,
    strategy_allowed_profiles, strategy_allowed_counts,

    minutes_of_day, day_index,day_of_week,
    entry_delay, s1_start, s1_end, s2_start, s2_end, s3_start, s3_end,
    max_gap_signal, max_gap_entry, candle_size_filter, min_size_pct, max_size_pct,
    prev_candle_direction, tp_pct, sl_pct, use_atr_sl_tp,
    tp_atr_mult, sl_atr_mult, allow_exit_on_entry_bar,
    exit_ema1, exit_ema2, use_ema1_tp, use_ema2_tp, use_ema_cross_tp,
    exit_signals, signal_tags, use_exit_signal, exit_delay,
    be_trigger_pct, be_offset_pct, be_delay_bars,
    trailing_trigger_pct, runner_trailing_mult,
    multi_entry, reverse_mode, me_max, me_period, me_reset_mode,
    MAX_TRADES, MAX_POS, track_mae_mfe,
    cooldown_entries, cooldown_bars, cooldown_mode,
    entry_on_close, entry_on_signal_close_price,
    max_holding_bars,
    forced_flat_mode, forced_flat_minute,
    max_tp, tp_period_mode, tp_period_bars,
    max_sl, sl_period_mode, sl_period_bars,
    n_state_per_pos,   # int — nb colonnes state par position
    n_state_global,    # int — nb colonnes state global
    n_strategies,      # int — nb stratégies (pour state_global)
    regime,            # np.ndarray (n_bars,) int32
    regime_exit_profile_override,
    regime_exit_strategy_override,
    use_regime,
    entry_limit_prices,    # np.ndarray (n_bars,) float64 — -1 = market
    limit_expiry_bars,     # int — nb barres avant expiration
    tp_price_array,        # np.ndarray (n_bars,) float64 — -1 = utiliser %
    sl_price_array,        # np.ndarray (n_bars,) float64 — -1 = utiliser %
    check_filters_on_fill, # bool — réappliquer filtres au fill
    has_limit_orders,      # bool — optimisation, évite les checks inutiles
    # ── Nouvelles matrices position rules ─────────────────────
    partial_rt_matrix,     # shape (n_profiles, max_partial_levels, N_PARTIAL_COLS)
    pyramid_rt_matrix,     # shape (n_profiles, max_pyramid_levels, N_PYRAMID_COLS)
    averaging_rt_matrix,   # shape (n_profiles, max_avg_levels, N_AVG_COLS)
    phase_rt_matrix,       # shape (n_profiles, max_phases, N_PHASE_COLS)
    rule_trigger_matrix,   # shape (n_profiles, max_rules, N_RULE_TRIGGER_COLS)
    rule_action_matrix,    # shape (n_profiles, max_rules, max_actions, N_RULE_ACTION_COLS)
    stateful_cfg_rt,       # shape (n_strategies, N_STATEFUL_CFG_COLS)
    max_partial_levels,    # int
    max_pyramid_levels,    # int
    max_avg_levels,        # int
    max_phases,            # int
    max_rules,             # int
    max_actions_per_rule,  # int
    has_partial,           # bool
    has_pyramid,           # bool
    has_averaging,         # bool
    has_phases,            # bool
    has_rules,             # bool
    has_stateful_cfg,      # bool
    sl_tp_be_priority, # check en priorité sl and tp et be avant de poursuivre sur un partiel 
):

    # entry_on_close : False = entrée au open[i] (défaut), True = entrée au close[i-1]

    n = opens.shape[0]

    trade_returns     = np.empty(MAX_TRADES, dtype=np.float64)
    trade_sides       = np.empty(MAX_TRADES, dtype=np.int8)
    trade_entry_idx   = np.empty(MAX_TRADES, dtype=np.int32)
    trade_exit_idx    = np.empty(MAX_TRADES, dtype=np.int32)
    trade_reasons     = np.empty(MAX_TRADES, dtype=np.int8)
    trade_exit_prices = np.empty(MAX_TRADES, dtype=np.float64)
    trade_mae         = np.empty(MAX_TRADES, dtype=np.float64)
    trade_mfe         = np.empty(MAX_TRADES, dtype=np.float64)
    trade_setup_ids = np.empty(MAX_TRADES, dtype=np.int32)
    trade_selected_score = np.empty(MAX_TRADES, dtype=np.float64)
    trade_exit_profile_ids = np.empty(MAX_TRADES, dtype=np.int32)
    trade_exit_strategy_ids = np.empty(MAX_TRADES, dtype=np.int32)
    trade_sizes      = np.ones(MAX_TRADES, dtype=np.float64)
    trade_regime_ids = np.full(MAX_TRADES, -1, dtype=np.int32)
    trade_phase      = np.zeros(MAX_TRADES, dtype=np.int32)
    trade_n_tp_hit   = np.zeros(MAX_TRADES, dtype=np.int32)
    trade_add_count  = np.zeros(MAX_TRADES, dtype=np.int32)
    trade_remaining  = np.ones(MAX_TRADES, dtype=np.float64)
    trade_avg_entry  = np.zeros(MAX_TRADES, dtype=np.float64)
    trade_bars       = np.zeros(MAX_TRADES, dtype=np.int32)
    n_trades          = 0

    # ── Event log — group_id ──────────────────────────────────────
    trade_group_ids = np.full(MAX_TRADES, -1, dtype=np.int32)

    # ── Event log — phases ────────────────────────────────────────
    MAX_PHASE_EVENTS = MAX_TRADES
    phase_ev_idx     = np.zeros(MAX_PHASE_EVENTS, dtype=np.int32)
    phase_ev_trade   = np.zeros(MAX_PHASE_EVENTS, dtype=np.int32)
    phase_ev_group   = np.zeros(MAX_PHASE_EVENTS, dtype=np.int32)
    phase_ev_from    = np.zeros(MAX_PHASE_EVENTS, dtype=np.int32)
    phase_ev_to      = np.zeros(MAX_PHASE_EVENTS, dtype=np.int32)
    phase_ev_profile = np.zeros(MAX_PHASE_EVENTS, dtype=np.int32)
    phase_ev_side    = np.zeros(MAX_PHASE_EVENTS, dtype=np.float64)
    n_phase_events   = 0

    if has_limit_orders:
        pending_orders = np.full((MAX_POS, N_PENDING_COLS), -1.0, dtype=np.float64)
    else:
        pending_orders = np.zeros((1, N_PENDING_COLS), dtype=np.float64)
    n_pending = 0

    pos   = np.zeros((MAX_POS, POS_N_COLS), dtype=np.float64)
    n_pos = 0

    # Allocation 
    if use_exit_system:
        exit_rt = np.zeros((MAX_POS, N_EXIT_RT_COLS), dtype=np.float64)
    else:
        exit_rt = np.zeros((1, 1), dtype=np.float64)

    # Nouveaux arrays state
    if use_exit_system and n_state_per_pos > 0:
        state_per_pos = np.zeros((MAX_POS, n_state_per_pos), dtype=np.float64)
    else:
        state_per_pos = np.zeros((1, 1), dtype=np.float64)

    if use_exit_system and n_state_global > 0 and n_strategies > 0:
        state_global = np.zeros((n_strategies, n_state_global), dtype=np.float64)
    else:
        state_global = np.zeros((1, 1), dtype=np.float64)


    recent_entries = np.zeros(me_max + 1, dtype=np.int32)
    re_head        = 0
    re_count       = 0
    last_day       = -1
    last_session   = -1
    cd_count       = 0
    cd_until       = -1
    tp_count       = 0
    tp_last_day    = -1
    tp_last_session = -1
    recent_tp_idx = np.zeros(max_tp + 1 if max_tp > 0 else 1, dtype=np.int32)
    tp_head       = 0
    tp_recent_count = 0
    sl_count       = 0
    sl_last_day    = -1
    sl_last_session = -1
    recent_sl_idx = np.zeros(max_sl + 1 if max_sl > 0 else 1, dtype=np.int32)
    sl_head       = 0
    sl_recent_count = 0
    next_group_id = 0

    ema_exit_mode = use_ema1_tp or use_ema2_tp or use_ema_cross_tp
    has_exit_sig  = use_exit_signal and exit_signals.shape[0] == n
    has_tags      = signal_tags.shape[0] == n
    has_ema_exit  = exit_ema1.shape[0] == n
    has_atr       = atrs.shape[0] == n

    delayed_signals = np.zeros(n, dtype=np.int8)
    delayed_setup_id = np.full(n, -1, dtype=np.int32)
    delayed_selected_score = np.zeros(n, dtype=np.float64)

    for i in range(entry_delay, n): # pour eviter index negatif si en dessous de 1 
        delayed_setup_id[i] = selected_setup_id[i - entry_delay]
        delayed_selected_score[i] = selected_score[i - entry_delay]

        o = opens[i]; h = highs[i]; l = lows[i]; c = closes[i]
        cur_day     = day_index[i]
        delayed_signals[i] = signals[i - entry_delay]
        sig = delayed_signals[i]

        #--- force flatt logic ---- 
        forced_flat_now = False
        if forced_flat_mode != 0 and n_pos > 0:
            prev_day = day_index[i - 1]
            crossed_time = (
                minutes_of_day[i] >= forced_flat_minute and
                (cur_day != prev_day or minutes_of_day[i - 1] < forced_flat_minute)
            )
            if crossed_time:
                if forced_flat_mode == 1:
                    forced_flat_now = True
                elif forced_flat_mode == 2 and day_of_week[i] == 4:
                    forced_flat_now = True
                    # Friday = dernier jour ouvré visible dans les données weekday
                    # ici on suppose données classiques lun-ven dans l'ordre
                    # plus robuste si tu ajoutes day_of_week plus tard
                    

        cur_session = -1
        if   s1_start >= 0 and s1_start <= minutes_of_day[i] <= s1_end: cur_session = 0
        elif s2_start >= 0 and s2_start <= minutes_of_day[i] <= s2_end: cur_session = 1
        elif s3_start >= 0 and s3_start <= minutes_of_day[i] <= s3_end: cur_session = 2
        
        #--- execution force flat now ----
        if forced_flat_now:
            k_ff = 0
            while k_ff < n_pos:
                cur_regime_id = int(regime[i]) if use_regime else -1
                if use_exit_system:
                    exit_profile_id_ff = int(exit_rt[k_ff, RT_EXIT_PROFILE_ID])
                    exit_strategy_id_ff = int(exit_rt[k_ff, RT_EXIT_STRATEGY_ID])
                else:
                    exit_profile_id_ff = -1
                    exit_strategy_id_ff= -1
                    
                ep_ff   = pos[k_ff, POS_ENTRY_PRICE]
                side_ff = pos[k_ff, POS_SIDE]

                if n_trades < MAX_TRADES:
                    setup_id_ff  = int(pos[k_ff, POS_SETUP_ID])
                    sel_score_ff = pos[k_ff, POS_SELECTED_SCORE]
                    n_trades = _record_trade(
                        trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                        trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                        trade_setup_ids, trade_selected_score, 
                        trade_exit_profile_ids,trade_exit_strategy_ids,
                        trade_regime_ids,
                        trade_phase, trade_n_tp_hit, trade_add_count,   # ← nouveau
                        trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,    # ← nouveau
                        n_trades, side_ff, o, ep_ff, int(pos[k_ff, POS_ENTRY_IDX]), i,
                        REASON_FORCED_FLAT, pos[k_ff, POS_MAE], pos[k_ff, POS_MFE],
                        setup_id_ff, sel_score_ff, exit_profile_id_ff, exit_strategy_id_ff,
                        cur_regime_id,  # regime_id ← nouveau
                        state_global,              # ← nouveau
                        n_state_global,            # ← nouveau
                        use_exit_system, state_per_pos, k_ff, n_state_per_pos,int(pos[k_ff, POS_GROUP_ID]),        # ← nouveau
                    )
                if use_exit_system:
                    exit_rt[k_ff] = exit_rt[n_pos - 1]
                if use_exit_system and n_state_per_pos > 0:    # ← manquant
                    state_per_pos[k_ff] = state_per_pos[n_pos - 1]
                    state_per_pos[n_pos - 1, :] = 0.0
                pos[k_ff] = pos[n_pos - 1]
                n_pos -= 1

            sig = 0

        # ── 1. REVERSE ───────────────────────────────────────────
        if reverse_mode and sig != 0:
            k = 0
            while k < n_pos:
                cur_regime_id = int(regime[i]) if use_regime else -1
                if use_exit_system:
                    exit_profile_id_rev = int(exit_rt[k, RT_EXIT_PROFILE_ID])
                    exit_strategy_id_rev = int(exit_rt[k, RT_EXIT_STRATEGY_ID])
                else:
                    exit_profile_id_rev = -1
                    exit_strategy_id_rev = -1 

                if int(pos[k, POS_SIDE]) == -int(sig):
                    ep = pos[k, POS_ENTRY_PRICE]; side = pos[k, POS_SIDE]
                    if n_trades < MAX_TRADES:
                        setup_id_rev  = int(pos[k, POS_SETUP_ID])
                        sel_score_rev = pos[k, POS_SELECTED_SCORE]
                         
                        n_trades = _record_trade(
                            trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                            trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                            trade_setup_ids, trade_selected_score,
                            trade_exit_profile_ids, trade_exit_strategy_ids,
                            trade_regime_ids,trade_phase, trade_n_tp_hit, trade_add_count,trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,                              # ← ajouter
                            n_trades, side, o, ep, int(pos[k, POS_ENTRY_IDX]), i,
                            REASON_REVERSE, pos[k, POS_MAE], pos[k, POS_MFE],
                            setup_id_rev, sel_score_rev, exit_profile_id_rev, exit_strategy_id_rev,
                            cur_regime_id,         # ← ajouter
                            state_global,                                  # ← ajouter
                            n_state_global,                                # ← ajouter
                            use_exit_system,state_per_pos, k, n_state_per_pos, int(pos[k, POS_GROUP_ID]),     # ← ajouter
                        )
                    if use_exit_system:
                        exit_rt[k] = exit_rt[n_pos - 1]
                    if use_exit_system and n_state_per_pos > 0:    # ← manquant
                        state_per_pos[k] = state_per_pos[n_pos - 1]
                        state_per_pos[n_pos - 1, :] = 0.0
                    pos[k] = pos[n_pos - 1]; n_pos -= 1
                else:
                    k += 1

        # ── 1.5 / 1.6 RESETS ────────────────────────────────────
        # Calculer les changements AVANT mise à jour des last_*
        day_changed     = last_day     != -1 and cur_day     != last_day
        session_changed = cur_session != -1 and (
            last_session == -1 or cur_session != last_session or day_changed
        )

        # Cooldown reset
        if cooldown_entries > 0:
            if   cooldown_mode == 3 and day_changed:     cd_count = 0; cd_until = -1
            elif cooldown_mode == 2 and session_changed: cd_count = 0; cd_until = -1

        # MaxEntries reset
        if me_reset_mode == 1:
            if day_changed: re_count = 0; re_head = 0
            last_day = cur_day
        elif me_reset_mode == 2:
            if session_changed: re_count = 0; re_head = 0
            if cur_session != -1: last_session = cur_session
        
        # Dans le bloc resets day/session
        if day_changed and use_exit_system and n_state_global >= N_SG_DEFAULT:
            for s in range(n_strategies):
                state_global[s, SG_DAILY_TRADE_COUNT] = 0.0

        if session_changed and use_exit_system and n_state_global >= N_SG_DEFAULT:
            for s in range(n_strategies):
                state_global[s, SG_SESSION_TRADE_COUNT] = 0.0
        # ── 1.7 max TP reset  ─────────────────────────────────────────────
        if me_reset_mode == 3:
            temp      = np.zeros(me_max + 1, dtype=np.int32)
            new_count = 0
            cutoff    = i - me_period
            for ri in range(re_count):
                idx_ri = (re_head - re_count + ri) % (me_max + 1)
                if recent_entries[idx_ri] >= cutoff:
                    temp[new_count] = recent_entries[idx_ri]
                    new_count += 1
            for ri in range(new_count):
                recent_entries[ri] = temp[ri]
            re_head  = new_count
            re_count = new_count

        # ── 1.8 CHECK PENDING ORDERS ─────────────────────────
        if has_limit_orders and n_pending > 0:
            p = 0
            while p < n_pending:
                pend_side   = pending_orders[p, PEND_SIDE]
                limit_price = pending_orders[p, PEND_LIMIT_PRICE]
                expiry_bar  = pending_orders[p, PEND_EXPIRY_BAR]
                pend_setup  = int(pending_orders[p, PEND_SETUP_ID])
                pend_score  = pending_orders[p, PEND_SCORE]
                pend_tp     = pending_orders[p, PEND_TP_PRICE]
                pend_sl     = pending_orders[p, PEND_SL_PRICE]
                pend_ep_id  = int(pending_orders[p, PEND_EXIT_PROF])
                pend_es_id  = int(pending_orders[p, PEND_EXIT_STRAT])

                # Expiré → annuler
                if float(i) > expiry_bar:
                    pending_orders[p] = pending_orders[n_pending - 1]
                    n_pending -= 1
                    continue

                # Check fill
                filled     = False
                fill_price = limit_price

                if pend_side == 1.0 and lows[i] <= limit_price:
                    filled = True
                elif pend_side == -1.0 and highs[i] >= limit_price:
                    filled = True

                if filled:
                    # Filtres optionnels au fill
                    fill_ok = True

                    if check_filters_on_fill:
                        # Session
                        in_session_fill = (
                            (s1_start >= 0 and s1_start <= minutes_of_day[i] <= s1_end) or
                            (s2_start >= 0 and s2_start <= minutes_of_day[i] <= s2_end) or
                            (s3_start >= 0 and s3_start <= minutes_of_day[i] <= s3_end)
                        )
                        if s1_start < 0 and s2_start < 0 and s3_start < 0:
                            in_session_fill = True
                        if not in_session_fill:
                            fill_ok = False

                        # Cooldown global
                        if fill_ok and cooldown_entries > 0 and cd_until >= 0 and i <= cd_until:
                            fill_ok = False

                        # Max entries
                        if fill_ok and me_reset_mode > 0 and re_count >= me_max:
                            fill_ok = False

                        # Cooldown stateful par strat
                        if fill_ok and use_exit_system and n_state_global >= N_SG_DEFAULT and pend_es_id >= 0:
                            if state_global[pend_es_id, SG_COOLDOWN_UNTIL] > 0.0:
                                if float(i) <= state_global[pend_es_id, SG_COOLDOWN_UNTIL]:
                                    fill_ok = False

                        # Max positions
                        if fill_ok and n_pos >= MAX_POS:
                            fill_ok = False

                    if fill_ok:
                        # TP/SL depuis pending ou depuis %
                        if pend_ep_id >= 0 and pend_ep_id < profile_rt_matrix.shape[0]:
                            rt_tp_pct_p        = profile_rt_matrix[pend_ep_id, RT_TP_PCT]
                            rt_sl_pct_p        = profile_rt_matrix[pend_ep_id, RT_SL_PCT]
                            rt_use_atr_p       = int(profile_rt_matrix[pend_ep_id, RT_USE_ATR_SL_TP])
                            rt_tp_atr_mult_p   = profile_rt_matrix[pend_ep_id, RT_TP_ATR_MULT]
                            rt_sl_atr_mult_p   = profile_rt_matrix[pend_ep_id, RT_SL_ATR_MULT]
                        else:
                            rt_tp_pct_p        = tp_pct
                            rt_sl_pct_p        = sl_pct
                            rt_use_atr_p       = use_atr_sl_tp
                            rt_tp_atr_mult_p   = tp_atr_mult
                            rt_sl_atr_mult_p   = sl_atr_mult

                        atr_val_p = atrs[i] if has_atr else 0.0

                        if pend_tp > 0.0:
                            tp2 = pend_tp
                        else:
                            tp2, _ = _compute_tp_sl(
                                int(pend_side), fill_price,
                                rt_tp_pct_p, rt_sl_pct_p,
                                rt_use_atr_p, rt_tp_atr_mult_p, rt_sl_atr_mult_p,
                                atr_val_p,
                            )

                        if pend_sl > 0.0:
                            sl2 = pend_sl
                        else:
                            _, sl2 = _compute_tp_sl(
                                int(pend_side), fill_price,
                                rt_tp_pct_p, rt_sl_pct_p,
                                rt_use_atr_p, rt_tp_atr_mult_p, rt_sl_atr_mult_p,
                                atr_val_p,
                            )

                        # Ouvrir position
                        pos[n_pos, POS_SIDE]             = pend_side
                        pos[n_pos, POS_ENTRY_PRICE]      = fill_price
                        pos[n_pos, POS_TP]               = tp2
                        pos[n_pos, POS_SL]               = sl2
                        pos[n_pos, POS_ENTRY_IDX]        = float(i)
                        pos[n_pos, POS_BE_ARMED]         = 0.0
                        pos[n_pos, POS_BE_ACTIVE]        = 0.0
                        pos[n_pos, POS_BE_ARM_IDX]       = -1.0
                        pos[n_pos, POS_RUNNER_ARMED]     = 0.0
                        pos[n_pos, POS_RUNNER_ACTIVE]    = 0.0
                        pos[n_pos, POS_RUNNER_SL]        = 0.0
                        pos[n_pos, POS_PENDING_BE_SL]    = 0.0
                        pos[n_pos, POS_RUNNER_THRESHOLD] = 0.0
                        pos[n_pos, POS_TAG]              = 0.0
                        pos[n_pos, POS_MAE]              = 0.0
                        pos[n_pos, POS_MFE]              = 0.0
                        pos[n_pos, POS_SETUP_ID]         = float(pend_setup)
                        pos[n_pos, POS_SELECTED_SCORE]   = pend_score
                        pos[n_pos, POS_REMAINING_SIZE]   = 1.0
                        pos[n_pos, POS_ORIGINAL_SIZE]    = 1.0
                        pos[n_pos, POS_GROUP_ID]         = float(next_group_id)
                        pos[n_pos, POS_GROUP_SL_MODE]    = 0.0
                        next_group_id += 1

                        if use_exit_system:
                            if pend_ep_id >= 0:
                                exit_rt[n_pos, :] = profile_rt_matrix[pend_ep_id, :]
                            exit_rt[n_pos, RT_EXIT_STRATEGY_ID] = float(pend_es_id)

                            if n_state_per_pos >= N_SP_DEFAULT:
                                state_per_pos[n_pos, SP_PHASE]            = 0.0
                                state_per_pos[n_pos, SP_N_TP_HIT]         = 0.0
                                state_per_pos[n_pos, SP_REMAINING_SIZE]   = 1.0
                                state_per_pos[n_pos, SP_LAST_HIGH]        = highs[i] if pend_side == 1.0 else lows[i]
                                state_per_pos[n_pos, SP_BARS_SINCE_ENTRY] = 0.0
                                state_per_pos[n_pos, SP_BARS_SINCE_TP]    = 0.0
                                state_per_pos[n_pos, SP_ADD_COUNT]        = 0.0
                                state_per_pos[n_pos, SP_AVG_ENTRY]        = fill_price
                                state_per_pos[n_pos, SP_ENTRY_VALID]      = 1.0
                                state_per_pos[n_pos, SP_REGIME_AT_ENTRY]  = float(regime[i]) if use_regime else -1.0

                            if n_state_global >= N_SG_DEFAULT and pend_es_id >= 0:
                                state_global[pend_es_id, SG_TOTAL_EXPOSURE] += 1.0
                                if use_regime:
                                    state_global[pend_es_id, SG_CURRENT_REGIME] = float(regime[i])

                        n_pos += 1

                        # Cooldown trigger
                        if cooldown_entries > 0:
                            cd_count += 1
                            if cd_count >= cooldown_entries:
                                cd_until = i + cooldown_bars
                                cd_count = 0

                        # Max entries update
                        if me_reset_mode > 0:
                            recent_entries[re_head] = i
                            re_head  = (re_head + 1) % (me_max + 1)
                            re_count += 1

                    # Supprimer le pending — fill ou fill_ok=False sur expiry
                    pending_orders[p] = pending_orders[n_pending - 1]
                    n_pending -= 1
                else:
                    p += 1

        #--- TP max ----
        tp_day_changed = tp_last_day != -1 and cur_day != tp_last_day
        tp_session_changed = tp_last_session != -1 and cur_session != tp_last_session and cur_session != -1
        if tp_period_mode == 1:
            if tp_day_changed:
                tp_count = 0
            tp_last_day = cur_day
        elif tp_period_mode == 2:
            if tp_session_changed:
                tp_count = 0
            if cur_session != -1:
                tp_last_session = cur_session
        elif tp_period_mode == 3 and max_tp > 0:
            new_recent_count = 0
            cutoff_tp = i - tp_period_bars
            temp_tp = np.zeros(max_tp + 1, dtype=np.int32)
            for ri in range(tp_recent_count):
                idx_ri = (tp_head - tp_recent_count + ri) % (max_tp + 1)
                if recent_tp_idx[idx_ri] >= cutoff_tp:
                    temp_tp[new_recent_count] = recent_tp_idx[idx_ri]
                    new_recent_count += 1
            for ri in range(new_recent_count):
                recent_tp_idx[ri] = temp_tp[ri]
            tp_head = new_recent_count
            tp_recent_count = new_recent_count

        sl_day_changed = sl_last_day != -1 and cur_day != sl_last_day
        sl_session_changed = sl_last_session != -1 and cur_session != sl_last_session and cur_session != -1
        if sl_period_mode == 1:
            if sl_day_changed:
                sl_count = 0
            sl_last_day = cur_day
        elif sl_period_mode == 2:
            if sl_session_changed:
                sl_count = 0
            if cur_session != -1:
                sl_last_session = cur_session
        elif sl_period_mode == 3 and max_sl > 0:
            new_recent_count = 0
            cutoff_sl = i - sl_period_bars
            temp_sl = np.zeros(max_sl + 1, dtype=np.int32)
            for ri in range(sl_recent_count):
                idx_ri = (sl_head - sl_recent_count + ri) % (max_sl + 1)
                if recent_sl_idx[idx_ri] >= cutoff_sl:
                    temp_sl[new_recent_count] = recent_sl_idx[idx_ri]
                    new_recent_count += 1
            for ri in range(new_recent_count):
                recent_sl_idx[ri] = temp_sl[ri]
            sl_head = new_recent_count
            sl_recent_count = new_recent_count
            
        # ── 2. ENTRY ─────────────────────────────────────────────
        if sig != 0:

            in_session = (
                (s1_start >= 0 and s1_start <= minutes_of_day[i] <= s1_end) or
                (s2_start >= 0 and s2_start <= minutes_of_day[i] <= s2_end) or
                (s3_start >= 0 and s3_start <= minutes_of_day[i] <= s3_end)
            )
            if s1_start < 0 and s2_start < 0 and s3_start < 0:
                in_session = True

            me_ok = True
            if me_reset_mode > 0 and re_count >= me_max:
                me_ok = False

            cd_ok = not (cooldown_entries > 0 and cd_until >= 0 and i <= cd_until)
            tp_ok = True
            if max_tp > 0:
                if tp_period_mode == 3:
                    tp_ok = tp_recent_count < max_tp
                elif tp_period_mode in (1, 2):
                    tp_ok = tp_count < max_tp
            sl_ok = True
            if max_sl > 0:
                if sl_period_mode == 3:
                    sl_ok = sl_recent_count < max_sl
                elif sl_period_mode in (1, 2):
                    sl_ok = sl_count < max_sl

            if in_session and me_ok and cd_ok and tp_ok and sl_ok:
                gap_ok = True
                if max_gap_entry > 0.0:
                    gap_ok = abs(opens[i] - closes[i - 1]) / closes[i - 1] < max_gap_entry
                if gap_ok and max_gap_signal > 0.0 and i - entry_delay >= 1:
                    gap_ok = abs(opens[i - entry_delay] - closes[i - entry_delay - 1]) / closes[i - entry_delay - 1] < max_gap_signal

                candle_ok = True
                if candle_size_filter:
                    body      = abs(opens[i - entry_delay] - closes[i - entry_delay]) / closes[i - entry_delay]
                    candle_ok = min_size_pct < body < max_size_pct
                    if candle_ok and prev_candle_direction:
                        dir_ok    = (sig == 1  and closes[i - entry_delay] > opens[i - entry_delay]) or \
                                    (sig == -1 and closes[i - entry_delay] < opens[i - entry_delay])
                        candle_ok = dir_ok

                multi_ok = multi_entry or n_pos == 0
                pos_ok   = n_pos < MAX_POS

                # ---- resolve exit runtime params for this incoming trade ----
                if use_exit_system:
                    setup_id_now = int(delayed_setup_id[i])

                    if setup_id_now < 0 or setup_id_now >= setup_to_exit_profile.shape[0]:
                        raise ValueError("setup_id out of bounds for setup_to_exit_profile")

                    exit_profile_id_now = setup_to_exit_profile[setup_id_now]
                    exit_strategy_id_now = setup_to_exit_strategy[setup_id_now]

                    if exit_profile_id_now < 0 or exit_profile_id_now >= profile_rt_matrix.shape[0]:
                        raise ValueError("resolved exit_profile_id out of bounds for profile_rt_matrix")

                    rt_tp_pct        = profile_rt_matrix[exit_profile_id_now, RT_TP_PCT]
                    rt_sl_pct        = profile_rt_matrix[exit_profile_id_now, RT_SL_PCT]
                    rt_use_atr_sl_tp = int(profile_rt_matrix[exit_profile_id_now, RT_USE_ATR_SL_TP])
                    rt_tp_atr_mult   = profile_rt_matrix[exit_profile_id_now, RT_TP_ATR_MULT]
                    rt_sl_atr_mult   = profile_rt_matrix[exit_profile_id_now, RT_SL_ATR_MULT]

                    if use_regime and use_exit_system:
                        r_now = int(regime[i])
                        if regime_exit_profile_override[r_now] >= 0:
                            exit_profile_id_now = regime_exit_profile_override[r_now]
                        if regime_exit_strategy_override[r_now] >= 0:
                            exit_strategy_id_now = regime_exit_strategy_override[r_now]

                        if exit_profile_id_now < 0 or exit_profile_id_now >= profile_rt_matrix.shape[0]:
                            raise ValueError("regime-overridden exit_profile_id out of bounds for profile_rt_matrix")

                        rt_tp_pct        = profile_rt_matrix[exit_profile_id_now, RT_TP_PCT]
                        rt_sl_pct        = profile_rt_matrix[exit_profile_id_now, RT_SL_PCT]
                        rt_use_atr_sl_tp = int(profile_rt_matrix[exit_profile_id_now, RT_USE_ATR_SL_TP])
                        rt_tp_atr_mult   = profile_rt_matrix[exit_profile_id_now, RT_TP_ATR_MULT]
                        rt_sl_atr_mult   = profile_rt_matrix[exit_profile_id_now, RT_SL_ATR_MULT]
                else:
                    setup_id_now = -1
                    exit_profile_id_now = -1
                    exit_strategy_id_now = -1

                    rt_tp_pct        = tp_pct
                    rt_sl_pct        = sl_pct
                    rt_use_atr_sl_tp = use_atr_sl_tp
                    rt_tp_atr_mult   = tp_atr_mult
                    rt_sl_atr_mult   = sl_atr_mult

                # Vérifier max_simultaneous_positions
                max_pos_ok = True
                if has_stateful_cfg and use_exit_system and exit_strategy_id_now >= 0:
                    if exit_strategy_id_now < stateful_cfg_rt.shape[0]:
                        max_pos_cfg = stateful_cfg_rt[exit_strategy_id_now, SCFG_MAX_POSITIONS]
                        if max_pos_cfg > 0.0 and n_state_global >= N_SG_DEFAULT:
                            if state_global[exit_strategy_id_now, SG_TOTAL_EXPOSURE] >= max_pos_cfg:
                                max_pos_ok = False

                # Vérifier cooldown stateful par strat
                strat_cooldown_ok = True
                if use_exit_system and n_state_global >= N_SG_DEFAULT and exit_strategy_id_now >= 0:
                    if state_global[exit_strategy_id_now, SG_COOLDOWN_UNTIL] > 0.0:
                        if float(i) <= state_global[exit_strategy_id_now, SG_COOLDOWN_UNTIL]:
                            strat_cooldown_ok = False

                atr_val = atrs[i - entry_delay] if (has_atr and rt_use_atr_sl_tp != 0) else 0.0
                atr_ok  = not (rt_use_atr_sl_tp != 0 and has_atr and atr_val != atr_val)

                if gap_ok and candle_ok and pos_ok and atr_ok and strat_cooldown_ok and max_pos_ok:
                    if multi_ok:
                        # ── ENTRY PRICE : open ou close de la bougie signal ──

                        open_position = False

                        if has_limit_orders and entry_limit_prices[i] > 0.0:
                            limit_price = entry_limit_prices[i]
                            immediate_fill = False
                            if sig == 1 and lows[i] <= limit_price:
                                immediate_fill = True
                            elif sig == -1 and highs[i] >= limit_price:
                                immediate_fill = True

                            if not immediate_fill:
                                if n_pending < MAX_POS:
                                    pending_orders[n_pending, PEND_SIDE]        = float(sig)
                                    pending_orders[n_pending, PEND_LIMIT_PRICE] = limit_price
                                    pending_orders[n_pending, PEND_EXPIRY_BAR]  = float(i + limit_expiry_bars)
                                    pending_orders[n_pending, PEND_SIGNAL_BAR]  = float(i)
                                    pending_orders[n_pending, PEND_SETUP_ID]    = float(setup_id_now)
                                    pending_orders[n_pending, PEND_SCORE]       = delayed_selected_score[i]
                                    pending_orders[n_pending, PEND_TP_PRICE]    = tp_price_array[i] if tp_price_array.shape[0] == n else -1.0
                                    pending_orders[n_pending, PEND_SL_PRICE]    = sl_price_array[i] if sl_price_array.shape[0] == n else -1.0
                                    pending_orders[n_pending, PEND_EXIT_PROF]   = float(exit_profile_id_now)
                                    pending_orders[n_pending, PEND_EXIT_STRAT]  = float(exit_strategy_id_now)
                                    n_pending += 1
                                # open_position reste False → skip
                            else:
                                ep2 = limit_price
                                open_position = True

                        else:
                            # Market order — comportement actuel
                            if entry_on_signal_close_price:
                                ep2 = closes[i - 1]
                            elif entry_on_close:
                                ep2 = closes[i]
                            else:
                                ep2 = o
                            open_position = True

                        if open_position:
                            # TP/SL
                            if tp_price_array.shape[0] == n and tp_price_array[i] > 0.0:
                                tp2 = tp_price_array[i]
                                _, sl2 = _compute_tp_sl(sig, ep2, rt_tp_pct, rt_sl_pct,
                                                        rt_use_atr_sl_tp, rt_tp_atr_mult, rt_sl_atr_mult, atr_val)
                            elif sl_price_array.shape[0] == n and sl_price_array[i] > 0.0:
                                tp2, _ = _compute_tp_sl(sig, ep2, rt_tp_pct, rt_sl_pct,
                                                        rt_use_atr_sl_tp, rt_tp_atr_mult, rt_sl_atr_mult, atr_val)
                                sl2 = sl_price_array[i]
                            else:
                                tp2, sl2 = _compute_tp_sl(sig, ep2, rt_tp_pct, rt_sl_pct,
                                                        rt_use_atr_sl_tp, rt_tp_atr_mult, rt_sl_atr_mult, atr_val)

                            pos[n_pos, POS_SIDE]             = float(sig)
                            pos[n_pos, POS_ENTRY_PRICE]      = ep2
                            pos[n_pos, POS_TP]               = tp2
                            pos[n_pos, POS_SL]               = sl2
                            pos[n_pos, POS_ENTRY_IDX]        = float(i)
                            pos[n_pos, POS_BE_ARMED]         = 0.0
                            pos[n_pos, POS_BE_ACTIVE]        = 0.0
                            pos[n_pos, POS_BE_ARM_IDX]       = -1.0
                            pos[n_pos, POS_RUNNER_ARMED]     = 0.0
                            pos[n_pos, POS_RUNNER_ACTIVE]    = 0.0
                            pos[n_pos, POS_RUNNER_SL]        = 0.0
                            pos[n_pos, POS_PENDING_BE_SL]    = 0.0
                            pos[n_pos, POS_RUNNER_THRESHOLD] = 0.0
                            pos[n_pos, POS_TAG]              = signal_tags[i] if has_tags else 0.0
                            pos[n_pos, POS_MAE]              = 0.0
                            pos[n_pos, POS_MFE]              = 0.0
                            pos[n_pos, POS_SETUP_ID]         = float(delayed_setup_id[i])
                            pos[n_pos, POS_SELECTED_SCORE]   = delayed_selected_score[i]

                            n_pos += 1

                            pos[n_pos - 1, POS_REMAINING_SIZE] = 1.0
                            pos[n_pos - 1, POS_ORIGINAL_SIZE]  = 1.0
                            pos[n_pos - 1, POS_GROUP_ID]       = float(next_group_id)
                            pos[n_pos - 1, POS_GROUP_SL_MODE]  = 0.0  # indépendant par défaut
                            next_group_id += 1

                            if use_exit_system:
                                exit_rt[n_pos - 1, :] = profile_rt_matrix[exit_profile_id_now, :]    # ← n_pos-1 pas n_pos
                                exit_rt[n_pos - 1, RT_EXIT_STRATEGY_ID] = exit_strategy_id_now

                                if n_state_per_pos >= N_SP_DEFAULT:
                                    state_per_pos[n_pos - 1, SP_PHASE]            = 0.0              # ← n_pos-1 pas n_pos
                                    state_per_pos[n_pos - 1, SP_N_TP_HIT]         = 0.0
                                    state_per_pos[n_pos - 1, SP_REMAINING_SIZE]   = 1.0
                                    state_per_pos[n_pos - 1, SP_LAST_HIGH]        = highs[i] if sig == 1 else lows[i]
                                    state_per_pos[n_pos - 1, SP_BARS_SINCE_ENTRY] = 0.0
                                    state_per_pos[n_pos - 1, SP_BARS_SINCE_TP]    = 0.0
                                    state_per_pos[n_pos - 1, SP_ADD_COUNT]        = 0.0
                                    state_per_pos[n_pos - 1, SP_AVG_ENTRY]        = ep2
                                    state_per_pos[n_pos - 1, SP_ENTRY_VALID]      = 1.0
                                    state_per_pos[n_pos - 1, SP_REGIME_AT_ENTRY]  = float(regime[i]) if use_regime else -1.0

                                # Update state global — exposition
                                if n_state_global >= N_SG_DEFAULT and exit_strategy_id_now >= 0:
                                    state_global[exit_strategy_id_now, SG_TOTAL_EXPOSURE] += 1.0
                                    if use_regime:
                                        state_global[exit_strategy_id_now, SG_CURRENT_REGIME] = float(regime[i])

                            # ── COOLDOWN TRIGGER ─────────────────────
                            if cooldown_entries > 0:
                                cd_count += 1
                                if cd_count >= cooldown_entries:
                                    cd_until = i + cooldown_bars
                                    cd_count = 0

                        if me_reset_mode > 0:
                            recent_entries[re_head] = i
                            re_head  = (re_head + 1) % (me_max + 1)
                            re_count += 1

        # ── 3. EXIT SIGNAL LIFO ──────────────────────────────────
        if has_exit_sig and n_pos > 0:
            es = exit_signals[i - exit_delay] if i >= exit_delay else 0
            if es == 1:
                last = n_pos - 1
                if use_exit_system:
                    exit_profile_id_last = int(exit_rt[last, RT_EXIT_PROFILE_ID])
                    exit_strategy_id_last = int(exit_rt[last, RT_EXIT_STRATEGY_ID])
                else:
                    exit_profile_id_last = -1
                    exit_strategy_id_last = -1 
                ep   = pos[last, POS_ENTRY_PRICE]
                side = pos[last, POS_SIDE]
                if n_trades < MAX_TRADES:
                    setup_id_last  = int(pos[last, POS_SETUP_ID])
                    sel_score_last = pos[last, POS_SELECTED_SCORE]

                     
                    n_trades = _record_trade(
                        trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                        trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                        trade_setup_ids, trade_selected_score,
                        trade_exit_profile_ids,trade_exit_strategy_ids,
                        trade_regime_ids,trade_phase, trade_n_tp_hit, trade_add_count,trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,
                        n_trades, side, o, ep, int(pos[last, POS_ENTRY_IDX]), i,
                        REASON_EXIT_SIG, pos[last, POS_MAE], pos[last, POS_MFE],
                        setup_id_last, sel_score_last,exit_profile_id_last,exit_strategy_id_last,
                        cur_regime_id,  # regime_id ← nouveau
                        state_global,              # ← nouveau
                        n_state_global,            # ← nouveau
                        use_exit_system,state_per_pos, last, n_state_per_pos, int(pos[last, POS_GROUP_ID]),           # ← nouveau
                    )


                if use_exit_system:
                    exit_rt[last] = exit_rt[n_pos - 1]
                if use_exit_system and n_state_per_pos > 0:    # ← manquant
                    state_per_pos[last] = state_per_pos[n_pos - 1]
                    state_per_pos[n_pos - 1, :] = 0.0
                pos[last] = pos[n_pos - 1]
                n_pos -= 1

        # ── 4. EXIT SL/TP ────────────────────────────────────────
        k = 0
        while k < n_pos:
            cur_regime_id = int(regime[i]) if use_regime else -1
            side       = pos[k, POS_SIDE];  ep = pos[k, POS_ENTRY_PRICE]
            tp         = pos[k, POS_TP];    sl = pos[k, POS_SL]
            entry_idx  = int(pos[k, POS_ENTRY_IDX])
            be_armed   = pos[k, POS_BE_ARMED];  be_active = pos[k, POS_BE_ACTIVE]
            be_arm_idx = pos[k, POS_BE_ARM_IDX]
            r_armed    = pos[k, POS_RUNNER_ARMED]; r_active = pos[k, POS_RUNNER_ACTIVE]
            r_sl       = pos[k, POS_RUNNER_SL];    tag      = pos[k, POS_TAG]
            setup_id   = int(pos[k, POS_SETUP_ID])
            sel_score  = pos[k, POS_SELECTED_SCORE]

            if use_exit_system:
                exit_profile_id         = int(exit_rt[k, RT_EXIT_PROFILE_ID])
                rt_be_trigger_pct       = exit_rt[k, RT_BE_TRIGGER_PCT]
                rt_be_offset_pct        = exit_rt[k, RT_BE_OFFSET_PCT]
                rt_be_delay_bars        = int(exit_rt[k, RT_BE_DELAY_BARS])
                rt_trailing_trigger_pct = exit_rt[k, RT_TRAILING_TRIGGER_PCT]
                rt_runner_trailing_mult = exit_rt[k, RT_RUNNER_TRAILING_MULT]
                rt_max_holding_bars     = int(exit_rt[k, RT_MAX_HOLDING_BARS])
                exit_strategy_id        = int(exit_rt[k, RT_EXIT_STRATEGY_ID])
            else:
                rt_be_trigger_pct       = be_trigger_pct
                rt_be_offset_pct        = be_offset_pct
                rt_be_delay_bars        = be_delay_bars
                rt_trailing_trigger_pct = trailing_trigger_pct
                rt_runner_trailing_mult = runner_trailing_mult
                rt_max_holding_bars     = max_holding_bars
                exit_profile_id         = -1
                exit_strategy_id        = -1

            # ── Stateful auto-update ──────────────────────────────
            if use_exit_system and n_state_per_pos >= N_SP_DEFAULT:
                state_per_pos[k, SP_BARS_SINCE_ENTRY] += 1.0
                state_per_pos[k, SP_BARS_SINCE_TP]    += 1.0
                if side == 1.0:
                    if highs[i] > state_per_pos[k, SP_LAST_HIGH]:
                        state_per_pos[k, SP_LAST_HIGH] = highs[i]
                else:
                    if lows[i] < state_per_pos[k, SP_LAST_HIGH]:
                        state_per_pos[k, SP_LAST_HIGH] = lows[i]
                if use_regime:
                    if exit_strategy_id >= 0 and n_state_global >= N_SG_DEFAULT:
                        state_global[exit_strategy_id, SG_CURRENT_REGIME] = float(regime[i])

            # ── Bloc A — StatefulConfig natif ────────────────────
            if has_stateful_cfg and use_exit_system and exit_strategy_id >= 0:
                if exit_strategy_id < stateful_cfg_rt.shape[0]:

                    # max_consec_sl → cooldown
                    max_csl = stateful_cfg_rt[exit_strategy_id, SCFG_MAX_CONSEC_SL]
                    cd_bars = stateful_cfg_rt[exit_strategy_id, SCFG_COOLDOWN_BARS]
                    if max_csl > 0.0 and cd_bars > 0.0:
                        if n_state_global >= N_SG_DEFAULT:
                            if state_global[exit_strategy_id, SG_CONSEC_SL] >= max_csl:
                                if state_global[exit_strategy_id, SG_COOLDOWN_UNTIL] <= float(i):
                                    state_global[exit_strategy_id, SG_COOLDOWN_UNTIL] = float(i) + cd_bars

                    # invalidate_on_regime_change
                    inv_regime = stateful_cfg_rt[exit_strategy_id, SCFG_INVALIDATE_ON_REGIME]
                    if inv_regime > 0.0 and use_regime and n_state_per_pos >= N_SP_DEFAULT:
                        regime_at_entry = state_per_pos[k, SP_REGIME_AT_ENTRY]
                        if regime_at_entry >= 0.0 and float(regime[i]) != regime_at_entry:
                            state_per_pos[k, SP_ENTRY_VALID] = 0.0

                    # min_rolling_winrate → cooldown
                    min_wr   = stateful_cfg_rt[exit_strategy_id, SCFG_MIN_WINRATE]
                    wr_cd    = stateful_cfg_rt[exit_strategy_id, SCFG_WINRATE_COOLDOWN]
                    if min_wr > 0.0 and wr_cd > 0.0 and n_state_global >= N_SG_DEFAULT:
                        cur_wr = state_global[exit_strategy_id, SG_ROLLING_WINRATE]
                        if cur_wr > 0.0 and cur_wr < min_wr:
                            if state_global[exit_strategy_id, SG_COOLDOWN_UNTIL] <= float(i):
                                state_global[exit_strategy_id, SG_COOLDOWN_UNTIL] = float(i) + wr_cd

            # ── Bloc B — Phase redéfinition params ───────────────
            if has_phases and use_exit_system and exit_profile_id >= 0:
                cur_phase = int(state_per_pos[k, SP_PHASE])
                if (exit_profile_id < phase_rt_matrix.shape[0] and
                        cur_phase < phase_rt_matrix.shape[1]):
                    ph_tp  = phase_rt_matrix[exit_profile_id, cur_phase, PHASE_TP_PCT]
                    ph_sl  = phase_rt_matrix[exit_profile_id, cur_phase, PHASE_SL_PCT]
                    ph_be  = phase_rt_matrix[exit_profile_id, cur_phase, PHASE_BE_TRIGGER]
                    ph_tr  = phase_rt_matrix[exit_profile_id, cur_phase, PHASE_TRAILING]
                    ph_mh  = phase_rt_matrix[exit_profile_id, cur_phase, PHASE_MAX_HOLD]

                    if ph_tp > 0.0:
                        pos[k, POS_TP] = ep * (1.0 + side * ph_tp)
                        tp = pos[k, POS_TP]
                    if ph_sl > 0.0:
                        pos[k, POS_SL] = ep * (1.0 - side * ph_sl)
                        sl = pos[k, POS_SL]
                    if ph_be >= 0.0:
                        rt_be_trigger_pct = ph_be
                    if ph_tr >= 0.0:
                        rt_trailing_trigger_pct = ph_tr
                    if ph_mh >= 0.0:
                        rt_max_holding_bars = int(ph_mh)

            # ── Invalidation structurelle ─────────────────────────
            if use_exit_system and n_state_per_pos >= N_SP_DEFAULT:
                if state_per_pos[k, SP_ENTRY_VALID] == 0.0:
                    if n_trades < MAX_TRADES:
                        n_trades = _record_trade(
                            trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                            trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                            trade_setup_ids, trade_selected_score,
                            trade_exit_profile_ids, trade_exit_strategy_ids,
                             trade_regime_ids,trade_phase, trade_n_tp_hit, trade_add_count,trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,
                            n_trades, side, o, ep, entry_idx, i,
                            REASON_INVALIDATED,
                            pos[k, POS_MAE], pos[k, POS_MFE],
                            setup_id, sel_score, exit_profile_id, exit_strategy_id,
                            cur_regime_id,
                            state_global, n_state_global, use_exit_system,state_per_pos, k, n_state_per_pos, int(pos[k, POS_GROUP_ID]),
                        )
                    if use_exit_system:
                        exit_rt[k] = exit_rt[n_pos - 1]
                    if use_exit_system and n_state_per_pos > 0:
                        state_per_pos[k] = state_per_pos[n_pos - 1]
                        state_per_pos[n_pos - 1, :] = 0.0
                    pos[k] = pos[n_pos - 1]; n_pos -= 1
                    continue

            # ── 1. Allow exit on entry bar ────────────────────────
            if not allow_exit_on_entry_bar and i == entry_idx:
                k += 1; continue

            # ── 2. Max holding bars ───────────────────────────────
            if rt_max_holding_bars > 0 and (i - entry_idx) >= rt_max_holding_bars:
                if n_trades < MAX_TRADES:
                    n_trades = _record_trade(
                        trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                        trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                        trade_setup_ids, trade_selected_score,
                        trade_exit_profile_ids, trade_exit_strategy_ids,
                         trade_regime_ids,trade_phase, trade_n_tp_hit, trade_add_count,trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,
                        n_trades, side, o, ep, entry_idx, i, REASON_MAX_HOLD,
                        pos[k, POS_MAE], pos[k, POS_MFE],
                        setup_id, sel_score, exit_profile_id, exit_strategy_id,
                        cur_regime_id,
                        state_global, n_state_global, use_exit_system,state_per_pos, k, n_state_per_pos, int(pos[k, POS_GROUP_ID]),
                    )
                if use_exit_system:
                    exit_rt[k] = exit_rt[n_pos - 1]
                if use_exit_system and n_state_per_pos > 0:
                    state_per_pos[k] = state_per_pos[n_pos - 1]
                    state_per_pos[n_pos - 1, :] = 0.0
                pos[k] = pos[n_pos - 1]
                n_pos -= 1
                continue

            # ── 3. MAE/MFE update ─────────────────────────────────
            if track_mae_mfe:
                if side == 1:
                    pos[k, POS_MFE] = max(pos[k, POS_MFE], (h - ep) / ep)
                    pos[k, POS_MAE] = min(pos[k, POS_MAE], (l - ep) / ep)
                else:
                    pos[k, POS_MFE] = max(pos[k, POS_MFE], (ep - l) / ep)
                    pos[k, POS_MAE] = min(pos[k, POS_MAE], (ep - h) / ep)

            # -- priorité sl tp si demandé —————————————————————————————
            if not sl_tp_be_priority:
                if rt_be_trigger_pct > 0.0:
                    pending_be = pos[k, POS_PENDING_BE_SL]
                    sl, be_armed, be_active, be_arm_idx, pending_be = _update_be(
                        side, ep, sl, i, entry_idx, be_armed, be_active,
                        be_arm_idx, pending_be, h, l,
                        rt_be_trigger_pct, rt_be_offset_pct, rt_be_delay_bars
                    )
                    pos[k, POS_SL]            = sl
                    pos[k, POS_BE_ARMED]      = be_armed
                    pos[k, POS_BE_ACTIVE]     = be_active
                    pos[k, POS_BE_ARM_IDX]    = be_arm_idx
                    pos[k, POS_PENDING_BE_SL] = pending_be

                # ── 3.3 Trailing update ────────────────────────────────
                if rt_trailing_trigger_pct > 0.0 and has_atr:
                    r_threshold = pos[k, POS_RUNNER_THRESHOLD]
                    r_armed     = pos[k, POS_RUNNER_ARMED]
                    r_active    = pos[k, POS_RUNNER_ACTIVE]
                    r_sl        = pos[k, POS_RUNNER_SL]
                    r_active_before = r_active
                    r_armed, r_active, r_sl, r_threshold = _update_runner(
                        side, ep, r_armed, r_active, r_sl, r_threshold,
                        h, l, c, atrs[i - 1],
                        rt_trailing_trigger_pct, rt_runner_trailing_mult
                    )
                    pos[k, POS_RUNNER_ARMED]     = r_armed
                    pos[k, POS_RUNNER_ACTIVE]    = r_active
                    pos[k, POS_RUNNER_SL]        = r_sl
                    pos[k, POS_RUNNER_THRESHOLD] = r_threshold
                    if r_active_before == 0.0 and r_active == 1.0:
                        k += 1; continue

            # ── 3.4 Check exit SL/TP ───────────────────────────────
            
                exit_price = -1.0; reason = 0

                if r_active and r_sl != 0.0:
                    r_threshold = pos[k, POS_RUNNER_THRESHOLD]
                    be_rsn = REASON_BE if be_active else REASON_SL
                    under_threshold = r_threshold > 0.0 and (
                        (side == 1  and r_sl <= r_threshold) or
                        (side == -1 and r_sl >= r_threshold)
                    )
                    if under_threshold:
                        if side == 1:
                            if o <= sl:   exit_price = o;  reason = be_rsn
                            elif l <= sl: exit_price = sl; reason = be_rsn
                        else:
                            if o >= sl:   exit_price = o;  reason = be_rsn
                            elif h >= sl: exit_price = sl; reason = be_rsn
                    else:
                        if side == 1:
                            if o <= r_sl:   exit_price = o;    reason = REASON_RUNNER_SL
                            elif l <= r_sl: exit_price = r_sl; reason = REASON_RUNNER_SL
                            elif o <= sl:   exit_price = o;    reason = be_rsn
                            elif l <= sl:   exit_price = sl;   reason = be_rsn
                        else:
                            if o >= r_sl:   exit_price = o;    reason = REASON_RUNNER_SL
                            elif h >= r_sl: exit_price = r_sl; reason = REASON_RUNNER_SL
                            elif o >= sl:   exit_price = o;    reason = be_rsn
                            elif h >= sl:   exit_price = sl;   reason = be_rsn

                elif ema_exit_mode and has_ema_exit:
                    e1 = exit_ema1[i]
                    e2 = exit_ema2[i] if exit_ema2.shape[0] == n else 0.0
                    exit_price, reason = _check_exit_ema(
                        side, o, h, l, c, tp, sl,
                        be_active == 1.0, ep, e1, e2,
                        use_ema1_tp, use_ema2_tp, use_ema_cross_tp
                    )
                else:
                    exit_price, reason = _check_exit_fixed(
                        side, o, h, l, tp, sl, be_active == 1.0
                    )

            # ── Bloc C — Partial TP natif ─────────────────────────
            if has_partial and use_exit_system and exit_profile_id >= 0:
                n_tp_hit = int(state_per_pos[k, SP_N_TP_HIT])
                if (exit_profile_id < partial_rt_matrix.shape[0] and
                        n_tp_hit < max_partial_levels and
                        n_tp_hit < partial_rt_matrix.shape[1]):
                    ttype = partial_rt_matrix[exit_profile_id, n_tp_hit, PART_TRIGGER_TYPE]
                    tval  = partial_rt_matrix[exit_profile_id, n_tp_hit, PART_TRIGGER_VALUE]
                    tfeat = partial_rt_matrix[exit_profile_id, n_tp_hit, PART_TRIGGER_FEAT]
                    top   = partial_rt_matrix[exit_profile_id, n_tp_hit, PART_TRIGGER_OP]

                    if ttype >= 0.0:
                        triggered = _eval_trigger_simple(
                            int(ttype), tval, tfeat, top,
                            i, k, pos, features, state_per_pos,
                            atrs, has_atr, ep, side,
                        )
                        if triggered:
                            dist_mode = int(partial_rt_matrix[exit_profile_id, n_tp_hit, PART_DIST_MODE])
                            dp1       = partial_rt_matrix[exit_profile_id, n_tp_hit, PART_DIST_PARAM1]
                            dp2       = partial_rt_matrix[exit_profile_id, n_tp_hit, PART_DIST_PARAM2]
                            fraction  = partial_rt_matrix[exit_profile_id, n_tp_hit, PART_FRACTION]

                            if dist_mode != DIST_MODE_EQUAL or dp2 > 0.0:
                                fraction = _eval_dist_fn(dist_mode, dp1, dp2, float(n_tp_hit))

                            ref_size = pos[k, POS_REMAINING_SIZE]
                            ref_mode = int(partial_rt_matrix[exit_profile_id, n_tp_hit, PART_REF])
                            if ref_mode == 1:
                                ref_size = pos[k, POS_ORIGINAL_SIZE]

                            part_size = fraction * ref_size
                            new_remaining = pos[k, POS_REMAINING_SIZE] - part_size

                            # ← ajouter ici
                            pos[k, POS_REMAINING_SIZE] = new_remaining
                            state_per_pos[k, SP_REMAINING_SIZE] = new_remaining

                            if n_trades < MAX_TRADES:
                                trade_sizes[n_trades] = part_size
                                n_trades = _record_trade(
                                    trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                                    trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                                    trade_setup_ids, trade_selected_score,
                                    trade_exit_profile_ids, trade_exit_strategy_ids,
                                     trade_regime_ids,trade_phase, trade_n_tp_hit, trade_add_count,trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,
                                    n_trades, side, o, ep, entry_idx, i,
                                    REASON_PARTIAL_TP,
                                    pos[k, POS_MAE], pos[k, POS_MFE],
                                    setup_id, sel_score, exit_profile_id, exit_strategy_id,
                                    cur_regime_id,
                                    state_global, n_state_global, use_exit_system,state_per_pos, k, n_state_per_pos, int(pos[k, POS_GROUP_ID]),
                                )
                                if max_tp > 0:
                                    if tp_period_mode == 3:
                                        recent_tp_idx[tp_head] = i
                                        tp_head = (tp_head + 1) % (max_tp + 1)
                                        tp_recent_count += 1
                                    else:
                                        tp_count += 1

                            state_per_pos[k, SP_N_TP_HIT]     += 1.0
                            state_per_pos[k, SP_BARS_SINCE_TP]  = 0.0

                            move_be = partial_rt_matrix[exit_profile_id, n_tp_hit, PART_MOVE_BE]
                            if move_be > 0.0:
                                pos[k, POS_SL] = ep
                                sl = ep

                            if new_remaining <= 0.01:
                                if use_exit_system:
                                    exit_rt[k] = exit_rt[n_pos - 1]
                                if n_state_per_pos > 0:
                                    state_per_pos[k] = state_per_pos[n_pos - 1]
                                    state_per_pos[n_pos - 1, :] = 0.0
                                pos[k] = pos[n_pos - 1]
                                n_pos -= 1
                                continue


            # ── Bloc D — Pyramiding natif ─────────────────────────
            if has_pyramid and use_exit_system and exit_profile_id >= 0 and n_pos < MAX_POS:
                add_count = int(state_per_pos[k, SP_ADD_COUNT])
                if (exit_profile_id < pyramid_rt_matrix.shape[0] and
                        add_count < max_pyramid_levels and
                        add_count < pyramid_rt_matrix.shape[1]):
                    ttype = pyramid_rt_matrix[exit_profile_id, add_count, PYR_TRIGGER_TYPE]
                    tval  = pyramid_rt_matrix[exit_profile_id, add_count, PYR_TRIGGER_VALUE]
                    tfeat = pyramid_rt_matrix[exit_profile_id, add_count, PYR_TRIGGER_FEAT]
                    top   = pyramid_rt_matrix[exit_profile_id, add_count, PYR_TRIGGER_OP]

                    if ttype >= 0.0:
                        triggered = _eval_trigger_simple(
                            int(ttype), tval, tfeat, top,
                            i, k, pos, features, state_per_pos,
                            atrs, has_atr, ep, side,
                        )
                        if triggered:
                            dist_mode = int(pyramid_rt_matrix[exit_profile_id, add_count, PYR_DIST_MODE])
                            dp1       = pyramid_rt_matrix[exit_profile_id, add_count, PYR_DIST_PARAM1]
                            dp2       = pyramid_rt_matrix[exit_profile_id, add_count, PYR_DIST_PARAM2]
                            base_frac = pyramid_rt_matrix[exit_profile_id, add_count, PYR_SIZE_FRACTION]
                            size_frac = _eval_dist_fn(dist_mode, dp1, dp2, float(add_count))
                            if size_frac <= 0.0:
                                size_frac = base_frac

                            sl_mode  = int(pyramid_rt_matrix[exit_profile_id, add_count, PYR_SL_MODE])
                            grp_mode = int(pyramid_rt_matrix[exit_profile_id, add_count, PYR_GROUP_SL_MODE])

                            # SL de la nouvelle position
                            if sl_mode == 0:   # breakeven
                                add_sl = ep
                            elif sl_mode == 1: # original
                                add_sl = pos[k, POS_SL]
                            elif sl_mode == 3: # atr_mult
                                atr_m = pyramid_rt_matrix[exit_profile_id, add_count, PYR_SL_ATR_MULT]
                                if has_atr and i >= 1:
                                    add_sl = o - side * atrs[i - 1] * atr_m
                                else:
                                    add_sl = pos[k, POS_SL]
                            else:
                                add_sl = pos[k, POS_SL]

                            pos[n_pos, POS_SIDE]             = side
                            pos[n_pos, POS_ENTRY_PRICE]      = o
                            pos[n_pos, POS_TP]               = pos[k, POS_TP]
                            pos[n_pos, POS_SL]               = add_sl
                            pos[n_pos, POS_ENTRY_IDX]        = float(i)
                            pos[n_pos, POS_REMAINING_SIZE]   = size_frac
                            pos[n_pos, POS_ORIGINAL_SIZE]    = size_frac
                            pos[n_pos, POS_GROUP_ID]         = pos[k, POS_GROUP_ID]
                            pos[n_pos, POS_GROUP_SL_MODE]    = float(grp_mode)
                            pos[n_pos, POS_MAE]              = 0.0
                            pos[n_pos, POS_MFE]              = 0.0
                            pos[n_pos, POS_SETUP_ID]         = float(setup_id)
                            pos[n_pos, POS_SELECTED_SCORE]   = sel_score
                            pos[n_pos, POS_BE_ARMED]         = 0.0
                            pos[n_pos, POS_BE_ACTIVE]        = 0.0
                            pos[n_pos, POS_BE_ARM_IDX]       = -1.0
                            pos[n_pos, POS_RUNNER_ARMED]     = 0.0
                            pos[n_pos, POS_RUNNER_ACTIVE]    = 0.0
                            pos[n_pos, POS_RUNNER_SL]        = 0.0
                            pos[n_pos, POS_PENDING_BE_SL]    = 0.0
                            pos[n_pos, POS_RUNNER_THRESHOLD] = 0.0

                            if use_exit_system:
                                exit_rt[n_pos, :] = exit_rt[k, :]

                            if n_state_per_pos >= N_SP_DEFAULT:
                                state_per_pos[n_pos, SP_REMAINING_SIZE]  = size_frac
                                state_per_pos[n_pos, SP_ENTRY_VALID]     = 1.0
                                state_per_pos[n_pos, SP_AVG_ENTRY]       = o
                                state_per_pos[n_pos, SP_REGIME_AT_ENTRY] = float(regime[i]) if use_regime else -1.0

                            state_per_pos[k, SP_ADD_COUNT] += 1.0

                            move_be = pyramid_rt_matrix[exit_profile_id, add_count, PYR_SL_MODE]
                            if sl_mode == 0:  # breakeven → aussi déplacer SL position parent
                                pos[k, POS_SL] = ep; sl = ep

                            n_pos += 1

            # ── Bloc E — Averaging natif ──────────────────────────
            if has_averaging and use_exit_system and exit_profile_id >= 0 and n_pos < MAX_POS:
                avg_count = int(state_per_pos[k, SP_ADD_COUNT])
                if (exit_profile_id < averaging_rt_matrix.shape[0] and
                        avg_count < max_avg_levels and
                        avg_count < averaging_rt_matrix.shape[1]):
                    ttype    = averaging_rt_matrix[exit_profile_id, avg_count, AVG_TRIGGER_TYPE]
                    tval     = averaging_rt_matrix[exit_profile_id, avg_count, AVG_TRIGGER_VALUE]
                    tfeat    = averaging_rt_matrix[exit_profile_id, avg_count, AVG_TRIGGER_FEAT]
                    top      = averaging_rt_matrix[exit_profile_id, avg_count, AVG_TRIGGER_OP]
                    max_down = averaging_rt_matrix[exit_profile_id, avg_count, AVG_MAX_DOWN]

                    # Vérifier max_avg_down_pct
                    down_ok = True
                    if max_down < 0.0:
                        down_ok = pos[k, POS_MAE] >= max_down

                    if ttype >= 0.0 and down_ok:
                        triggered = _eval_trigger_simple(
                            int(ttype), tval, tfeat, top,
                            i, k, pos, features, state_per_pos,
                            atrs, has_atr, ep, side,
                        )
                        if triggered:
                            dist_mode = int(averaging_rt_matrix[exit_profile_id, avg_count, AVG_DIST_MODE])
                            dp1       = averaging_rt_matrix[exit_profile_id, avg_count, AVG_DIST_PARAM1]
                            dp2       = averaging_rt_matrix[exit_profile_id, avg_count, AVG_DIST_PARAM2]
                            base_frac = averaging_rt_matrix[exit_profile_id, avg_count, AVG_SIZE_FRACTION]
                            size_frac = _eval_dist_fn(dist_mode, dp1, dp2, float(avg_count))
                            if size_frac <= 0.0:
                                size_frac = base_frac

                            pos[n_pos, POS_SIDE]             = side
                            pos[n_pos, POS_ENTRY_PRICE]      = o
                            pos[n_pos, POS_TP]               = pos[k, POS_TP]
                            pos[n_pos, POS_SL]               = pos[k, POS_SL]
                            pos[n_pos, POS_ENTRY_IDX]        = float(i)
                            pos[n_pos, POS_REMAINING_SIZE]   = size_frac
                            pos[n_pos, POS_ORIGINAL_SIZE]    = size_frac
                            pos[n_pos, POS_GROUP_ID]         = pos[k, POS_GROUP_ID]
                            pos[n_pos, POS_GROUP_SL_MODE]    = 1.0
                            pos[n_pos, POS_MAE]              = 0.0
                            pos[n_pos, POS_MFE]              = 0.0
                            pos[n_pos, POS_SETUP_ID]         = float(setup_id)
                            pos[n_pos, POS_SELECTED_SCORE]   = sel_score
                            pos[n_pos, POS_BE_ARMED]         = 0.0
                            pos[n_pos, POS_BE_ACTIVE]        = 0.0
                            pos[n_pos, POS_BE_ARM_IDX]       = -1.0
                            pos[n_pos, POS_RUNNER_ARMED]     = 0.0
                            pos[n_pos, POS_RUNNER_ACTIVE]    = 0.0
                            pos[n_pos, POS_RUNNER_SL]        = 0.0
                            pos[n_pos, POS_PENDING_BE_SL]    = 0.0
                            pos[n_pos, POS_RUNNER_THRESHOLD] = 0.0

                            if use_exit_system:
                                exit_rt[n_pos, :] = exit_rt[k, :]

                            if n_state_per_pos >= N_SP_DEFAULT:
                                state_per_pos[n_pos, SP_REMAINING_SIZE]  = size_frac
                                state_per_pos[n_pos, SP_ENTRY_VALID]     = 1.0
                                state_per_pos[n_pos, SP_AVG_ENTRY]       = o
                                state_per_pos[n_pos, SP_REGIME_AT_ENTRY] = float(regime[i]) if use_regime else -1.0

                            state_per_pos[k, SP_ADD_COUNT] += 1.0
                            n_pos += 1

            # ── Bloc F — Rules custom (phases) ────────────────────
            if has_rules and use_exit_system and exit_profile_id >= 0:
                cur_phase = int(state_per_pos[k, SP_PHASE])
                if exit_profile_id < rule_trigger_matrix.shape[0]:
                    for rid in range(max_rules):
                        if rid >= rule_trigger_matrix.shape[1]:
                            break

                        phase_filter = int(rule_trigger_matrix[exit_profile_id, rid, RT_RULE_PHASE_FILTER])
                        if phase_filter >= 0 and phase_filter != cur_phase:
                            continue

                        ttype = rule_trigger_matrix[exit_profile_id, rid, RT_RULE_TRIGGER_TYPE]
                        if ttype < 0.0:
                            continue

                        tval  = rule_trigger_matrix[exit_profile_id, rid, RT_RULE_TRIGGER_VALUE]
                        tfeat = rule_trigger_matrix[exit_profile_id, rid, RT_RULE_TRIGGER_FEAT1]
                        top   = rule_trigger_matrix[exit_profile_id, rid, RT_RULE_TRIGGER_OP]

                        triggered = _eval_trigger_simple(
                            int(ttype), tval, tfeat, top,
                            i, k, pos, features, state_per_pos,
                            atrs, has_atr, ep, side,
                        )
                        if not triggered:
                            continue

                        n_actions = int(rule_trigger_matrix[exit_profile_id, rid, RT_RULE_N_ACTIONS])
                        rule_closed = False

                        for aid in range(n_actions):
                            if aid >= max_actions_per_rule:
                                break
                            if aid >= rule_action_matrix.shape[2]:
                                break

                            atype  = int(rule_action_matrix[exit_profile_id, rid, aid, RA_ACTION_TYPE])
                            p1     = rule_action_matrix[exit_profile_id, rid, aid, RA_PARAM1]
                            p2     = rule_action_matrix[exit_profile_id, rid, aid, RA_PARAM2]
                            p3     = rule_action_matrix[exit_profile_id, rid, aid, RA_PARAM3]
                            fidx   = int(rule_action_matrix[exit_profile_id, rid, aid, RA_FEAT_IDX])

                            if atype == ACTION_TYPE_EXIT_PARTIAL:
                                ref_size = pos[k, POS_REMAINING_SIZE]
                                if int(p2) == 1:
                                    ref_size = pos[k, POS_ORIGINAL_SIZE]
                                part_size = p1 * ref_size
                                new_rem   = pos[k, POS_REMAINING_SIZE] - part_size
                                if n_trades < MAX_TRADES:
                                    trade_sizes[n_trades] = part_size
                                    n_trades = _record_trade(
                                        trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                                        trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                                        trade_setup_ids, trade_selected_score,
                                        trade_exit_profile_ids, trade_exit_strategy_ids,
                                         trade_regime_ids,trade_phase, trade_n_tp_hit, trade_add_count,trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,
                                        n_trades, side, o, ep, entry_idx, i,
                                        REASON_PARTIAL_TP,
                                        pos[k, POS_MAE], pos[k, POS_MFE],
                                        setup_id, sel_score, exit_profile_id, exit_strategy_id,
                                        cur_regime_id,
                                        state_global, n_state_global, use_exit_system,state_per_pos, k, n_state_per_pos, int(pos[k, POS_GROUP_ID]),
                                    )
                                    if max_tp > 0:
                                        if tp_period_mode == 3:
                                            recent_tp_idx[tp_head] = i
                                            tp_head = (tp_head + 1) % (max_tp + 1)
                                            tp_recent_count += 1
                                        else:
                                            tp_count += 1
                                state_per_pos[k, SP_N_TP_HIT]     += 1.0
                                state_per_pos[k, SP_BARS_SINCE_TP]  = 0.0
                                if new_rem <= 0.01:
                                    if use_exit_system:
                                        exit_rt[k] = exit_rt[n_pos - 1]
                                    if n_state_per_pos > 0:
                                        state_per_pos[k] = state_per_pos[n_pos - 1]
                                        state_per_pos[n_pos - 1, :] = 0.0
                                    pos[k] = pos[n_pos - 1]
                                    n_pos -= 1
                                    rule_closed = True
                                    break
                                else:
                                    pos[k, POS_REMAINING_SIZE] = new_rem

                            elif atype == ACTION_TYPE_MOVE_SL_BE:
                                offset = p1
                                pos[k, POS_SL] = ep + side * ep * offset
                                sl = pos[k, POS_SL]

                            elif atype == ACTION_TYPE_MOVE_SL_FEAT:
                                if fidx >= 0 and fidx < features.shape[1]:
                                    feat_sl = features[i, fidx]
                                    offset  = p1
                                    pos[k, POS_SL] = feat_sl + side * feat_sl * offset
                                    sl = pos[k, POS_SL]

                            elif atype == ACTION_TYPE_SET_TP:
                                rr = p1; atr_m = p2
                                if rr > 0.0:
                                    sl_dist = abs(ep - pos[k, POS_SL])
                                    pos[k, POS_TP] = ep + side * sl_dist * rr
                                    tp = pos[k, POS_TP]
                                elif atr_m > 0.0 and has_atr and i >= 1:
                                    pos[k, POS_TP] = ep + side * atrs[i - 1] * atr_m
                                    tp = pos[k, POS_TP]
                                elif fidx >= 0 and fidx < features.shape[1]:
                                    pos[k, POS_TP] = features[i, fidx]
                                    tp = pos[k, POS_TP]

                            elif atype == ACTION_TYPE_ADD_POSITION:
                                if n_pos < MAX_POS:
                                    size_frac = p1
                                    sl_mode   = int(p2)
                                    grp_mode  = int(p3)
                                    add_sl = ep if sl_mode == 0 else pos[k, POS_SL]
                                    pos[n_pos, POS_SIDE]             = side
                                    pos[n_pos, POS_ENTRY_PRICE]      = o
                                    pos[n_pos, POS_TP]               = pos[k, POS_TP]
                                    pos[n_pos, POS_SL]               = add_sl
                                    pos[n_pos, POS_ENTRY_IDX]        = float(i)
                                    pos[n_pos, POS_REMAINING_SIZE]   = size_frac
                                    pos[n_pos, POS_ORIGINAL_SIZE]    = size_frac
                                    pos[n_pos, POS_GROUP_ID]         = pos[k, POS_GROUP_ID]
                                    pos[n_pos, POS_GROUP_SL_MODE]    = float(grp_mode)
                                    pos[n_pos, POS_MAE]              = 0.0
                                    pos[n_pos, POS_MFE]              = 0.0
                                    pos[n_pos, POS_SETUP_ID]         = float(setup_id)
                                    pos[n_pos, POS_SELECTED_SCORE]   = sel_score
                                    pos[n_pos, POS_BE_ARMED]         = 0.0
                                    pos[n_pos, POS_BE_ACTIVE]        = 0.0
                                    pos[n_pos, POS_BE_ARM_IDX]       = -1.0
                                    pos[n_pos, POS_RUNNER_ARMED]     = 0.0
                                    pos[n_pos, POS_RUNNER_ACTIVE]    = 0.0
                                    pos[n_pos, POS_RUNNER_SL]        = 0.0
                                    pos[n_pos, POS_PENDING_BE_SL]    = 0.0
                                    pos[n_pos, POS_RUNNER_THRESHOLD] = 0.0
                                    if use_exit_system:
                                        exit_rt[n_pos, :] = exit_rt[k, :]
                                    if n_state_per_pos >= N_SP_DEFAULT:
                                        state_per_pos[n_pos, SP_REMAINING_SIZE]  = size_frac
                                        state_per_pos[n_pos, SP_ENTRY_VALID]     = 1.0
                                        state_per_pos[n_pos, SP_AVG_ENTRY]       = o
                                        state_per_pos[n_pos, SP_REGIME_AT_ENTRY] = float(regime[i]) if use_regime else -1.0
                                    state_per_pos[k, SP_ADD_COUNT] += 1.0
                                    n_pos += 1

                            elif atype == ACTION_TYPE_SET_PHASE:
                                old_phase = int(state_per_pos[k, SP_PHASE])
                                new_phase = int(p1)
                                state_per_pos[k, SP_PHASE] = p1

                                # ← logger l'événement phase
                                if n_phase_events < MAX_PHASE_EVENTS:
                                    phase_ev_idx[n_phase_events]     = i
                                    phase_ev_trade[n_phase_events]   = int(pos[k, POS_ENTRY_IDX])
                                    phase_ev_group[n_phase_events]   = int(pos[k, POS_GROUP_ID])
                                    phase_ev_from[n_phase_events]    = old_phase
                                    phase_ev_to[n_phase_events]      = new_phase
                                    phase_ev_profile[n_phase_events] = exit_profile_id
                                    phase_ev_side[n_phase_events]    = side
                                    n_phase_events += 1

                            elif atype == ACTION_TYPE_INVALIDATE:
                                state_per_pos[k, SP_ENTRY_VALID] = 0.0

                        if rule_closed:
                            continue

            # ── 4. Exit signal forced ─────────────────────────────
            if has_exit_sig:
                es     = exit_signals[i - exit_delay] if i >= exit_delay else 0
                forced = False
                if   es == 2  and side == 1:  forced = True
                elif es == -2 and side == -1: forced = True
                elif es == 3:                 forced = True
                elif has_tags and es == tag:  forced = True
                if forced:
                    if n_trades < MAX_TRADES:
                        n_trades = _record_trade(
                            trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                            trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                            trade_setup_ids, trade_selected_score,
                            trade_exit_profile_ids, trade_exit_strategy_ids,
                             trade_regime_ids,trade_phase, trade_n_tp_hit, trade_add_count,trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,
                            n_trades, side, o, ep, entry_idx, i, REASON_EXIT_SIG,
                            pos[k, POS_MAE], pos[k, POS_MFE],
                            setup_id, sel_score, exit_profile_id, exit_strategy_id,
                            cur_regime_id,
                            state_global, n_state_global, use_exit_system,state_per_pos, k, n_state_per_pos, int(pos[k, POS_GROUP_ID]),
                        )
                    if use_exit_system:
                        exit_rt[k] = exit_rt[n_pos - 1]
                    if use_exit_system and n_state_per_pos > 0:
                        state_per_pos[k] = state_per_pos[n_pos - 1]
                        state_per_pos[n_pos - 1, :] = 0.0
                    pos[k] = pos[n_pos - 1]; n_pos -= 1
                    continue

            # ── 5. Dispatch strat ─────────────────────────────────
            if use_exit_system and exit_strategy_id >= 0:
                action = _run_exit_strategy_dispatch(
                    exit_strategy_id, strategy_rt_matrix,
                    i, k,
                    opens, highs, lows, closes,
                    features, pos, exit_rt,
                    state_per_pos, state_global,
                )
                action_type = int(action[ACT_TYPE])

                if action_type == EXIT_ACT_SWITCH_PROFILE:
                    target_profile_id = int(action[ACT_TARGET_PROFILE_ID])
                    if _is_profile_allowed(
                        exit_strategy_id, target_profile_id,
                        strategy_allowed_profiles, strategy_allowed_counts
                    ):
                        exit_rt[k, :] = profile_rt_matrix[target_profile_id, :]
                        exit_rt[k, RT_EXIT_STRATEGY_ID] = exit_strategy_id
                        exit_profile_id         = target_profile_id
                        rt_be_trigger_pct       = exit_rt[k, RT_BE_TRIGGER_PCT]
                        rt_be_offset_pct        = exit_rt[k, RT_BE_OFFSET_PCT]
                        rt_be_delay_bars        = int(exit_rt[k, RT_BE_DELAY_BARS])
                        rt_trailing_trigger_pct = exit_rt[k, RT_TRAILING_TRIGGER_PCT]
                        rt_runner_trailing_mult = exit_rt[k, RT_RUNNER_TRAILING_MULT]
                        rt_max_holding_bars     = int(exit_rt[k, RT_MAX_HOLDING_BARS])

                elif action_type == EXIT_ACT_OVERWRITE_PRICE:
                    new_tp = action[ACT_NEW_TP_PRICE]
                    new_sl = action[ACT_NEW_SL_PRICE]
                    if new_tp > 0.0:
                        pos[k, POS_TP] = new_tp; tp = new_tp
                    if new_sl > 0.0:
                        pos[k, POS_SL] = new_sl; sl = new_sl

                elif action_type == EXIT_ACT_FORCE_EXIT:
                    force_reason = int(action[ACT_FORCE_EXIT_REASON])
                    if force_reason <= 0:
                        force_reason = REASON_EXIT_STRAT_FORCE
                    if n_trades < MAX_TRADES:
                        n_trades = _record_trade(
                            trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                            trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                            trade_setup_ids, trade_selected_score,
                            trade_exit_profile_ids, trade_exit_strategy_ids,
                             trade_regime_ids,trade_phase, trade_n_tp_hit, trade_add_count,trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,
                            n_trades, side, o, ep, entry_idx, i, force_reason,
                            pos[k, POS_MAE], pos[k, POS_MFE],
                            setup_id, sel_score, exit_profile_id, exit_strategy_id,
                            cur_regime_id,
                            state_global, n_state_global, use_exit_system,state_per_pos, k, n_state_per_pos, int(pos[k, POS_GROUP_ID]),
                        )
                    if use_exit_system:
                        exit_rt[k] = exit_rt[n_pos - 1]
                    if use_exit_system and n_state_per_pos > 0:
                        state_per_pos[k] = state_per_pos[n_pos - 1]
                        state_per_pos[n_pos - 1, :] = 0.0
                    pos[k] = pos[n_pos - 1]; n_pos -= 1
                    continue

                elif action_type == EXIT_ACT_PARTIAL_EXIT:
                    fraction   = action[ACT_PARTIAL_FRACTION]
                    part_price = action[ACT_PARTIAL_PRICE]
                    if part_price <= 0.0:
                        part_price = o

                    remaining     = pos[k, POS_REMAINING_SIZE]
                    new_remaining = remaining - fraction

                    if n_trades < MAX_TRADES:
                        trade_sizes[n_trades] = fraction
                        n_trades = _record_trade(
                            trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                            trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                            trade_setup_ids, trade_selected_score,
                            trade_exit_profile_ids, trade_exit_strategy_ids,
                             trade_regime_ids,trade_phase, trade_n_tp_hit, trade_add_count,trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,
                            n_trades, side, part_price, ep, entry_idx, i,
                            REASON_PARTIAL_TP,
                            pos[k, POS_MAE], pos[k, POS_MFE],
                            setup_id, sel_score, exit_profile_id, exit_strategy_id,
                            cur_regime_id,
                            state_global, n_state_global, use_exit_system,state_per_pos, k, n_state_per_pos, int(pos[k, POS_GROUP_ID]),
                        )

                        # ← ajouter
                        if max_tp > 0:
                            if tp_period_mode == 3:
                                recent_tp_idx[tp_head] = i
                                tp_head = (tp_head + 1) % (max_tp + 1)
                                tp_recent_count += 1
                            else:
                                tp_count += 1

                    if new_remaining <= 0.01:
                        if use_exit_system:
                            exit_rt[k] = exit_rt[n_pos - 1]
                        if use_exit_system and n_state_per_pos > 0:
                            state_per_pos[k] = state_per_pos[n_pos - 1]
                            state_per_pos[n_pos - 1, :] = 0.0
                        pos[k] = pos[n_pos - 1]
                        n_pos -= 1
                    else:
                        pos[k, POS_REMAINING_SIZE]      = new_remaining
                        state_per_pos[k, SP_N_TP_HIT]  += 1.0
                        state_per_pos[k, SP_BARS_SINCE_TP] = 0.0
                        if action[ACT_PARTIAL_PRICE] > 0.0:
                            pos[k, POS_SL] = ep; sl = ep
                        k += 1
                    continue

                elif action_type == EXIT_ACT_ADD_POSITION:
                    if n_pos < MAX_POS:
                        size_frac = action[ACT_ADD_SIZE_FRACTION]
                        add_sl    = action[ACT_ADD_SL_PRICE]
                        add_tp    = action[ACT_ADD_TP_PRICE]
                        sl_mode   = int(action[ACT_ADD_GROUP_SL_MODE])

                        if add_sl <= 0.0: add_sl = pos[k, POS_SL]
                        if add_tp <= 0.0: add_tp = pos[k, POS_TP]

                        pos[n_pos, POS_SIDE]             = side
                        pos[n_pos, POS_ENTRY_PRICE]      = o
                        pos[n_pos, POS_TP]               = add_tp
                        pos[n_pos, POS_SL]               = add_sl
                        pos[n_pos, POS_ENTRY_IDX]        = float(i)
                        pos[n_pos, POS_REMAINING_SIZE]   = size_frac
                        pos[n_pos, POS_ORIGINAL_SIZE]    = size_frac
                        pos[n_pos, POS_GROUP_ID]         = pos[k, POS_GROUP_ID]
                        pos[n_pos, POS_GROUP_SL_MODE]    = float(sl_mode)
                        pos[n_pos, POS_MAE]              = 0.0
                        pos[n_pos, POS_MFE]              = 0.0
                        pos[n_pos, POS_SETUP_ID]         = float(setup_id)
                        pos[n_pos, POS_SELECTED_SCORE]   = sel_score
                        pos[n_pos, POS_BE_ARMED]         = 0.0
                        pos[n_pos, POS_BE_ACTIVE]        = 0.0
                        pos[n_pos, POS_BE_ARM_IDX]       = -1.0
                        pos[n_pos, POS_RUNNER_ARMED]     = 0.0
                        pos[n_pos, POS_RUNNER_ACTIVE]    = 0.0
                        pos[n_pos, POS_RUNNER_SL]        = 0.0
                        pos[n_pos, POS_PENDING_BE_SL]    = 0.0
                        pos[n_pos, POS_RUNNER_THRESHOLD] = 0.0

                        if use_exit_system:
                            exit_rt[n_pos, :] = exit_rt[k, :]

                        if n_state_per_pos >= N_SP_DEFAULT:
                            state_per_pos[n_pos, SP_REMAINING_SIZE]  = size_frac
                            state_per_pos[n_pos, SP_ENTRY_VALID]     = 1.0
                            state_per_pos[n_pos, SP_AVG_ENTRY]       = o
                            state_per_pos[n_pos, SP_REGIME_AT_ENTRY] = float(regime[i]) if use_regime else -1.0

                        state_per_pos[k, SP_ADD_COUNT] += 1.0
                        n_pos += 1

                    k += 1
                    continue

            # ── 6. BE update ──────────────────────────────────────
            if not sl_tp_be_priority:
                if rt_be_trigger_pct > 0.0:
                    pending_be = pos[k, POS_PENDING_BE_SL]
                    sl, be_armed, be_active, be_arm_idx, pending_be = _update_be(
                        side, ep, sl, i, entry_idx, be_armed, be_active,
                        be_arm_idx, pending_be, h, l,
                        rt_be_trigger_pct, rt_be_offset_pct, rt_be_delay_bars
                    )
                    pos[k, POS_SL]            = sl
                    pos[k, POS_BE_ARMED]      = be_armed
                    pos[k, POS_BE_ACTIVE]     = be_active
                    pos[k, POS_BE_ARM_IDX]    = be_arm_idx
                    pos[k, POS_PENDING_BE_SL] = pending_be

                # ── 7. Trailing update ────────────────────────────────
                if rt_trailing_trigger_pct > 0.0 and has_atr:
                    r_threshold = pos[k, POS_RUNNER_THRESHOLD]
                    r_armed     = pos[k, POS_RUNNER_ARMED]
                    r_active    = pos[k, POS_RUNNER_ACTIVE]
                    r_sl        = pos[k, POS_RUNNER_SL]
                    r_active_before = r_active
                    r_armed, r_active, r_sl, r_threshold = _update_runner(
                        side, ep, r_armed, r_active, r_sl, r_threshold,
                        h, l, c, atrs[i - 1],
                        rt_trailing_trigger_pct, rt_runner_trailing_mult
                    )
                    pos[k, POS_RUNNER_ARMED]     = r_armed
                    pos[k, POS_RUNNER_ACTIVE]    = r_active
                    pos[k, POS_RUNNER_SL]        = r_sl
                    pos[k, POS_RUNNER_THRESHOLD] = r_threshold
                    if r_active_before == 0.0 and r_active == 1.0:
                        k += 1; continue

            # ── 8. Check exit SL/TP ───────────────────────────────
            
                exit_price = -1.0; reason = 0

                if r_active and r_sl != 0.0:
                    r_threshold = pos[k, POS_RUNNER_THRESHOLD]
                    be_rsn = REASON_BE if be_active else REASON_SL
                    under_threshold = r_threshold > 0.0 and (
                        (side == 1  and r_sl <= r_threshold) or
                        (side == -1 and r_sl >= r_threshold)
                    )
                    if under_threshold:
                        if side == 1:
                            if o <= sl:   exit_price = o;  reason = be_rsn
                            elif l <= sl: exit_price = sl; reason = be_rsn
                        else:
                            if o >= sl:   exit_price = o;  reason = be_rsn
                            elif h >= sl: exit_price = sl; reason = be_rsn
                    else:
                        if side == 1:
                            if o <= r_sl:   exit_price = o;    reason = REASON_RUNNER_SL
                            elif l <= r_sl: exit_price = r_sl; reason = REASON_RUNNER_SL
                            elif o <= sl:   exit_price = o;    reason = be_rsn
                            elif l <= sl:   exit_price = sl;   reason = be_rsn
                        else:
                            if o >= r_sl:   exit_price = o;    reason = REASON_RUNNER_SL
                            elif h >= r_sl: exit_price = r_sl; reason = REASON_RUNNER_SL
                            elif o >= sl:   exit_price = o;    reason = be_rsn
                            elif h >= sl:   exit_price = sl;   reason = be_rsn

                elif ema_exit_mode and has_ema_exit:
                    e1 = exit_ema1[i]
                    e2 = exit_ema2[i] if exit_ema2.shape[0] == n else 0.0
                    exit_price, reason = _check_exit_ema(
                        side, o, h, l, c, tp, sl,
                        be_active == 1.0, ep, e1, e2,
                        use_ema1_tp, use_ema2_tp, use_ema_cross_tp
                    )
                else:
                    exit_price, reason = _check_exit_fixed(
                        side, o, h, l, tp, sl, be_active == 1.0
                    )

            # ── 9. Group SL + record + fermeture ─────────────────
            if exit_price > 0.0:

                # Group SL — fermer tout le groupe si mode partagé
                if reason == REASON_SL or reason == REASON_BE:
                    group_id  = int(pos[k, POS_GROUP_ID])
                    sl_mode_g = int(pos[k, POS_GROUP_SL_MODE])
                    if sl_mode_g == 1:
                        j = 0
                        while j < n_pos:
                            ep_id_j = int(exit_rt[j, RT_EXIT_PROFILE_ID]) if use_exit_system else -1
                            es_id_j = int(exit_rt[j, RT_EXIT_STRATEGY_ID]) if use_exit_system else -1
                            if j != k and int(pos[j, POS_GROUP_ID]) == group_id:
                                if n_trades < MAX_TRADES:
                                    n_trades = _record_trade(
                                        trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                                        trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                                        trade_setup_ids, trade_selected_score,
                                        trade_exit_profile_ids, trade_exit_strategy_ids,
                                         trade_regime_ids,trade_phase, trade_n_tp_hit, trade_add_count,trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,
                                        n_trades, pos[j, POS_SIDE], exit_price,
                                        pos[j, POS_ENTRY_PRICE], int(pos[j, POS_ENTRY_IDX]),
                                        i, REASON_GROUP_SL,
                                        pos[j, POS_MAE], pos[j, POS_MFE],
                                        int(pos[j, POS_SETUP_ID]), pos[j, POS_SELECTED_SCORE],
                                        ep_id_j,
                                        es_id_j,
                                        cur_regime_id,
                                        state_global, n_state_global, use_exit_system,state_per_pos, j, n_state_per_pos, int(pos[j, POS_GROUP_ID]),
                                    )
                                if use_exit_system:
                                    exit_rt[j] = exit_rt[n_pos - 1]
                                if use_exit_system and n_state_per_pos > 0:
                                    state_per_pos[j] = state_per_pos[n_pos - 1]
                                    state_per_pos[n_pos - 1, :] = 0.0
                                pos[j] = pos[n_pos - 1]
                                n_pos -= 1
                                if j < k: k -= 1
                            else:
                                j += 1

                # Record trade normal
                if n_trades < MAX_TRADES:
                    n_trades = _record_trade(
                        trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                        trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                        trade_setup_ids, trade_selected_score,
                        trade_exit_profile_ids, trade_exit_strategy_ids,
                         trade_regime_ids,trade_phase, trade_n_tp_hit, trade_add_count,trade_remaining, trade_avg_entry, trade_bars,trade_group_ids,
                        n_trades, side, exit_price, ep, entry_idx, i, reason,
                        pos[k, POS_MAE], pos[k, POS_MFE],
                        setup_id, sel_score, exit_profile_id, exit_strategy_id,
                        cur_regime_id,
                        state_global, n_state_global, use_exit_system,state_per_pos, k, n_state_per_pos, int(pos[k, POS_GROUP_ID]),
                    )
                    if reason == REASON_TP and max_tp > 0:
                        if tp_period_mode == 3:
                            recent_tp_idx[tp_head] = i
                            tp_head = (tp_head + 1) % (max_tp + 1)
                            tp_recent_count += 1
                        else:
                            tp_count += 1
                    if reason == REASON_SL and max_sl > 0:
                        if sl_period_mode == 3:
                            recent_sl_idx[sl_head] = i
                            sl_head = (sl_head + 1) % (max_sl + 1)
                            sl_recent_count += 1
                        else:
                            sl_count += 1

                if use_exit_system:
                    exit_rt[k] = exit_rt[n_pos - 1]
                if use_exit_system and n_state_per_pos > 0:
                    state_per_pos[k] = state_per_pos[n_pos - 1]
                    state_per_pos[n_pos - 1, :] = 0.0
                pos[k] = pos[n_pos - 1]; n_pos -= 1
            else:
                k += 1

    return (
        trade_returns[:n_trades],
        trade_sides[:n_trades],
        trade_entry_idx[:n_trades],
        trade_exit_idx[:n_trades],
        trade_reasons[:n_trades],
        trade_exit_prices[:n_trades],
        trade_mae[:n_trades],
        trade_mfe[:n_trades],
        trade_setup_ids[:n_trades],
        trade_selected_score[:n_trades],
        trade_exit_profile_ids[:n_trades],
        trade_exit_strategy_ids[:n_trades],
        trade_regime_ids[:n_trades],
        trade_sizes[:n_trades],
        trade_phase[:n_trades],
        trade_n_tp_hit[:n_trades],
        trade_add_count[:n_trades],
        trade_remaining[:n_trades],
        trade_avg_entry[:n_trades],
        trade_bars[:n_trades],
        trade_group_ids[:n_trades],         # ← nouveau
        phase_ev_idx[:n_phase_events],      # ← nouveau
        phase_ev_trade[:n_phase_events],    # ← nouveau
        phase_ev_group[:n_phase_events],    # ← nouveau
        phase_ev_from[:n_phase_events],     # ← nouveau
        phase_ev_to[:n_phase_events],       # ← nouveau
        phase_ev_profile[:n_phase_events],  # ← nouveau
        phase_ev_side[:n_phase_events],     # ← nouveau
    )


# ══════════════════════════════════════════════════════════════════
# 5. MÉTRIQUES COMPLÈTES
# ══════════════════════════════════════════════════════════════════
def compute_metrics_full(
    trade_returns, trade_sides, trade_entry_idx,
    trade_exit_idx, trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
    bar_index,
    highs=None, lows=None,
    hold_minutes=0, bar_duration_min=5,
    commission_pct=0.0, commission_per_lot_usd=0.0, contract_size=1.0,
    spread_pct=0.0, spread_abs=0.0, slippage_pct=0.0,
    alpha=5, period_freq="ME",
    entry_on_signal_close_price=False,
    trade_entry_prices=None,
    trade_setup_ids=None,
    trade_selected_score=None,
    trade_exit_profile_ids=None,
    trade_exit_strategy_ids=None, trade_regime_ids=None, trade_sizes=None,
    trade_phase=None,
    trade_n_tp_hit=None,
    trade_add_count=None,
    trade_remaining=None,
    trade_avg_entry=None,
    trade_bars=None,
    trade_group_ids=None,
    phase_events_df=None,
):
    if len(trade_returns) == 0:
        return None

    trade_returns = np.asarray(trade_returns, dtype=np.float64)
    if trade_entry_prices is None:
        entry_price_arr = None
    else:
        entry_price_arr = np.asarray(trade_entry_prices, dtype=np.float64)
        if len(entry_price_arr) != len(trade_returns):
            raise ValueError("trade_entry_prices must have the same length as trade_returns")

    spread_cost_arr = np.full(len(trade_returns), float(spread_pct), dtype=np.float64)
    if spread_abs > 0.0:
        if entry_price_arr is None:
            raise ValueError("trade_entry_prices is required when spread_abs > 0")
        spread_cost_arr = float(spread_abs) / np.maximum(np.abs(entry_price_arr), 1e-12)

    commission_cost_arr = np.full(len(trade_returns), float(commission_pct) * 2.0, dtype=np.float64)
    if commission_per_lot_usd > 0.0:
        if entry_price_arr is None:
            raise ValueError("trade_entry_prices is required when commission_per_lot_usd > 0")
        if contract_size <= 0.0:
            raise ValueError("contract_size must be > 0 when commission_per_lot_usd > 0")
        commission_cost_arr = float(commission_per_lot_usd) / (
            float(contract_size) * np.maximum(np.abs(entry_price_arr), 1e-12)
        )

    slippage_cost_arr = np.full(len(trade_returns), float(slippage_pct) * 2.0, dtype=np.float64)
    total_cost_arr = commission_cost_arr + spread_cost_arr + slippage_cost_arr
    ret_arr = trade_returns - total_cost_arr

    _sizes           = trade_sizes if trade_sizes is not None else np.ones(len(ret_arr))
    weighted_returns = ret_arr * _sizes

    if entry_on_signal_close_price:
        entry_times = bar_index[trade_entry_idx - 1]
    else:
        entry_times = bar_index[trade_entry_idx]
    exit_times = bar_index[trade_exit_idx]

    # ── Returns économiques ───────────────────────────────────────
    if trade_remaining is not None:
        _remaining    = np.asarray(trade_remaining, dtype=np.float64)
        weighted_legs = ret_arr * _remaining

        sort_order   = np.argsort(trade_entry_idx, kind="stable")
        sorted_entry = trade_entry_idx[sort_order]
        sorted_wlegs = weighted_legs[sort_order]
        sorted_exit  = trade_exit_idx[sort_order]

        _, group_starts, _ = np.unique(sorted_entry, return_index=True, return_counts=True)

        eco_ret_arr  = np.add.reduceat(sorted_wlegs, group_starts)
        eco_exit_idx = np.maximum.reduceat(sorted_exit, group_starts)

        chrono_order   = np.argsort(eco_exit_idx, kind="stable")
        eco_ret_arr    = eco_ret_arr[chrono_order]
        eco_exit_idx   = eco_exit_idx[chrono_order]
        eco_exit_times = bar_index[eco_exit_idx]
    else:
        eco_ret_arr    = ret_arr
        eco_exit_idx   = trade_exit_idx
        eco_exit_times = exit_times

    # ── Métriques globales sur eco_ret_arr ───────────────────────
    pos_mask = eco_ret_arr > 0
    neg_mask = eco_ret_arr < 0
    cum      = np.cumprod(1 + eco_ret_arr)
    roll_max = np.maximum.accumulate(cum)
    dd_curve = (cum - roll_max) / roll_max
    mdd      = dd_curve.min()

    max_uw = current_uw = 0
    for d in dd_curve < 0:
        current_uw = current_uw + 1 if d else 0
        max_uw     = max(max_uw, current_uw)

    cum_return = cum[-1] - 1
    n_years    = (eco_exit_times[-1] - eco_exit_times[0]).days / 365
    ann_return = (1 + cum_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    std        = eco_ret_arr.std()
    wins_sum   = eco_ret_arr[pos_mask].sum()
    loss_sum   = abs(eco_ret_arr[neg_mask].sum())
    var_t      = -np.percentile(eco_ret_arr, alpha)
    n_tpy      = len(eco_ret_arr) / n_years

    sharpe  = eco_ret_arr.mean() / std * np.sqrt(n_tpy) if std > 0 else np.nan
    t_stat, p_value = scipy_stats.ttest_1samp(eco_ret_arr, 0)
    p_binom = scipy_stats.binomtest(pos_mask.sum(), len(eco_ret_arr), 0.5).pvalue

    sum_to_max_ddr = eco_ret_arr.sum() / mdd if mdd != 0 else np.nan

    # ── Period returns sur eco ────────────────────────────────────
    eco_series = pd.Series(eco_ret_arr, index=pd.DatetimeIndex(eco_exit_times))
    period_ret = eco_series.resample(period_freq).sum()
    period_ret = period_ret[period_ret != 0].to_numpy()
    pr_pos  = period_ret > 0
    pr_neg  = period_ret < 0
    pr_var  = np.percentile(period_ret, alpha) if len(period_ret) > 0 else np.nan
    pr_cvar = period_ret[period_ret <= pr_var].mean() \
              if len(period_ret) > 0 and (period_ret <= pr_var).any() else np.nan

    # ── MAE/MFE intra — sur legs individuels ─────────────────────
    has_mae_mfe = (trade_mae is not None and trade_mfe is not None
                   and len(trade_mae) == len(trade_returns))
    mae_intra = trade_mae if has_mae_mfe else np.full(len(ret_arr), np.nan)
    mfe_intra = trade_mfe if has_mae_mfe else np.full(len(ret_arr), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        capture_ratio_intra = np.where(mfe_intra != 0, ret_arr / mfe_intra, np.nan)

    # ── MAE/MFE hold — sur legs individuels ──────────────────────
    hold_bars = int(hold_minutes // bar_duration_min) if hold_minutes > 0 else 0
    n_bars    = len(highs) if highs is not None else 0

    if hold_bars > 0 and highs is not None and lows is not None:
        mae_hold = np.empty(len(ret_arr), dtype=np.float64)
        mfe_hold = np.empty(len(ret_arr), dtype=np.float64)
        for t in range(len(ret_arr)):
            side   = int(trade_sides[t])
            start  = trade_exit_idx[t] + 1
            end    = min(start + hold_bars, n_bars)
            ep_ref = trade_exit_prices[t]
            if start >= n_bars or start >= end:
                mae_hold[t] = 0.0; mfe_hold[t] = 0.0; continue
            h_win = highs[start:end]; l_win = lows[start:end]
            if side == 1:
                mfe_hold[t] = (h_win.max() - ep_ref) / ep_ref
                mae_hold[t] = (l_win.min() - ep_ref) / ep_ref
            else:
                mfe_hold[t] = (ep_ref - l_win.min()) / ep_ref
                mae_hold[t] = (ep_ref - h_win.max()) / ep_ref
        with np.errstate(divide="ignore", invalid="ignore"):
            capture_ratio_hold = np.where(
                (ret_arr > 0) & (mfe_hold > 0), ret_arr / (ret_arr + mfe_hold), np.nan
            )
    else:
        mae_hold           = np.full(len(ret_arr), np.nan)
        mfe_hold           = np.full(len(ret_arr), np.nan)
        capture_ratio_hold = np.full(len(ret_arr), np.nan)

    # ── trades_df — legs individuels ─────────────────────────────
    trades_df = pd.DataFrame({
        "entry_time":          entry_times,
        "exit_time":           exit_times,
        "entry_idx":           trade_entry_idx,
        "exit_idx":            trade_exit_idx,
        "entry":               np.nan,
        "exit":                trade_exit_prices,
        "side":                trade_sides,
        "return":              ret_arr,
        "raw_return":          trade_returns,
        "commission_cost":     commission_cost_arr,
        "spread_cost":         spread_cost_arr,
        "slippage_cost":       slippage_cost_arr,
        "total_cost":          total_cost_arr,
        "reason":              [REASON_LABELS.get(int(r), str(int(r))) for r in trade_reasons],
        "mae_intra":           mae_intra,
        "mfe_intra":           mfe_intra,
        "capture_ratio_intra": capture_ratio_intra,
        "mae_hold":            mae_hold,
        "mfe_hold":            mfe_hold,
        "capture_ratio_hold":  capture_ratio_hold,
    })

    if trade_setup_ids is not None:
        trades_df["setup_id"]        = trade_setup_ids
    if trade_selected_score is not None:
        trades_df["selected_score"]  = trade_selected_score
    if trade_exit_profile_ids is not None:
        trades_df["exit_profile_id"] = trade_exit_profile_ids
    if trade_exit_strategy_ids is not None:
        trades_df["exit_strategy_id"]= trade_exit_strategy_ids
    if trade_regime_ids is not None:
        trades_df["regime_id"]       = trade_regime_ids

    trades_df["weighted_return"] = weighted_returns
    trades_df["bars_in_trade"]   = trade_bars if trade_bars is not None \
                                   else (trade_exit_idx - trade_entry_idx)

    if trade_phase is not None:
        trades_df["phase_at_exit"]  = trade_phase
    if trade_n_tp_hit is not None:
        trades_df["n_tp_hit"]       = trade_n_tp_hit
    if trade_add_count is not None:
        trades_df["add_count"]      = trade_add_count
    if trade_remaining is not None:
        trades_df["remaining_size"] = trade_remaining
    if trade_avg_entry is not None:
        trades_df["avg_entry"]      = trade_avg_entry

    # ── win rates sur eco ─────────────────────────────────────────
    winrate_eco      = round(float(pos_mask.mean()), 3)
    winrate_weighted = round(float((eco_ret_arr > 0).mean()), 3)

    #-- historique des events ----
    if trade_group_ids is not None:
        trades_df["group_id"] = trade_group_ids

    # ── by_* sur legs individuels ─────────────────────────────────
    def _group_metrics(df, group_col, ret_col="return"):
        if group_col not in df.columns:
            return {}
        g = df.groupby(group_col)
        return {
            str(k): {
                "n_trades":   len(v),
                "win_rate":   round((v[ret_col] > 0).mean(), 3),
                "avg_return": round(v[ret_col].mean(), 4),
                "avg_mae":    round(v["mae_intra"].mean(), 4) if "mae_intra" in v else None,
                "avg_mfe":    round(v["mfe_intra"].mean(), 4) if "mfe_intra" in v else None,
                "avg_bars":   round(v["bars_in_trade"].mean(), 1) if "bars_in_trade" in v else None,
            }
            for k, v in g
        }

    return {
        "n_trades"             : len(eco_ret_arr),
        "n_trades_raw"         : len(ret_arr),
        "win_rate"             : winrate_eco,
        "total_return_sum"     : round(eco_ret_arr.sum(), 4),
        "cum_return"           : round(cum_return, 4),
        "ann_return"           : round(ann_return, 4) if not np.isnan(ann_return) else np.nan,
        "max_drawdown"         : round(mdd, 4),
        "RetSum_to_Mddr"       : round(sum_to_max_ddr, 4),
        "max_underwater_trades": max_uw,
        "calmar"               : round(ann_return / abs(mdd), 3) if mdd != 0 else np.nan,
        "sharpe"               : round(sharpe, 4),
        "profit_factor"        : round(wins_sum / loss_sum, 3) if loss_sum != 0 else np.nan,
        "avg_win"              : round(eco_ret_arr[pos_mask].mean(), 4) if pos_mask.any() else np.nan,
        "avg_loss"             : round(eco_ret_arr[neg_mask].mean(), 4) if neg_mask.any() else np.nan,
        "VaR"                  : round(var_t, 4),
        "CVaR"                 : round(-eco_ret_arr[eco_ret_arr <= -var_t].mean(), 4)
                                 if (eco_ret_arr <= -var_t).any() else np.nan,
        "t_stat"               : round(t_stat, 3),
        "p_value"              : round(p_value, 4),
        "p_binom"              : round(p_binom, 4),
        "period_freq"          : period_freq,
        "n_periods"            : len(period_ret),
        "n_periods_positive"   : int(pr_pos.sum()),
        "n_periods_negative"   : int(pr_neg.sum()),
        "pct_periods_positive" : round(pr_pos.mean(), 3) if len(period_ret) > 0 else np.nan,
        "worst_period"         : round(period_ret.min(), 4) if len(period_ret) > 0 else np.nan,
        "best_period"          : round(period_ret.max(), 4) if len(period_ret) > 0 else np.nan,
        "period_cvar"          : round(pr_cvar, 4) if pr_cvar is not None
                                 and not np.isnan(pr_cvar) else np.nan,
        "avg_mae_intra"        : round(float(np.nanmean(mae_intra)), 4) if has_mae_mfe else np.nan,
        "avg_mfe_intra"        : round(float(np.nanmean(mfe_intra)), 4) if has_mae_mfe else np.nan,
        "avg_capture_intra"    : round(float(np.nanmean(capture_ratio_intra)), 4) if has_mae_mfe else np.nan,
        "avg_mae_hold"         : round(float(np.nanmean(mae_hold)), 4) if hold_bars > 0 else np.nan,
        "avg_mfe_hold"         : round(float(np.nanmean(mfe_hold)), 4) if hold_bars > 0 else np.nan,
        "avg_capture_hold"     : round(float(np.nanmean(capture_ratio_hold)), 4) if hold_bars > 0 else np.nan,
        "trades_df"            : trades_df,
        "by_phase"             : _group_metrics(trades_df, "phase_at_exit"),
        "by_regime"            : _group_metrics(trades_df, "regime_id"),
        "by_profile"           : _group_metrics(trades_df, "exit_profile_id"),
        "by_setup"             : _group_metrics(trades_df, "setup_id"),
        "by_reason"            : _group_metrics(trades_df, "reason"),
        "avg_n_tp_hit"         : round(float(trade_n_tp_hit.mean()), 2) if trade_n_tp_hit is not None else None,
        "avg_add_count"        : round(float(trade_add_count.mean()), 2) if trade_add_count is not None else None,
        "avg_bars_in_trade"    : round(float(trade_bars.mean()), 1) if trade_bars is not None else None,
        "win_rate_full_trades" : winrate_eco,
        "win_rate_weighted"    : winrate_weighted,
        "weighted_pnl"         : round(float(eco_ret_arr.sum()), 4),
        "phase_events_df": phase_events_df if phase_events_df is not None 
                           else pd.DataFrame(),
    }
