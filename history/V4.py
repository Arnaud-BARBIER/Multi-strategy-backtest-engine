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



"""
╔══════════════════════════════════════════════════════════════════╗
║      NJIT ENGINE CORE — engine multi_setups + metrics            ║
║                                                                  ║
║  Includes :                                                      ║
║    - Constantes                                                  ║
║    - Indicateurs (EMA, ATR)                                      ║
║    - Générateurs de signaux                                      ║
║    - Helpers exit/entry                                          ║
║    - backtest_njit  ← moteur principal                           ║
║    - compute_metrics_full                                        ║
║    - NJITEngine                                                  ║
║                                                                  ║
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
POS_N_COLS           = 18

REASON_SL          = 1
REASON_TP          = 2
REASON_BE          = 3
REASON_EMA1_TP     = 4
REASON_EMA2_TP     = 5
REASON_EMACROSS_TP = 6
REASON_RUNNER_SL   = 7
REASON_EXIT_SIG    = 8
REASON_REVERSE     = 9
REASON_FORCED_FLAT = 10
REASON_MAX_HOLD    = 11

REASON_LABELS = {
    1: "SL", 2: "TP", 3: "BE",
    4: "EMA1_TP", 5: "EMA2_TP", 6: "EMA_CROSS_TP",
    7: "RUNNER_SL", 8: "EXIT_SIGNAL", 9: "REVERSE",
    10: "FORCED_FLAT", 11: "MAX_HOLD"
}


# ══════════════════════════════════════════════════════════════════
# 1. INDICATEURS
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def ema_njit(closes, span):
    n      = closes.shape[0]
    ema    = np.empty(n, dtype=np.float64)
    alpha  = 2.0 / (span + 1.0)
    ema[0] = closes[0]
    for i in range(1, n):
        ema[i] = alpha * closes[i] + (1.0 - alpha) * ema[i - 1]
    return ema


@njit(cache=True)
def atr_wilder_njit(highs, lows, closes, period):
    n   = highs.shape[0]
    tr  = np.empty(n, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1])
        )
    total = 0.0
    for i in range(period):
        total += tr[i]
    atr[period - 1] = total / period
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


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


@njit(cache=True)
def _record_trade(
    trade_returns, trade_sides, trade_entry_idx, trade_exit_idx, trade_reasons,
    trade_exit_prices, trade_mae, trade_mfe,
    trade_setup_ids, trade_selected_score,
    n_trades, side, exit_price, ep, entry_idx, exit_idx, reason, mae, mfe,
    setup_id, sel_score
):
    trade_returns[n_trades] = side * (exit_price - ep) / ep
    trade_sides[n_trades] = int(side)
    trade_entry_idx[n_trades] = entry_idx
    trade_exit_idx[n_trades] = exit_idx
    trade_reasons[n_trades] = reason
    trade_exit_prices[n_trades] = exit_price
    trade_mae[n_trades] = mae
    trade_mfe[n_trades] = mfe
    trade_setup_ids[n_trades] = setup_id
    trade_selected_score[n_trades] = sel_score
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
# 4. MOTEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def backtest_njit(
    opens, highs, lows, closes, atrs, signals, selected_setup_id, selected_score, minutes_of_day, day_index,day_of_week,
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
    max_tp, tp_period_mode, tp_period_bars
):
    # entry_on_close : False = entrée au open[i] (défaut), True = entrée au close[i-1]

    n = opens.shape[0]

    #---- 




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
    n_trades          = 0

    pos   = np.zeros((MAX_POS, POS_N_COLS), dtype=np.float64)
    n_pos = 0

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
                ep_ff   = pos[k_ff, POS_ENTRY_PRICE]
                side_ff = pos[k_ff, POS_SIDE]
                if n_trades < MAX_TRADES:
                    setup_id_ff  = int(pos[k_ff, POS_SETUP_ID])
                    sel_score_ff = pos[k_ff, POS_SELECTED_SCORE]

                    n_trades = _record_trade(
                        trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                        trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                        trade_setup_ids, trade_selected_score,
                        n_trades, side_ff, o, ep_ff, int(pos[k_ff, POS_ENTRY_IDX]), i,
                        REASON_FORCED_FLAT, pos[k_ff, POS_MAE], pos[k_ff, POS_MFE],
                        setup_id_ff, sel_score_ff
                    )
                pos[k_ff] = pos[n_pos - 1]
                n_pos -= 1

            sig = 0

        # ── 1. REVERSE ───────────────────────────────────────────
        if reverse_mode and sig != 0:
            k = 0
            while k < n_pos:
                if int(pos[k, POS_SIDE]) == -int(sig):
                    ep = pos[k, POS_ENTRY_PRICE]; side = pos[k, POS_SIDE]
                    if n_trades < MAX_TRADES:
                        setup_id_rev  = int(pos[k, POS_SETUP_ID])
                        sel_score_rev = pos[k, POS_SELECTED_SCORE]

                        n_trades = _record_trade(
                            trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                            trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                            trade_setup_ids, trade_selected_score,
                            n_trades, side, o, ep, int(pos[k, POS_ENTRY_IDX]), i,
                            REASON_REVERSE, pos[k, POS_MAE], pos[k, POS_MFE],
                            setup_id_rev, sel_score_rev
                        )
                    pos[k] = pos[n_pos - 1]; n_pos -= 1
                else:
                    k += 1

        # ── 1.5 / 1.6 RESETS ────────────────────────────────────
        # Calculer les changements AVANT mise à jour des last_*
        day_changed     = last_day     != -1 and cur_day     != last_day
        session_changed = last_session != -1 and cur_session != last_session and cur_session != -1

        # Cooldown reset
        if cooldown_entries > 0:
            if   cooldown_mode == 3 and day_changed:     cd_count = 0; cd_until = -1
            elif cooldown_mode == 2 and session_changed: cd_count = 0; cd_until = -1

        # MaxEntries reset
        if me_reset_mode == 1 or me_reset_mode == 5:
            if day_changed: re_count = 0; re_head = 0
            last_day = cur_day
        elif me_reset_mode == 2 or me_reset_mode == 4:
            if session_changed: re_count = 0; re_head = 0
            if cur_session != -1: last_session = cur_session
        # ── 1.7 max TP reset  ─────────────────────────────────────────────
        if me_reset_mode == 3 or me_reset_mode == 4 or me_reset_mode == 5:
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

        #--- TP max ----
        tp_day_changed = tp_last_day != -1 and cur_day != tp_last_day
        tp_session_changed = tp_last_session != -1 and cur_session != tp_last_session and cur_session != -1
        if tp_period_mode == 2:
            if tp_session_changed:
                tp_count = 0
            if cur_session != -1:
                tp_last_session = cur_session
        elif tp_period_mode == 3:
            if tp_day_changed:
                tp_count = 0
            tp_last_day = cur_day
        elif tp_period_mode == 1 and max_tp > 0:
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
                if tp_period_mode == 1:
                    tp_ok = tp_recent_count < max_tp
                elif tp_period_mode in (2, 3):
                    tp_ok = tp_count < max_tp

            if in_session and me_ok and cd_ok and tp_ok:
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
                atr_val  = atrs[i - entry_delay] if (has_atr and use_atr_sl_tp != 0) else 0.0
                atr_ok   = not (use_atr_sl_tp != 0 and has_atr and atr_val != atr_val)

                if gap_ok and candle_ok and pos_ok and atr_ok:
                    if multi_ok:
                        # ── ENTRY PRICE : open ou close de la bougie signal ──
                        if entry_on_signal_close_price:
                            ep2 = closes[i - 1]
                        elif entry_on_close:
                            ep2 = closes[i]
                        else :
                            ep2 = o

                        tp2, sl2 = _compute_tp_sl(sig, ep2, tp_pct, sl_pct,
                                                   use_atr_sl_tp, tp_atr_mult, sl_atr_mult, atr_val)
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
                        pos[n_pos, POS_SETUP_ID]         = delayed_setup_id[i]
                        pos[n_pos, POS_SELECTED_SCORE]   = delayed_selected_score[i]
                        n_pos += 1

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
                ep   = pos[last, POS_ENTRY_PRICE]
                side = pos[last, POS_SIDE]
                if n_trades < MAX_TRADES:
                    setup_id_last  = int(pos[last, POS_SETUP_ID])
                    sel_score_last = pos[last, POS_SELECTED_SCORE]

                    n_trades = _record_trade(
                        trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                        trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                        trade_setup_ids, trade_selected_score,
                        n_trades, side, o, ep, int(pos[last, POS_ENTRY_IDX]), i,
                        REASON_EXIT_SIG, pos[last, POS_MAE], pos[last, POS_MFE],
                        setup_id_last, sel_score_last
                    )
                n_pos -= 1

        # ── 4. EXIT SL/TP ────────────────────────────────────────
        k = 0
        while k < n_pos:
            side       = pos[k, POS_SIDE];  ep = pos[k, POS_ENTRY_PRICE]
            tp         = pos[k, POS_TP];    sl = pos[k, POS_SL]
            entry_idx  = int(pos[k, POS_ENTRY_IDX])
            be_armed   = pos[k, POS_BE_ARMED];  be_active = pos[k, POS_BE_ACTIVE]
            be_arm_idx = pos[k, POS_BE_ARM_IDX]
            r_armed    = pos[k, POS_RUNNER_ARMED]; r_active = pos[k, POS_RUNNER_ACTIVE]
            r_sl       = pos[k, POS_RUNNER_SL];    tag      = pos[k, POS_TAG]
            setup_id  = int(pos[k, POS_SETUP_ID])
            sel_score = pos[k, POS_SELECTED_SCORE]

            if not allow_exit_on_entry_bar and i == entry_idx:
                k += 1; continue
            #-- Max holding bar logic ---
            if max_holding_bars > 0 and (i - entry_idx) >= max_holding_bars:
                if n_trades < MAX_TRADES:
                    n_trades = _record_trade(
                        trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                        trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                        trade_setup_ids, trade_selected_score,
                        n_trades, side, o, ep, entry_idx, i, REASON_MAX_HOLD,
                        pos[k, POS_MAE], pos[k, POS_MFE],
                        setup_id, sel_score
                    )
                pos[k] = pos[n_pos - 1]
                n_pos -= 1
                continue

            if track_mae_mfe:
                if side == 1:
                    pos[k, POS_MFE] = max(pos[k, POS_MFE], (h - ep) / ep)
                    pos[k, POS_MAE] = min(pos[k, POS_MAE], (l - ep) / ep)
                else:
                    pos[k, POS_MFE] = max(pos[k, POS_MFE], (ep - l) / ep)
                    pos[k, POS_MAE] = min(pos[k, POS_MAE], (ep - h) / ep)

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
                            n_trades, side, o, ep, entry_idx, i, REASON_EXIT_SIG,
                            pos[k, POS_MAE], pos[k, POS_MFE],
                            setup_id, sel_score
                        )
                    pos[k] = pos[n_pos - 1]; n_pos -= 1
                    continue

            if be_trigger_pct > 0.0:
                pending_be = pos[k, POS_PENDING_BE_SL]
                sl, be_armed, be_active, be_arm_idx, pending_be = _update_be(
                    side, ep, sl, i, entry_idx, be_armed, be_active,
                    be_arm_idx, pending_be, h, l,
                    be_trigger_pct, be_offset_pct, be_delay_bars
                )
                pos[k, POS_SL]            = sl
                pos[k, POS_BE_ARMED]      = be_armed
                pos[k, POS_BE_ACTIVE]     = be_active
                pos[k, POS_BE_ARM_IDX]    = be_arm_idx
                pos[k, POS_PENDING_BE_SL] = pending_be

            if trailing_trigger_pct > 0.0 and has_atr:
                r_threshold     = pos[k, POS_RUNNER_THRESHOLD]
                r_armed         = pos[k, POS_RUNNER_ARMED]
                r_active        = pos[k, POS_RUNNER_ACTIVE]
                r_sl            = pos[k, POS_RUNNER_SL]
                r_active_before = r_active
                r_armed, r_active, r_sl, r_threshold = _update_runner(
                    side, ep, r_armed, r_active, r_sl, r_threshold,
                    h, l, c, atrs[i - 1],
                    trailing_trigger_pct, runner_trailing_mult
                )
                pos[k, POS_RUNNER_ARMED]     = r_armed
                pos[k, POS_RUNNER_ACTIVE]    = r_active
                pos[k, POS_RUNNER_SL]        = r_sl
                pos[k, POS_RUNNER_THRESHOLD] = r_threshold
                if r_active_before == 0.0 and r_active == 1.0:
                    k += 1; continue

            exit_price = -1.0; reason = 0

            if r_active and r_sl != 0.0:
                r_threshold   = pos[k, POS_RUNNER_THRESHOLD]
                be_rsn        = REASON_BE if be_active else REASON_SL
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

            if exit_price > 0.0:
                if n_trades < MAX_TRADES:
                    n_trades = _record_trade(
                        trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                        trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                        trade_setup_ids, trade_selected_score,
                        n_trades, side, exit_price, ep, entry_idx, i, reason,
                        pos[k, POS_MAE], pos[k, POS_MFE],
                        setup_id, sel_score
                    )
                    #-----
                    # TP EMA ne compte pas
                    # EMA1_TP / EMA2_TP / EMA_CROSS_TP ne comptent pas
                    #-----

                    if reason == REASON_TP and max_tp > 0:    
                        if tp_period_mode == 1:
                            recent_tp_idx[tp_head] = i
                            tp_head = (tp_head + 1) % (max_tp + 1)
                            tp_recent_count += 1
                        else:
                            tp_count += 1
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
    commission_pct=0.0, spread_pct=0.0, slippage_pct=0.0,
    alpha=5, period_freq="ME",
    entry_on_signal_close_price=False,
    trade_setup_ids=None,
    trade_selected_score=None,
):
    if len(trade_returns) == 0:
        return None

    cost    = commission_pct * 2 + spread_pct + slippage_pct * 2
    ret_arr = trade_returns - cost

    if entry_on_signal_close_price:
        entry_times = bar_index[trade_entry_idx - 1]
    else:
        entry_times = bar_index[trade_entry_idx]

    exit_times  = bar_index[trade_exit_idx]

    has_mae_mfe = (trade_mae is not None and trade_mfe is not None
                   and len(trade_mae) == len(trade_returns))
    mae_intra = trade_mae if has_mae_mfe else np.full(len(ret_arr), np.nan)
    mfe_intra = trade_mfe if has_mae_mfe else np.full(len(ret_arr), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        capture_ratio_intra = np.where(mfe_intra != 0, ret_arr / mfe_intra, np.nan)

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

    trades_df = pd.DataFrame({
        "entry_time": entry_times,
        "exit_time": exit_times,
        "entry_idx": trade_entry_idx,
        "exit_idx": trade_exit_idx,
        "entry": np.nan,
        "exit": trade_exit_prices,
        "side": trade_sides,
        "return": ret_arr,
        "reason": [REASON_LABELS.get(int(r), str(int(r))) for r in trade_reasons],
        "mae_intra": mae_intra,
        "mfe_intra": mfe_intra,
        "capture_ratio_intra": capture_ratio_intra,
        "mae_hold": mae_hold,
        "mfe_hold": mfe_hold,
        "capture_ratio_hold": capture_ratio_hold,
    })

    #-- multi setup mode --- 
    if trade_setup_ids is not None:
        trades_df["setup_id"] = trade_setup_ids

    if trade_selected_score is not None:
        trades_df["selected_score"] = trade_selected_score
    #------------------------

    pos_mask = ret_arr > 0; neg_mask = ret_arr < 0
    cum      = np.cumprod(1 + ret_arr)
    roll_max = np.maximum.accumulate(cum)
    dd_curve = (cum - roll_max) / roll_max
    mdd      = dd_curve.min()

    max_uw = current_uw = 0
    for d in dd_curve < 0:
        current_uw = current_uw + 1 if d else 0
        max_uw     = max(max_uw, current_uw)

    cum_return = cum[-1] - 1
    n_years    = (exit_times[-1] - entry_times[0]).days / 365
    ann_return = (1 + cum_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    std        = ret_arr.std()
    wins_sum   = ret_arr[pos_mask].sum()
    loss_sum   = abs(ret_arr[neg_mask].sum())
    var_t      = -np.percentile(ret_arr, alpha)

    n_tpy   = len(ret_arr) / n_years
    sharpe  = ret_arr.mean() / std * np.sqrt(n_tpy) if std > 0 else np.nan
    t_stat, p_value = scipy_stats.ttest_1samp(ret_arr, 0)
    p_binom = scipy_stats.binomtest(pos_mask.sum(), len(ret_arr), 0.5).pvalue

    t          = trades_df.set_index("exit_time")
    period_ret = t["return"].resample(period_freq).sum()
    period_ret = period_ret[period_ret != 0].to_numpy()
    pr_pos  = period_ret > 0; pr_neg = period_ret < 0
    pr_var  = np.percentile(period_ret, alpha) if len(period_ret) > 0 else np.nan
    pr_cvar = period_ret[period_ret <= pr_var].mean() \
              if len(period_ret) > 0 and (period_ret <= pr_var).any() else np.nan

    return {
        "n_trades"              : len(ret_arr),
        "win_rate"              : round(pos_mask.mean(), 3),
        "total_return_sum"      : round(ret_arr.sum(), 4),
        "cum_return"            : round(cum_return, 4),
        "ann_return"            : round(ann_return, 4) if not np.isnan(ann_return) else np.nan,
        "max_drawdown"          : round(mdd, 4),
        "max_underwater_trades" : max_uw,
        "calmar"                : round(ann_return / abs(mdd), 3) if mdd != 0 else np.nan,
        "sharpe"                : round(sharpe, 4),
        "profit_factor"         : round(wins_sum / loss_sum, 3) if loss_sum != 0 else np.nan,
        "avg_win"               : round(ret_arr[pos_mask].mean(), 4) if pos_mask.any() else np.nan,
        "avg_loss"              : round(ret_arr[neg_mask].mean(), 4) if neg_mask.any() else np.nan,
        "VaR"                   : round(var_t, 4),
        "CVaR"                  : round(-ret_arr[ret_arr <= -var_t].mean(), 4)
                                  if (ret_arr <= -var_t).any() else np.nan,
        "t_stat"                : round(t_stat, 3),
        "p_value"               : round(p_value, 4),
        "p_binom"               : round(p_binom, 4),
        "period_freq"           : period_freq,
        "n_periods"             : len(period_ret),
        "n_periods_positive"    : int(pr_pos.sum()),
        "n_periods_negative"    : int(pr_neg.sum()),
        "pct_periods_positive"  : round(pr_pos.mean(), 3) if len(period_ret) > 0 else np.nan,
        "worst_period"          : round(period_ret.min(), 4) if len(period_ret) > 0 else np.nan,
        "best_period"           : round(period_ret.max(), 4) if len(period_ret) > 0 else np.nan,
        "period_cvar"           : round(pr_cvar, 4) if pr_cvar is not None
                                  and not np.isnan(pr_cvar) else np.nan,
        "avg_mae_intra"         : round(float(np.nanmean(mae_intra)), 4) if has_mae_mfe else np.nan,
        "avg_mfe_intra"         : round(float(np.nanmean(mfe_intra)), 4) if has_mae_mfe else np.nan,
        "avg_capture_intra"     : round(float(np.nanmean(capture_ratio_intra)), 4) if has_mae_mfe else np.nan,
        "avg_mae_hold"          : round(float(np.nanmean(mae_hold)), 4)           if hold_bars > 0 else np.nan,
        "avg_mfe_hold"          : round(float(np.nanmean(mfe_hold)), 4)           if hold_bars > 0 else np.nan,
        "avg_capture_hold"      : round(float(np.nanmean(capture_ratio_hold)), 4) if hold_bars > 0 else np.nan,
        "trades_df"             : trades_df,
    }


# ══════════════════════════════════════════════════════════════════
# 6. NJITEngine
# ══════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class BacktestConfig:
    # Multi_setup structure activation
    multi_setup_mode: bool = True

    # Data / preprocessing
    timezone_shift: int = 1
    atr_period: int = 14

    # Default signal params
    period_1: int = 50
    period_2: int = 100

    # Entry / signal execution
    entry_delay: int = 1
    session_1: Optional[Tuple[str, str]] = None
    session_2: Optional[Tuple[str, str]] = None
    session_3: Optional[Tuple[str, str]] = None

    max_gap_signal: float = 0.0
    max_gap_entry: float = 0.0

    candle_size_filter: bool = False
    min_size_pct: float = 0.0
    max_size_pct: float = 1.0
    prev_candle_direction: bool = False

    multi_entry: bool = True
    reverse_mode: bool = False

    # Cooldown
    cooldown_entries: int = 0   # nb entrées avant déclenchement du cooldown
    cooldown_bars:    int = 0   # durée du cooldown en barres
    cooldown_mode:    int = 1   # 1=indépendant, 2=reset session, 3=reset jour

    # Entry cap logic
    me_max: int = 0
    me_period: int = 0
    me_reset_mode: int = 0

    # Entry price mode
    entry_on_close: bool = False  # False=open[i] (défaut), True=close[i]
    entry_on_signal_close_price: bool= False #True=close[i-1]

    # Exit — fixed / ATR TP-SL
    tp_pct: float = 0.01
    sl_pct: float = 0.005
    use_atr_sl_tp: int = 0
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    allow_exit_on_entry_bar: bool = True

    # Exit — EMA mode
    use_ema1_tp: bool = False
    use_ema2_tp: bool = False
    use_ema_cross_tp: bool = False

    # Exit — external signal
    use_exit_signal: bool = False
    exit_delay: int = 1

    # Break-even
    be_trigger_pct: float = 0.0
    be_offset_pct: float = 0.0
    be_delay_bars: int = 0

    # Runner trailing
    trailing_trigger_pct: float = 0.0
    runner_trailing_mult: float = 2.0

    # Metrics defaults
    track_mae_mfe: bool = True
    hold_minutes: int = 0
    bar_duration_min: int = 5
    commission_pct: float = 0.0
    spread_pct: float = 0.0
    slippage_pct: float = 0.0
    alpha: float = 5.0
    period_freq: str = "ME"

    # Inspection / plotting defaults
    return_df_after: bool = False
    plot: bool = False
    crypto: bool = False
    full_df_after: bool = False
    window_before: int = 200
    window_after: int = 50

    # Trade lifetime / forced flat / TP quota
    max_holding_bars: int = 0

    forced_flat_frequency: Optional[str] = None   # None, "day", "weekend"
    forced_flat_time: Optional[str] = None        # ex: "21:30"

    max_tp: int = 0
    tp_period_mode: int = 0   # 0=off, 1=rolling bars, 2=session, 3=day
    tp_period_bars: int = 0

    def __post_init__(self):
        if self.period_1 <= 0:       raise ValueError("period_1 must be > 0")
        if self.period_2 <= 0:       raise ValueError("period_2 must be > 0")
        if self.atr_period <= 0:     raise ValueError("atr_period must be > 0")
        if self.entry_delay <= 0:    raise ValueError("entry_delay below 1 implies look-ahead bias")
        if self.max_gap_signal < 0:  raise ValueError("max_gap_signal must be >= 0")
        if self.max_gap_entry < 0:   raise ValueError("max_gap_entry must be >= 0")
        if self.min_size_pct < 0:    raise ValueError("min_size_pct must be >= 0")
        if self.max_size_pct <= 0:   raise ValueError("max_size_pct must be > 0")
        if self.min_size_pct > self.max_size_pct: raise ValueError("min_size_pct cannot exceed max_size_pct")
        if self.tp_pct < 0:          raise ValueError("tp_pct must be >= 0")
        if self.sl_pct < 0:          raise ValueError("sl_pct must be >= 0")
        if self.use_atr_sl_tp not in (-1, 0, 1, 2): raise ValueError("use_atr_sl_tp must be in {-1,0,1,2}")
        if self.tp_atr_mult < 0:     raise ValueError("tp_atr_mult must be >= 0")
        if self.sl_atr_mult < 0:     raise ValueError("sl_atr_mult must be >= 0")
        if self.me_max < 0:          raise ValueError("me_max must be >= 0")
        if self.me_period < 0:       raise ValueError("me_period must be >= 0")
        if self.me_reset_mode not in (0,1,2,3,4,5): raise ValueError("me_reset_mode must be in {0..5}")
        if self.cooldown_entries < 0: raise ValueError("cooldown_entries must be >= 0")
        if self.cooldown_bars < 0:    raise ValueError("cooldown_bars must be >= 0")
        if self.cooldown_mode not in (1, 2, 3): raise ValueError("cooldown_mode must be 1, 2 or 3")
        if self.be_trigger_pct < 0:  raise ValueError("be_trigger_pct must be >= 0")
        if self.be_offset_pct < 0:   raise ValueError("be_offset_pct must be >= 0")
        if self.be_delay_bars < 0:   raise ValueError("be_delay_bars must be >= 0")
        if self.trailing_trigger_pct < 0: raise ValueError("trailing_trigger_pct must be >= 0")
        if self.runner_trailing_mult < 0: raise ValueError("runner_trailing_mult must be >= 0")
        if self.commission_pct < 0:  raise ValueError("commission_pct must be >= 0")
        if self.spread_pct < 0:      raise ValueError("spread_pct must be >= 0")
        if self.slippage_pct < 0:    raise ValueError("slippage_pct must be >= 0")
        if not (0 < self.alpha < 100): raise ValueError("alpha must be between 0 and 100")
        if self.bar_duration_min <= 0: raise ValueError("bar_duration_min must be > 0")
        if self.window_before < 0:   raise ValueError("window_before must be >= 0")
        if self.window_after < 0:    raise ValueError("window_after must be >= 0")
        if self.entry_on_close and self.entry_on_signal_close_price: raise ValueError("entry_on_close and entry_on_signal_close_price cannot both be True")
        if self.max_holding_bars < 0: raise ValueError("max_holding_bars must be >= 0")
        if self.forced_flat_frequency not in (None, "day", "weekend"):raise ValueError('forced_flat_frequency must be None, "day" or "weekend"')
        if (self.forced_flat_frequency is None) != (self.forced_flat_time is None):raise ValueError("forced_flat_frequency and forced_flat_time must be both set or both None")
        if self.max_tp < 0:raise ValueError("max_tp must be >= 0")
        if self.tp_period_mode not in (0, 1, 2, 3):raise ValueError("tp_period_mode must be in {0,1,2,3}")
        if self.tp_period_bars < 0:raise ValueError("tp_period_bars must be >= 0")
        if self.tp_period_mode == 1 and self.tp_period_bars <= 0:raise ValueError("tp_period_bars must be > 0 when tp_period_mode == 1")

class DataPipeline:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def fetchdata(self, ticker: str, start: str, end: str, timezone_shift: int = 0) -> pd.DataFrame:
        df = pd.read_csv(
            f"{self.base_path}/{ticker}.csv",
            header=None,
            names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
        )
        df["Datetime"] = pd.to_datetime(df["Datetime"]) + pd.Timedelta(hours=timezone_shift)
        df = df.set_index("Datetime").sort_index()
        return df.loc[start:end]

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)

        out = df.copy()
        out["ATR"] = tr.rolling(period).mean()
        return out

    def prepare_df(self, ticker: str, start: str, end: str, timezone_shift: int = 0, atr_period: int = 14) -> pd.DataFrame:
        df = self.fetchdata(ticker, start, end, timezone_shift=timezone_shift)
        df = self.compute_atr(df, atr_period)
        return df

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
        bls = 0.0
        bss = 0.0
        bl_id = -1
        bs_id = -1

        if allow_long:
            for j in range(n_setups):
                score = long_scores_matrix[i, j] * long_active_matrix[i, j]
                if score > bls:
                    bls = score
                    bl_id = setup_ids[j]

        if allow_short:
            for j in range(n_setups):
                score = short_scores_matrix[i, j] * short_active_matrix[i, j]
                if score > bss:
                    bss = score
                    bs_id = setup_ids[j]

        best_long_score[i] = bls
        best_short_score[i] = bss
        best_long_setup_id[i] = bl_id
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
        signals,
        selected_setup_id,
        selected_score,
        best_long_score,
        best_short_score,
        best_long_setup_id,
        best_short_setup_id,
    )


# ══════════════════════════════════════════════════════════════════
# 6. PYTHON WRAPPER
# ══════════════════════════════════════════════════════════════════
def aggregate_and_decide(
    setup_dfs: list[pd.DataFrame],
    decision_cfg: DecisionConfig,
) -> dict[str, np.ndarray]:
    mats = _build_setup_matrices(setup_dfs)

    (
        signals,
        selected_setup_id,
        selected_score,
        best_long_score,
        best_short_score,
        best_long_setup_id,
        best_short_setup_id,
    ) = aggregate_and_decide_njit(
        mats["long_scores_matrix"],
        mats["short_scores_matrix"],
        mats["long_active_matrix"],
        mats["short_active_matrix"],
        mats["setup_ids"],
        float(decision_cfg.min_score),
        decision_cfg.allow_long,
        decision_cfg.allow_short,
        decision_cfg.tie_policy,
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


class NJITEngine:

    def __init__(self, pipeline, ticker, start, end, cfg,
                atr_period=14, MAX_TRADES=50_000, MAX_POS=500):
        self.cfg        = cfg
        self.MAX_TRADES = MAX_TRADES
        self.MAX_POS    = MAX_POS

        df = pipeline.prepare_df(
            ticker=ticker,
            start=start,
            end=end,
            timezone_shift=cfg.timezone_shift,
            atr_period=cfg.atr_period
            )

        self.bar_index      = df.index
        self.opens          = df["Open"].to_numpy(dtype=np.float64)
        self.highs          = df["High"].to_numpy(dtype=np.float64)
        self.lows           = df["Low"].to_numpy(dtype=np.float64)
        self.closes         = df["Close"].to_numpy(dtype=np.float64)
        self.atrs           = df["ATR"].to_numpy(dtype=np.float64)
        self.minutes_of_day = (df.index.hour * 60 + df.index.minute).to_numpy(dtype=np.int16)
        self.day_index      = ((df.index - df.index[0]).days).to_numpy(dtype=np.int32)
        self.day_of_week = df.index.dayofweek.to_numpy(dtype=np.int8)
        self.last_signal_df  = None
        self.last_signal_col = "Signal"

        self._warmup()

# ══════════════════════════════════════════════════════════════════
# 6.1. df inspection pre and post engine execution
# ══════════════════════════════════════════════════════════════════

    def signal_generation_inspection(self, strategy_fn=None, signal_col="Signal", plot=False,
                                     crypto=False, return_df_signals=True, **kwargs):
        df = pd.DataFrame({
            "Open": self.opens, "High": self.highs,
            "Low":  self.lows,  "Close": self.closes, "ATR": self.atrs,
        }, index=self.bar_index)

        if strategy_fn is not None:
            df = strategy_fn(df.copy(), **kwargs)
        else:
            ema1 = ema_njit(self.closes, self.cfg.period_1)
            ema2 = ema_njit(self.closes, self.cfg.period_2)
            _, _, sig = signals_ema_vs_close_njit(
                self.opens, self.closes, self.cfg.period_1, self.cfg.period_2
            )
            print('Automatic Fallback on default ema_close strategy; pass strategy_fn=my_strategy to use yours.')
            df["EMA1"] = ema1; df["EMA2"] = ema2; df[signal_col] = sig

        self.last_signal_df  = df.copy()
        self.last_signal_col = signal_col

        if plot:
            self._plot_signal_df(df, signal_col=signal_col, crypto=crypto)

        if return_df_signals:
            return df
        return df[signal_col].to_numpy(dtype=np.int8)

    def _build_after_run_df(
        self,
        trades_df,
        full_df=False,
        window_before=200,
        window_after=50,
        entry_on_signal_close_price=False
    ):
        if self.last_signal_df is not None:
            df = self.last_signal_df.copy()
        else:
            df = pd.DataFrame({
                "Open": self.opens, "High": self.highs,
                "Low":  self.lows,  "Close": self.closes, "ATR": self.atrs,
            }, index=self.bar_index)

        df["EntryTradeID"] = np.nan
        df["ExitTradeID"]  = np.nan
        long_id = 0
        short_id = 0
        trade_ids = []

        trades_df = trades_df.copy()

        if entry_on_signal_close_price:
            trades_df["plot_entry_idx"] = np.maximum(trades_df["entry_idx"].astype(int) - 1, 0)
        else:
            trades_df["plot_entry_idx"] = trades_df["entry_idx"].astype(int)

        for _, tr in trades_df.iterrows():
            if tr["side"] == 1:
                long_id += 1
                tid = long_id
            else:
                short_id += 1
                tid = -short_id

            trade_ids.append(tid)

            pei = int(tr["plot_entry_idx"])
            exi = int(tr["exit_idx"])

            df.iloc[pei, df.columns.get_loc("EntryTradeID")] = tid
            df.iloc[exi, df.columns.get_loc("ExitTradeID")]  = tid

        trades_df["trade_id"] = trade_ids

        if len(trades_df) == 0 or full_df:
            return df.copy(), trades_df

        first_idx = max(0, int(trades_df["plot_entry_idx"].min()) - window_before)
        last_idx  = min(len(df), int(trades_df["exit_idx"].max()) + window_after + 1)

        return df.iloc[first_idx:last_idx].copy(), trades_df

    @staticmethod
    def _plot_signal_df(df, signal_col="Signal", title="Signal preparation", crypto=False):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
        if signal_col not in df.columns:
            raise ValueError(f"'{signal_col}' not found in df columns: {list(df.columns)}")
        long_signals  = df[df[signal_col] == 1]
        short_signals = df[df[signal_col] == -1]
        fig.add_trace(go.Scatter(x=long_signals.index,  y=long_signals["Low"]  * 0.999,
            mode="markers", marker=dict(symbol="triangle-up",   size=8, color="lime"), name="Long signal"))
        fig.add_trace(go.Scatter(x=short_signals.index, y=short_signals["High"] * 1.001,
            mode="markers", marker=dict(symbol="triangle-down", size=8, color="red"),  name="Short signal"))
        rangebreaks = [] if crypto else [dict(bounds=["sat", "mon"])]
        fig.update_layout(title=title, template="plotly_dark", xaxis_rangeslider_visible=False,
            xaxis=dict(rangebreaks=rangebreaks), hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0), height=750)
        fig.show()

    @staticmethod
    def _plot_backtest(df, trades, title="Backtest results", crypto=False):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
        long_entries  = trades[trades["side"] == 1]
        short_entries = trades[trades["side"] == -1]
        fig.add_trace(go.Scatter(x=long_entries["entry_time"],  y=long_entries["entry"],
            mode="markers", marker=dict(symbol="triangle-up",   size=10, color="lime"), name="Long entry"))
        fig.add_trace(go.Scatter(x=short_entries["entry_time"], y=short_entries["entry"],
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),  name="Short entry"))
        for _, r in trades.iterrows():
            color = "lime" if r["side"] == 1 else "red"
            fig.add_trace(go.Scatter(x=[r["entry_time"], r["exit_time"]], y=[r["entry"], r["exit"]],
                mode="lines", line=dict(color=color, width=1), opacity=0.5, showlegend=False))
            fig.add_trace(go.Scatter(x=[r["exit_time"]], y=[r["exit"]],
                mode="markers", marker=dict(symbol="x", size=9, color=color), showlegend=False))
        rangebreaks = [] if crypto else [dict(bounds=["sat", "mon"])]
        fig.update_layout(title=title, template="plotly_dark", xaxis_rangeslider_visible=False,
            xaxis=dict(rangebreaks=rangebreaks), hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0), height=700)
        fig.show()

    @staticmethod
    def _parse_session(s):
        if s is None: return -1
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    def _warmup(self):
        n = min(500, len(self.opens))
        sig = np.zeros(n, dtype=np.int8)
        sig[10] = 1
        sig[20] = -1

        selected_setup_id = np.full(n, -1, dtype=np.int32)
        selected_score = np.zeros(n, dtype=np.float64)

        backtest_njit(
            self.opens[:n], self.highs[:n], self.lows[:n], self.closes[:n], self.atrs[:n],
            sig, selected_setup_id, selected_score,
            self.minutes_of_day[:n], self.day_index[:n], self.day_of_week[:n],

            1,      # entry_delay
            -1, -1, -1, -1, -1, -1,   # sessions
            0.0, 0.0,                  # max_gap_signal, max_gap_entry
            False, 0.0, 1.0,           # candle_size_filter, min_size_pct, max_size_pct
            False,                     # prev_candle_direction
            0.001, 0.01, 0,            # tp_pct, sl_pct, use_atr_sl_tp
            2.0, 1.0, True,            # tp_atr_mult, sl_atr_mult, allow_exit_on_entry_bar

            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
            False, False, False,

            np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.float64),
            False, 1,                  # use_exit_signal, exit_delay
            0.0, 0.0, 0,               # BE
            0.0, 2.0,                  # trailing

            True, False,               # multi_entry, reverse_mode
            0, 0, 0,                   # me_max, me_period, me_reset_mode
            1000, 10, True,            # MAX_TRADES, MAX_POS, track_mae_mfe
            0, 0, 1,                   # cooldown_entries, cooldown_bars, cooldown_mode
            False, False,              # entry_on_close, entry_on_signal_close_price
            0,                         # max_holding_bars
            0, -1,                     # forced_flat_mode, forced_flat_minute
            0, 0, 0                    # max_tp, tp_period_mode, tp_period_bars
        )
        print("NJITEngine — JIT warmup done ✓")

    def signals_ema(self, span1=None, span2=None, mode="close_vs_ema"):
        s1 = span1 or self.cfg.period_1
        s2 = span2 or self.cfg.period_2
        if mode == "close_vs_ema":
            _, _, sig = signals_ema_vs_close_njit(self.opens, self.closes, s1, s2)
        else:
            _, _, sig = signals_ema_cross_njit(self.closes, s1, s2)
        return sig

    def signals_from_strategy(self, strategy_fn, signal_col="Signal", **kwargs):
        df = pd.DataFrame({
            "Open": self.opens, "High": self.highs,
            "Low":  self.lows,  "Close": self.closes,
        }, index=self.bar_index)
        df_out = strategy_fn(df, **kwargs)
        if signal_col not in df_out.columns:
            raise ValueError(f"strategy_fn must return a DataFrame with '{signal_col}'")
        return df_out[signal_col].to_numpy(dtype=np.int8)

    def run(
        self,
        signals,
        tp_pct=None, sl_pct=None,
        use_atr_sl_tp=None, tp_atr_mult=None, sl_atr_mult=None,
        entry_delay=None,
        session_1=None, session_2=None, session_3=None,
        max_gap_signal=None, max_gap_entry=None,
        candle_size_filter=None, min_size_pct=None, max_size_pct=None,
        prev_candle_direction=None,
        allow_exit_on_entry_bar=None,
        exit_ema1=None, exit_ema2=None, use_ema1_tp=None, use_ema2_tp=None,
        use_ema_cross_tp=None,
        exit_signals=None, signal_tags=None,
        use_exit_signal=None, exit_delay=None,
        be_trigger_pct=None, be_offset_pct=None, be_delay_bars=None,
        trailing_trigger_pct=None, runner_trailing_mult=None,
        multi_entry=None, reverse_mode=None,
        me_max=None, me_period=None, me_reset_mode=None,
        cooldown_entries=None, cooldown_bars=None, cooldown_mode=None,
        entry_on_close=None,
        track_mae_mfe=None, hold_minutes=None, bar_duration_min=None,
        commission_pct=None, spread_pct=None, slippage_pct=None,
        alpha=None, period_freq=None,
        return_df_after=None, plot=None, crypto=None,
        full_df_after=None, window_before=None, window_after=None,entry_on_signal_close_price=None,
        max_holding_bars=None,
        forced_flat_frequency=None, forced_flat_time=None,
        max_tp=None, tp_period_mode=None, tp_period_bars=None,selected_setup_id=None, selected_score=None,multi_setup_mode=None,
    ):
        cfg = self.cfg

       


        tp_pct               = tp_pct               if tp_pct               is not None else cfg.tp_pct
        sl_pct               = sl_pct               if sl_pct               is not None else cfg.sl_pct
        use_atr_sl_tp        = use_atr_sl_tp        if use_atr_sl_tp        is not None else cfg.use_atr_sl_tp
        tp_atr_mult          = tp_atr_mult          if tp_atr_mult          is not None else cfg.tp_atr_mult
        sl_atr_mult          = sl_atr_mult          if sl_atr_mult          is not None else cfg.sl_atr_mult
        entry_delay          = entry_delay          if entry_delay          is not None else cfg.entry_delay
        session_1            = session_1            if session_1            is not None else cfg.session_1
        session_2            = session_2            if session_2            is not None else cfg.session_2
        session_3            = session_3            if session_3            is not None else cfg.session_3
        max_gap_signal       = max_gap_signal       if max_gap_signal       is not None else cfg.max_gap_signal
        max_gap_entry        = max_gap_entry        if max_gap_entry        is not None else cfg.max_gap_entry
        candle_size_filter   = candle_size_filter   if candle_size_filter   is not None else cfg.candle_size_filter
        min_size_pct         = min_size_pct         if min_size_pct         is not None else cfg.min_size_pct
        max_size_pct         = max_size_pct         if max_size_pct         is not None else cfg.max_size_pct
        prev_candle_direction= prev_candle_direction if prev_candle_direction is not None else cfg.prev_candle_direction
        allow_exit_on_entry_bar = allow_exit_on_entry_bar if allow_exit_on_entry_bar is not None else cfg.allow_exit_on_entry_bar
        use_ema1_tp          = use_ema1_tp          if use_ema1_tp          is not None else cfg.use_ema1_tp
        use_ema2_tp          = use_ema2_tp          if use_ema2_tp          is not None else cfg.use_ema2_tp
        use_ema_cross_tp     = use_ema_cross_tp     if use_ema_cross_tp     is not None else cfg.use_ema_cross_tp
        use_exit_signal      = use_exit_signal      if use_exit_signal      is not None else cfg.use_exit_signal
        exit_delay           = exit_delay           if exit_delay           is not None else cfg.exit_delay
        be_trigger_pct       = be_trigger_pct       if be_trigger_pct       is not None else cfg.be_trigger_pct
        be_offset_pct        = be_offset_pct        if be_offset_pct        is not None else cfg.be_offset_pct
        be_delay_bars        = be_delay_bars        if be_delay_bars        is not None else cfg.be_delay_bars
        trailing_trigger_pct = trailing_trigger_pct if trailing_trigger_pct is not None else cfg.trailing_trigger_pct
        runner_trailing_mult = runner_trailing_mult if runner_trailing_mult is not None else cfg.runner_trailing_mult
        multi_entry          = multi_entry          if multi_entry          is not None else cfg.multi_entry
        reverse_mode         = reverse_mode         if reverse_mode         is not None else cfg.reverse_mode
        me_max               = me_max               if me_max               is not None else cfg.me_max
        me_period            = me_period            if me_period            is not None else cfg.me_period
        me_reset_mode        = me_reset_mode        if me_reset_mode        is not None else cfg.me_reset_mode
        cooldown_entries     = cooldown_entries     if cooldown_entries     is not None else cfg.cooldown_entries
        cooldown_bars        = cooldown_bars        if cooldown_bars        is not None else cfg.cooldown_bars
        cooldown_mode        = cooldown_mode        if cooldown_mode        is not None else cfg.cooldown_mode
        entry_on_close       = entry_on_close       if entry_on_close       is not None else cfg.entry_on_close
        track_mae_mfe        = track_mae_mfe        if track_mae_mfe        is not None else cfg.track_mae_mfe
        hold_minutes         = hold_minutes         if hold_minutes         is not None else cfg.hold_minutes
        bar_duration_min     = bar_duration_min     if bar_duration_min     is not None else cfg.bar_duration_min
        commission_pct       = commission_pct       if commission_pct       is not None else cfg.commission_pct
        spread_pct           = spread_pct           if spread_pct           is not None else cfg.spread_pct
        slippage_pct         = slippage_pct         if slippage_pct         is not None else cfg.slippage_pct
        alpha                = alpha                if alpha                is not None else cfg.alpha
        period_freq          = period_freq          if period_freq          is not None else cfg.period_freq
        return_df_after      = return_df_after      if return_df_after      is not None else cfg.return_df_after
        plot                 = plot                 if plot                 is not None else cfg.plot
        crypto               = crypto               if crypto               is not None else cfg.crypto
        full_df_after        = full_df_after        if full_df_after        is not None else cfg.full_df_after
        window_before        = window_before        if window_before        is not None else cfg.window_before
        window_after         = window_after         if window_after         is not None else cfg.window_after
        entry_on_signal_close_price = entry_on_signal_close_price if entry_on_signal_close_price is not None else cfg.entry_on_signal_close_price
        max_holding_bars            = max_holding_bars if max_holding_bars is not None else cfg.max_holding_bars
        forced_flat_frequency       = forced_flat_frequency if forced_flat_frequency is not None else cfg.forced_flat_frequency
        forced_flat_time            = forced_flat_time if forced_flat_time is not None else cfg.forced_flat_time
        max_tp                      = max_tp if max_tp is not None else cfg.max_tp
        tp_period_mode              = tp_period_mode if tp_period_mode is not None else cfg.tp_period_mode
        tp_period_bars              = tp_period_bars if tp_period_bars is not None else cfg.tp_period_bars

        # signals est pas sensé etre retourné avant l'appel du moteur ? tant mieux si il se fait de maniere automatique pour muli setup mode
        signals = np.asarray(signals, dtype=np.int8)

         #----- Multi Set-up mode ------
        multi_setup_mode = multi_setup_mode if multi_setup_mode is not None else cfg.multi_setup_mode

        if multi_setup_mode:
            if selected_setup_id is None or selected_score is None:
                raise ValueError("In multi_setup_mode, selected_setup_id and selected_score must be provided")

            selected_setup_id = np.asarray(selected_setup_id, dtype=np.int32)
            selected_score = np.asarray(selected_score, dtype=np.float64)

            if len(selected_setup_id) != len(signals):
                raise ValueError("selected_setup_id must have same length as signals")
            if len(selected_score) != len(signals):
                raise ValueError("selected_score must have same length as signals")
        else:
            if selected_setup_id is None:
                selected_setup_id = np.full(len(signals), -1, dtype=np.int32)
            else:
                selected_setup_id = np.asarray(selected_setup_id, dtype=np.int32)

            if selected_score is None:
                selected_score = np.zeros(len(signals), dtype=np.float64)
            else:
                selected_score = np.asarray(selected_score, dtype=np.float64)

        s1_start = self._parse_session(session_1[0]) if session_1 else -1
        s1_end   = self._parse_session(session_1[1]) if session_1 else -1
        s2_start = self._parse_session(session_2[0]) if session_2 else -1
        s2_end   = self._parse_session(session_2[1]) if session_2 else -1
        s3_start = self._parse_session(session_3[0]) if session_3 else -1
        s3_end   = self._parse_session(session_3[1]) if session_3 else -1

        _e_ema1 = exit_ema1    if exit_ema1    is not None else np.empty(0, dtype=np.float64)
        _e_ema2 = exit_ema2    if exit_ema2    is not None else np.empty(0, dtype=np.float64)
        _e_sig  = exit_signals if exit_signals is not None else np.zeros(0, dtype=np.int8)
        _s_tags = signal_tags  if signal_tags  is not None else np.zeros(0, dtype=np.float64)
        
        if entry_on_close and entry_on_signal_close_price:
            raise ValueError("entry_on_close and entry_on_signal_close_price cannot both be True")
        if forced_flat_frequency is None or forced_flat_time is None:
            forced_flat_mode = 0
            forced_flat_minute = -1
        else:
            if forced_flat_frequency == "day":
                forced_flat_mode = 1
            elif forced_flat_frequency == "weekend":
                forced_flat_mode = 2
            else:
                raise ValueError('forced_flat_frequency must be None, "day" or "weekend"')
            forced_flat_minute = self._parse_session(forced_flat_time)

        rets, sides, entry_idx, exit_idx, reasons, exit_prices, mae, mfe, trade_setup_ids, trade_selected_score = backtest_njit(
            self.opens, self.highs, self.lows, self.closes, self.atrs,
            signals, selected_setup_id, selected_score, self.minutes_of_day, self.day_index, self.day_of_week,
            entry_delay, s1_start, s1_end, s2_start, s2_end, s3_start, s3_end,
            max_gap_signal, max_gap_entry, candle_size_filter, min_size_pct, max_size_pct,
            prev_candle_direction, tp_pct, sl_pct, use_atr_sl_tp,
            tp_atr_mult, sl_atr_mult, allow_exit_on_entry_bar,
            _e_ema1, _e_ema2, use_ema1_tp, use_ema2_tp, use_ema_cross_tp,
            _e_sig, _s_tags, use_exit_signal, exit_delay,
            be_trigger_pct, be_offset_pct, be_delay_bars,
            trailing_trigger_pct, runner_trailing_mult,
            multi_entry, reverse_mode,
            me_max, me_period, me_reset_mode,
            self.MAX_TRADES, self.MAX_POS, track_mae_mfe,
            cooldown_entries, cooldown_bars, cooldown_mode,
            entry_on_close, entry_on_signal_close_price,
            max_holding_bars,
            forced_flat_mode, forced_flat_minute,
            max_tp, tp_period_mode, tp_period_bars
        )

        metrics = compute_metrics_full(
            rets, sides, entry_idx, exit_idx, reasons, exit_prices, mae, mfe,
            self.bar_index,
            highs=self.highs, lows=self.lows,
            hold_minutes=hold_minutes, bar_duration_min=bar_duration_min,
            commission_pct=commission_pct, spread_pct=spread_pct,
            slippage_pct=slippage_pct, alpha=alpha, period_freq=period_freq,
            entry_on_signal_close_price=entry_on_signal_close_price,
            trade_setup_ids=trade_setup_ids if multi_setup_mode else None,
            trade_selected_score=trade_selected_score if multi_setup_mode else None,
        )
        # -- security if no trades ----
        if metrics is None:
            return rets, {"trades_df": pd.DataFrame()}
        
        if entry_on_signal_close_price:
            metrics["trades_df"]["entry"] = self.closes[entry_idx - 1]
        elif entry_on_close:
            metrics["trades_df"]["entry"] = self.closes[entry_idx]
        else:
            metrics["trades_df"]["entry"] = self.opens[entry_idx]
        metrics["trades_df"]["exit"]      = exit_prices
        metrics["trades_df"]["entry_idx"] = entry_idx
        metrics["trades_df"]["exit_idx"]  = exit_idx

        # Il faudrait que trades_df contienne un plot_entry_idx ou que _build_after_run_df() sache quel mode d’entrée a été utilisé.
        if return_df_after:
            df_after, trades_df_annot = self._build_after_run_df(
                metrics["trades_df"],
                full_df=full_df_after,
                window_before=window_before,
                window_after=window_after,
                entry_on_signal_close_price=entry_on_signal_close_price
            )
            metrics["trades_df"] = trades_df_annot
            metrics["df_after"]  = df_after

        if plot:
            df_price = pd.DataFrame({
                "Open": self.opens, "High": self.highs,
                "Low":  self.lows,  "Close": self.closes,
            }, index=self.bar_index)
            self._plot_backtest(df_price, metrics["trades_df"], crypto=crypto)

        return rets, metrics

    def run_with_inspection(self, signals, signal_df=None, plot_before=False, plot_after=False,
                            signal_col="Signal", crypto=False, **run_kwargs):
        if signal_df is not None and plot_before:
            self._plot_signal_df(signal_df, signal_col=signal_col, crypto=crypto)

        rets, metrics = self.run(signals, **run_kwargs)
        trades_df = metrics["trades_df"].copy()

        if "entry" not in trades_df.columns:
            raise ValueError("trades_df must contain 'entry' and 'exit' columns for plotting.")

        if plot_after:
            df_price = pd.DataFrame({
                "Open": self.opens, "High": self.highs,
                "Low":  self.lows,  "Close": self.closes,
            }, index=self.bar_index)
            self._plot_backtest(df_price, trades_df, crypto=crypto)

        return rets, metrics

    def _base_price_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Open": self.opens,
            "High": self.highs,
            "Low": self.lows,
            "Close": self.closes,
            "ATR": self.atrs,
        }, index=self.bar_index)

    def prepare_multi_setup_signals(
        self,
        setup_specs: list[SetupSpec],
        decision_cfg: DecisionConfig,
        include_price_cols: bool = True,
    ) -> dict[str, np.ndarray]:
        if len(setup_specs) == 0:
            raise ValueError("setup_specs cannot be empty")

        df = self._base_price_df() if include_price_cols else pd.DataFrame(index=self.bar_index)

        setup_dfs: list[pd.DataFrame] = []
        for k, spec in enumerate(setup_specs):
            setup_name = spec.name if spec.name else f"setup[{k}]"
            sdf = spec.fn(df.copy(), **spec.params)
            _validate_setup_df(sdf, expected_index=self.bar_index, setup_name=setup_name)
            setup_dfs.append(sdf)

        return aggregate_and_decide(
            setup_dfs=setup_dfs,
            decision_cfg=decision_cfg,
        )


#------------------------------- tests ------------------------------------

pipeline = DataPipeline("/Users/arnaudbarbier/Desktop/Quant reaserch/Metals/CSV/Metals")
cfg = BacktestConfig(
    # signal defaults
    period_1=30,
    period_2=100,

    # entry logic
    entry_delay=1,
    prev_candle_direction=False,

    # session filters
    session_1=("08:00", "12:00"),
    session_2=("13:00", "17:00"),
    session_3=None,

    # candle filter
    candle_size_filter=True,
    min_size_pct=0.000,
    max_size_pct=0.1,

    # TP / SL
    tp_pct=0.002,
    sl_pct=0.01,

    # break-even
    be_trigger_pct=0.005,
    be_offset_pct=0.001,
    be_delay_bars=5,

    # runner trailing
    trailing_trigger_pct=0.005,
    runner_trailing_mult=3,

    # entry cap logic
    me_max=3,
    me_period=10,
    me_reset_mode=3,

    # metrics
    track_mae_mfe=True,
    hold_minutes=2 * 60,
    bar_duration_min=5,

    # optional preprocessing
    timezone_shift=1,
)

# appel type avec stratégie ema close par défaut 
njit_engine = NJITEngine(
    pipeline, "XAUUSD_M5", "2026-01-02", "2026-01-10", cfg,    MAX_TRADES   = 50_000,
    MAX_POS      = 600
)


signals = njit_engine.signals_ema(
    span1 = 30,#cfg.period_1,   # 50
    span2 = cfg.period_2,   # 100
    mode  = "close_vs_ema", # même logique que Strategy_Signal.ema_cross()
)

rets, metrics_v2 = njit_engine.run(
    signals,
    tp_pct=0.002,
    sl_pct=0.01,
    be_trigger_pct=0.005,
    be_offset_pct=0.001,
    be_delay_bars=5,
    me_max=1,
    me_reset_mode=2,
    me_period=100,
    session_1 = ("08:00", "12:00"),
    session_2 = ("13:00", "17:00"),
    session_3 = ("01:30", "06:00"),  # désactivée
    track_mae_mfe=True,
    hold_minutes=2*60,
    bar_duration_min=5,
    candle_size_filter=True, min_size_pct=0.000, max_size_pct=0.1,
    entry_delay=1,  
    prev_candle_direction=False,
    trailing_trigger_pct=0.005,
    runner_trailing_mult=3,
    plot=True,
    #entry_on_close=True,
    entry_on_signal_close_price=True,
    cooldown_entries=1, cooldown_bars=1000, cooldown_mode=2,
    multi_setup_mode=False,
)

print(metrics_v2)
#print("cfg trailing:", cfg.trailing_trigger_pct, cfg.runner_trailing_mult)
print("V2 reasons:", metrics_v2["trades_df"]["reason"].value_counts())
