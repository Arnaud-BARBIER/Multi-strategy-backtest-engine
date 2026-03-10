import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numba import njit
from scipy import stats as scipy_stats

from .signals import (
    ema_njit,
    atr_wilder_njit,
    signals_ema_vs_close_njit,
    signals_ema_cross_njit
)
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
def _record_trade(trade_returns, trade_sides, trade_entry_idx, trade_exit_idx, trade_reasons,
                  trade_exit_prices, trade_mae, trade_mfe,
                  n_trades, side, exit_price, ep, entry_idx, exit_idx, reason, mae, mfe):
    trade_returns[n_trades]   = side * (exit_price - ep) / ep
    trade_sides[n_trades]     = int(side)
    trade_entry_idx[n_trades] = entry_idx
    trade_exit_idx[n_trades]  = exit_idx
    trade_reasons[n_trades]   = reason
    trade_mae[n_trades]       = mae
    trade_mfe[n_trades]       = mfe
    trade_exit_prices[n_trades] = exit_price
    return n_trades + 1


@njit(cache=True)
def _update_runner(side, ep, runner_armed, runner_active, runner_sl,
                   runner_threshold, h, l, close, atr_value,
                   trailing_trigger_pct, runner_trailing_mult):
    if trailing_trigger_pct <= 0.0:
        return runner_armed, runner_active, runner_sl, runner_threshold

    # Armement
    if not runner_armed and not runner_active:
        threshold = ep * (1.0 + side * trailing_trigger_pct)
        if (side == 1 and h >= threshold) or (side == -1 and l <= threshold):
            runner_armed     = 1.0
            runner_threshold = threshold
            return runner_armed, runner_active, runner_sl, runner_threshold  # ← stop ici

    # Activation bougie suivante
    if runner_armed and not runner_active:
        runner_active = 1.0; runner_armed = 0.0
        init_sl   = close - side * atr_value * runner_trailing_mult
        runner_sl = max(init_sl, ep) if side == 1 else min(init_sl, ep)
        return runner_armed, runner_active, runner_sl, runner_threshold

    # Trailing dynamique
    if runner_active:
        new_sl = close - side * atr_value * runner_trailing_mult
        if side == 1: runner_sl = max(runner_sl, new_sl, ep)
        else:         runner_sl = min(runner_sl, new_sl, ep)

    return runner_armed, runner_active, runner_sl, runner_threshold


# ══════════════════════════════════════════════════════════════════
# 4. Main Engine   
# ══════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def backtest_njit(
    opens, highs, lows, closes, atrs, signals, minutes_of_day, day_index,
    entry_delay, s1_start, s1_end, s2_start, s2_end, s3_start, s3_end,
    max_gap_signal, max_gap_entry, candle_size_filter, min_size_pct, max_size_pct,
    prev_candle_direction, tp_pct, sl_pct, use_atr_sl_tp,
    tp_atr_mult, sl_atr_mult, allow_exit_on_entry_bar,
    exit_ema1, exit_ema2, use_ema1_tp, use_ema2_tp, use_ema_cross_tp,
    exit_signals, signal_tags, use_exit_signal, exit_delay,
    be_trigger_pct, be_offset_pct, be_delay_bars,
    trailing_trigger_pct, runner_trailing_mult,
    multi_entry, reverse_mode, me_max, me_period, me_reset_mode,
    MAX_TRADES, MAX_POS, track_mae_mfe
):
    n = opens.shape[0]

    trade_returns   = np.empty(MAX_TRADES, dtype=np.float64)
    trade_sides     = np.empty(MAX_TRADES, dtype=np.int8)
    trade_entry_idx = np.empty(MAX_TRADES, dtype=np.int32)
    trade_exit_idx  = np.empty(MAX_TRADES, dtype=np.int32)
    trade_reasons   = np.empty(MAX_TRADES, dtype=np.int8)
    trade_exit_prices = np.empty(MAX_TRADES, dtype=np.float64)
    trade_mae       = np.empty(MAX_TRADES, dtype=np.float64)
    trade_mfe       = np.empty(MAX_TRADES, dtype=np.float64)
    n_trades        = 0

    pos   = np.zeros((MAX_POS, POS_N_COLS), dtype=np.float64)
    n_pos = 0

    recent_entries = np.zeros(me_max + 1, dtype=np.int32)
    re_head        = 0
    re_count       = 0
    last_day       = -1
    last_session   = -2

    ema_exit_mode = use_ema1_tp or use_ema2_tp or use_ema_cross_tp
    has_exit_sig  = use_exit_signal and exit_signals.shape[0] == n
    has_tags      = signal_tags.shape[0] == n
    has_ema_exit  = exit_ema1.shape[0] == n
    has_atr       = atrs.shape[0] == n

    delayed_signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        o = opens[i]; h = highs[i]; l = lows[i]; c = closes[i]
        cur_day     = day_index[i]
        delayed_signals[i] = signals[i - entry_delay]
        sig = delayed_signals[i]
        cur_session = -1
        if   s1_start >= 0 and s1_start <= minutes_of_day[i] <= s1_end: cur_session = 0
        elif s2_start >= 0 and s2_start <= minutes_of_day[i] <= s2_end: cur_session = 1
        elif s3_start >= 0 and s3_start <= minutes_of_day[i] <= s3_end: cur_session = 2


        # ── 1. REVERSE ───────────────────────────────────────────
        if reverse_mode and sig != 0:
            k = 0
            while k < n_pos:
                if int(pos[k, POS_SIDE]) == -int(sig):
                    ep = pos[k, POS_ENTRY_PRICE]; side = pos[k, POS_SIDE]
                    if n_trades < MAX_TRADES:
                        n_trades = _record_trade(
                            trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                            trade_reasons,trade_exit_prices, trade_mae, trade_mfe,
                            n_trades, side, o, ep, int(pos[k, POS_ENTRY_IDX]), i,
                            REASON_REVERSE, pos[k, POS_MAE], pos[k, POS_MFE]
                        )
                    pos[k] = pos[n_pos - 1]; n_pos -= 1
                else:
                    k += 1

        # ── 1.5 MAX ENTRIES RESET / SLIDING ─────────────────────────────
        if me_reset_mode == 1 or me_reset_mode == 5:
            if last_day == -1:
                last_day = cur_day
            elif cur_day != last_day:
                re_count = 0
                re_head  = 0
                last_day = cur_day

        elif me_reset_mode == 2 or me_reset_mode == 4:
            if last_session == -2:
                last_session = cur_session
            elif cur_session != last_session:
                re_count = 0
                re_head  = 0
                last_session = cur_session

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

            if in_session:
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
                atr_val = atrs[i - entry_delay] if (has_atr and use_atr_sl_tp != 0) else 0.0
                atr_ok  = not (use_atr_sl_tp != 0 and has_atr and atr_val != atr_val)

                if gap_ok and candle_ok and pos_ok and atr_ok and me_ok:
                    if multi_ok:
                        ep2     = o
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
                        n_pos += 1

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
                    n_trades = _record_trade(
                        trade_returns, trade_sides, trade_entry_idx, trade_exit_idx,
                        trade_reasons,trade_exit_prices, trade_mae, trade_mfe,
                        n_trades, side, o, ep, int(pos[last, POS_ENTRY_IDX]), i,
                        REASON_EXIT_SIG, pos[last, POS_MAE], pos[last, POS_MFE]
                    )
                n_pos -= 1

        # ── 4. EXIT SL/TP ────────────────────────────────────────
        k = 0
        while k < n_pos:
            side      = pos[k, POS_SIDE];  ep = pos[k, POS_ENTRY_PRICE]
            tp        = pos[k, POS_TP];    sl = pos[k, POS_SL]
            entry_idx = int(pos[k, POS_ENTRY_IDX])
            be_armed  = pos[k, POS_BE_ARMED];  be_active = pos[k, POS_BE_ACTIVE]
            be_arm_idx = pos[k, POS_BE_ARM_IDX]
            r_armed   = pos[k, POS_RUNNER_ARMED]; r_active = pos[k, POS_RUNNER_ACTIVE]
            r_sl      = pos[k, POS_RUNNER_SL];    tag      = pos[k, POS_TAG]

            if not allow_exit_on_entry_bar and i == entry_idx:
                k += 1; continue

            # MAE/MFE intra-trade
            if track_mae_mfe:
                if side == 1:
                    pos[k, POS_MFE] = max(pos[k, POS_MFE], (h - ep) / ep)
                    pos[k, POS_MAE] = min(pos[k, POS_MAE], (l - ep) / ep)
                else:
                    pos[k, POS_MFE] = max(pos[k, POS_MFE], (ep - l) / ep)
                    pos[k, POS_MAE] = min(pos[k, POS_MAE], (ep - h) / ep)

            # Exit signal ALL/ciblé
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
                            trade_reasons,trade_exit_prices, trade_mae, trade_mfe,
                            n_trades, side, o, ep, entry_idx, i, REASON_EXIT_SIG,
                            pos[k, POS_MAE], pos[k, POS_MFE]
                        )
                    pos[k] = pos[n_pos - 1]; n_pos -= 1
                    continue

            # BE update
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

            # Runner update
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
                        trade_reasons,trade_exit_prices, trade_mae, trade_mfe,
                        n_trades, side, exit_price, ep, entry_idx, i, reason,
                        pos[k, POS_MAE], pos[k, POS_MFE]
                    )
                pos[k] = pos[n_pos - 1]; n_pos -= 1
            else:
                k += 1

    return (trade_returns[:n_trades], trade_sides[:n_trades],
            trade_entry_idx[:n_trades], trade_exit_idx[:n_trades],
            trade_reasons[:n_trades], trade_exit_prices[:n_trades],
            trade_mae[:n_trades], trade_mfe[:n_trades])


# ══════════════════════════════════════════════════════════════════
# 5. METRICS          
# ══════════════════════════════════════════════════════════════════

def compute_metrics_full(trade_returns, trade_sides, trade_entry_idx,
                         trade_exit_idx, trade_reasons, trade_exit_prices, trade_mae, trade_mfe,
                         bar_index,
                         highs=None, lows=None,
                         hold_minutes=0, bar_duration_min=5,
                         commission_pct=0.0, spread_pct=0.0, slippage_pct=0.0,
                         alpha=5, period_freq="ME"):
    if len(trade_returns) == 0:
        return None

    cost    = commission_pct * 2 + spread_pct + slippage_pct * 2
    ret_arr = trade_returns - cost

    entry_times = bar_index[trade_entry_idx]
    exit_times  = bar_index[trade_exit_idx]

    # ── MAE/MFE intra ────────────────────────────────────────────
    has_mae_mfe = (trade_mae is not None and trade_mfe is not None
                   and len(trade_mae) == len(trade_returns))
    mae_intra = trade_mae if has_mae_mfe else np.full(len(ret_arr), np.nan)
    mfe_intra = trade_mfe if has_mae_mfe else np.full(len(ret_arr), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        capture_ratio_intra = np.where(mfe_intra != 0, ret_arr / mfe_intra, np.nan)

    # ── MAE/MFE hold (post-sortie) ───────────────────────────────
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
                mae_hold[t] = 0.0
                mfe_hold[t] = 0.0
                continue

            h_win = highs[start:end]
            l_win = lows[start:end]

            if side == 1:
                mfe_hold[t] = (h_win.max() - ep_ref) / ep_ref
                mae_hold[t] = (l_win.min() - ep_ref) / ep_ref
            else:
                mfe_hold[t] = (ep_ref - l_win.min()) / ep_ref
                mae_hold[t] = (ep_ref - h_win.max()) / ep_ref

        with np.errstate(divide="ignore", invalid="ignore"):
            capture_ratio_hold = np.where(
                (ret_arr > 0) & (mfe_hold > 0),
                ret_arr / (ret_arr + mfe_hold),
                np.nan
            )
    else:
        mae_hold           = np.full(len(ret_arr), np.nan)
        mfe_hold           = np.full(len(ret_arr), np.nan)
        capture_ratio_hold = np.full(len(ret_arr), np.nan)

    trades_df = pd.DataFrame({
        "entry_time"          : entry_times,
        "exit_time"           : exit_times,
        "entry_idx"           : trade_entry_idx,
        "exit_idx"            : trade_exit_idx,
        "entry"               : bar_index[trade_entry_idx].map(lambda _: np.nan),  # placeholder
        "exit"                : trade_exit_prices,
        "side"                : trade_sides,
        "return"              : ret_arr,
        "reason"              : [REASON_LABELS.get(int(r), str(int(r))) for r in trade_reasons],
        "mae_intra"           : mae_intra,
        "mfe_intra"           : mfe_intra,
        "capture_ratio_intra" : capture_ratio_intra,
        "mae_hold"            : mae_hold,
        "mfe_hold"            : mfe_hold,
        "capture_ratio_hold"  : capture_ratio_hold,
    })

    trades_df["entry"] = np.nan

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
class NJITEngine:

    def __init__(self, pipeline, ticker, start, end, cfg,
                atr_period=14, MAX_TRADES=50_000, MAX_POS=500):
        self.cfg        = cfg # in case using the previous pandas engine  
        self.MAX_TRADES = MAX_TRADES
        self.MAX_POS    = MAX_POS

        df = pipeline.fetchdata(ticker, start, end,
                                timezone_shift=cfg.timezone_shift) 
        df = pipeline.compute_atr(df, atr_period)

        self.bar_index      = df.index
        self.opens          = df["Open"].to_numpy(dtype=np.float64)
        self.highs          = df["High"].to_numpy(dtype=np.float64)
        self.lows           = df["Low"].to_numpy(dtype=np.float64)
        self.closes         = df["Close"].to_numpy(dtype=np.float64)
        self.atrs           = df["ATR"].to_numpy(dtype=np.float64)
        self.minutes_of_day = (df.index.hour * 60 + df.index.minute).to_numpy(dtype=np.int16)
        self.day_index      = ((df.index - df.index[0]).days).to_numpy(dtype=np.int32)

        self.last_signal_df = None
        self.last_signal_col = "Signal"

        self._warmup()

# ══════════════════════════════════════════════════════════════════
# 6.1. df inspection pre and post engine execution
# ══════════════════════════════════════════════════════════════════

    def signal_generation_inspection(self, strategy_fn=None, signal_col="Signal", plot=False,
                        crypto=False, return_df_signals=True, **kwargs):
        df = pd.DataFrame({
            "Open": self.opens,
            "High": self.highs,
            "Low": self.lows,
            "Close": self.closes,
            "ATR": self.atrs,
        }, index=self.bar_index) # extractions and arrays compilation  

        if strategy_fn is not None: 
            df = strategy_fn(df.copy(), **kwargs)
        else:
            ema1 = ema_njit(self.closes, self.cfg.period_1) # fallback on default strategie 
            ema2 = ema_njit(self.closes, self.cfg.period_2)
            _, _, sig = signals_ema_vs_close_njit(
                self.opens, self.closes, self.cfg.period_1, self.cfg.period_2
            )
            print('Automatic Fallback on default ema_close strategy; to run your strategy please enter strategy_fn=my_strategy \
            as an argument of signal_df = njit_engine.signal_generation_inspection(...) call ')

            df["EMA1"] = ema1
            df["EMA2"] = ema2
            df[signal_col] = sig

        self.last_signal_df = df.copy()
        self.last_signal_col = signal_col

        if plot:
            self._plot_signal_df(df, signal_col=signal_col, crypto=crypto)

        if return_df_signals:
            return df
        return df[signal_col].to_numpy(dtype=np.int8)

    def _build_after_run_df(self, trades_df, full_df=False, window_before=200, window_after=50):
        if self.last_signal_df is not None:
            df = self.last_signal_df.copy()
        else:
            df = pd.DataFrame({
                "Open": self.opens,
                "High": self.highs,
                "Low": self.lows,
                "Close": self.closes,
                "ATR": self.atrs,
            }, index=self.bar_index)

        df["EntryTradeID"] = np.nan # linking entries to exit via an id attribution mecanism 
        df["ExitTradeID"] = np.nan

        long_id = 0
        short_id = 0

        trade_ids = []

        for _, tr in trades_df.iterrows():
            if tr["side"] == 1:
                long_id += 1
                tid = long_id
            else:
                short_id += 1
                tid = -short_id

            trade_ids.append(tid)

            eidx = int(tr["entry_idx"])
            xidx = int(tr["exit_idx"])

            df.iloc[eidx, df.columns.get_loc("EntryTradeID")] = tid
            df.iloc[xidx, df.columns.get_loc("ExitTradeID")] = tid

        trades_df = trades_df.copy()
        trades_df["trade_id"] = trade_ids

        if len(trades_df) == 0 or full_df:
            return df.copy(), trades_df

        first_idx = max(0, int(trades_df["entry_idx"].min()) - window_before)
        last_idx  = min(len(df), int(trades_df["exit_idx"].max()) + window_after + 1)

        return df.iloc[first_idx:last_idx].copy(), trades_df

    @staticmethod
    def _plot_signal_df(df, signal_col="Signal", title="Signal preparation", crypto=False):
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ))

        if signal_col not in df.columns:
            raise ValueError(f"'{signal_col}' not found in df columns: {list(df.columns)}")

        long_signals = df[df[signal_col] == 1]
        short_signals = df[df[signal_col] == -1]

        fig.add_trace(go.Scatter(
            x=long_signals.index,
            y=long_signals["Low"] * 0.999,
            mode="markers",
            marker=dict(symbol="triangle-up", size=8, color="lime"),
            name="Long signal"
        ))

        fig.add_trace(go.Scatter(
            x=short_signals.index,
            y=short_signals["High"] * 1.001,
            mode="markers",
            marker=dict(symbol="triangle-down", size=8, color="red"),
            name="Short signal"
        ))

        rangebreaks = [] if crypto else [dict(bounds=["sat", "mon"])]

        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            xaxis=dict(rangebreaks=rangebreaks),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0),
            height=750
        )

        fig.show()

    @staticmethod
    def _plot_backtest(df, trades, title="Backtest results", crypto=False):
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ))

        long_entries  = trades[trades["side"] == 1]
        short_entries = trades[trades["side"] == -1]

        fig.add_trace(go.Scatter(
            x=long_entries["entry_time"],
            y=long_entries["entry"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="lime"),
            name="Long entry"
        ))

        fig.add_trace(go.Scatter(
            x=short_entries["entry_time"],
            y=short_entries["entry"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="red"),
            name="Short entry"
        ))

        for _, r in trades.iterrows():
            color = "lime" if r["side"] == 1 else "red"

            fig.add_trace(go.Scatter(
                x=[r["entry_time"], r["exit_time"]],
                y=[r["entry"], r["exit"]],
                mode="lines",
                line=dict(color=color, width=1),
                opacity=0.5,
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=[r["exit_time"]],
                y=[r["exit"]],
                mode="markers",
                marker=dict(symbol="x", size=9, color=color),
                showlegend=False
            ))

        rangebreaks = [] if crypto else [dict(bounds=["sat", "mon"])]

        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            xaxis=dict(rangebreaks=rangebreaks),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0),
            height=700
        )

        fig.show()


    @staticmethod
    def _parse_session(s):
        if s is None: return -1
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    def _warmup(self):
        n   = min(500, len(self.opens))
        sig = np.zeros(n, dtype=np.int8)
        sig[10] = 1; sig[20] = -1
        backtest_njit(
            self.opens[:n], self.highs[:n], self.lows[:n],
            self.closes[:n], self.atrs[:n],
            sig, self.minutes_of_day[:n], self.day_index[:n],
            1, -1, -1, -1, -1, -1, -1, 0.0, 0.0,
            False, 0.001, 0.01, True,
            0.01, 0.005, 0, 2.0, 1.0, True,
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
            False, False, False,
            np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.float64),
            False, 1, 0.0, 0.0, 0, 0.0, 2.0,
            True, False, 0, 0, 0, 1000, 10, True
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
        exit_signals=None,
        signal_tags=None,
        use_exit_signal=None, exit_delay=None,
        be_trigger_pct=None, be_offset_pct=None, be_delay_bars=None,
        trailing_trigger_pct=None, runner_trailing_mult=None,
        multi_entry=None, reverse_mode=None,
        me_max=None, me_period=None, me_reset_mode=None,
        track_mae_mfe=None, hold_minutes=None, bar_duration_min=None,
        commission_pct=None, spread_pct=None, slippage_pct=None,
        alpha=None, period_freq=None,
        return_df_after=None, plot=None,crypto=None,
        full_df_after=None, window_before=None, window_after=None,
    ):

        cfg = self.cfg

        tp_pct = tp_pct if tp_pct is not None else cfg.tp_pct
        sl_pct = sl_pct if sl_pct is not None else cfg.sl_pct

        use_atr_sl_tp = use_atr_sl_tp if use_atr_sl_tp is not None else cfg.use_atr_sl_tp
        tp_atr_mult = tp_atr_mult if tp_atr_mult is not None else cfg.tp_atr_mult
        sl_atr_mult = sl_atr_mult if sl_atr_mult is not None else cfg.sl_atr_mult

        entry_delay = entry_delay if entry_delay is not None else cfg.entry_delay

        session_1 = session_1 if session_1 is not None else cfg.session_1
        session_2 = session_2 if session_2 is not None else cfg.session_2
        session_3 = session_3 if session_3 is not None else cfg.session_3

        max_gap_signal = max_gap_signal if max_gap_signal is not None else cfg.max_gap_signal
        max_gap_entry = max_gap_entry if max_gap_entry is not None else cfg.max_gap_entry

        candle_size_filter = candle_size_filter if candle_size_filter is not None else cfg.candle_size_filter
        min_size_pct = min_size_pct if min_size_pct is not None else cfg.min_size_pct
        max_size_pct = max_size_pct if max_size_pct is not None else cfg.max_size_pct
        prev_candle_direction = prev_candle_direction if prev_candle_direction is not None else cfg.prev_candle_direction

        allow_exit_on_entry_bar = (
            allow_exit_on_entry_bar
            if allow_exit_on_entry_bar is not None
            else cfg.allow_exit_on_entry_bar
        )

        use_ema1_tp = use_ema1_tp if use_ema1_tp is not None else cfg.use_ema1_tp
        use_ema2_tp = use_ema2_tp if use_ema2_tp is not None else cfg.use_ema2_tp
        use_ema_cross_tp = use_ema_cross_tp if use_ema_cross_tp is not None else cfg.use_ema_cross_tp

        use_exit_signal = use_exit_signal if use_exit_signal is not None else cfg.use_exit_signal
        exit_delay = exit_delay if exit_delay is not None else cfg.exit_delay

        be_trigger_pct = be_trigger_pct if be_trigger_pct is not None else cfg.be_trigger_pct
        be_offset_pct = be_offset_pct if be_offset_pct is not None else cfg.be_offset_pct
        be_delay_bars = be_delay_bars if be_delay_bars is not None else cfg.be_delay_bars

        trailing_trigger_pct = (
            trailing_trigger_pct if trailing_trigger_pct is not None else cfg.trailing_trigger_pct
        )
        runner_trailing_mult = (
            runner_trailing_mult if runner_trailing_mult is not None else cfg.runner_trailing_mult
        )

        multi_entry = multi_entry if multi_entry is not None else cfg.multi_entry
        reverse_mode = reverse_mode if reverse_mode is not None else cfg.reverse_mode

        track_mae_mfe = track_mae_mfe if track_mae_mfe is not None else cfg.track_mae_mfe
        hold_minutes = hold_minutes if hold_minutes is not None else cfg.hold_minutes
        bar_duration_min = bar_duration_min if bar_duration_min is not None else cfg.bar_duration_min
        commission_pct = commission_pct if commission_pct is not None else cfg.commission_pct
        spread_pct = spread_pct if spread_pct is not None else cfg.spread_pct
        slippage_pct = slippage_pct if slippage_pct is not None else cfg.slippage_pct
        alpha = alpha if alpha is not None else cfg.alpha
        period_freq = period_freq if period_freq is not None else cfg.period_freq

        return_df_after = return_df_after if return_df_after is not None else cfg.return_df_after
        plot = plot if plot is not None else cfg.plot
        crypto = crypto if crypto is not None else cfg.crypto
        full_df_after = full_df_after if full_df_after is not None else cfg.full_df_after
        window_before = window_before if window_before is not None else cfg.window_before
        window_after = window_after if window_after is not None else cfg.window_after

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

        me_max = me_max if me_max is not None else cfg.me_max
        me_period = me_period if me_period is not None else cfg.me_period
        me_reset_mode = me_reset_mode if me_reset_mode is not None else cfg.me_reset_mode

        rets, sides, entry_idx, exit_idx, reasons, exit_prices, mae, mfe = backtest_njit(
            self.opens, self.highs, self.lows, self.closes, self.atrs,
            signals, self.minutes_of_day, self.day_index,
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
            self.MAX_TRADES, self.MAX_POS, track_mae_mfe
        )

        metrics = compute_metrics_full(
            rets, sides, entry_idx, exit_idx, reasons, exit_prices, mae, mfe,
            self.bar_index,
            highs=self.highs, lows=self.lows,
            hold_minutes=hold_minutes, bar_duration_min=bar_duration_min,
            commission_pct=commission_pct, spread_pct=spread_pct,
            slippage_pct=slippage_pct, alpha=alpha, period_freq=period_freq
        )

        metrics["trades_df"]["entry"] = self.opens[entry_idx]
        metrics["trades_df"]["exit"] = exit_prices
        metrics["trades_df"]["entry_idx"] = entry_idx
        metrics["trades_df"]["exit_idx"] = exit_idx

        if return_df_after:
            df_after, trades_df_annot = self._build_after_run_df(
                metrics["trades_df"],
                full_df=full_df_after,
                window_before=window_before,
                window_after=window_after
            )
            metrics["trades_df"] = trades_df_annot
            metrics["df_after"] = df_after
        
        if plot:
            df_price = pd.DataFrame({
                "Open": self.opens,
                "High": self.highs,
                "Low": self.lows,
                "Close": self.closes,
            }, index=self.bar_index)
            self._plot_backtest(df_price, metrics["trades_df"], crypto=crypto)

        return rets, metrics

    def run_with_inspection(self, signals, signal_df=None, plot_before=False, plot_after=False,
                            signal_col="Signal", crypto=False, **run_kwargs):
        """
        Lance le backtest et permet de visualiser :
        - avant : les signaux / indicateurs
        - après : les trades
        """
        if signal_df is not None and plot_before:
            self._plot_signal_df(signal_df, signal_col=signal_col, crypto=crypto)

        rets, metrics = self.run(signals, **run_kwargs)

        trades_df = metrics["trades_df"].copy()

        if "entry" not in trades_df.columns:
            raise ValueError("trades_df must contain 'entry' and 'exit' columns for plotting.")

        if plot_after:
            df_price = pd.DataFrame({
                "Open": self.opens,
                "High": self.highs,
                "Low": self.lows,
                "Close": self.closes,
            }, index=self.bar_index)

            self._plot_backtest(df_price, trades_df, crypto=crypto)

        return rets, metrics


    def grid_search(self, signals,
                    tp_values, sl_values,
                    commission_pct=0.0, spread_pct=0.0, slippage_pct=0.0,
                    n_trades_per_year=None,
                    entry_delay=None, session_1=None, session_2=None, session_3=None,
                    use_session=False, max_gap_size=None,
                    candle_size_filter=None, min_size_pct=None,
                    max_size_pct=None, prev_candle_direction=None,
                    use_atr_sl_tp=0, tp_atr_mult=2.0, sl_atr_mult=1.0,
                    allow_exit_on_entry_bar=None, multi_entry=None,
                    me_max=0, me_period=0, me_reset_mode=0,
                    be_trigger_pct=None, be_offset_pct=None, be_delay_bars=None,
                    trailing_trigger_pct=None, runner_trailing_mult=None,
                    sort_by="sharpe"):
        cfg = self.cfg
        entry_delay   = entry_delay   if entry_delay   is not None else cfg.entry_delay
        max_gap_size  = max_gap_size  if max_gap_size  is not None else (cfg.max_gap_size or 0.0)
        candle_size_filter    = candle_size_filter    if candle_size_filter    is not None else cfg.Candle_Size_filter
        min_size_pct          = min_size_pct          if min_size_pct          is not None else cfg.min_size_pct
        max_size_pct          = max_size_pct          if max_size_pct          is not None else cfg.max_size_pct
        prev_candle_direction = prev_candle_direction if prev_candle_direction is not None else cfg.Previous_Candle_same_direction
        allow_exit_on_entry_bar = allow_exit_on_entry_bar if allow_exit_on_entry_bar is not None else cfg.allow_exit_on_entry_bar
        multi_entry   = multi_entry   if multi_entry   is not None else cfg.multi_entry
        be_trigger_pct= be_trigger_pct if be_trigger_pct is not None else (cfg.be_trigger_pct or 0.0)
        be_offset_pct = be_offset_pct  if be_offset_pct  is not None else cfg.be_offset_pct
        be_delay_bars = be_delay_bars  if be_delay_bars  is not None else cfg.be_delay_bars
        trailing_trigger_pct = trailing_trigger_pct if trailing_trigger_pct is not None else (cfg.trailing_trigger_pct or 0.0)
        runner_trailing_mult = runner_trailing_mult if runner_trailing_mult is not None else cfg.runner_trailing_mult

        s1_start = self._parse_session(session_1[0]) if session_1 else -1
        s1_end   = self._parse_session(session_1[1]) if session_1 else -1
        s2_start = self._parse_session(session_2[0]) if session_2 else -1
        s2_end   = self._parse_session(session_2[1]) if session_2 else -1
        s3_start = self._parse_session(session_3[0]) if session_3 else -1
        s3_end   = self._parse_session(session_3[1]) if session_3 else -1

        cost = commission_pct * 2 + spread_pct + slippage_pct * 2

        if n_trades_per_year is None:
            n_days  = (self.bar_index[-1] - self.bar_index[0]).days
            n_years = max(n_days / 365, 0.1)
            n_trades_per_year = float(int((signals != 0).sum())) / n_years

        results = grid_search_njit(
            self.opens, self.highs, self.lows, self.closes, self.atrs,
            signals, self.minutes_of_day,
            tp_values.astype(np.float64), sl_values.astype(np.float64),
            cost, n_trades_per_year,
            entry_delay, s1_start, s1_end, s2_start, s2_end, s3_start, s3_end,
            max_gap_size, candle_size_filter, min_size_pct, max_size_pct,
            prev_candle_direction, use_atr_sl_tp, tp_atr_mult, sl_atr_mult,
            allow_exit_on_entry_bar, multi_entry,
            be_trigger_pct, be_offset_pct, be_delay_bars,
            trailing_trigger_pct, runner_trailing_mult,
            me_max, me_period, me_reset_mode, self.MAX_TRADES, self.MAX_POS, False,
        )

        df = pd.DataFrame(
            results,
            columns=["tp", "sl", "sharpe", "win_rate", "mdd", "profit_factor", "cum_return"]
        )
        return df.sort_values(sort_by, ascending=False).reset_index(drop=True)
  
