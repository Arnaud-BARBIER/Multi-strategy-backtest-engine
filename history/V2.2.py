# Version 2.2, Numpy Optimised 
import pandas as pd 
import numpy as np
import seaborn as sns
from collections import namedtuple

"""
V2.2 – Procedural Backtest Engine 

Major structural upgrade over previous versions.

Key improvements:
- Position management redesigned: boolean state replaced by a dynamic list
  of position dictionaries storing full trade state (entry, side, TP/SL,
  MAE/MFE intratrade and hold during x candle, break-even, trailing).
- Trailing stop ATR, armed if price > trigger_Atr, else tp
- Strategy Candle gap proof, making stock trading compatible with the signal 
  generation.
- A fast mode, preventing MAE & MFE from being computed.

This version spans ~800 lines and represents the final procedural
architecture and trading features before the transition to an 
object-oriented engine (V3).

Known limitation:
- Isn't a object-oriented engine.
"""


def fetchdata(ticker, start, end):
    df=pd.read_csv(
        f'/Users/arnaudbarbier/Desktop/Quant reaserch/Metals/{ticker}.csv',
        header=None,
        names=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    df['Datetime']=pd.to_datetime(df['Datetime'])+pd.Timedelta(hours=1) 
    df=df.set_index('Datetime').sort_index() 
    df=df.loc[start:end]
    return df

def EMA_signal(df,period_1=50,period_2=100,  max_gap_size=None):
    d=df.copy()

    #EMA creation
    d['EMA1']=d['Close'].ewm(span=period_1,adjust=False).mean()
    d['EMA2']=d['Close'].ewm(span=period_2,adjust=False).mean()

    if max_gap_size is not None:
        gap_pct = abs(d['Open'] - d['Close'].shift(1)) / d['Close'].shift(1)
        gap_filter = gap_pct < max_gap_size
    else:
        gap_filter = pd.Series(True, index=d.index)

    # Signal Generation EMA1 
    d['Signal'] = np.where(
        (
            (d['EMA1'].shift(1) < d['Close'].shift(1)) &
            (d['EMA1'] > d['Close']) &
            gap_filter
        ),
        -1,
        np.where(
            (
                (d['EMA1'].shift(1) > d['Close'].shift(1)) &
                (d['EMA1'] < d['Close']) &
                gap_filter
            ),
            1,
            0
        )
    )

    # Signal EMA2
    d['Signal2']=0
    bull = (d['EMA2'].shift(1)>d['Close'].shift(1))&(d['EMA2']<d['Close'])&(gap_filter)
    bear = (d['EMA2'].shift(1)<d['Close'].shift(1))&(d['EMA2']>d['Close'])&(gap_filter)

    d.loc[bull, 'Signal2']=1
    d.loc[bear, 'Signal2']=-1

    # EMA Cross 
    d['EMA_CROSS']=0
    down_cross = (d['EMA1'].shift(1)>d['EMA2'].shift(1))& (d['EMA1']<d['EMA2'])&(gap_filter)
    upper_cross = (d['EMA1'].shift(1)<d['EMA2'].shift(1))& (d['EMA1']>d['EMA2'])&(gap_filter)
    d.loc[upper_cross, 'EMA_CROSS']=1
    d.loc[down_cross, 'EMA_CROSS']=-1

    # Entry Price 
    d['Entry_Price']=d['Open'].where(d['Signal'].shift(1)!=0)

    return d
#
def compute_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(period).mean()
    return df

def Tp_Sl_prices(side, entry_price, tp_pct=None, sl_pct=None,
                 use_atr=False, atr_value=None, tp_atr_mult=None, sl_atr_mult=None):

    if use_atr is True:
        # both sl and tp repriced by the ATR
        if atr_value is None:
            return None, None
        tp = entry_price + side * atr_value * tp_atr_mult
        sl = entry_price - side * atr_value * sl_atr_mult

    elif use_atr == 1:
        # TP repriced by ATR, SL pct based 
        if atr_value is None:
            return None, None
        tp = entry_price + side * atr_value * tp_atr_mult
        sl = entry_price * (1 - side * sl_pct)

    elif use_atr == -1:
        # SL repriced by ATR, TP pct based 
        if atr_value is None:
            return None, None
        tp = entry_price * (1 + side * tp_pct)
        sl = entry_price - side * atr_value * sl_atr_mult

    else:
        # Both pct based
        tp = entry_price * (1 + side * tp_pct)
        sl = entry_price * (1 - side * sl_pct)

    return tp, sl

def time_in_window_fast(ts, parsed_window):
    if parsed_window is None:
        return False
    start, end = parsed_window
    current = ts.time()
    if start <= end:
        return start <= current <= end
    else:
        return current >= start or current <= end

def entry_logic(i, min_size_pct, max_size_pct,
                Previous_Candle_same_direction,
                Candle_Size_filter, ts,
                tp_pct, sl_pct, bar, sig_prev, 
                prev_open,prev_close,atr_value,
                parsed_windows=None,
                use_atr_sl_tp=False,
                tp_atr_mult=None,
                sl_atr_mult=None,
                ):
    
    #-----Session/day/period count cap trade------#
    # 1 check previous signal (for an entry, sig_prev)
    # 2 check time filters
    # 3 if True → new position is allowed to be created
    # 4 else → return None
    #---------------------------------------------#
                  
    # Time filter
    windows = parsed_windows or [None, None, None]

    # if all w are non -> no filter
    if any(w is not None for w in windows):
        in_any_window = any(time_in_window_fast(ts, w) for w in windows if w is not None)
        if not in_any_window:
            return None

    if sig_prev == 0:
        return None

    body_pct = abs(prev_open - prev_close) / prev_close
    size_ok = min_size_pct < body_pct < max_size_pct

    direction_ok = (
        (sig_prev == 1 and prev_close > prev_open) or
        (sig_prev == -1 and prev_close < prev_open)
    )

    # Candle size Filtre 
    if Candle_Size_filter:
        if not (size_ok and (not Previous_Candle_same_direction or direction_ok)):
            return None

    # entrée exécutée si filtre OK ou filtre désactivé
    entry_price = bar.Open

    if use_atr_sl_tp and atr_value is not None and np.isnan(atr_value):
        return None

    tp, sl = Tp_Sl_prices(
        side=sig_prev,
        entry_price=entry_price,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        use_atr=use_atr_sl_tp,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
        atr_value=atr_value
)


    new_position_dict={
    "side": sig_prev,
    "entry_price": bar.Open,
    "entry_time": ts,
    "entry_index": i,
    "tp": tp,
    "sl": sl,
    "be_armed": False,
    "pending_be_sl": None,
    "be_arm_index": None,
    "be_active": False,
    "runner_active": False,
    "runner_armed":  False,   
    "runner_sl":     None,
    "mae":0.0,
    "mfe":0.0       
    }


    return new_position_dict
#
def parse_window(w):
    if w is None: return None
    s, e = w.split("-")
    return pd.to_datetime(s).time(), pd.to_datetime(e).time()
#
def apply_reverse(positions, sig_prev, bar_open, ts, reverse_mode, fast=False):
    if not reverse_mode or sig_prev == 0:
        return positions, []

    closed = []
    surviving = []
    for pos in positions:
        if pos["side"] == -sig_prev:
            if fast:
                closed.append(trade_analysis(
                    position=pos["side"],
                    exit_price=bar_open,
                    entry_price=pos["entry_price"],
                    entry_time=pos["entry_time"],
                    ts=ts,
                    exit_reason="REVERSE",
                    mae=None, mfe=None, mae_h=None, mfe_h=None
                ))
            else:
                closed.append(trade_analysis(
                    position=pos["side"],
                    exit_price=bar_open,
                    entry_price=pos["entry_price"],
                    entry_time=pos["entry_time"],
                    ts=ts,
                    exit_reason="REVERSE",
                    mae=pos.get("mae", 0.0),
                    mfe=pos.get("mfe", 0.0)
                ))
        else:
            surviving.append(pos)

    return surviving, closed

def update_be_logic(
    position,
    entry_price,
    base_sl,
    i,
    entry_index,
    bar,
    be_armed,
    pending_be_sl,
    be_arm_index,
    be_active,
    be_trigger_pct=None,
    be_offset_pct=0.0,
    be_delay_bars=0
):

    if be_trigger_pct is None or position == 0:
        return base_sl, be_armed, pending_be_sl, be_arm_index, be_active

    delay_ok = (i - entry_index) >= be_delay_bars

    # ===== LONG =====
    if position == 1 and delay_ok and not be_armed:
        trigger_price = entry_price * (1 + be_trigger_pct)

        if bar.High >= trigger_price:
            be_armed = True
            be_arm_index = i
            pending_be_sl = entry_price * (1 + be_offset_pct)

    # ===== SHORT =====
    if position == -1 and delay_ok and not be_armed:
        trigger_price = entry_price * (1 - be_trigger_pct)

        if bar.Low <= trigger_price:
            be_armed = True
            be_arm_index = i
            pending_be_sl = entry_price * (1 - be_offset_pct)

    # ===== Activation bougie suivante =====
    if be_armed and i > be_arm_index:

        if position == 1:
            base_sl = max(base_sl, pending_be_sl)

        elif position == -1:
            base_sl = min(base_sl, pending_be_sl)

        be_armed = False
        be_active = True
        pending_be_sl = None
        be_arm_index = None

    return base_sl, be_armed, pending_be_sl, be_arm_index, be_active

def exit_logic(
    pos, bar, i,
    ema1, ema2,
    allow_exit_on_entry_bar,
    exit_mode,
    EMA1_TP, EMA2_TP, EMA_CROSS_TP, EMA_SL,
):
    if not allow_exit_on_entry_bar and i == pos['entry_index']:
        return None

    side  = pos['side']
    tp    = pos['tp']
    sl    = pos['sl']
    o     = bar.Open
    h     = bar.High
    l     = bar.Low
    close = bar.Close

    # ── Active Runner ATR ─────────────────────────────────────────────
    if pos.get("runner_active") and pos["runner_sl"] is not None:
        runner_sl       = pos["runner_sl"]
        threshold_price = pos.get("runner_threshold")
        be_reason       = "BE" if pos.get("be_active", False) else "SL"

        under_threshold = threshold_price is not None and (
            (side == 1 and runner_sl <= threshold_price) or
            (side == -1 and runner_sl >= threshold_price)
        )

        if under_threshold:
            # SL fixe prend le relais
            if side == 1:
                if o <= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                elif l <= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
            else:
                if o >= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                elif h >= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
            return None  # TP bloqué

        # runner_sl valide → logique runner
        if side == 1:
            if o <= runner_sl:   return {"exit_price": o,         "exit_time": bar.name, "reason": "RUNNER_SL"}
            elif l <= runner_sl: return {"exit_price": runner_sl, "exit_time": bar.name, "reason": "RUNNER_SL"}
            elif o <= sl:        return {"exit_price": o,         "exit_time": bar.name, "reason": be_reason}
            elif l <= sl:        return {"exit_price": sl,        "exit_time": bar.name, "reason": be_reason}
        else:
            if o >= runner_sl:   return {"exit_price": o,         "exit_time": bar.name, "reason": "RUNNER_SL"}
            elif h >= runner_sl: return {"exit_price": runner_sl, "exit_time": bar.name, "reason": "RUNNER_SL"}
            elif o >= sl:        return {"exit_price": o,         "exit_time": bar.name, "reason": be_reason}
            elif h >= sl:        return {"exit_price": sl,        "exit_time": bar.name, "reason": be_reason}

        return None  # TP bloqué
            
    # =========================
    # FIXED MODE
    # =========================
    if exit_mode == "fixed":
        be_reason = "BE" if pos.get("be_active", False) else "SL"

        if side == 1:
            if o <= sl:
                return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
            elif l <= sl:
                return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
            elif h >= tp:                                                      # ← TP enfin accessible
                return {"exit_price": tp, "exit_time": bar.name, "reason": "TP"}

        elif side == -1:
            if o >= sl:
                return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
            elif h >= sl:
                return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
            elif l <= tp:                                                      # ← TP enfin accessible
                return {"exit_price": tp, "exit_time": bar.name, "reason": "TP"}

    # =========================
    # EMA MODE
    # =========================
    elif exit_mode == "ema":

        entry_price = pos["entry_price"]

        # =========================
        # 1️⃣ SL FIXE PRIORITAIRE (intrabar)
        # =========================

        be_reason = "BE" if pos.get("be_active", False) else "SL"

        if side == 1:
            if o <= sl:
                return {"exit_price": o, "exit_time": bar.name, "reason": be_reason}
            if l <= sl:
                return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}

        elif side == -1:
            if o >= sl:
                return {"exit_price": o, "exit_time": bar.name, "reason": be_reason}
            if h >= sl:
                return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}

        # =========================
        # 2️⃣ EMA TP UNIQUEMENT SI PROFIT
        # =========================

        trade_positive = (
            (side == 1 and close > entry_price) or
            (side == -1 and close < entry_price)
        )

        if not trade_positive:
            return None   # on bloque toute sortie EMA en perte

        # ----- EMA1 TP -----
        if EMA1_TP:
            if side == 1 and close < ema1:
                return {"exit_price": close, "exit_time": bar.name, "reason": "EMA1_TP"}
            if side == -1 and close > ema1:
                return {"exit_price": close, "exit_time": bar.name, "reason": "EMA1_TP"}

        # ----- EMA2 TP -----
        if EMA2_TP:
            if side == 1 and close < ema2:
                return {"exit_price": close, "exit_time": bar.name, "reason": "EMA2_TP"}
            if side == -1 and close > ema2:
                return {"exit_price": close, "exit_time": bar.name, "reason": "EMA2_TP"}

        # ----- EMA CROSS TP -----
        if EMA_CROSS_TP:
            if side == 1 and ema1 < ema2:
                return {"exit_price": close, "exit_time": bar.name, "reason": "EMA_CROSS_TP"}
            if side == -1 and ema1 > ema2:
                return {"exit_price": close, "exit_time": bar.name, "reason": "EMA_CROSS_TP"}


    return None
#
def update_trailing_runner(pos, bar, atr_value, trailing_trigger_pct, trailing_mult):
    side = pos["side"]
    ep   = pos["entry_price"]

    # 1. Armement — threshold en % du prix
    if not pos["runner_active"] and not pos.get("runner_armed", False):
        threshold = ep * (1 + side * trailing_trigger_pct)
        if (side == 1 and bar.High >= threshold) or \
           (side == -1 and bar.Low <= threshold):
            pos["runner_armed"]     = True
            pos["runner_threshold"] = threshold  # ← stocké pour contraindre runner_sl

    # Activation bougie suivante
    if pos.get("runner_armed") and not pos["runner_active"]:
        pos["runner_active"] = True
        pos["runner_armed"]  = False
        initial_sl = bar.Close - side * atr_value * trailing_mult
        if side == 1:
            pos["runner_sl"] = max(initial_sl, ep)
        else:
            pos["runner_sl"] = min(initial_sl, ep)
        return pos  # ← on ne recalcule pas le trailing sur cette même barre

    # Trailing ATR dynamique — seulement barres suivantes
    if pos["runner_active"]:
        new_sl = bar.Close - side * atr_value * trailing_mult
        if side == 1:
            pos["runner_sl"] = max(pos["runner_sl"], new_sl, ep)
        else:
            pos["runner_sl"] = min(pos["runner_sl"], new_sl, ep)

    return pos
#   
def trade_analysis(position, exit_price, entry_price, entry_time, ts, exit_reason,
                   mae=None, mfe=None, mae_h=None, mfe_h=None):

    ret = position * (exit_price - entry_price) / entry_price

    trade = {
        "entry_time": entry_time,
        "exit_time": ts,
        "side": position,
        "entry": entry_price,
        "exit": exit_price,
        "return": ret,
        "reason": exit_reason,
    }

    # --- Ajouter MAE / MFE uniquement si présents ---
    if (mae is not None) and (mfe is not None):
        trade["mae"] = mae
        trade["mfe"] = mfe
        trade["capture_ratio"] = ret / mfe if mfe > 0 else None

    # --- Ajouter Hold metrics uniquement si présents ---
    if (mae_h is not None) and (mfe_h is not None):
        trade["hold_mfe"] = mfe_h
        trade["hold_mae"] = mae_h
        trade["capture_ratio_hold"] = (
            round(ret / mfe_h, 3) if mfe_h > 0 else None
        )

    return trade
#
def get_active_session(ts, windows):
    for idx, w in enumerate(windows):
        if w is not None and time_in_window_fast(ts, w):
            return idx
    return None

def filtre_AND_Exit_StratV2(
    ticker,
    start,
    end,
    period_1=50,
    period_2=100,  
    max_gap_size=None,
    period_atr=14,
    Candle_Size_filter=True,
    Previous_Candle_same_direction=True,
    min_size_pct=0.001,
    max_size_pct=0.02,
    tp_pct=0.05,
    sl_pct=0.02,
    EMA1_TP=False,
    EMA2_TP=False,
    EMA_CROSS_TP=False,
    EMA_SL=False,
    MaxEntries4Periods=True,
    ME_X=2,
    ME_Period_Y=8,
    ME_reset_mode=None,
    allow_exit_on_entry_bar=True,
    multi_entry=True,
    reverse_mode=False,
    be_trigger_pct=None,
    be_offset_pct=0.0,
    be_delay_bars=0,
    time_window_1=None, 
    time_window_2=None, 
    time_window_3=None,
    use_atr_sl_tp=False,
    tp_atr_mult=2.0,
    sl_atr_mult=1.0,
    trailing_trigger_pct=None,   # None = désactivé, ex: 0.003 = +0.3%
    runner_trailing_mult=2.0,
    observation_hours=None,
    timeframe_minutes=5,
    fast=True,
):
    
    df=fetchdata(ticker=ticker, start=start, end=end)
    df=EMA_signal(df,period_1=period_1,period_2=period_2,max_gap_size=max_gap_size)
    df=compute_atr(df, period=period_atr)
    positions=[]
    use_observation = observation_hours is not None
    pending_observations = []
    observation_bars = int(observation_hours * 60 / timeframe_minutes) if use_observation else 0
    trades=[]
    recent_entries=[] 
    last_session_id = None
    last_day = None
    opens   = df['Open'].to_numpy()
    highs   = df['High'].to_numpy()
    lows    = df['Low'].to_numpy()
    closes  = df['Close'].to_numpy()
    signals = df['Signal'].to_numpy()
    ema1s   = df['EMA1'].to_numpy()
    ema2s   = df['EMA2'].to_numpy()
    atrs = df['ATR'].to_numpy() if ('ATR' in df.columns) else None
    index   = df.index
    Bar = namedtuple('Bar', ['Open', 'High', 'Low', 'Close', 'name'])
    
    parsed_windows = [
    parse_window(time_window_1),
    parse_window(time_window_2),
    parse_window(time_window_3)
]
    
    # be_armed = False    On ne fait pas ça car = sinon chaque trade sera concerné 
    # pending_be_sl = None

    exit_mode = "ema" if sum([EMA1_TP, EMA2_TP, EMA_CROSS_TP]) == 1 else "fixed"
    for i in range(1, len(df)):
        ts  = index[i]          # ← index déjà extrait
        bar = Bar(Open=opens[i], High=highs[i], Low=lows[i], Close=closes[i], name=ts)

        sig_prev = signals[i-1]

        # ======================
        # REVERSE MODE
        # ======================

        positions, closed_trades = apply_reverse(
            positions=positions,
            sig_prev=sig_prev,
            bar_open=bar.Open,
            ts=ts,
            reverse_mode=reverse_mode,
            fast=fast
        )
        trades.extend(closed_trades)

        # ======================
        # ENTRY LOGIC
        # ======================
    
        entry_event = entry_logic(
            i=i,
            min_size_pct=min_size_pct,
            max_size_pct=max_size_pct,
            Previous_Candle_same_direction=Previous_Candle_same_direction,
            Candle_Size_filter=Candle_Size_filter,
            ts=ts,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            bar=bar,
            sig_prev=sig_prev,
            use_atr_sl_tp=use_atr_sl_tp,
            tp_atr_mult=tp_atr_mult,
            sl_atr_mult=sl_atr_mult,
            prev_open=opens[i-1],                                   
            prev_close=closes[i-1],                             
            atr_value=atrs[i] if use_atr_sl_tp else None, 
            parsed_windows=parsed_windows
        )

        current_day = ts.date()

        # --- Daily reset ---
        if ME_reset_mode == "day":
            if last_day is None:
                last_day = current_day
            elif current_day != last_day:
                recent_entries = []
                last_day = current_day

        # --- Session reset ---
        elif ME_reset_mode == "session":
            current_session = get_active_session(ts, parsed_windows) 

            if last_session_id is None:
                last_session_id = current_session
            elif current_session != last_session_id:
                recent_entries = []
                last_session_id = current_session

        # Cleaning of previous entries
        if MaxEntries4Periods:
            cutoff = i - ME_Period_Y
            j = 0
            while j < len(recent_entries) and recent_entries[j] < cutoff:
                j += 1
            recent_entries = recent_entries[j:]

            # Count verifictation
            if len(recent_entries) >= ME_X:
                entry_event = None
            
        if entry_event is not None:
                if multi_entry:
                    positions.append(entry_event)
                elif len(positions) == 0:
                    positions.append(entry_event)
                if MaxEntries4Periods:
                    recent_entries.append(i)

                    
        # ======================
        # EXIT LOGIC
        # ======================
        surviving = []
        for pos in positions:
            if not fast:
                ep = pos["entry_price"]
                side = pos["side"]
                favorable = (bar.High - ep) / ep * side
                adverse   = (bar.Low  - ep) / ep * side
                pos["mfe"] = max(pos["mfe"], favorable)
                pos["mae"] = min(pos["mae"], adverse)
            
            if be_trigger_pct is not None:
            # BE 
                pos["sl"], \
                pos["be_armed"], \
                pos["pending_be_sl"], \
                pos["be_arm_index"], \
                pos["be_active"] = update_be_logic(
                    position=pos["side"],
                    entry_price=pos["entry_price"],
                    base_sl=pos["sl"],
                    i=i,
                    entry_index=pos["entry_index"],
                    bar=bar,
                    be_armed=pos["be_armed"],
                    pending_be_sl=pos["pending_be_sl"],
                    be_arm_index=pos["be_arm_index"],
                    be_active=pos["be_active"],
                    be_trigger_pct=be_trigger_pct,
                    be_offset_pct=be_offset_pct,
                    be_delay_bars=be_delay_bars
                )

            if trailing_trigger_pct is not None and atrs is not None:
                pos = update_trailing_runner(
                    pos=pos,
                    bar=bar,
                    atr_value=atrs[i],
                    trailing_trigger_pct=trailing_trigger_pct,
                    trailing_mult=runner_trailing_mult
                )

            exit_event = exit_logic(
                            pos,
                            bar,
                            i,
                            ema1s[i],
                            ema2s[i],
                            allow_exit_on_entry_bar,
                            exit_mode,
                            EMA1_TP,
                            EMA2_TP,
                            EMA_CROSS_TP,
                            EMA_SL
            )

            if exit_event is not None:
                if fast:
                    trades.append(trade_analysis(
                        position=pos["side"],
                        exit_price=exit_event["exit_price"],
                        entry_price=pos["entry_price"],
                        entry_time=pos["entry_time"],
                        ts=exit_event["exit_time"],
                        exit_reason=exit_event["reason"],
                    ))
                else:
                    trades.append(trade_analysis(
                        position=pos["side"],
                        exit_price=exit_event["exit_price"],
                        entry_price=pos["entry_price"],
                        entry_time=pos["entry_time"],
                        ts=exit_event["exit_time"],
                        exit_reason=exit_event["reason"],
                        mae=pos["mae"],
                        mfe=pos["mfe"],
                        mae_h=pos["mae"] if use_observation else None,
                        mfe_h=pos["mfe"] if use_observation else None
                    ))
                    if use_observation:
                        pending_observations.append({
                            "idx": len(trades) - 1,
                            "ep": pos["entry_price"],
                            "side": pos["side"],
                            "end_i": i + observation_bars
                        })

            else:
                surviving.append(pos)
        # ← un niveau de moins
        positions = surviving

        # ── POST-TRADE OBSERVATION ───────────────────────
        if (not fast) and use_observation and pending_observations:
            still_pending = []
            for obs in pending_observations:
                favorable = (bar.High - obs["ep"]) / obs["ep"] * obs["side"]
                adverse   = (bar.Low  - obs["ep"]) / obs["ep"] * obs["side"]
                t = trades[obs["idx"]]
                t["hold_mfe"] = max(t["hold_mfe"], favorable)
                t["hold_mae"] = min(t["hold_mae"], adverse)
                if i < obs["end_i"]:
                    still_pending.append(obs)

            pending_observations = still_pending


    return pd.DataFrame(trades)


trades = filtre_AND_Exit_StratV2(
    'XAGUSD_M5',#ticker
    '2021-01-01',#start
    '2026-01-30',#end
    period_1=50,#EMA period 1 
    period_2=100,#EMA Period 2 
    max_gap_size=None, # ema gap size consideration (for intraday gap)
    period_atr=14,# Atr period
    Candle_Size_filter=True,
    Previous_Candle_same_direction=False,
    min_size_pct=0.0003,
    max_size_pct=0.01,
    tp_pct=0.01,
    sl_pct=0.004,
    EMA1_TP=False,
    EMA2_TP=False,
    EMA_CROSS_TP=False,
    EMA_SL=False, # si False alors sl fix, si True alors sorti a la meme ema que tp
    MaxEntries4Periods=True,
    ME_X=5,
    ME_Period_Y=50,
    ME_reset_mode='session', # reset par 'day' ou 'session'
    allow_exit_on_entry_bar = True,
    reverse_mode=False,
    multi_entry=True,
    be_trigger_pct=0.003,#0.0015
    be_offset_pct=0.0005,
    be_delay_bars=5,
    time_window_1='08:00-12:00', 
    time_window_2='01:00-06:00', #'01:00-06:00'
    time_window_3='13:30-18:30',
    use_atr_sl_tp=None, #2-> atr TP / fixed sl #-1 ->atr SL / fixed tp
    tp_atr_mult=10.0,
    sl_atr_mult=5.0,
    trailing_trigger_pct=0.01,  #0.003 # None = désactivé, ex: 0.003 = +0.3%
    runner_trailing_mult=4,
    observation_hours=24,
    timeframe_minutes=5,
    fast=False,
    )
print(trades['return'].sum())
trades 
