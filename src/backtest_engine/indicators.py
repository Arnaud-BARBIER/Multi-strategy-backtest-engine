import numpy as np
from numba import njit

@njit(cache=True)
def ema_njit(closes, span):
    n      = closes.shape[0]
    ema    = np.empty(n, dtype=np.float64)
    alpha  = 2.0 / (span + 1.0)
    ema[0] = closes[0]
    for i in range(1, n):
        ema[i] = alpha * closes[i] + (1.0 - alpha) * ema[i - 1]
    return ema
def ema_feature(opens, highs, lows, closes, volumes=None, span=20):
    closes = np.asarray(closes, dtype=np.float64)
    return ema_njit(closes, int(span))

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

@njit(cache=True)
def _rsi_wilder_core(closes: np.ndarray, period: int) -> np.ndarray:
    n = closes.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan

    if period <= 0 or n <= period:
        return out

    gains = 0.0
    losses = 0.0

    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        if delta > 0.0:
            gains += delta
        else:
            losses -= delta

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss == 0.0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        delta = closes[i] - closes[i - 1]
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0

        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out
def rsi_feature(opens, highs, lows, closes, volumes=None, period=14):
    closes = np.asarray(closes, dtype=np.float64)
    return _rsi_wilder_core(closes, int(period))

@njit(cache=True)
def consecutive_candle_signal_strict(opens: np.ndarray,
                                     closes: np.ndarray,
                                     streak_len: int) -> np.ndarray:
    n = closes.shape[0]
    signals = np.zeros(n, dtype=np.int8)

    if streak_len <= 0:
        return signals

    for i in range(streak_len - 1, n):
        long_ok = True
        short_ok = True

        for j in range(i - streak_len + 1, i + 1):
            if closes[j] <= opens[j]:
                long_ok = False
            if closes[j] >= opens[j]:
                short_ok = False

            if j > i - streak_len + 1:
                if closes[j] <= closes[j - 1]:
                    long_ok = False
                if closes[j] >= closes[j - 1]:
                    short_ok = False

        if long_ok:
            signals[i] = 1
        elif short_ok:
            signals[i] = -1

    return signals
