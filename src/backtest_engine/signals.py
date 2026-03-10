# ══════════════════════════════════════════════════════════════════
# 1. Default Indicators
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
# 2. Default strategy
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

