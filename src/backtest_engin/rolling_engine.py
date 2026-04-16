# Backtest_Framework/rolling_engine.py
from __future__ import annotations
import numpy as np
from typing import Callable


def rolling_apply(
    fn: Callable[[np.ndarray], float],
    arr: np.ndarray,
    lookback: int,
    min_periods: int | None = None,
) -> np.ndarray:
    """
    Applique fn sur une fenêtre glissante de taille lookback.
    
    Parameters
    ----------
    fn       : fonction qui prend un array 1D de taille lookback → float
    arr      : array source 1D
    lookback : taille de la fenêtre
    min_periods : nb minimum de barres valides avant de calculer
                  None = lookback (fenêtre complète requise)
    
    Returns
    -------
    np.ndarray de même longueur que arr, NaN avant min_periods
    
    Example
    -------
    result = rolling_apply(np.mean, closes, lookback=20)
    result = rolling_apply(lambda w: w[-1] / w[0] - 1, closes, lookback=5)
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    mp = lookback if min_periods is None else min_periods

    for i in range(mp - 1, n):
        start = max(0, i - lookback + 1)
        window = arr[start:i + 1]
        out[i] = fn(window)

    return out


def rolling_apply_2d(
    fn: Callable[[np.ndarray], float],
    arrays: list[np.ndarray],
    lookback: int,
    min_periods: int | None = None,
) -> np.ndarray:
    """
    Comme rolling_apply mais fn reçoit une matrice (lookback, n_arrays).
    Utile pour des indicateurs qui combinent plusieurs séries.
    
    Example
    -------
    # Corrélation rolling entre deux séries
    def rolling_corr(w):  # w shape (lookback, 2)
        return np.corrcoef(w[:, 0], w[:, 1])[0, 1]
    
    result = rolling_apply_2d(rolling_corr, [closes_a, closes_b], lookback=20)
    """
    n = len(arrays[0])
    out = np.full(n, np.nan, dtype=np.float64)
    mp = lookback if min_periods is None else min_periods
    matrix = np.column_stack(arrays)

    for i in range(mp - 1, n):
        start = max(0, i - lookback + 1)
        window = matrix[start:i + 1]
        out[i] = fn(window)

    return out


class RollingIndicator:
    """
    Décorateur qui transforme une fonction window → float
    en un indicateur compatible FeatureSpec.
    
    Usage
    -----
    @RollingIndicator(lookback=14)
    def my_rsi_variant(window: np.ndarray) -> float:
        gains = window[window > 0].mean() or 0
        losses = abs(window[window < 0].mean()) or 1e-10
        return 100 - 100 / (1 + gains / losses)

    # Puis dans FeatureSpec
    FeatureSpec(name="my_rsi", fn=my_rsi_variant, params={"lookback": 14})
    
    # Ou directement
    result = my_rsi_variant(opens=o, highs=h, lows=l, closes=c, lookback=14)
    """

    def __init__(self, lookback: int, source: str = "closes", min_periods: int | None = None):
        """
        Parameters
        ----------
        lookback    : taille de la fenêtre
        source      : quelle série passer à fn — "closes", "opens", "highs", "lows"
        min_periods : nb minimum de barres avant de calculer (défaut = lookback)
        """
        self.lookback    = lookback
        self.source      = source
        self.min_periods = min_periods

    def __call__(self, fn: Callable) -> Callable:
        lookback    = self.lookback
        source      = self.source
        min_periods = self.min_periods

        def wrapper(
            opens=None, highs=None, lows=None, closes=None,
            volumes=None, lookback=lookback, **kwargs
        ):
            source_map = {
                "closes":  closes,
                "opens":   opens,
                "highs":   highs,
                "lows":    lows,
                "volumes": volumes,
            }
            arr = source_map.get(source)
            if arr is None:
                raise ValueError(f"RollingIndicator: source='{source}' is None")
            arr = np.asarray(arr, dtype=np.float64)
            return rolling_apply(fn, arr, lookback=lookback, min_periods=min_periods)

        wrapper.__name__ = fn.__name__
        wrapper.__doc__  = fn.__doc__
        return wrapper