from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


# ==========================================================
# Helpers retournés par les strats Python
# ==========================================================

def no_action() -> dict:
    return {"type": 0}

def switch_profile(profile_id: int) -> dict:
    return {"type": 1, "target_profile_id": int(profile_id)}

def overwrite_tp_sl(tp: float | None = None, sl: float | None = None) -> dict:
    return {"type": 2, "tp": float(tp) if tp is not None else -1.0,
                       "sl": float(sl) if sl is not None else -1.0}

def force_exit(reason: int = 11) -> dict:
    return {"type": 3, "reason": int(reason)}


# ==========================================================
# Contexte position
# ==========================================================

@dataclass(slots=True)
class PosCtx:
    side: float          # 1.0 = long, -1.0 = short
    entry_price: float
    tp: float
    sl: float
    bars_in_trade: int
    mae: float
    mfe: float
    be_active: bool
    runner_active: bool
    setup_id: int
    selected_score: float
    exit_profile_id: int
    exit_strategy_id: int


# ==========================================================
# Contexte bar courante
# ==========================================================

@dataclass(slots=True)
class BarCtx:
    i: int               # index absolu dans le dataset
    open: float
    high: float
    low: float
    close: float
    atr: float


# ==========================================================
# Contexte features
# ==========================================================

class FeatCtx:
    """
    Accès aux features par nom.
    
    Mode instant  : feat.ema_20        → float (valeur bar courante)
    Mode window   : feat.ema_20        → np.ndarray shape (window_bars,)
                    feat.ema_20[0]     → bar courante
                    feat.ema_20[-1]    → bar précédente
    """
    __slots__ = ("_data", "_names", "_window_mode")

    def __init__(self, data: np.ndarray, names: tuple[str, ...], window_mode: bool = False):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_window_mode", window_mode)

    def __getattr__(self, name: str):
        names = object.__getattribute__(self, "_names")
        data  = object.__getattribute__(self, "_data")
        window_mode = object.__getattribute__(self, "_window_mode")

        if name not in names:
            raise AttributeError(f"Feature '{name}' not found. Available: {names}")

        idx = names.index(name)

        if window_mode:
            # data shape : (window_bars, n_features)
            return data[:, idx]
        else:
            # data shape : (n_features,)
            return float(data[idx])

    def available(self) -> tuple[str, ...]:
        return object.__getattribute__(self, "_names")


# ==========================================================
# Contexte params
# ==========================================================

class ParamsCtx:
    """
    Accès aux params par nom : params.threshold → float
    
    Exemple déclaration dans ExitStrategySpec :
        params={"threshold_force": 70.0, "target_profile": 1}
    """
    __slots__ = ("_params",)

    def __init__(self, params: dict[str, Any]):
        object.__setattr__(self, "_params", params)

    def __getattr__(self, name: str):
        params = object.__getattribute__(self, "_params")
        if name not in params:
            raise AttributeError(f"Param '{name}' not found. Available: {list(params.keys())}")
        return params[name]

    def get(self, name: str, default: Any = None) -> Any:
        return object.__getattribute__(self, "_params").get(name, default)