from .config import BacktestConfig
from .engine import NJITEngine
from .signals import signals_ema

__all__ = [
    "BacktestConfig",
    "NJITEngine",
    "signals_ema",
]
__version__ = "4.0.0"
