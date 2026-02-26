from dataclasses import dataclass
from typing import Optional, Union

@dataclass(slots=True)
class BacktestConfig:

    # --- Signal Generation and strategy Choice ---
    strategy: str = "ema_cross"

    # --- Data / indicators ---
    period_1: int = 50
    period_2: int = 100
    max_gap_size: Optional[float] = None
    period_atr: int = 14
    
    # --- Filters ---
    Candle_Size_filter: bool = True
    Previous_Candle_same_direction: bool = True
    min_size_pct: float = 0.001
    max_size_pct: float = 0.02
    Exit_filter_EMA1: int = 50
    Exit_filter_EMA2: int = 100

    # --- TP/SL ---
    tp_pct: float = 0.05
    sl_pct: float = 0.02
    use_atr_sl_tp: int = 0        # 0=fixed, 1=ATR TP only, -1=ATR SL only, 2=both
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0

    # --- Exit mode toggles ---
    EMA1_TP: bool = False
    EMA2_TP: bool = False
    EMA_CROSS_TP: bool = False
    EMA_SL: bool = False

    # --- Entries cap ---
    MaxEntries4Periods: bool = True
    ME_X: int = 2
    ME_Period_Y: int = 8
    ME_reset_mode: Optional[str] = None  # "day" / "session" / None

    # --- Engine behavior ---
    allow_exit_on_entry_bar: bool = True
    multi_entry: bool = True
    reverse_mode: bool = False

    # --- BE / runner trailing ---
    be_trigger_pct: Optional[float] = None
    be_offset_pct: float = 0.0
    be_delay_bars: int = 0
    trailing_trigger_pct: Optional[float] = None
    runner_trailing_mult: float = 2.0

    # --- Time windows ---
    time_window_1: Optional[str] = None
    time_window_2: Optional[str] = None
    time_window_3: Optional[str] = None

    # --- Observation for MAE-MFE hold ---
    observation_hours: Optional[float] = None
    timeframe_minutes: int = 5

    # --- Fast preventing MAE MFE calculations ---
    fast: bool = True

    def __post_init__(self):
        if self.period_1 <= 0 or self.period_2 <= 0:
            raise ValueError("EMA periods must be positive")
        if self.period_1 >= self.period_2:
            raise ValueError("period_1 must be less than period_2")
        if self.tp_pct <= self.sl_pct:
            raise ValueError("tp_pct must be greater than sl_pct")
        if self.ME_X <= 0:
            raise ValueError("ME_X must be positive")
        if self.use_atr_sl_tp not in (0, 1, -1, 2):
            raise ValueError("use_atr_sl_tp must be 0, 1, -1 or 2")
