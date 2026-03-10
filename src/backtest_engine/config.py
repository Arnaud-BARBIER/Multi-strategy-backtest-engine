from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(slots=True)
class BacktestConfig:
    # Data / preprocessing
    timezone_shift: int = 0
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

    # Entry cap logic
    me_max: int = 0
    me_period: int = 0
    me_reset_mode: int = 0

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


    def __post_init__(self):

        # --- Basic parameter checks ---
        if self.period_1 <= 0:
            raise ValueError("period_1 must be > 0")

        if self.period_2 <= 0:
            raise ValueError("period_2 must be > 0")

        if self.atr_period <= 0:
            raise ValueError("atr_period must be > 0")

        if self.entry_delay <= 0:
            raise ValueError("Putting your entry delay below 1 implies a look ahead bias")

        # --- Gap filters ---
        if self.max_gap_signal < 0:
            raise ValueError("max_gap_signal must be >= 0")

        if self.max_gap_entry < 0:
            raise ValueError("max_gap_entry must be >= 0")

        # --- Candle filter consistency ---
        if self.min_size_pct < 0:
            raise ValueError("min_size_pct must be >= 0")

        if self.max_size_pct <= 0:
            raise ValueError("max_size_pct must be > 0")

        if self.min_size_pct > self.max_size_pct:
            raise ValueError("min_size_pct cannot exceed max_size_pct")

        # --- TP / SL logic ---
        if self.tp_pct < 0:
            raise ValueError("tp_pct must be >= 0")

        if self.sl_pct < 0:
            raise ValueError("sl_pct must be >= 0")

        if self.use_atr_sl_tp not in (-1, 0, 1, 2):
            raise ValueError("use_atr_sl_tp must be one of {-1, 0, 1, 2}")

        if self.tp_atr_mult < 0:
            raise ValueError("tp_atr_mult must be >= 0")

        if self.sl_atr_mult < 0:
            raise ValueError("sl_atr_mult must be >= 0")

        # --- Entry cap logic ---
        if self.me_max < 0:
            raise ValueError("me_max must be >= 0")

        if self.me_period < 0:
            raise ValueError("me_period must be >= 0")

        if self.me_reset_mode not in (0, 1, 2, 3, 4, 5):
            raise ValueError("me_reset_mode must be in {0,1,2,3,4,5}")

        # --- Break-even ---
        if self.be_trigger_pct < 0:
            raise ValueError("be_trigger_pct must be >= 0")

        if self.be_offset_pct < 0:
            raise ValueError("be_offset_pct must be >= 0")

        if self.be_delay_bars < 0:
            raise ValueError("be_delay_bars must be >= 0")

        # --- Runner trailing ---
        if self.trailing_trigger_pct < 0:
            raise ValueError("trailing_trigger_pct must be >= 0")

        if self.runner_trailing_mult < 0:
            raise ValueError("runner_trailing_mult must be >= 0")

        # --- Metrics ---
        if self.commission_pct < 0:
            raise ValueError("commission_pct must be >= 0")

        if self.spread_pct < 0:
            raise ValueError("spread_pct must be >= 0")

        if self.slippage_pct < 0:
            raise ValueError("slippage_pct must be >= 0")

        if not (0 < self.alpha < 100):
            raise ValueError("alpha must be between 0 and 100 (percentile for VaR/CVaR)")

        if self.bar_duration_min <= 0:
            raise ValueError("bar_duration_min for observed MAE MFE hold must be > 0")

        if self.window_before < 0:
            raise ValueError("window_before must be >= 0")

        if self.window_after < 0:
            raise ValueError("window_after must be >= 0")
