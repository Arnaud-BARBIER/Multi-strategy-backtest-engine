# from a .ipynb 
%pip install git+https://github.com/Arnaud-BARBIER/Multi-strategy-backtest-engine.git
from backtest_engine import BacktestConfig, DataPipeline, NJITEngine


pipeline = DataPipeline("/Users/arnaudbarbier/Desktop/Quant reaserch/Metals")


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

njit_engine = NJITEngine(
    pipeline,
    "XAUUSD_M5",
    "2023-01-02",
    "2026-02-16",
    cfg,
    MAX_TRADES=50_000,
    MAX_POS=600,
)

# Built-in default EMA signal generation
signals = njit_engine.signals_ema(
    span1=cfg.period_1,
    span2=cfg.period_2,
    mode="close_vs_ema",
)

rets, metrics = njit_engine.run(signals)

print(metrics)
print("Exit reasons:")
print(metrics["trades_df"]["reason"].value_counts())
