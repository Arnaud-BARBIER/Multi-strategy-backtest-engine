# run_example.py
from backtest_engine.config import BacktestConfig
from backtest_engine.pipeline import DataPipeline
from backtest_engine.engine import BacktestEngine

def main():
    # 1. Config
    cfg = BacktestConfig(
        strategy="ema_cross", 
        period_1=50,
        period_2=100,
        tp_pct=0.01,
        sl_pct=0.004,
        fast=True,
    )

    # 2. Pipeline
    pipeline = DataPipeline("/path/to/your/data")

    # 3. Run
    engine = BacktestEngine.from_ticker(
        pipeline=pipeline,
        ticker="XAUUSD_M5",
        start="2021-01-01",
        end="2024-01-01",
        cfg=cfg,
    )
    trades = engine.run()

    # 4. Metrics
    print(f"Trades     : {len(trades)}")
    print(f"Mean return: {trades['return'].mean():.4f}")
    print(f"Total return: {trades['return'].sum():.4f}")
    print(f"Win rate   : {(trades['return'] > 0).mean():.2%}")

if __name__ == "__main__":
    main()
