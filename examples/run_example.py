# from a .ipynb 
%pip install git+https://github.com/Arnaud-BARBIER/Multi-strategy-backtest-engine.git
from backtest_engine import BacktestConfig, DataPipeline, BacktestEngine #,Strategy_Signal if you want to use a build in strategy

# Plug in your own strategy and add as many parameters as you need ! 
# The engine only reads the 'Signal' column (1 / -1 / 0) from the returned df.
def My_strategy(df, rsi_period=14, oversold=20, overbought=80):
    d = df.copy()
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(rsi_period).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    d["Signal"] = 0
    d.loc[rsi < oversold,  "Signal"] = 1
    d.loc[rsi > overbought, "Signal"] = -1
    return d


pipeline = DataPipeline("Your/path/")
cfg = BacktestConfig(tp_pct=0.01, sl_pct=0.004,timezone_shift=1) # add or delete hours from datetime if your csv is not at the right time zone
engine = BacktestEngine.from_df(
    pipeline, "XAUUSD_M5", "2021-01-01", "2026-01-01", cfg,
    strategy_fn=My_strategy, #<- omit this if using a built-in strategy (e.g. strategy='ema_cross' in cfg)
    rsi_period=14,
    oversold=20,
    overbought=80,
)
trades = engine.run()
trades 
