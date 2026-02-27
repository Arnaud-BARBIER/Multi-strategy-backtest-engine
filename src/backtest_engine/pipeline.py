import pandas as pd
import numpy as np
from .config import BacktestConfig
from .signals import Strategy_Signal

class DataPipeline:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def fetchdata(self, ticker: str, start: str, end: str, timezone_shift=1) -> pd.DataFrame:
        df = pd.read_csv(
            f"{self.base_path}/{ticker}.csv",
            header=None,
            names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
        )
        df["Datetime"] = pd.to_datetime(df["Datetime"]) + pd.Timedelta(hours=timezone_shift)
        df = df.set_index("Datetime").sort_index()
        return df.loc[start:end]

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)

        df = df.copy()
        df["ATR"] = tr.rolling(period).mean()
        return df

    @staticmethod
    def apply_exitfilter_indicators(df, cfg):
        if any([cfg.EMA1_TP, cfg.EMA2_TP, cfg.EMA_CROSS_TP]):
            df["EMA1_exit"] = df["Close"].ewm(span=cfg.Exit_filter_EMA1, adjust=False).mean()
            df["EMA2_exit"] = df["Close"].ewm(span=cfg.Exit_filter_EMA2, adjust=False).mean()
        return df


    def build(self, ticker, start, end, cfg):
        df = self.fetchdata(ticker, start, end, timezone_shift=cfg.timezone_shift)
        df = self.compute_atr(df, cfg.period_atr)
        df = Strategy_Signal.apply(df, cfg)
        df = self.apply_exitfilter_indicators(df, cfg)  # ← self, no Strategy_Signal
        return df
