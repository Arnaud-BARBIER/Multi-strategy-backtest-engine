import pandas as pd


class DataPipeline:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def fetchdata(self, ticker: str, start: str, end: str, timezone_shift: int = 0) -> pd.DataFrame:
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

        out = df.copy()
        out["ATR"] = tr.rolling(period).mean()
        return out
