class DataPipeline:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def fetchdata(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        df = pd.read_csv(
            f"{self.base_path}/{ticker}.csv",
            header=None,
            names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
        )
        df["Datetime"] = pd.to_datetime(df["Datetime"]) + pd.Timedelta(hours=1)
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

    def build(self, ticker: str, start: str, end: str, cfg: BacktestConfig) -> pd.DataFrame:
        df = self.fetchdata(ticker, start, end)
        df = self.compute_atr(df, cfg.period_atr)
        df = Strategy_Signal.apply(df, cfg)   # ← dispatch automatique
        df = Strategy_Signal.apply_exitfilter_indicators(df, cfg)
        return df
