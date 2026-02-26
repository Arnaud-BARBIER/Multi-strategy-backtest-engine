
class Strategy_Signal:
        @staticmethod
        def apply(df, cfg):
            if cfg.strategy == "ema_cross":
                return Strategy_Signal.ema_cross(df, cfg.period_1, cfg.period_2, cfg.max_gap_size)
            else:
                raise ValueError(f"Unknown strategy: {cfg.strategy}")
            
        @staticmethod
        def ema_cross(df: pd.DataFrame, period_1=50, period_2=100, max_gap_size=None) -> pd.DataFrame:
            d = df.copy()
            d["EMA1"] = d["Close"].ewm(span=period_1, adjust=False).mean()
            d["EMA2"] = d["Close"].ewm(span=period_2, adjust=False).mean()

            if max_gap_size is not None:
                gap_pct = (d["Open"] - d["Close"].shift(1)).abs() / d["Close"].shift(1)
                gap_filter = gap_pct < max_gap_size
            else:
                gap_filter = pd.Series(True, index=d.index)

            d["Signal"] = np.where(
                (d["EMA1"].shift(1) < d["Close"].shift(1)) & (d["EMA1"] > d["Close"]) & gap_filter,
                -1,
                np.where(
                    (d["EMA1"].shift(1) > d["Close"].shift(1)) & (d["EMA1"] < d["Close"]) & gap_filter,
                    1,
                    0,
                ),
            )

            d["Signal2"] = 0
            bull = (d["EMA2"].shift(1) > d["Close"].shift(1)) & (d["EMA2"] < d["Close"]) & gap_filter
            bear = (d["EMA2"].shift(1) < d["Close"].shift(1)) & (d["EMA2"] > d["Close"]) & gap_filter
            d.loc[bull, "Signal2"] = 1
            d.loc[bear, "Signal2"] = -1

            d["EMA_CROSS"] = 0
            down_cross = (d["EMA1"].shift(1) > d["EMA2"].shift(1)) & (d["EMA1"] < d["EMA2"]) & gap_filter
            up_cross = (d["EMA1"].shift(1) < d["EMA2"].shift(1)) & (d["EMA1"] > d["EMA2"]) & gap_filter
            d.loc[up_cross, "EMA_CROSS"] = 1
            d.loc[down_cross, "EMA_CROSS"] = -1

            d["Entry_Price"] = d["Open"].where(d["Signal"].shift(1) != 0)
            return d
 
