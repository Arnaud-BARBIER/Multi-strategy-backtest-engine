from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from _support import prepare_import_path


prepare_import_path()

import Backtest_Git as bt  # noqa: E402


class TestDataPipeline(unittest.TestCase):
    def test_fetchdata_loads_standard_ohlcv_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)
            csv_path = base_path / "XAUUSD_M5.csv"

            df = pd.DataFrame(
                [
                    ["2024-01-01 00:00:00", 100.0, 100.2, 99.8, 100.1, 1_000],
                    ["2024-01-01 00:05:00", 100.1, 100.3, 99.9, 100.2, 1_100],
                    ["2024-01-01 00:10:00", 100.2, 100.4, 100.0, 100.3, 1_200],
                ]
            )
            df.to_csv(csv_path, header=False, index=False)

            pipeline = bt.DataPipeline(str(base_path))
            loaded = pipeline.fetchdata(
                ticker="XAUUSD_M5",
                start="2024-01-01 00:00:00",
                end="2024-01-01 00:10:00",
            )

        self.assertEqual(list(loaded.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(len(loaded), 3)
        self.assertAlmostEqual(float(loaded["Close"].iloc[-1]), 100.3)


if __name__ == "__main__":
    unittest.main()
