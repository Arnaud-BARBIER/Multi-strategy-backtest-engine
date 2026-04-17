from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from _support import make_price_df, prepare_import_path


prepare_import_path()

import Backtest_Git as bt  # noqa: E402


def _fake_backtest_njit(
    opens,
    highs,
    lows,
    closes,
    atrs,
    signals,
    selected_setup_id,
    selected_score,
    features,
    *args,
):
    return (
        np.array([0.01], dtype=np.float64),
        np.array([1], dtype=np.int8),
        np.array([5], dtype=np.int32),
        np.array([8], dtype=np.int32),
        np.array(["TP"], dtype=object),
        np.array([float(closes[8])], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.02], dtype=np.float64),
        np.array([0], dtype=np.int32),
        np.array([1.0], dtype=np.float64),
        np.array([-1], dtype=np.int32),
        np.array([-1], dtype=np.int32),
        np.array([-1], dtype=np.int32),
        np.array([1.0], dtype=np.float64),
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([1.0], dtype=np.float64),
        np.array([float(opens[5])], dtype=np.float64),
        np.array([3], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
    )


def _fake_compute_metrics_full(
    rets,
    sides,
    entry_idx,
    exit_idx,
    reasons,
    exit_prices,
    mae,
    mfe,
    bar_index,
    **kwargs,
):
    return {
        "n_trades": int(len(rets)),
        "win_rate": 1.0,
        "sharpe": 1.23,
        "trades_df": pd.DataFrame(
            {
                "entry_time": bar_index[entry_idx],
                "exit_time": bar_index[exit_idx],
                "side": sides,
                "return": rets,
                "reason": reasons,
            }
        ),
    }


class TestBasicRun(unittest.TestCase):
    def test_run_returns_metrics_and_annotated_dataframe(self) -> None:
        njit_module = importlib.import_module("Backtest_Git.NJITEngine")

        cfg = bt.BacktestConfig(
            multi_setup_mode=False,
            tp_pct=0.002,
            sl_pct=0.002,
            plot=False,
            plot_results=False,
        )
        df = make_price_df()

        with patch.object(njit_module.NJITEngine, "_warmup", lambda self: None), patch.object(
            njit_module, "backtest_njit", side_effect=_fake_backtest_njit
        ), patch.object(
            njit_module, "compute_metrics_full", side_effect=_fake_compute_metrics_full
        ):
            engine = bt.NJITEngine(main_df=df, cfg=cfg, MAX_TRADES=2_000, MAX_POS=50)
            signals = np.zeros(len(df), dtype=np.int8)
            signals[5] = 1

            rets, metrics = engine.run(
                signals=signals,
                cfg=cfg,
                multi_setup_mode=False,
                return_df_after=True,
            )

        self.assertEqual(rets.shape, (1,))
        self.assertIn("trades_df", metrics)
        self.assertIn("df_after", metrics)
        self.assertEqual(metrics["n_trades"], 1)
        self.assertAlmostEqual(metrics["win_rate"], 1.0)
        self.assertIn("EntryTradeID", metrics["df_after"].columns)
        self.assertIn("ExitTradeID", metrics["df_after"].columns)
        self.assertEqual(int(metrics["trades_df"]["trade_id"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
