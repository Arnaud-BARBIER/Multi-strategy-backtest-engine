from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

import numpy as np

from _support import make_price_df, prepare_import_path


prepare_import_path()

import Backtest_Git as bt  # noqa: E402


def _simple_signal_df(df):
    out = df.copy()
    out["Signal"] = 0
    out.iloc[3, out.columns.get_loc("Signal")] = 1
    out.iloc[7, out.columns.get_loc("Signal")] = -1
    return out


class TestSetupPipeline(unittest.TestCase):
    def test_prepare_signal_inputs_and_execution_context(self) -> None:
        njit_module = importlib.import_module("Backtest_Git.NJITEngine")

        with patch.object(njit_module.NJITEngine, "_warmup", lambda self: None):
            engine = bt.NJITEngine(main_df=make_price_df(20), cfg=bt.BacktestConfig())

            wrapped = bt.NJITEngine.wrap_signal_strategy(_simple_signal_df)
            setup_specs = [
                bt.SetupSpec(
                    fn=wrapped,
                    params={"setup_id": 0, "score": 1.0},
                    name="demo_setup",
                )
            ]

            regime = np.zeros(len(engine.opens), dtype=np.int32)
            regime_policy = bt.RegimePolicy(
                n_regimes=1,
                score_multiplier={
                    0: {"demo_setup": {"long": 1.0, "short": 1.0}},
                },
            )

            prep = engine.prepare_signal_inputs(
                setup_specs=setup_specs,
                decision_cfg=bt.DecisionConfig(min_score=0.1),
                regime=regime,
                regime_policy=regime_policy,
            )

            exit_profiles = [
                bt.ExitProfileSpec(name="default_profile", tp_pct=0.01, sl_pct=0.01)
            ]
            execution_context = bt.build_execution_context(
                cfg=bt.BacktestConfig(),
                exit_profile_specs=exit_profiles,
                setup_exit_binding={
                    0: {
                        "exit_profile_id": 0,
                        "exit_strategy_id": -1,
                    }
                },
                strategy_profile_binding={},
                n_setups=1,
                exit_strategy_specs=[],
                n_strategies=0,
            )

        self.assertEqual(int(prep.signals[3]), 1)
        self.assertEqual(int(prep.signals[7]), -1)
        self.assertEqual(int(prep.selected_setup_id[3]), 0)
        self.assertAlmostEqual(float(prep.selected_score[3]), 1.0)
        self.assertEqual(execution_context.profile_rt_matrix.shape[0], 1)
        self.assertEqual(execution_context.setup_to_exit_profile.tolist(), [0])
        self.assertEqual(execution_context.setup_to_exit_strategy.tolist(), [-1])


if __name__ == "__main__":
    unittest.main()
