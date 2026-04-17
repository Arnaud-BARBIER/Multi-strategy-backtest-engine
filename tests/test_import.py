from __future__ import annotations

import unittest

from _support import prepare_import_path


prepare_import_path()

import Backtest_Git as bt  # noqa: E402


class TestPublicImports(unittest.TestCase):
    def test_core_research_api_is_exposed(self) -> None:
        expected_names = [
            "BacktestConfig",
            "DataPipeline",
            "Data",
            "Feature",
            "NJITEngine",
            "DecisionConfig",
            "SetupSpec",
            "RegimePolicy",
            "ExitProfileSpec",
            "build_execution_context",
            "build_event_log",
        ]

        for name in expected_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(bt, name), f"Missing public export: {name}")


if __name__ == "__main__":
    unittest.main()
