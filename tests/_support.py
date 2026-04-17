from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def prepare_import_path() -> None:
    """Make the local package importable without requiring installation."""
    mpl_cache = Path(tempfile.gettempdir()) / "backtest_engine_mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def make_price_df(n_bars: int = 40, start: str = "2024-01-01") -> pd.DataFrame:
    """Small deterministic OHLCV sample for smoke tests."""
    index = pd.date_range(start, periods=n_bars, freq="5min")
    close = np.linspace(100.0, 101.0, n_bars)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    volume = np.full(n_bars, 1_000.0)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )
