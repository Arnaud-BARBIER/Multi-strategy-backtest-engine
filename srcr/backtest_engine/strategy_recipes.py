from __future__ import annotations

import numpy as np
import pandas as pd

from .Exit_system import ExitProfileSpec
from .partial_config import DistributionFn, PartialConfig
from .position_rules import OnRR


def _wilder_rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def low_vol_bb_rsi_reversion_df(
    df: pd.DataFrame,
    price_col: str = "Close",
    bb_length: int = 20,
    bb_std: float = 2.0,
    rsi_length: int = 14,
    rsi_low: float = 35.0,
    rsi_high: float = 65.0,
    rsi_memory: int = 5,
    bandwidth_lookback: int = 120,
    bandwidth_quantile: float = 0.35,
) -> pd.DataFrame:
    """
    Low-volatility mean-reversion helper.

    Logic:
    - use Bollinger BandWidth as a low-volatility filter
    - wait for price to stretch outside the band
    - trigger only when price closes back inside the band
    - confirm the move with a recent RSI extreme
    """
    if price_col not in df.columns:
        raise KeyError(f"Missing required price column: {price_col}")
    if bb_length < 2:
        raise ValueError("bb_length must be >= 2")
    if rsi_length < 2:
        raise ValueError("rsi_length must be >= 2")
    if rsi_memory < 1:
        raise ValueError("rsi_memory must be >= 1")
    if bandwidth_lookback < 2:
        raise ValueError("bandwidth_lookback must be >= 2")
    if not 0.0 < bandwidth_quantile < 1.0:
        raise ValueError("bandwidth_quantile must be in (0, 1)")

    out = df.copy()
    close = out[price_col].astype("float64")

    mid = close.rolling(bb_length, min_periods=bb_length).mean()
    std = close.rolling(bb_length, min_periods=bb_length).std(ddof=0)
    upper = mid + bb_std * std
    lower = mid - bb_std * std

    bandwidth = ((upper - lower) / mid.replace(0.0, np.nan)).abs()
    bw_threshold = bandwidth.rolling(
        bandwidth_lookback,
        min_periods=bandwidth_lookback,
    ).quantile(bandwidth_quantile)

    rsi = _wilder_rsi(close, rsi_length)

    low_vol = bandwidth <= bw_threshold
    rsi_long_ready = rsi.rolling(rsi_memory, min_periods=1).min() <= rsi_low
    rsi_short_ready = rsi.rolling(rsi_memory, min_periods=1).max() >= rsi_high

    # Re-entry avoids chasing the initial expansion outside the bands.
    long_reentry = (close.shift(1) < lower.shift(1)) & (close >= lower)
    short_reentry = (close.shift(1) > upper.shift(1)) & (close <= upper)

    long_active = (low_vol & rsi_long_ready & long_reentry).fillna(False).astype("int8")
    short_active = (low_vol & rsi_short_ready & short_reentry).fillna(False).astype("int8")

    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bb_width"] = bandwidth
    out["bb_width_threshold"] = bw_threshold
    out["low_vol"] = low_vol.fillna(False).astype("int8")
    out["rsi"] = rsi
    out["long_active"] = long_active
    out["short_active"] = short_active
    out["Signal"] = np.where(
        long_active == 1,
        1,
        np.where(short_active == 1, -1, 0),
    ).astype("int8")
    return out


def low_vol_bb_rsi_reversion_setup(
    df: pd.DataFrame,
    setup_id: int = 0,
    score: float = 1.0,
    price_col: str = "Close",
    bb_length: int = 20,
    bb_std: float = 2.0,
    rsi_length: int = 14,
    rsi_low: float = 35.0,
    rsi_high: float = 65.0,
    rsi_memory: int = 5,
    bandwidth_lookback: int = 120,
    bandwidth_quantile: float = 0.35,
) -> pd.DataFrame:
    signal_df = low_vol_bb_rsi_reversion_df(
        df=df,
        price_col=price_col,
        bb_length=bb_length,
        bb_std=bb_std,
        rsi_length=rsi_length,
        rsi_low=rsi_low,
        rsi_high=rsi_high,
        rsi_memory=rsi_memory,
        bandwidth_lookback=bandwidth_lookback,
        bandwidth_quantile=bandwidth_quantile,
    )

    long_active = signal_df["long_active"].astype("int8")
    short_active = signal_df["short_active"].astype("int8")

    return pd.DataFrame(
        {
            "long_score": np.where(long_active == 1, score, 0.0).astype("float64"),
            "short_score": np.where(short_active == 1, score, 0.0).astype("float64"),
            "long_active": long_active,
            "short_active": short_active,
            "setup_id": np.full(len(signal_df), setup_id, dtype=np.int32),
        },
        index=signal_df.index,
    )


def make_low_vol_bb_reversion_profile(
    name: str = "low_vol_bb_reversion",
    tp_pct: float = 0.0030,
    sl_pct: float = 0.0012,
    partial_rr: float = 1.0,
    partial_fraction: float = 0.5,
    max_holding_bars: int = 24,
) -> ExitProfileSpec:
    """
    Default exit profile for a short-horizon mean-reversion trade:
    - take one partial around 1R
    - move stop to break-even after the first partial
    - let the remainder aim for the full fixed TP
    - cut stale trades quickly
    """
    if not 0.0 < partial_fraction <= 1.0:
        raise ValueError("partial_fraction must be in (0, 1]")
    if partial_rr <= 0.0:
        raise ValueError("partial_rr must be > 0")

    return ExitProfileSpec(
        name=name,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        max_holding_bars=max_holding_bars,
        partial_config=PartialConfig(
            n_levels=1,
            spacing=OnRR(partial_rr),
            sizing=DistributionFn(mode="equal", start=partial_fraction),
            move_sl_to_be_after_first=True,
        ),
    )
