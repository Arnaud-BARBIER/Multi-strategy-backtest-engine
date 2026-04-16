from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence, Any

import numpy as np
import pandas as pd


# ==========================================================
# Helpers
# ==========================================================


def _ensure_indexed_like(context_df: pd.DataFrame, price_index: pd.Index) -> pd.DataFrame:
    if not context_df.index.equals(price_index):
        raise ValueError("context_df.index must exactly match the engine/bar index")
    return context_df



def _require_trade_cols(trades_df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in trades_df.columns]
    if missing:
        raise ValueError(f"trades_df is missing required columns: {missing}")



def _safe_iloc(df: pd.DataFrame, start: int, end_exclusive: int) -> pd.DataFrame:
    start = max(0, int(start))
    end_exclusive = min(len(df), int(end_exclusive))
    if start >= end_exclusive:
        return df.iloc[0:0].copy()
    return df.iloc[start:end_exclusive].copy()


# ==========================================================
# Segment + Context objects exposed to the user
# ==========================================================


@dataclass(slots=True)
class PriceSegment:
    df: pd.DataFrame
    side: int
    reference_price: float

    @property
    def len(self) -> int:
        return len(self.df)

    @property
    def open(self) -> np.ndarray:
        return self.df["Open"].to_numpy(dtype=np.float64)

    @property
    def high(self) -> np.ndarray:
        return self.df["High"].to_numpy(dtype=np.float64)

    @property
    def low(self) -> np.ndarray:
        return self.df["Low"].to_numpy(dtype=np.float64)

    @property
    def close(self) -> np.ndarray:
        return self.df["Close"].to_numpy(dtype=np.float64)

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    def values(self, col: str) -> np.ndarray:
        return self.df[col].to_numpy()

    def max_high(self) -> float:
        return float(self.df["High"].max()) if self.len else np.nan

    def min_low(self) -> float:
        return float(self.df["Low"].min()) if self.len else np.nan

    def adverse_excursion_pct(self) -> float:
        if self.len == 0:
            return np.nan
        if self.side == 1:
            return float((self.reference_price - self.min_low()) / self.reference_price)
        return float((self.max_high() - self.reference_price) / self.reference_price)

    def favorable_excursion_pct(self) -> float:
        if self.len == 0:
            return np.nan
        if self.side == 1:
            return float((self.max_high() - self.reference_price) / self.reference_price)
        return float((self.reference_price - self.min_low()) / self.reference_price)

    def first_bar_opposite_color(self) -> float:
        if self.len == 0:
            return np.nan
        o = float(self.open[0])
        c = float(self.close[0])
        if self.side == 1:
            return float(c < o)
        return float(c > o)

    def first_bar_same_color(self) -> float:
        if self.len == 0:
            return np.nan
        o = float(self.open[0])
        c = float(self.close[0])
        if self.side == 1:
            return float(c > o)
        return float(c < o)


@dataclass(slots=True)
class TradeAnalysisContext:
    trade: pd.Series
    trades_df: pd.DataFrame
    price_df: pd.DataFrame
    context_df: pd.DataFrame | None = None

    @property
    def entry_idx(self) -> int:
        return int(self.trade["entry_idx"])

    @property
    def exit_idx(self) -> int:
        return int(self.trade["exit_idx"])

    @property
    def side(self) -> int:
        return int(self.trade["side"])

    @property
    def entry_price(self) -> float:
        return float(self.trade["entry"])

    @property
    def exit_price(self) -> float:
        return float(self.trade["exit"])

    @property
    def trade_return(self) -> float:
        return float(self.trade["return"])

    @property
    def is_winner(self) -> bool:
        return self.trade_return > 0

    @property
    def is_loser(self) -> bool:
        return self.trade_return < 0

    def before_entry(self, n_bars: int) -> PriceSegment:
        seg = _safe_iloc(self.price_df, self.entry_idx - n_bars, self.entry_idx)
        return PriceSegment(seg, self.side, self.entry_price)

    def after_entry(self, n_bars: int, include_entry_bar: bool = True) -> PriceSegment:
        start = self.entry_idx if include_entry_bar else self.entry_idx + 1
        seg = _safe_iloc(self.price_df, start, start + n_bars)
        return PriceSegment(seg, self.side, self.entry_price)

    def during_trade(self, include_exit_bar: bool = True) -> PriceSegment:
        end_excl = self.exit_idx + 1 if include_exit_bar else self.exit_idx
        seg = _safe_iloc(self.price_df, self.entry_idx, end_excl)
        return PriceSegment(seg, self.side, self.entry_price)

    def after_exit(self, n_bars: int, include_exit_bar: bool = False) -> PriceSegment:
        start = self.exit_idx if include_exit_bar else self.exit_idx + 1
        seg = _safe_iloc(self.price_df, start, start + n_bars)
        return PriceSegment(seg, self.side, self.exit_price)

    def entry_context(self) -> pd.Series | None:
        if self.context_df is None:
            return None
        return self.context_df.iloc[self.entry_idx]

    def exit_context(self) -> pd.Series | None:
        if self.context_df is None:
            return None
        return self.context_df.iloc[self.exit_idx]

    def context_window_before_entry(self, n_bars: int) -> pd.DataFrame | None:
        if self.context_df is None:
            return None
        return _safe_iloc(self.context_df, self.entry_idx - n_bars, self.entry_idx)

    def context_window_after_entry(self, n_bars: int, include_entry_bar: bool = True) -> pd.DataFrame | None:
        if self.context_df is None:
            return None
        start = self.entry_idx if include_entry_bar else self.entry_idx + 1
        return _safe_iloc(self.context_df, start, start + n_bars)

    def context_during_trade(self, include_exit_bar: bool = True) -> pd.DataFrame | None:
        if self.context_df is None:
            return None
        end_excl = self.exit_idx + 1 if include_exit_bar else self.exit_idx
        return _safe_iloc(self.context_df, self.entry_idx, end_excl)


# ==========================================================
# Declarative path specs
# ==========================================================


@dataclass(slots=True)
class PathAggSpec:
    col: str
    aggs: Sequence[str]
    prefix: str = "ctx_path"


@dataclass(slots=True)
class TradeFeatureSpec:
    name: str
    fn: Callable[[TradeAnalysisContext], Any]
    expand_dict: bool = True


# ==========================================================
# Main engine
# ==========================================================


@dataclass(slots=True)
class TradeContextEngine:
    trades_df: pd.DataFrame
    price_df: pd.DataFrame
    context_df: pd.DataFrame | None = None
    default_price_cols: tuple[str, ...] = ("Open", "High", "Low", "Close")

    def __post_init__(self) -> None:
        _require_trade_cols(self.trades_df, ["entry_idx", "exit_idx", "entry", "exit", "side", "return"])
        missing_price = [c for c in self.default_price_cols if c not in self.price_df.columns]
        if missing_price:
            raise ValueError(f"price_df is missing required OHLC columns: {missing_price}")
        if self.context_df is not None:
            _ensure_indexed_like(self.context_df, self.price_df.index)

    # ------------------------------------------------------
    # Basic joins on entry / exit bars
    # ------------------------------------------------------
    def attach_entry_context(
        self,
        cols: Sequence[str] | None = None,
        prefix: str = "ctx_entry_",
        inplace: bool = False,
    ) -> pd.DataFrame:
        out = self.trades_df if inplace else self.trades_df.copy()
        if self.context_df is None:
            return out

        src = self.context_df if cols is None else self.context_df[list(cols)]
        joined = src.iloc[out["entry_idx"].to_numpy()].reset_index(drop=True).add_prefix(prefix)
        out = pd.concat([out.reset_index(drop=True), joined], axis=1)
        if inplace:
            self.trades_df = out
        return out

    def attach_exit_context(
        self,
        cols: Sequence[str] | None = None,
        prefix: str = "ctx_exit_",
        inplace: bool = False,
    ) -> pd.DataFrame:
        out = self.trades_df if inplace else self.trades_df.copy()
        if self.context_df is None:
            return out

        src = self.context_df if cols is None else self.context_df[list(cols)]
        joined = src.iloc[out["exit_idx"].to_numpy()].reset_index(drop=True).add_prefix(prefix)
        out = pd.concat([out.reset_index(drop=True), joined], axis=1)
        if inplace:
            self.trades_df = out
        return out

    # ------------------------------------------------------
    # Context aggregations during trade
    # ------------------------------------------------------
    def attach_path_aggregations(
        self,
        specs: Sequence[PathAggSpec],
        include_exit_bar: bool = True,
        inplace: bool = False,
    ) -> pd.DataFrame:
        out = self.trades_df if inplace else self.trades_df.copy()
        if self.context_df is None:
            return out

        rows: list[dict[str, Any]] = []
        for _, tr in out.iterrows():
            start = int(tr["entry_idx"])
            end_excl = int(tr["exit_idx"]) + 1 if include_exit_bar else int(tr["exit_idx"])
            win = _safe_iloc(self.context_df, start, end_excl)
            row: dict[str, Any] = {}
            for spec in specs:
                if spec.col not in win.columns:
                    raise ValueError(f"Column '{spec.col}' not found in context_df")
                s = win[spec.col]
                for agg in spec.aggs:
                    key = f"{spec.prefix}_{spec.col}_{agg}"
                    if len(s) == 0:
                        row[key] = np.nan
                    elif agg == "mean":
                        row[key] = float(s.mean())
                    elif agg == "max":
                        row[key] = float(s.max())
                    elif agg == "min":
                        row[key] = float(s.min())
                    elif agg == "std":
                        row[key] = float(s.std())
                    elif agg == "first":
                        row[key] = s.iloc[0]
                    elif agg == "last":
                        row[key] = s.iloc[-1]
                    elif agg == "median":
                        row[key] = float(s.median())
                    else:
                        raise ValueError(f"Unsupported aggregation '{agg}'")
            rows.append(row)

        agg_df = pd.DataFrame(rows)
        out = pd.concat([out.reset_index(drop=True), agg_df], axis=1)
        if inplace:
            self.trades_df = out
        return out

    # ------------------------------------------------------
    # User-defined features: only business logic
    # ------------------------------------------------------
    def compute_trade_features(
        self,
        specs: Sequence[TradeFeatureSpec],
        inplace: bool = False,
    ) -> pd.DataFrame:
        out = self.trades_df if inplace else self.trades_df.copy()
        produced_rows: list[dict[str, Any]] = []

        for _, tr in out.iterrows():
            ctx = TradeAnalysisContext(
                trade=tr,
                trades_df=out,
                price_df=self.price_df,
                context_df=self.context_df,
            )
            row: dict[str, Any] = {}
            for spec in specs:
                res = spec.fn(ctx)
                if isinstance(res, Mapping) and spec.expand_dict:
                    for k, v in res.items():
                        row[str(k)] = v
                else:
                    row[spec.name] = res
            produced_rows.append(row)

        feat_df = pd.DataFrame(produced_rows)
        out = pd.concat([out.reset_index(drop=True), feat_df], axis=1)
        if inplace:
            self.trades_df = out
        return out

    # ------------------------------------------------------
    # Convenience views for a single trade
    # ------------------------------------------------------
    def get_trade_context(self, trade_row_or_idx: int | pd.Series) -> TradeAnalysisContext:
        tr = self.trades_df.iloc[int(trade_row_or_idx)] if not isinstance(trade_row_or_idx, pd.Series) else trade_row_or_idx
        return TradeAnalysisContext(
            trade=tr,
            trades_df=self.trades_df,
            price_df=self.price_df,
            context_df=self.context_df,
        )


# ==========================================================
# Optional defaults builder
# ==========================================================


def build_default_context_df(
    price_df: pd.DataFrame,
    extra_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=price_df.index)
    out["bar_return"] = price_df["Close"].pct_change()
    out["bar_range_pct"] = (price_df["High"] - price_df["Low"]) / price_df["Close"]
    out["bar_body_pct"] = (price_df["Close"] - price_df["Open"]).abs() / price_df["Close"]
    out["bar_direction"] = np.sign(price_df["Close"] - price_df["Open"])
    out["minute_of_day"] = price_df.index.hour * 60 + price_df.index.minute
    out["day_of_week"] = price_df.index.dayofweek

    if extra_context is not None:
        _ensure_indexed_like(extra_context, price_df.index)
        overlap = set(out.columns).intersection(extra_context.columns)
        if overlap:
            raise ValueError(f"extra_context has overlapping columns with defaults: {sorted(overlap)}")
        out = pd.concat([out, extra_context], axis=1)

    return out


# ==========================================================
# Example user features
# ==========================================================


def feat_first_bar_opposite_then_win(ctx: TradeAnalysisContext) -> dict[str, float]:
    seg = ctx.after_entry(1, include_entry_bar=False)
    first_bar_opp = seg.first_bar_opposite_color()
    return {
        "path_first_bar_opposite": first_bar_opp,
        "path_first_bar_opposite_then_win": float(first_bar_opp == 1.0 and ctx.is_winner),
    }



def feat_initial_pullback_first_3(ctx: TradeAnalysisContext) -> dict[str, float]:
    seg = ctx.after_entry(3, include_entry_bar=False)
    return {
        "path_adverse_excursion_first_3": seg.adverse_excursion_pct(),
        "path_favorable_excursion_first_3": seg.favorable_excursion_pct(),
    }


# ==========================================================
# Example usage
# ==========================================================

"""
# price_df should be indexed exactly like your engine bar_index
price_df = pd.DataFrame({
    "Open": opens,
    "High": highs,
    "Low": lows,
    "Close": closes,
}, index=bar_index)

context_df = build_default_context_df(price_df, extra_context=my_custom_context_df)

engine = TradeContextEngine(
    trades_df=metrics["trades_df"],
    price_df=price_df,
    context_df=context_df,
)

trades_enriched = engine.attach_entry_context(cols=["bar_range_pct", "day_of_week"])
trades_enriched = TradeContextEngine(trades_enriched, price_df, context_df).attach_exit_context(cols=["bar_direction"])

path_specs = [
    PathAggSpec(col="bar_range_pct", aggs=["mean", "max"]),
    PathAggSpec(col="bar_return", aggs=["mean", "std"]),
]
trades_enriched = TradeContextEngine(trades_enriched, price_df, context_df).attach_path_aggregations(path_specs)

feature_specs = [
    TradeFeatureSpec(name="first_bar_opposite_then_win", fn=feat_first_bar_opposite_then_win),
    TradeFeatureSpec(name="initial_pullback_first_3", fn=feat_initial_pullback_first_3),
]
trades_enriched = TradeContextEngine(trades_enriched, price_df, context_df).compute_trade_features(feature_specs)
"""
