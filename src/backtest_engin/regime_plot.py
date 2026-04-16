from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go


DEFAULT_REGIME_COLORS: dict[int, str] = {
    0: "#7f8c8d",  # range / neutral
    1: "#27ae60",  # trend up
    2: "#c0392b",  # trend down
    3: "#f39c12",
    4: "#2980b9",
}


def plot_price_with_regime(
    df: pd.DataFrame,
    regime_col: str = "regime",
    price_col: str = "Close",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    title: str | None = None,
    crypto: bool = False,
    show_ohlc: bool = True,
    show: bool = True,
    regime_colors: Mapping[int, str] | None = None,
) -> go.Figure:
    """
    Plot OHLC/price data and automatically color the price by regime id.

    Expected input:
    - `df` contains the OHLC columns (`Open`, `High`, `Low`, `Close`)
    - `df[regime_col]` already exists, for example after `df_reg = my_regime(df)`
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if regime_col not in df.columns:
        raise ValueError(f"Missing regime column: '{regime_col}'.")

    if price_col not in df.columns:
        raise ValueError(f"Missing price column: '{price_col}'.")

    ohlc_cols = ("Open", "High", "Low", "Close")
    if show_ohlc:
        missing_ohlc = [col for col in ohlc_cols if col not in df.columns]
        if missing_ohlc:
            raise ValueError(
                "Missing OHLC columns required for candlestick plot: "
                + ", ".join(missing_ohlc)
            )

    plot_df = df.copy()
    plot_df = plot_df.sort_index()

    if start is not None:
        plot_df = plot_df.loc[pd.Timestamp(start):]

    if end is not None:
        plot_df = plot_df.loc[:pd.Timestamp(end)]

    if plot_df.empty:
        raise ValueError("No rows left to plot after applying start/end filters.")

    colors = dict(DEFAULT_REGIME_COLORS)
    if regime_colors is not None:
        colors.update({int(k): v for k, v in regime_colors.items()})

    fig = go.Figure()

    if show_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                name="Price",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                opacity=0.8,
            )
        )

    unique_regimes = [
        int(reg)
        for reg in pd.Series(plot_df[regime_col]).dropna().unique().tolist()
    ]
    unique_regimes.sort()

    for regime_id in unique_regimes:
        color = colors.get(regime_id, "#8e44ad")
        regime_mask = plot_df[regime_col].eq(regime_id)
        y = plot_df[price_col].where(regime_mask)

        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=y,
                mode="lines",
                name=f"Regime {regime_id}",
                line=dict(color=color, width=2.8),
                connectgaps=False,
                hovertemplate=(
                    "Date: %{x}<br>"
                    f"{price_col}: "
                    "%{y:.4f}<br>"
                    f"{regime_col}: {regime_id}"
                    "<extra></extra>"
                ),
            )
        )

    rangebreaks = [] if crypto else [dict(bounds=["sat", "mon"])]

    fig.update_layout(
        title=title or f"{price_col} colored by {regime_col}",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title=price_col,
        xaxis_rangeslider_visible=False,
        xaxis=dict(rangebreaks=rangebreaks),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
        height=700,
    )

    if show:
        fig.show()

    return fig
