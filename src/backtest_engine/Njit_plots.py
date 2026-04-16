"""
╔══════════════════════════════════════════════════════════════════╗
║              NJIT PLOTS — post-run visualization                ║
║                                                                  ║
║  Input: metrics dict returned by compute_metrics_full()          ║
║                                                                  ║
║  Standalone usage:                                               ║
║    from njit_plots import plot_results                           ║
║    plot_results(metrics)                                         ║
║                                                                  ║
║  Usage via run():                                                ║
║    rets, metrics = engine.run(signals, plot_results=True)        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Palette ───────────────────────────────────────────────────────
_C = dict(
    bg      = "#0e1117",
    panel   = "#161b22",
    border  = "#21262d",
    text    = "#e6edf3",
    sub     = "#8b949e",
    green   = "#3fb950",
    red     = "#f85149",
    blue    = "#58a6ff",
    orange  = "#d29922",
    purple  = "#bc8cff",
    zero    = "#30363d",
)

_AXIS = dict(gridcolor=_C["border"], zerolinecolor=_C["zero"],
             tickfont=dict(color=_C["sub"]))


def _base_layout(title="", height=450):
    return dict(
        paper_bgcolor = _C["bg"],
        plot_bgcolor  = _C["panel"],
        font          = dict(family="monospace", size=11, color=_C["text"]),
        margin        = dict(l=55, r=25, t=45, b=45),
        height        = height,
        title         = dict(text=title, font=dict(size=13, color=_C["text"])),
        legend        = dict(bgcolor=_C["panel"], bordercolor=_C["border"],
                             borderwidth=1, font=dict(size=10)),
        hovermode     = "x unified",
    )


def _fig(title="", height=450, rows=1, cols=1, **subplot_kwargs):
    if rows == 1 and cols == 1:
        fig = go.Figure()
    else:
        fig = make_subplots(rows=rows, cols=cols, **subplot_kwargs)
    fig.update_layout(**_base_layout(title, height))
    for attr in list(fig.layout):
        if attr.startswith("xaxis") or attr.startswith("yaxis"):
            fig.layout[attr].update(**_AXIS)
    return fig


# ══════════════════════════════════════════════════════════════════
# 1. EDGE CURVE
# ══════════════════════════════════════════════════════════════════

def plot_edge(metrics: dict, label: str = "", show: bool = True) -> go.Figure:
    """
    Cumulative sum of individual trade returns under equal stake per trade.
    This is the raw edge without compounding effects.
    Uses total_return_sum as the final reference value.
    """
    df    = metrics["trades_df"]
    ret   = df["return"].to_numpy()
    t     = df["exit_time"]

    cum      = np.cumsum(ret)
    roll_max = np.maximum.accumulate(np.maximum(cum, 0))
    dd       = cum - roll_max

    total = metrics.get("total_return_sum", cum[-1])
    n     = len(ret)
    wr    = metrics.get("win_rate", np.nan)
    pf    = metrics.get("profit_factor", np.nan)
    title = f"Edge curve {'— ' + label if label else ''}"

    fig = _fig(title, height=500, rows=2, cols=1,
               row_heights=[0.72, 0.28], shared_xaxes=True,
               vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=t, y=cum * 100,
        mode="lines",
        line=dict(color=_C["blue"], width=2),
        name=f"Σ returns  {total*100:+.2f}%",
        hovertemplate="%{y:.3f}%<extra></extra>",
    ), row=1, col=1)

    # zero line
    fig.add_hline(y=0, line_color=_C["zero"], line_width=1, row=1, col=1)

    # annotation KPIs
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.97, showarrow=False,
        text=(f"N={n}  |  WR={wr:.1%}  |  "
              f"Avg={ret.mean()*100:+.3f}%  |  "
              f"PF={pf:.2f}  |  "
              f"∑={total*100:+.2f}%"),
        font=dict(size=10, color=_C["sub"]),
        align="left",
    )

    fig.add_trace(go.Scatter(
        x=t, y=dd * 100,
        mode="lines", fill="tozeroy",
        line=dict(color=_C["red"], width=1),
        fillcolor="rgba(248,81,73,0.15)",
        name="Drawdown Σret",
        hovertemplate="%{y:.3f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_yaxes(title_text="∑ ret %", row=1, col=1)
    fig.update_yaxes(title_text="DD %",    row=2, col=1)

    if show:
        fig.show()
    return fig


# ══════════════════════════════════════════════════════════════════
# 2. EQUITY CURVE  (compound)
# ══════════════════════════════════════════════════════════════════

def plot_equity(metrics: dict, label: str = "", show: bool = True) -> go.Figure:
    """Compounded equity curve."""
    df       = metrics["trades_df"]
    ret      = df["return"].to_numpy()
    t        = df["exit_time"]

    cum      = np.cumprod(1 + ret)
    roll_max = np.maximum.accumulate(cum)
    dd_pct   = (cum - roll_max) / roll_max * 100

    mdd    = metrics.get("max_drawdown", dd_pct.min() / 100)
    cagr   = metrics.get("ann_return",   np.nan)
    sharpe = metrics.get("sharpe",       np.nan)
    calmar = metrics.get("calmar",       np.nan)
    title  = f"Equity curve (compound) {'— ' + label if label else ''}"

    fig = _fig(title, height=500, rows=2, cols=1,
               row_heights=[0.72, 0.28], shared_xaxes=True,
               vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=t, y=cum,
        mode="lines",
        line=dict(color=_C["green"], width=2),
        name=f"Equity  CAGR={cagr:.1%}",
        hovertemplate="%{y:.4f}x<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=1, line_color=_C["zero"], line_width=1, row=1, col=1)

    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.97, showarrow=False,
        text=(f"CAGR={cagr:.2%}  |  Sharpe={sharpe:.2f}  |  "
              f"Calmar={calmar:.2f}  |  MDD={mdd:.2%}"),
        font=dict(size=10, color=_C["sub"]),
        align="left",
    )

    fig.add_trace(go.Scatter(
        x=t, y=dd_pct,
        mode="lines", fill="tozeroy",
        line=dict(color=_C["red"], width=1),
        fillcolor="rgba(248,81,73,0.15)",
        name=f"Drawdown  max={mdd:.2%}",
        hovertemplate="%{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_yaxes(title_text="Factor", row=1, col=1)
    fig.update_yaxes(title_text="DD %",    row=2, col=1)

    if show:
        fig.show()
    return fig


# ══════════════════════════════════════════════════════════════════
# 3. RETURN DISTRIBUTION
# ══════════════════════════════════════════════════════════════════

def plot_returns_dist(metrics: dict, bins: int = 60,
                      show: bool = True) -> go.Figure:
    """Win/loss histogram with average returns and profit factor."""
    df     = metrics["trades_df"]
    ret    = df["return"].to_numpy() * 100

    wins   = ret[ret > 0]
    losses = ret[ret <= 0]
    pf     = metrics.get("profit_factor", np.nan)
    var_   = metrics.get("VaR",  np.nan)
    cvar_  = metrics.get("CVaR", np.nan)

    fig = _fig("Return distribution by trade", height=400)

    fig.add_trace(go.Histogram(
        x=losses, nbinsx=bins,
        name=f"Loss  avg={losses.mean():.3f}%  N={len(losses)}",
        marker_color=_C["red"],   opacity=0.75,
    ))
    fig.add_trace(go.Histogram(
        x=wins,   nbinsx=bins,
        name=f"Win   avg={wins.mean():.3f}%   N={len(wins)}",
        marker_color=_C["green"], opacity=0.75,
    ))

    fig.add_vline(x=0,     line_color=_C["zero"],   line_width=1)
    fig.add_vline(x=-var_, line_color=_C["orange"], line_width=1,
                  annotation_text=f"VaR {var_:.3f}%",
                  annotation_font_color=_C["orange"])

    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.97, showarrow=False,
        text=f"PF={pf:.2f}  VaR={var_:.3f}%  CVaR={cvar_:.3f}%",
        font=dict(size=10, color=_C["sub"]), align="right",
    )

    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Return %")
    fig.update_yaxes(title_text="Frequency")

    if show:
        fig.show()
    return fig


# ══════════════════════════════════════════════════════════════════
# 4. P&L BY EXIT REASON
# ══════════════════════════════════════════════════════════════════

def plot_by_reason(metrics: dict, show: bool = True) -> go.Figure:
    """Bars for average return and win rate by exit reason."""
    by = metrics.get("by_reason", {})
    if not by:
        print("by_reason is empty — skipping")
        return None

    reasons  = list(by.keys())
    avg_rets = [by[r]["avg_return"] * 100 for r in reasons]
    winrates = [by[r]["win_rate"]   * 100 for r in reasons]
    n_trades = [by[r]["n_trades"]        for r in reasons]
    colors   = [_C["green"] if v >= 0 else _C["red"] for v in avg_rets]

    fig = _fig("P&L by exit reason", height=400, rows=1, cols=2,
               column_titles=["Avg return %", "Win rate %"],
               horizontal_spacing=0.12)

    fig.add_trace(go.Bar(
        x=reasons, y=avg_rets,
        marker_color=colors,
        text=[f"{v:.3f}%" for v in avg_rets],
        textposition="outside",
        textfont=dict(size=9),
        name="Avg return",
        customdata=n_trades,
        hovertemplate="%{x}<br>avg=%{y:.3f}%<br>N=%{customdata}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=reasons, y=winrates,
        marker_color=_C["blue"],
        text=[f"{v:.0f}%" for v in winrates],
        textposition="outside",
        textfont=dict(size=9),
        name="Win rate",
        customdata=n_trades,
        hovertemplate="%{x}<br>wr=%{y:.1f}%<br>N=%{customdata}<extra></extra>",
    ), row=1, col=2)

    fig.add_hline(y=0,  line_color=_C["zero"], line_width=1, row=1, col=1)
    fig.add_hline(y=50, line_color=_C["zero"], line_width=1, row=1, col=2)

    if show:
        fig.show()
    return fig


# ══════════════════════════════════════════════════════════════════
# 5. MAE / MFE SCATTER
# ══════════════════════════════════════════════════════════════════

def plot_mae_mfe(metrics: dict, show: bool = True) -> go.Figure:
    """MAE vs MFE scatter, colored by win/loss."""
    df = metrics["trades_df"]
    if "mae_intra" not in df.columns or df["mae_intra"].isna().all():
        print("MAE/MFE not available — is track_mae_mfe=False?")
        return None

    ret     = df["return"].to_numpy() * 100
    mae     = df["mae_intra"].to_numpy() * 100
    mfe     = df["mfe_intra"].to_numpy() * 100
    reasons = df["reason"].to_numpy() if "reason" in df.columns else [""] * len(ret)
    col     = [_C["green"] if r > 0 else _C["red"] for r in ret]

    avg_cap = metrics.get("avg_capture_intra", np.nan)

    fig = _fig(f"MAE / MFE intra  —  capture avg={avg_cap:.1%}", height=440)

    fig.add_trace(go.Scatter(
        x=mae, y=mfe,
        mode="markers",
        marker=dict(color=col, size=4, opacity=0.60),
        customdata=np.stack([ret, reasons], axis=-1),
        hovertemplate=(
            "MAE=%{x:.3f}%  MFE=%{y:.3f}%<br>"
            "ret=%{customdata[0]:.3f}%  %{customdata[1]}<extra></extra>"
        ),
        showlegend=False,
    ))

    lim = max(abs(mae.min()), mfe.max()) if len(mae) else 1
    fig.add_trace(go.Scatter(
        x=[0, -lim], y=[0, lim],
        mode="lines",
        line=dict(color=_C["sub"], width=1, dash="dot"),
        name="100% capture",
    ))

    fig.add_hline(y=0, line_color=_C["zero"], line_width=1)
    fig.add_vline(x=0, line_color=_C["zero"], line_width=1)

    fig.update_xaxes(title_text="MAE %  (adverse)")
    fig.update_yaxes(title_text="MFE %  (favorable)")

    if show:
        fig.show()
    return fig


# ══════════════════════════════════════════════════════════════════
# 6. RETURNS BY PERIOD
# ══════════════════════════════════════════════════════════════════

def plot_period_returns(metrics: dict, show: bool = True) -> go.Figure:
    """Bars of aggregated returns by period (ME / WE / ...)."""
    df    = metrics["trades_df"]
    freq  = metrics.get("period_freq", "ME")
    t_idx = pd.DatetimeIndex(df["exit_time"])

    series = pd.Series(df["return"].to_numpy(), index=t_idx).resample(freq).sum()
    series = series[series != 0]

    if len(series) == 0:
        print("No period return data available")
        return None

    colors  = [_C["green"] if v >= 0 else _C["red"] for v in series.values]
    pct_pos = metrics.get("pct_periods_positive", np.nan)
    worst   = metrics.get("worst_period", np.nan)
    best    = metrics.get("best_period",  np.nan)

    fig = _fig(
        f"Returns by period ({freq})  —  "
        f"pos={pct_pos:.0%}  worst={worst:.2%}  best={best:.2%}",
        height=380,
    )

    fig.add_trace(go.Bar(
        x=series.index,
        y=series.values * 100,
        marker_color=colors,
        hovertemplate="%{x|%Y-%m}<br>%{y:.3f}%<extra></extra>",
        name="Period return",
    ))

    fig.add_hline(y=0, line_color=_C["zero"], line_width=1)
    fig.update_yaxes(title_text="Return %")

    if show:
        fig.show()
    return fig


# ══════════════════════════════════════════════════════════════════
# 7. SUMMARY CONSOLE
# ══════════════════════════════════════════════════════════════════

def print_summary(metrics: dict, label: str = "") -> None:
    """Compact console summary — always shown."""
    m   = metrics
    sep = "─" * 55
    tag = f"  [{label}]" if label else ""
    print(f"\n{sep}")
    print(f"  BACKTEST SUMMARY{tag}")
    print(sep)
    print(f"  Trades            : {m.get('n_trades', '?')}")
    print(f"  Win rate          : {m.get('win_rate', 0):.1%}")
    print(f"  Σ returns (edge)  : {m.get('total_return_sum', 0)*100:+.2f}%  ← equal stake/trade")
    print(f"  Cum ret (compound): {m.get('cum_return', 0)*100:+.2f}%")
    print(f"  Ann return        : {m.get('ann_return', 0)*100:+.2f}%")
    print(f"  Max drawdown      : {m.get('max_drawdown', 0):.2%}")
    print(f"  Sharpe            : {m.get('sharpe', float('nan')):.3f}")
    print(f"  Profit factor     : {m.get('profit_factor', float('nan')):.3f}")
    print(f"  Calmar            : {m.get('calmar', float('nan')):.3f}")
    print(f"  Avg win / loss    : "
          f"{m.get('avg_win', 0)*100:+.3f}% / "
          f"{m.get('avg_loss', 0)*100:+.3f}%")
    print(f"  VaR / CVaR        : "
          f"{m.get('VaR', float('nan')):.4f} / "
          f"{m.get('CVaR', float('nan')):.4f}")
    print(f"  p-value (t/binom) : "
          f"{m.get('p_value', float('nan')):.4f} / "
          f"{m.get('p_binom', float('nan')):.4f}")
    freq = m.get('period_freq', '?')
    pp   = m.get('pct_periods_positive', float('nan'))
    print(f"  Periods ({freq})    : {m.get('n_periods','?')}  pos={pp:.0%}")
    print(sep + "\n")


# ══════════════════════════════════════════════════════════════════
# 8. MAIN ENTRY — plot_results()
# ══════════════════════════════════════════════════════════════════

def plot_results(
    metrics,
    label:       str  = "",
    edge:        bool = True,
    equity:      bool = True,
    dist:        bool = True,
    by_reason:   bool = True,
    mae_mfe:     bool = True,
    period_ret:  bool = True,
    summary:     bool = True,
) -> dict:
    """
    Single entry point — displays all requested charts.

    Parameters
    ----------
    metrics    : dict returned by engine.run()
    label      : optional text displayed in titles
    edge       : raw cumulative returns under equal stake per trade
    equity     : compounded equity curve
    dist       : win/loss histogram
    by_reason  : bars by exit reason
    mae_mfe    : MAE/MFE scatter
    period_ret : bars by aggregation period
    summary    : console summary

    Returns
    -------
    dict {name: go.Figure}  — useful for later reuse
    """
    if metrics is None:
        print("metrics is None — no trades?")
        return {}

    figs = {}

    if summary:
        print_summary(metrics, label)

    if edge:
        figs["edge"]       = plot_edge(metrics, label)
    if equity:
        figs["equity"]     = plot_equity(metrics, label)
    if dist:
        figs["dist"]       = plot_returns_dist(metrics)
    if by_reason and metrics.get("by_reason"):
        figs["by_reason"]  = plot_by_reason(metrics)
    if mae_mfe:
        figs["mae_mfe"]    = plot_mae_mfe(metrics)
    if period_ret:
        figs["period_ret"] = plot_period_returns(metrics)

    return figs
