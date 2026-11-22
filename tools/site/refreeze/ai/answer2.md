```python
import numpy as np
import pandas as pd
from numba import njit
from Backtest_Git import *

# bt:features
# Indicators drawn on each asset's candles in the Inspection, plus the values the allocator used.
def sma(f, window=50):
    return f.col('Close').rolling(int(window), min_periods=int(window)).mean()

def rolling_mean(f, window=126):
    return f.col('Close').pct_change().rolling(int(window)).mean()

def rolling_vol(f, window=126):
    return f.col('Close').pct_change().rolling(int(window)).std()

def features(feature, df):
    p = settings.get('parameters', {})
    lookback = int(p.get('lookback', 126))
    fast = feature('sma', on=df, window=int(p.get('sma_fast', 50))).series()
    slow = feature('sma', on=df, window=int(p.get('sma_slow', 200))).series()
    mean = feature('rolling_mean', on=df, window=lookback).series()
    vol = feature('rolling_vol', on=df, window=lookback).series()
    close = df['Close']
    band = close.rolling(20, min_periods=20).std()
    mid = close.rolling(20, min_periods=20).mean()
    return pd.DataFrame(dict(sma_fast=fast, sma_slow=slow, band_up=mid + 2 * band, band_down=mid - 2 * band,
                             rolling_mean=mean, rolling_vol=vol * np.sqrt(252.), individual_sharpe=mean / vol),
                        index=df.index)

feature = Feature()
feature.add(sma, 'sma')
feature.add(rolling_mean, 'rolling_mean')
feature.add(rolling_vol, 'rolling_vol')

# bt:allocator
# Allocator-owned state only.
S_SEEDED = 0
INITIAL_WEIGHTS_APPLIED = 63
INITIAL_WEIGHT_SLOTS = 20

META = MetaSpec(
    values=("sharpe", "observations", "budget"),
    assets=("best_asset", "worst_asset"),
    # rolling_window: returns behind rolling_mean / rolling_vol at that rebalance (dates common to the
    # sample the value comes from: ranking sample, or the chosen sample for an optimised asset).
    per_asset=("individual_sharpe", "excluded", "rolling_mean", "rolling_vol", "rolling_window"),
    reasons=("warming_up", "not_enough_common_returns", "no_positive_expected_returns"),
)
M = META.ids


class _CalendarIntervalTrigger(StatefulTriggerSpec):
    """Python calendar cadence metadata; the compiled trigger ABI stays unchanged."""
    __slots__ = ("period_count",)

    def __init__(self, frequency: str, period_count: int):
        super().__init__(kind="calendar", frequency=frequency)
        self.period_count = int(period_count)


@njit(cache=False)
def _std(x):
    n = x.shape[0]
    if n < 2:
        return np.nan
    m = x.mean()
    s = 0.0
    for v in x:
        s += (v - m) * (v - m)
    return np.sqrt(s / (n - 1))


@njit(cache=False)
def _common_returns(r, cols, k, lookback):
    """The last `lookback` dates where every selected line has a return, each return compounded since
    the previous such date, so that all lines' returns span the same interval: a Sunday FX bar is
    folded into Friday -> Monday like the index, a stock's overnight H1 bar faces the 17 crypto hours
    of the same night. When every row is common this is exactly the rows themselves."""
    n = r.shape[0]
    rows = np.empty(lookback + 1, dtype=np.int64)          # common rows, latest first
    found = 0
    t = n - 1
    while t >= 0 and found <= lookback:
        common = True
        for i in range(k):
            if not np.isfinite(r[t, cols[i]]):
                common = False
                break
        if common:
            rows[found] = t
            found += 1
        t -= 1
    m = min(found, lookback)
    out = np.empty((m, k))
    for s in range(m):
        end = rows[m - 1 - s]
        start = rows[m - s] + 1 if m - s < found else end  # the oldest without a previous date: its own row
        for i in range(k):
            acc = 1.0
            count = 0
            last = 0.0
            for u in range(start, end + 1):
                v = r[u, cols[i]]
                if np.isfinite(v):
                    acc *= 1.0 + v
                    count += 1
                    last = v
            out[s, i] = last if count == 1 else acc - 1.0
    return out


@njit(cache=False)
def _project(w, cap, target, n):
    if target == cap * n:
        return np.full(n, cap)
    low = np.min(w - cap)
    high = np.max(w)
    for _ in range(50):
        shift = (low + high) / 2.0
        if np.clip(w - shift, 0.0, cap).sum() > target:
            low = shift
        else:
            high = shift
    return np.clip(w - (low + high) / 2.0, 0.0, cap)


@njit(cache=False)
def _portfolio_sharpe(w, mean, cov):
    v = w @ cov @ w
    return (w @ mean) / np.sqrt(v) if v > 0 else -np.inf


@njit(cache=False)
def _max_sharpe(sample, cap, target, shrink):
    n = sample.shape[1]
    mean = np.empty(n)
    for k in range(n):
        mean[k] = sample[:, k].mean()
    if n == 1:
        cov = np.full((1, 1), _std(sample[:, 0]) ** 2)
    else:
        cov = np.cov(sample.T)
    cov = (1.0 - shrink) * cov + shrink * np.diag(np.diag(cov))
    w = _project(np.maximum(mean, 0.0), cap, target, n)
    score = _portfolio_sharpe(w, mean, cov)
    for _ in range(250):
        v = w @ cov @ w
        if v <= 0:
            break
        g = mean / np.sqrt(v) - score * (cov @ w) / v
        scale = np.max(np.abs(g))
        if not np.isfinite(scale) or scale < 1e-12:
            break
        step = 0.2
        improved = False
        for _ in range(20):
            c = _project(w + step * g / scale, cap, target, n)
            s = _portfolio_sharpe(c, mean, cov)
            if s > score + 1e-12:
                w = c
                score = s
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    return w, score, mean


@njit(cache=False)
def _financed_gross(ctx, borrow_enabled):
    """Book size per unit of equity: 1 without borrow; with the Borrow panel on, equity plus debt plus
    the draws decided this bar, over equity. Weights are book shares multiplied by it."""
    if borrow_enabled > 0.0 and ctx.equity > 0.0:
        return (ctx.equity + ctx.borrowed + ctx.borrow_pending) / ctx.equity
    return 1.0


@njit(cache=False)
def _above_average(r, col, window):
    """Price above its `window`-bar average, from returns only: P_(t-k) / P_t walked back from the last close."""
    n = r.shape[0]
    if n < window:
        return True
    rel = 1.0
    total = 1.0
    for k in range(1, window):
        v = r[n - k, col]
        if np.isfinite(v):
            rel /= 1.0 + v
        total += rel
    return total / window < 1.0


@njit(cache=False)
def max_sharpe(ctx, mem, out):
    full = ctx.return_df
    lookback = int(ctx.f.lookback)
    if full.shape[0] < lookback:
        out.reason = M.warming_up
        return
    a = full.shape[1]
    seeded = mem[S_SEEDED] > 0.0
    fixed = int(ctx.f.fixed_asset)
    fixed_w = min(max(ctx.f.fixed_weight, 0.0), 1.0) if fixed >= 0 else 0.0
    keep_fixed_holdings = ctx.f.fixed_mode == 0.0
    preserve_best = seeded and ctx.f.preserve_best > 0.0
    n_worst = int(ctx.f.exclude_worst) if seeded else 0

    # Rank by individual Sharpe on the market universe (never the holdings).
    # Candidate only with `lookback` returns of its own (an asset listed mid-run waits); estimates use
    # the last `lookback` dates where every candidate has a return (`_common_returns`): a day or an
    # hour when one line does not trade is skipped and folded into the next common return.
    history = ctx.history_bars_by_asset
    cand = np.empty(a, dtype=np.int64)
    n_cand = 0
    for j in range(a):
        if j == fixed or history[j] < lookback:
            continue
        if ctx.f.trend_filter > 0.0 and not _above_average(full, j, int(ctx.f.sma_slow)):
            out.meta_per_asset[M.excluded, j] = 1.0
            continue
        cand[n_cand] = j
        n_cand += 1
    if n_cand == 0:
        out.reason = M.not_enough_common_returns
        return
    ranking = _common_returns(full, cand, n_cand, lookback)
    if ranking.shape[0] < 2:
        out.reason = M.not_enough_common_returns
        return
    ind = np.full(n_cand, -np.inf)
    for k in range(n_cand):
        s = _std(ranking[:, k])
        out.meta_per_asset[M.rolling_mean, cand[k]] = ranking[:, k].mean()
        out.meta_per_asset[M.rolling_vol, cand[k]] = s
        out.meta_per_asset[M.rolling_window, cand[k]] = ranking.shape[0]
        if s > 0:
            ind[k] = ranking[:, k].mean() / s
            out.meta_per_asset[M.individual_sharpe, cand[k]] = ind[k]
    order = np.argsort(-ind)                          # best first; ties keep column order
    best = cand[order[0]]
    worst = cand[order[n_cand - 1]]
    out.meta_asset[M.best_asset] = best
    out.meta_asset[M.worst_asset] = worst

    # Selection: drop the n worst, set the best aside when it is preserved.
    chosen = np.empty(n_cand, dtype=np.int64)
    n_sel = 0
    for k in range(n_cand):
        j = cand[order[k]]
        if k >= n_cand - n_worst:
            out.meta_per_asset[M.excluded, j] = 1.0
            continue
        if preserve_best and j == best:
            continue
        chosen[n_sel] = j
        n_sel += 1
    if n_sel == 0:
        out.reason = M.not_enough_common_returns
        return
    sample = _common_returns(full, chosen, n_sel, lookback)
    if sample.shape[0] <= n_sel:
        out.reason = M.not_enough_common_returns
        return

    # Budget: what the kept holdings leave free, in shares of the book.
    drift = ctx.current_drift_weight_by_asset
    g = _financed_gross(ctx, ctx.f.borrow_enabled)
    budget = 1.0
    if fixed >= 0:
        budget -= drift[fixed] / g if (seeded and keep_fixed_holdings) else fixed_w
    if preserve_best:
        budget -= drift[best] / g
    budget = max(budget, 0.0)
    cap = min(max(ctx.f.max_weight, 0.0), 1.0)
    shrink = min(max(ctx.f.shrinkage, 0.0), 1.0)
    w, score, mean = _max_sharpe(sample, cap, min(budget, cap * n_sel), shrink)
    for k in range(n_sel):
        out.meta_per_asset[M.rolling_mean, chosen[k]] = mean[k]
        out.meta_per_asset[M.rolling_vol, chosen[k]] = _std(sample[:, k])
        out.meta_per_asset[M.rolling_window, chosen[k]] = sample.shape[0]
    if not np.any(mean > 0):
        out.reason = M.no_positive_expected_returns
        return

    out.action = REBALANCE
    out.scope = REPLACE
    for k in range(n_sel):
        out.by_asset[chosen[k]] = w[k] * g
    if fixed >= 0:
        if seeded and keep_fixed_holdings:
            out.preserve_assets[fixed] = True
        else:
            out.by_asset[fixed] = fixed_w * g
    if preserve_best:
        out.preserve_assets[best] = True
    out.portfolio_leverage = g if ctx.f.borrow_enabled > 0.0 else ctx.f.leverage
    out.reserve_cash_pct = ctx.f.reserve_cash_pct
    out.meta[M.sharpe] = score
    out.meta[M.observations] = sample.shape[0]
    out.meta[M.budget] = budget
    mem[S_SEEDED] = 1.0


@njit(cache=False)
def equal_weight(ctx, mem, out):
    """1/n of the lines already listed (at least one return of their own); a line listed later joins
    at the next rebalance."""
    if ctx.return_df.shape[0] < int(ctx.f.lookback):
        out.reason = M.warming_up
        return
    history = ctx.history_bars_by_asset
    listed = 0
    for j in range(history.shape[0]):
        if history[j] >= 1:
            listed += 1
    if listed == 0:
        out.reason = M.warming_up
        return
    g = _financed_gross(ctx, ctx.f.borrow_enabled)
    out.action = REBALANCE
    out.scope = REPLACE
    for j in range(history.shape[0]):
        if history[j] >= 1:
            out.by_asset[j] = g / listed
    out.portfolio_leverage = g if ctx.f.borrow_enabled > 0.0 else ctx.f.leverage
    out.reserve_cash_pct = ctx.f.reserve_cash_pct


@njit(cache=False)
def white_portfolio_hold(ctx, mem, out):
    """Reapply named initial weights whenever the White portfolio trigger fires."""
    own_count = ctx.own_count
    listed_weight = 0.0
    listed = 0
    for j in range(own_count.shape[1]):
        if own_count[ctx.bar, j] > 0:
            listed += 1
            weight = 0.0
            if j == 0: weight = ctx.f.initial_weight_0
            elif j == 1: weight = ctx.f.initial_weight_1
            elif j == 2: weight = ctx.f.initial_weight_2
            elif j == 3: weight = ctx.f.initial_weight_3
            elif j == 4: weight = ctx.f.initial_weight_4
            elif j == 5: weight = ctx.f.initial_weight_5
            elif j == 6: weight = ctx.f.initial_weight_6
            elif j == 7: weight = ctx.f.initial_weight_7
            elif j == 8: weight = ctx.f.initial_weight_8
            elif j == 9: weight = ctx.f.initial_weight_9
            elif j == 10: weight = ctx.f.initial_weight_10
            elif j == 11: weight = ctx.f.initial_weight_11
            elif j == 12: weight = ctx.f.initial_weight_12
            elif j == 13: weight = ctx.f.initial_weight_13
            elif j == 14: weight = ctx.f.initial_weight_14
            elif j == 15: weight = ctx.f.initial_weight_15
            elif j == 16: weight = ctx.f.initial_weight_16
            elif j == 17: weight = ctx.f.initial_weight_17
            elif j == 18: weight = ctx.f.initial_weight_18
            elif j == 19: weight = ctx.f.initial_weight_19
            out.by_asset[j] = max(0.0, weight)
            listed_weight += max(0.0, weight)
    if listed == 0:
        return
    if listed_weight <= 0.0 and ctx.f.initial_cash_weight < 1.0:
        fallback = (1.0 - max(0.0, ctx.f.initial_cash_weight)) / listed
        for j in range(own_count.shape[1]):
            if own_count[ctx.bar, j] > 0:
                out.by_asset[j] = fallback
    out.action = REBALANCE
    out.scope = REPLACE


_INITIAL_ALLOCATOR_WRAPPERS = {}


def initial_weights_then_allocator(allocator):
    """Apply the initial basket once, then pass each later trigger to the allocator."""
    if allocator not in _INITIAL_ALLOCATOR_WRAPPERS:
        def callback(ctx, mem, out):
            if ctx.f.initial_weights_enabled <= 0.0:
                return allocator(ctx, mem, out)
            if mem[INITIAL_WEIGHTS_APPLIED] > 0.0:
                return allocator(ctx, mem, out)
            own_count = ctx.own_count
            listed = 0
            total = 0.0
            for j in range(own_count.shape[1]):
                if own_count[ctx.bar, j] > 0:
                    listed += 1
                    weight = 0.0
                    if j == 0: weight = ctx.f.initial_weight_0
                    elif j == 1: weight = ctx.f.initial_weight_1
                    elif j == 2: weight = ctx.f.initial_weight_2
                    elif j == 3: weight = ctx.f.initial_weight_3
                    elif j == 4: weight = ctx.f.initial_weight_4
                    elif j == 5: weight = ctx.f.initial_weight_5
                    elif j == 6: weight = ctx.f.initial_weight_6
                    elif j == 7: weight = ctx.f.initial_weight_7
                    elif j == 8: weight = ctx.f.initial_weight_8
                    elif j == 9: weight = ctx.f.initial_weight_9
                    elif j == 10: weight = ctx.f.initial_weight_10
                    elif j == 11: weight = ctx.f.initial_weight_11
                    elif j == 12: weight = ctx.f.initial_weight_12
                    elif j == 13: weight = ctx.f.initial_weight_13
                    elif j == 14: weight = ctx.f.initial_weight_14
                    elif j == 15: weight = ctx.f.initial_weight_15
                    elif j == 16: weight = ctx.f.initial_weight_16
                    elif j == 17: weight = ctx.f.initial_weight_17
                    elif j == 18: weight = ctx.f.initial_weight_18
                    elif j == 19: weight = ctx.f.initial_weight_19
                    out.by_asset[j] = max(0.0, weight)
                    total += max(0.0, weight)
            if listed == 0:
                return
            if total <= 0.0 and ctx.f.initial_cash_weight < 1.0:
                fallback = (1.0 - max(0.0, ctx.f.initial_cash_weight)) / listed
                for j in range(own_count.shape[1]):
                    if own_count[ctx.bar, j] > 0:
                        out.by_asset[j] = fallback
            out.action = REBALANCE
            out.scope = REPLACE
            mem[INITIAL_WEIGHTS_APPLIED] = 1.0
        from numba import config, njit
        _INITIAL_ALLOCATOR_WRAPPERS[allocator] = callback if config.DISABLE_JIT else njit(cache=True)(callback)
    return _INITIAL_ALLOCATOR_WRAPPERS[allocator]


def initial_weight_parameters(weights: dict, asset_names: list[str]) -> dict[str, float]:
    """Bind a name-keyed initial basket to the allocator's stable named parameter slots."""
    if len(asset_names) > INITIAL_WEIGHT_SLOTS:
        raise ValueError(f"White portfolios support at most {INITIAL_WEIGHT_SLOTS} selected assets.")
    values = {}
    for name, weight in weights.items():
        if name == "Cash":
            if not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight < 0:
                raise ValueError("initial weights must be finite non-negative numbers")
            continue
        if name not in asset_names:
            raise ValueError(f"initial weight asset {name!r} is not selected")
        if not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight < 0:
            raise ValueError("initial weights must be finite non-negative numbers")
    for index, name in enumerate(asset_names):
        values[f"initial_weight_{index}"] = float(weights.get(name, 0.0))
    for index in range(len(asset_names), INITIAL_WEIGHT_SLOTS):
        values[f"initial_weight_{index}"] = 0.0
    values["initial_cash_weight"] = float(weights.get("Cash", 0.0))
    return values


def default_initial_weight_parameters(asset_names: list[str], existing: dict | None = None) -> dict[str, float]:
    """Use 1/N when no initial basket was supplied; retain explicit named values."""
    values = dict(existing or {})
    if not any(str(name).startswith("initial_weight_") for name in values):
        count = len(asset_names)
        values.update(initial_weight_parameters(
            {name: 1.0 / count for name in asset_names} if count else {}, asset_names))
    else:
        defaults = initial_weight_parameters({}, asset_names)
        for name, value in defaults.items():
            values.setdefault(name, value)
    values.setdefault("initial_weights_enabled", 1.0)
    return values


def white_rebalance_settings(settings: dict):
    """Build White's calendar/event policy from the Results controls."""
    strategy = settings.get("strategy", {})
    periodic = bool(strategy.get("periodic_rebalancing", False))
    drift_enabled = bool(strategy.get("drift_enabled", False))
    drift = float(strategy.get("drift", 0.0)) if drift_enabled else 0.0
    if not periodic:
        # A very long interval leaves only the first calendar seed; drift checks remain live.
        trigger = _CalendarIntervalTrigger(frequency="monthly", period_count=2_000_000_000)
        return trigger, drift
    unit = strategy.get("rebalancing_unit", "monthly")
    count = max(1, int(strategy.get("rebalancing_count", 1)))
    months = {"quarterly": 3, "semester": 6, "yearly": 12}
    if unit == "every_n_bars":
        trigger = StatefulTriggerSpec(kind="calendar", frequency="every_n_bars",
                                      every_n_bars=max(1, count))
    else:
        frequency = "monthly" if unit in months else unit
        interval = count * months.get(unit, 1)
        trigger = _CalendarIntervalTrigger(frequency=frequency, period_count=interval)
    return trigger, drift


@njit(cache=False)
def allocator(ctx, mem, out):
    if ctx.f.model > 0.5:
        max_sharpe(ctx, mem, out)
    else:
        equal_weight(ctx, mem, out)

# bt:view
def view(plot, columns):
    # Over the candles: 50 / 200-day averages and 2-sigma bands; below: what the optimiser ranks on.
    plot.price().line(columns.sma_fast, stroke=Stroke(color='#29c5ff', width=2)) \
        .line(columns.sma_slow, stroke=Stroke(color='#ffffff', width=2)) \
        .line(columns.band_up, stroke=Stroke(color='#8592a6', width=1, pattern='dotted')) \
        .line(columns.band_down, stroke=Stroke(color='#8592a6', width=1, pattern='dotted'))
    plot.panel('individual_sharpe').line(columns.individual_sharpe, stroke=Stroke(color='#29c5ff', width=2)).hline(0.)
    plot.panel('rolling_vol').line(columns.rolling_vol, stroke=Stroke(color='#b78bfa', width=2))

# bt:specs
alloc_spec = AllocSpec(on_rebalance=allocator, metadata=META,
    trigger=StatefulTriggerSpec(frequency='every_n_bars', every_n_bars=21))
allocation = AllocationSpec(initial_balance=100_000.)
cfg = BacktestConfig(entry_delay=1, tp_pct=0., sl_pct=0., session_1=None,
    candle_size_filter=False, execution_costs=ExecutionCostSpec(product_type='spot',
    contract_spec=ContractSpec(product_type='spot', contract_size=1., min_lot=1e-8, lot_step=1e-8)))
```

```toml
["declare"]
"type" = "allocation"
"universe" = "Allocations"
"assets" = ["BRENTCMDUSD", "GBPUSD", "NFLXUSUSD", "TSLAUSUSD", "USA500IDXUSD", "XAUUSD", "DEUIDXEUR"]
"timeframe" = "D1"
"portfolio_first" = false
"common_period" = true
"initial_assets" = ["BRENTCMDUSD", "GBPUSD", "NFLXUSUSD", "TSLAUSUSD", "USA500IDXUSD", "XAUUSD", "DEUIDXEUR"]
"finite_universe" = true
"data" = { "invested" = { "folder" = "Allocations", "timeframe" = "D1", "period" = "common" }, "validation" = { "folders" = ["validation_CA"], "period" = "common", "assets" = {  } }, "fx" = { "source" = "FX" }, "rates" = { "source" = "Overnights_Rates" } }

["parameters"]
"initial_weight_0" = 0.14285714285714285
"initial_weight_1" = 0.14285714285714285
"initial_cash_weight" = 0.0
"initial_weight_2" = 0.14285714285714285
"initial_weight_3" = 0.14285714285714285
"initial_weight_4" = 0.14285714285714285
"initial_weight_5" = 0.14285714285714285
"initial_weight_6" = 0.14285714285714285
"initial_weight_7" = 0.0
"initial_weight_8" = 0.0
"initial_weight_9" = 0.0
"initial_weight_10" = 0.0
"initial_weight_11" = 0.0
"initial_weight_12" = 0.0
"initial_weight_13" = 0.0
"initial_weight_14" = 0.0
"initial_weight_15" = 0.0
"initial_weight_16" = 0.0
"initial_weight_17" = 0.0
"initial_weight_18" = 0.0
"initial_weight_19" = 0.0
"initial_weights_enabled" = 1.0
"model" = "max_sharpe"
"lookback" = 126
"max_weight" = 0.2
"shrinkage" = 0.2
"fixed_asset" = "none"
"fixed_weight" = 0.1
"fixed_mode" = "keep_holdings"
"preserve_best" = false
"exclude_worst" = 0
"leverage" = 1.0
"reserve_cash_pct" = 0.0
"sma_fast" = 50
"sma_slow" = 200
"trend_filter" = 1.0

["ranges"]
"initial_weight_0" = [0.0, 1.0, 0.01]
"initial_weight_1" = [0.0, 1.0, 0.01]
"initial_cash_weight" = [0.0, 1.0, 0.01]
"initial_weight_2" = [0.0, 1.0, 0.01]
"initial_weight_3" = [0.0, 1.0, 0.01]
"initial_weight_4" = [0.0, 1.0, 0.01]
"initial_weight_5" = [0.0, 1.0, 0.01]
"initial_weight_6" = [0.0, 1.0, 0.01]
"initial_weights_enabled" = [0.0, 1.0, 1.0]
"model" = ["equal_weight", "max_sharpe"]
"lookback" = [10, 250, 5]
"max_weight" = [0.0, 1.0, 0.01]
"shrinkage" = [0.0, 1.0, 0.01]
"fixed_asset" = ["none", "USA500IDXUSD", "GBPUSD"]
"fixed_weight" = [0.0, 1.0, 0.01]
"fixed_mode" = ["keep_holdings", "reset_to_weight"]
"preserve_best" = [0, 1, 1]
"exclude_worst" = [0, 5, 1]
"leverage" = [1.0, 3.0, 0.05]
"reserve_cash_pct" = [0.0, 1.0, 0.01]
"sma_fast" = [10, 100, 5]
"sma_slow" = [50, 300, 10]
"trend_filter" = [0, 1, 1]

["validation"]
"oos_ratio" = 0.3
"oos_direction" = "right"
"draws" = 1000
```
