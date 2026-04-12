# Backtest_Framework/adaptive_engine.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# RÉSULTATS
# ══════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveResults:
    trades_df: pd.DataFrame
    params_history: pd.DataFrame      # évolution params par fenêtre
    metrics_history: list[dict]       # métriques par fenêtre
    window_ranges: list[tuple]        # (start, end) par fenêtre


# ══════════════════════════════════════════════════════════════════
# ADAPTIVE ENGINE
# ══════════════════════════════════════════════════════════════════

class AdaptiveEngine:

    def __init__(
        self,
        engine,
        warmup_bars: int,
        step_bars: int,
        adaptation_fn: Callable,
        initial_params: dict[str, Any],
        include_warmup_trades: bool = False,
        min_trades_to_adapt: int = 10,
        max_param_change_pct: float | None = None,
    ):
        if warmup_bars <= 0:
            raise ValueError("warmup_bars must be > 0")
        if step_bars <= 0:
            raise ValueError("step_bars must be > 0")
        if not callable(adaptation_fn):
            raise TypeError("adaptation_fn must be callable")

        self.engine                = engine
        self.warmup_bars           = warmup_bars
        self.step_bars             = step_bars
        self.adaptation_fn         = adaptation_fn
        self.initial_params        = initial_params.copy()
        self.include_warmup_trades = include_warmup_trades
        self.min_trades_to_adapt   = min_trades_to_adapt
        self.max_param_change_pct  = max_param_change_pct

    def _apply_param_guard(
        self,
        old_params: dict,
        new_params: dict,
    ) -> dict:
        if self.max_param_change_pct is None:
            return new_params

        guarded = old_params.copy()
        for k, new_val in new_params.items():
            if k not in old_params:
                guarded[k] = new_val
                continue
            old_val = old_params[k]
            if not isinstance(new_val, (int, float)):
                guarded[k] = new_val
                continue
            if old_val == 0:
                guarded[k] = new_val
                continue
            change_pct = abs(new_val - old_val) / abs(old_val)
            if change_pct > self.max_param_change_pct:
                direction = 1 if new_val > old_val else -1
                guarded[k] = old_val * (1 + direction * self.max_param_change_pct)
            else:
                guarded[k] = new_val

        return guarded

    def run(
        self,
        build_bundle_fn: Callable,
        run_kwargs: dict | None = None,
    ) -> AdaptiveResults:
        """
        Parameters
        ----------
        build_bundle_fn : Callable
            fn(params, bar_slice) → BacktestBundle
            L'user reconstruit le bundle depuis les params courants.
            bar_slice = {
                "opens": ..., "highs": ..., "lows": ...,
                "closes": ..., "atrs": ..., "bar_index": ...
            }

        run_kwargs : dict | None
            Kwargs supplémentaires passés à engine.run()
            (session_1, track_mae_mfe, etc.)
        """
        run_kwargs = run_kwargs or {}

        opens     = self.engine.opens
        highs     = self.engine.highs
        lows      = self.engine.lows
        closes    = self.engine.closes
        atrs      = self.engine.atrs
        bar_index = self.engine.bar_index
        n         = len(closes)

        if n < self.warmup_bars + self.step_bars:
            raise ValueError(
                f"Not enough bars ({n}) for warmup_bars={self.warmup_bars} "
                f"+ step_bars={self.step_bars}"
            )

        current_params = self.initial_params.copy()
        all_trades     = []
        params_history = []
        metrics_history = []
        window_ranges  = []

        # ── Calibration initiale sur warmup ──────────────────────
        warmup_slice = {
            "opens":     opens[:self.warmup_bars],
            "highs":     highs[:self.warmup_bars],
            "lows":      lows[:self.warmup_bars],
            "closes":    closes[:self.warmup_bars],
            "atrs":      atrs[:self.warmup_bars],
            "bar_index": bar_index[:self.warmup_bars],
        }

        warmup_bundle = build_bundle_fn(current_params, warmup_slice)

        warmup_rets, warmup_metrics = self.engine.run(
            bundle=warmup_bundle,
            **run_kwargs,
        )

        warmup_trades = warmup_metrics.get("trades_df", pd.DataFrame())

        if include := self.include_warmup_trades:
            if len(warmup_trades) > 0:
                warmup_trades = warmup_trades.copy()
                warmup_trades["window"] = 0
                warmup_trades["is_warmup"] = True
                all_trades.append(warmup_trades)

        # Première adaptation sur le warmup
        if len(warmup_trades) >= self.min_trades_to_adapt:
            try:
                new_params = self.adaptation_fn(
                    trades_df=warmup_trades,
                    current_params=current_params,
                    bar_data=warmup_slice,
                )
                current_params = self._apply_param_guard(
                    current_params, new_params
                )
            except Exception as e:
                print(f"[AdaptiveEngine] adaptation_fn failed on warmup: {e}")

        params_history.append({
            "window": 0,
            "start": 0,
            "end": self.warmup_bars,
            "n_trades": len(warmup_trades),
            "is_warmup": True,
            **{k: v for k, v in current_params.items()
               if isinstance(v, (int, float))},
        })

        # ── Boucle rolling ───────────────────────────────────────
        window_idx = 1
        start = self.warmup_bars

        while start + self.step_bars <= n:
            end = min(start + self.step_bars, n)

            bar_slice = {
                "opens":     opens[start:end],
                "highs":     highs[start:end],
                "lows":      lows[start:end],
                "closes":    closes[start:end],
                "atrs":      atrs[start:end],
                "bar_index": bar_index[start:end],
            }

            # Build bundle avec params courants
            try:
                bundle = build_bundle_fn(current_params, bar_slice)
            except Exception as e:
                print(f"[AdaptiveEngine] build_bundle_fn failed on window {window_idx}: {e}")
                start += self.step_bars
                window_idx += 1
                continue

            # Run backtest sur la fenêtre
            try:
                rets, metrics = self.engine.run(
                    bundle=bundle,
                    **run_kwargs,
                )
            except Exception as e:
                print(f"[AdaptiveEngine] engine.run failed on window {window_idx}: {e}")
                start += self.step_bars
                window_idx += 1
                continue

            trades_window = metrics.get("trades_df", pd.DataFrame())
            window_ranges.append((start, end))

            # Tag des trades
            if len(trades_window) > 0:
                trades_window = trades_window.copy()
                trades_window["window"]    = window_idx
                trades_window["is_warmup"] = False
                all_trades.append(trades_window)

            # Métriques de la fenêtre
            window_metrics = {
                k: v for k, v in metrics.items()
                if k != "trades_df" and isinstance(v, (int, float))
            }
            window_metrics["window"] = window_idx
            window_metrics["start"]  = start
            window_metrics["end"]    = end
            window_metrics["n_trades"] = len(trades_window)
            metrics_history.append(window_metrics)

            # Adaptation sur la fenêtre courante
            # Recalibre sur les dernières warmup_bars
            # (rolling lookback = i - warmup_bars)
            lookback_start = max(0, start - self.warmup_bars)
            lookback_slice = {
                "opens":     opens[lookback_start:end],
                "highs":     highs[lookback_start:end],
                "lows":      lows[lookback_start:end],
                "closes":    closes[lookback_start:end],
                "atrs":      atrs[lookback_start:end],
                "bar_index": bar_index[lookback_start:end],
            }

            if len(trades_window) >= self.min_trades_to_adapt:
                try:
                    new_params = self.adaptation_fn(
                        trades_df=trades_window,
                        current_params=current_params,
                        bar_data=lookback_slice,
                    )
                    current_params = self._apply_param_guard(
                        current_params, new_params
                    )
                except Exception as e:
                    print(
                        f"[AdaptiveEngine] adaptation_fn failed "
                        f"on window {window_idx}: {e}"
                    )
            else:
                print(
                    f"[AdaptiveEngine] window {window_idx}: "
                    f"only {len(trades_window)} trades "
                    f"(min={self.min_trades_to_adapt}) — skipping adaptation"
                )

            # Log params
            params_history.append({
                "window":    window_idx,
                "start":     start,
                "end":       end,
                "n_trades":  len(trades_window),
                "is_warmup": False,
                **{k: v for k, v in current_params.items()
                   if isinstance(v, (int, float))},
            })

            start      += self.step_bars
            window_idx += 1

        # ── Assemblage final ─────────────────────────────────────
        if all_trades:
            final_trades = pd.concat(all_trades, ignore_index=True)
        else:
            final_trades = pd.DataFrame()

        return AdaptiveResults(
            trades_df      = final_trades,
            params_history = pd.DataFrame(params_history),
            metrics_history= metrics_history,
            window_ranges  = window_ranges,
        )
