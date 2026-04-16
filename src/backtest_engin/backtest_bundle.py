from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .execution_binding import build_execution_context
from .feature_compiler import compile_features


@dataclass(slots=True)
class BacktestBundle:
    # objets déclaratifs utilisateur
    cfg: Any
    setup_specs: list[Any]
    decision_cfg: Any
    feature_specs: list[Any] = field(default_factory=list)
    exit_profile_specs: list[Any] = field(default_factory=list)
    exit_strategy_specs: list[Any] = field(default_factory=list)
    setup_exit_binding: dict[int, dict[str, int]] = field(default_factory=dict)
    strategy_profile_binding: dict[Any, Any] = field(default_factory=dict)
    strategy_param_names: tuple[str, ...] = field(default_factory=tuple)

    # options de préparation
    include_price_cols: bool = True
    volumes: np.ndarray | None = None

    # sorties préparées
    multi_out: dict[str, Any] | None = None
    signals: np.ndarray | None = None
    selected_setup_id: np.ndarray | None = None
    selected_score: np.ndarray | None = None

    opens: np.ndarray | None = None
    highs: np.ndarray | None = None
    lows: np.ndarray | None = None
    closes: np.ndarray | None = None

    compiled_features: Any = None
    features: np.ndarray | None = None

    execution_context: Any = None

    # debug / inspection
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_prepared(self) -> bool:
        return (
            self.signals is not None
            and self.selected_setup_id is not None
            and self.selected_score is not None
            and self.execution_context is not None
        )


def _extract_price_arrays(engine, multi_out: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if all(k in multi_out for k in ("open", "high", "low", "close")):
        opens = np.asarray(multi_out["open"], dtype=np.float64)
        highs = np.asarray(multi_out["high"], dtype=np.float64)
        lows = np.asarray(multi_out["low"], dtype=np.float64)
        closes = np.asarray(multi_out["close"], dtype=np.float64)
        return opens, highs, lows, closes

    return (
        np.asarray(engine.opens, dtype=np.float64),
        np.asarray(engine.highs, dtype=np.float64),
        np.asarray(engine.lows, dtype=np.float64),
        np.asarray(engine.closes, dtype=np.float64),
    )


def prepare_backtest_bundle(
    engine,
    setup_specs,
    decision_cfg,
    cfg=None,
    feature_specs=None,
    exit_profile_specs=None,
    setup_exit_binding=None,
    strategy_profile_binding=None,
    exit_strategy_specs=None,
    strategy_param_names=(),
    include_price_cols: bool = True,
    volumes=None,
    regime: np.ndarray | None = None,
    regime_policy=None,
    regime_context=None,
):
    cfg = cfg if cfg is not None else engine.cfg
    feature_specs = feature_specs if feature_specs is not None else []
    exit_profile_specs = exit_profile_specs if exit_profile_specs is not None else []
    setup_exit_binding = setup_exit_binding if setup_exit_binding is not None else {}
    strategy_profile_binding = strategy_profile_binding if strategy_profile_binding is not None else {}
    exit_strategy_specs = exit_strategy_specs if exit_strategy_specs is not None else []

    bundle = BacktestBundle(
        cfg=cfg,
        setup_specs=setup_specs,
        decision_cfg=decision_cfg,
        feature_specs=feature_specs,
        exit_profile_specs=exit_profile_specs,
        exit_strategy_specs=exit_strategy_specs,
        setup_exit_binding=setup_exit_binding,
        strategy_profile_binding=strategy_profile_binding,
        strategy_param_names=tuple(strategy_param_names),
        include_price_cols=include_price_cols,
        volumes=volumes,
    )

    if regime_context is None and regime is not None and regime_policy is not None:
        from .regime_policy import build_regime_context

        regime_context = build_regime_context(
            regime=regime,
            policy=regime_policy,
            setup_specs=setup_specs,
        )

    multi_out = engine.prepare_multi_setup_signals(
        setup_specs=setup_specs,
        decision_cfg=decision_cfg,
        include_price_cols=include_price_cols,
        regime_context=regime_context,
    )

    signals = np.asarray(multi_out["signals"], dtype=np.int8)
    selected_setup_id = np.asarray(multi_out["selected_setup_id"], dtype=np.int32)
    selected_score = np.asarray(multi_out["selected_score"], dtype=np.float64)

    opens, highs, lows, closes = _extract_price_arrays(engine, multi_out)

    if len(feature_specs) > 0:
        compiled_features = compile_features(
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            feature_specs=feature_specs,
        )
        features = np.asarray(compiled_features.matrix, dtype=np.float64)
    else:
        compiled_features = None
        features = np.zeros((len(signals), 0), dtype=np.float64)

    if len(exit_profile_specs) > 0 or len(exit_strategy_specs) > 0:
        execution_context = build_execution_context(
            cfg=cfg,
            exit_profile_specs=exit_profile_specs,
            setup_exit_binding=setup_exit_binding,
            strategy_profile_binding=strategy_profile_binding,
            n_setups=len(setup_specs),
            exit_strategy_specs=exit_strategy_specs,
            n_strategies=len(exit_strategy_specs),
            compiled_features=compiled_features,
            strategy_param_names=tuple(strategy_param_names),
        )
    else:
        execution_context = None

    bundle.multi_out = multi_out
    bundle.signals = signals
    bundle.selected_setup_id = selected_setup_id
    bundle.selected_score = selected_score

    bundle.opens = opens
    bundle.highs = highs
    bundle.lows = lows
    bundle.closes = closes

    bundle.compiled_features = compiled_features
    bundle.features = features
    bundle.execution_context = execution_context

    bundle.meta = {
        "n_bars": len(signals),
        "n_setups": len(setup_specs),
        "n_features": 0 if features is None else features.shape[1],
        "n_exit_profiles": len(exit_profile_specs),
        "n_exit_strategies": len(exit_strategy_specs),
    }

    return bundle


def bundle_signal_df(engine, bundle: BacktestBundle) -> pd.DataFrame:
    df = engine._base_price_df().copy()
    df["signal"] = bundle.signals
    df["selected_setup_id"] = bundle.selected_setup_id
    df["selected_score"] = bundle.selected_score
    return df


def bundle_feature_df(bundle: BacktestBundle) -> pd.DataFrame:
    if bundle.compiled_features is None:
        return pd.DataFrame()

    return bundle.compiled_features.dataframe()

@dataclass(slots=True)
class SignalPrep:
    multi_out: dict
    signals: np.ndarray
    selected_setup_id: np.ndarray
    selected_score: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    df_signal: pd.DataFrame | None = None

def print_bundle_summary(bundle: BacktestBundle) -> None:
    print("Bundle prepared:", bundle.is_prepared)
    print("Bars:", bundle.meta.get("n_bars"))
    print("Setups:", bundle.meta.get("n_setups"))
    print("Features:", bundle.meta.get("n_features"))
    print("Exit profiles:", bundle.meta.get("n_exit_profiles"))
    print("Exit strategies:", bundle.meta.get("n_exit_strategies"))
    if bundle.signals is not None:
        print("Non-zero signals:", int((bundle.signals != 0).sum()))
    if bundle.selected_setup_id is not None:
        used = np.unique(bundle.selected_setup_id[bundle.selected_setup_id >= 0])
        print("Used setup ids:", used)
