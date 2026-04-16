"""
StrategyEngine — Interface haut niveau au-dessus de NJITEngine.

Simplifie le câblage sans rien sacrifier en personnalisation.
Tout ce qui existait avant reste accessible via NJITEngine directement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .NJITEngine import NJITEngine
from .pipeline_config import BacktestConfig, DataPipeline
from .multi_setup_layer import SetupSpec, DecisionConfig
from .feature_compiler import FeatureSpec, compile_features
from .Exit_system import ExitProfileSpec
from .exit_strategy_system import ExitStrategySpec
from .execution_binding import build_execution_context


# ══════════════════════════════════════════════════════════════
# Structures internes (non exposées à l'user)
# ══════════════════════════════════════════════════════════════

@dataclass
class _SetupEntry:
    fn: Callable
    params: dict
    name: str


@dataclass
class _FeatureEntry:
    name: str
    fn: Callable
    params: dict


@dataclass
class _Binding:
    setup_name: str
    profile_name: str
    strategy_name: str
    allowed_profiles: list[str]


# ══════════════════════════════════════════════════════════════
# StrategyEngine
# ══════════════════════════════════════════════════════════════

class StrategyEngine(NJITEngine):
    """
    Interface haut niveau pour construire et runner un backtest.

    Usage:
        engine = StrategyEngine(pipeline, "XAUUSD_M5", "2023-01-02", "2025-01-01",
            tp_pct=0.004, sl_pct=0.004, session_1=("08:00","12:00"), ...)

        engine.add_setup(fn=my_setup, ema_period=30, score=1.0, name="fast")
        engine.add_feature("rsi_14", rsi_feature, period=14)
        engine.add_profile("p0", tp_pct=0.004, sl_pct=0.004, max_holding_bars=30)
        engine.add_strategy("s0", type=EXIT_STRAT_INSTANT, features=["rsi_14"], params={...})
        engine.bind(setup="fast", profile="p0", strategy="s0", allowed_profiles=["p0"])

        rets, metrics = engine.run(min_score=0.5)
    """

    def __init__(
        self,
        pipeline: DataPipeline,
        ticker: str,
        start: str,
        end: str,
        # ── Params BacktestConfig ──────────────────────────
        atr_period: int = 14,
        timezone_shift: int = 1,
        period_1: int = 50,
        period_2: int = 100,
        entry_delay: int = 1,
        session_1=None,
        session_2=None,
        session_3=None,
        max_gap_signal: float = 0.0,
        max_gap_entry: float = 0.0,
        candle_size_filter: bool = False,
        min_size_pct: float = 0.0,
        max_size_pct: float = 1.0,
        prev_candle_direction: bool = False,
        multi_entry: bool = True,
        reverse_mode: bool = False,
        cooldown_entries: int = 0,
        cooldown_bars: int = 0,
        cooldown_mode: int = 1,
        me_max: int = 0,
        me_period: int = 0,
        me_reset_mode: int = 0,
        entry_on_close: bool = False,
        entry_on_signal_close_price: bool = False,
        tp_pct: float = 0.01,
        sl_pct: float = 0.005,
        use_atr_sl_tp: int = 0,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.0,
        allow_exit_on_entry_bar: bool = True,
        use_ema1_tp: bool = False,
        use_ema2_tp: bool = False,
        use_ema_cross_tp: bool = False,
        use_exit_signal: bool = False,
        exit_delay: int = 1,
        be_trigger_pct: float = 0.0,
        be_offset_pct: float = 0.0,
        be_delay_bars: int = 0,
        trailing_trigger_pct: float = 0.0,
        runner_trailing_mult: float = 2.0,
        track_mae_mfe: bool = True,
        hold_minutes: int = 0,
        bar_duration_min: int = 5,
        commission_pct: float = 0.0,
        commission_per_lot_usd: float = 0.0,
        contract_size: float = 1.0,
        spread_pct: float = 0.0,
        spread_abs: float = 0.0,
        slippage_pct: float = 0.0,
        alpha: float = 5.0,
        period_freq: str = "ME",
        return_df_after: bool = False,
        plot: bool = False,
        crypto: bool = False,
        full_df_after: bool = False,
        window_before: int = 200,
        window_after: int = 50,
        max_holding_bars: int = 0,
        forced_flat_frequency=None,
        forced_flat_time=None,
        max_tp: int = 0,
        tp_period_mode: int = 0,
        tp_period_bars: int = 0,
        max_sl: int = 0,
        sl_period_mode: int = 0,
        sl_period_bars: int = 0,
        # ── Params NJITEngine ─────────────────────────────
        MAX_TRADES: int = 50_000,
        MAX_POS: int = 500,
    ):
        # Construire BacktestConfig en interne
        cfg = BacktestConfig(
            multi_setup_mode=True,  # détecté automatiquement au run()
            timezone_shift=timezone_shift,
            atr_period=atr_period,
            period_1=period_1,
            period_2=period_2,
            entry_delay=entry_delay,
            session_1=session_1,
            session_2=session_2,
            session_3=session_3,
            max_gap_signal=max_gap_signal,
            max_gap_entry=max_gap_entry,
            candle_size_filter=candle_size_filter,
            min_size_pct=min_size_pct,
            max_size_pct=max_size_pct,
            prev_candle_direction=prev_candle_direction,
            multi_entry=multi_entry,
            reverse_mode=reverse_mode,
            cooldown_entries=cooldown_entries,
            cooldown_bars=cooldown_bars,
            cooldown_mode=cooldown_mode,
            me_max=me_max,
            me_period=me_period,
            me_reset_mode=me_reset_mode,
            entry_on_close=entry_on_close,
            entry_on_signal_close_price=entry_on_signal_close_price,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            use_atr_sl_tp=use_atr_sl_tp,
            tp_atr_mult=tp_atr_mult,
            sl_atr_mult=sl_atr_mult,
            allow_exit_on_entry_bar=allow_exit_on_entry_bar,
            use_ema1_tp=use_ema1_tp,
            use_ema2_tp=use_ema2_tp,
            use_ema_cross_tp=use_ema_cross_tp,
            use_exit_signal=use_exit_signal,
            exit_delay=exit_delay,
            be_trigger_pct=be_trigger_pct,
            be_offset_pct=be_offset_pct,
            be_delay_bars=be_delay_bars,
            trailing_trigger_pct=trailing_trigger_pct,
            runner_trailing_mult=runner_trailing_mult,
            track_mae_mfe=track_mae_mfe,
            hold_minutes=hold_minutes,
            bar_duration_min=bar_duration_min,
            commission_pct=commission_pct,
            commission_per_lot_usd=commission_per_lot_usd,
            contract_size=contract_size,
            spread_pct=spread_pct,
            spread_abs=spread_abs,
            slippage_pct=slippage_pct,
            alpha=alpha,
            period_freq=period_freq,
            return_df_after=return_df_after,
            plot=plot,
            crypto=crypto,
            full_df_after=full_df_after,
            window_before=window_before,
            window_after=window_after,
            max_holding_bars=max_holding_bars,
            forced_flat_frequency=forced_flat_frequency,
            forced_flat_time=forced_flat_time,
            max_tp=max_tp,
            tp_period_mode=tp_period_mode,
            tp_period_bars=tp_period_bars,
            max_sl=max_sl,
            sl_period_mode=sl_period_mode,
            sl_period_bars=sl_period_bars,
        )

        super().__init__(
            pipeline=pipeline,
            ticker=ticker,
            start=start,
            end=end,
            cfg=cfg,
            atr_period=atr_period,
            MAX_TRADES=MAX_TRADES,
            MAX_POS=MAX_POS,
        )

        # ── Registres internes ────────────────────────────
        self._setups: list[_SetupEntry]       = []
        self._features: list[_FeatureEntry]   = []
        self._profiles: dict[str, ExitProfileSpec]    = {}
        self._strategies: dict[str, ExitStrategySpec] = {}
        self._bindings: list[_Binding]        = []

        # Index pour les IDs numériques (ordre d'ajout)
        self._profile_name_to_id: dict[str, int]   = {}
        self._strategy_name_to_id: dict[str, int]  = {}
        self._setup_name_to_id: dict[str, int]     = {}

    # ══════════════════════════════════════════════════════════
    # API publique — construction
    # ══════════════════════════════════════════════════════════

    def add_setup(self, fn: Callable, name: str | None = None, **params) -> "StrategyEngine":
        """
        Enregistre un setup.

        engine.add_setup(fn=my_setup, name="fast", ema_period=30, score=1.0)
        """
        setup_id = len(self._setups)
        # setup_id injecté automatiquement dans params pour le moteur
        params["setup_id"] = setup_id

        auto_name = name if name else f"setup_{setup_id}"
        self._setups.append(_SetupEntry(fn=fn, params=params, name=auto_name))
        self._setup_name_to_id[auto_name] = setup_id
        return self

    def add_feature(self, name: str, fn: Callable, **params) -> "StrategyEngine":
        """
        Enregistre une feature.

        engine.add_feature("rsi_14", rsi_feature, period=14)
        """
        self._features.append(_FeatureEntry(name=name, fn=fn, params=params))
        return self

    def add_profile(self, name: str, **kwargs) -> "StrategyEngine":
        """
        Enregistre un exit profile.
        Accepte tous les champs de ExitProfileSpec comme kwargs,
        plus partial=, pyramid=, averaging=, phases= directement.

        engine.add_profile("p0",
            tp_pct=0.004, sl_pct=0.004, max_holding_bars=30,
            partial=PartialConfig(...),
        )
        """

        # Extraire les configs de position management
        partial   = kwargs.pop("partial",   None)
        pyramid   = kwargs.pop("pyramid",   None)
        averaging = kwargs.pop("averaging", None)
        phases    = kwargs.pop("phases",    [])

        kwargs.setdefault("be_trigger_pct",       0.0)
        kwargs.setdefault("be_offset_pct",        0.0)
        kwargs.setdefault("be_delay_bars",        0)
        kwargs.setdefault("trailing_trigger_pct", 0.0)
        kwargs.setdefault("runner_trailing_mult", 0.0)
        kwargs.setdefault("use_atr_sl_tp",        0)
        kwargs.setdefault("tp_atr_mult",          2.0)
        kwargs.setdefault("sl_atr_mult",          1.0)
        kwargs.setdefault("max_holding_bars",     0)

        spec = ExitProfileSpec(
            name=name,
            partial_config=partial,
            pyramid_config=pyramid,
            averaging_config=averaging,
            phases=phases,
            **kwargs,
        )

        self._profiles[name] = spec
        self._profile_name_to_id[name] = len(self._profile_name_to_id)
        return self

    def add_strategy(
        self,
        name: str,
        strategy_type: int,
        features: list[str],
        params: dict,
        backend: int = 0,  # STRAT_BACKEND_NUMBA
        stateful=None,
    ) -> "StrategyEngine":
        """
        Enregistre une exit strategy.

        engine.add_strategy("s0",
            strategy_type=EXIT_STRAT_INSTANT,
            features=["rsi_14"],
            params={"target_profile_id": 0, "threshold_1": 70.0},
            stateful=StatefulConfig(...),
        )
        """
        strategy_id = len(self._strategies)

        spec = ExitStrategySpec(
            strategy_id=strategy_id,
            name=name,
            strategy_type=strategy_type,
            backend=backend,
            feature_names=features,
            params=params,
            stateful_config=stateful,
        )

        self._strategies[name] = spec
        self._strategy_name_to_id[name] = strategy_id
        return self

    def bind(
        self,
        setup: str,
        profile: str,
        strategy: str,
        allowed_profiles: list[str],
    ) -> "StrategyEngine":
        """
        Câble un setup à un profil de sortie et une stratégie.

        engine.bind(
            setup="fast",
            profile="p0",
            strategy="s0",
            allowed_profiles=["p0", "p1"],
        )
        """
        # Validation noms
        if setup not in self._setup_name_to_id:
            raise ValueError(f"Setup '{setup}' not registered. Call add_setup() first.")
        if profile not in self._profiles:
            raise ValueError(f"Profile '{profile}' not registered. Call add_profile() first.")
        if strategy not in self._strategies:
            raise ValueError(f"Strategy '{strategy}' not registered. Call add_strategy() first.")
        for ap in allowed_profiles:
            if ap not in self._profiles:
                raise ValueError(f"Allowed profile '{ap}' not registered. Call add_profile() first.")

        self._bindings.append(_Binding(
            setup_name=setup,
            profile_name=profile,
            strategy_name=strategy,
            allowed_profiles=allowed_profiles,
        ))
        return self

    # ══════════════════════════════════════════════════════════
    # run() — compile tout et délègue à NJITEngine.run()
    # ══════════════════════════════════════════════════════════

    def run(
        self,
        # ── Décision de run ───────────────────────────────
        min_score: float = 0.5,
        allow_long: bool = True,
        allow_short: bool = True,
        tie_policy: int = 0,
        # ── Overrides run-time (tous optionnels) ──────────
        signals=None,          # override si tu veux passer tes propres signaux
        regime=None,
        regime_policy=None,
        regime_context=None,
        entry_limit_prices=None,
        limit_expiry_bars: int = 5,
        tp_prices=None,
        sl_prices=None,
        check_filters_on_fill: bool = True,
        exit_signals=None,
        signal_tags=None,
        exit_ema1=None,
        exit_ema2=None,
        # ── Overrides params (tous optionnels) ────────────
        tp_pct=None, sl_pct=None,
        be_trigger_pct=None, be_offset_pct=None, be_delay_bars=None,
        trailing_trigger_pct=None, runner_trailing_mult=None,
        max_holding_bars=None,
        session_1=None, session_2=None, session_3=None,
        track_mae_mfe=None, hold_minutes=None, bar_duration_min=None,
        commission_pct=None, commission_per_lot_usd=None, contract_size=None,
        spread_pct=None, spread_abs=None, slippage_pct=None,
        max_tp=None, tp_period_mode=None, tp_period_bars=None,
        max_sl=None, sl_period_mode=None, sl_period_bars=None,
        plot: bool = False,
        return_df_after: bool = False,
        crypto: bool = False,
        period_freq=None,
        inspection=None,
        **kwargs,  # tout autre override passé directement à NJITEngine.run()
    ):
        # ── Validation ────────────────────────────────────
        multi_setup_mode = len(self._setups) > 1

        if len(self._setups) == 0:
            raise ValueError("No setups registered. Call add_setup() first.")
        if len(self._bindings) > 0:
            if len(self._profiles) == 0:
                raise ValueError("Bindings registered but no profiles. Call add_profile() first.")
            if len(self._strategies) == 0:
                raise ValueError("Bindings registered but no strategies. Call add_strategy() first.")

        use_exit_system = len(self._bindings) > 0

        # ── 1. Compiler les setup specs ───────────────────
        setup_specs = [
            SetupSpec(fn=e.fn, params=e.params, name=e.name)
            for e in self._setups
        ]

        decision_cfg = DecisionConfig(
            min_score=min_score,
            allow_long=allow_long,
            allow_short=allow_short,
            tie_policy=tie_policy,
        )

        # ── 2. Préparer les signaux ───────────────────────
        if signals is None:
            prep = self.prepare_signal_inputs(
                setup_specs=setup_specs,
                decision_cfg=decision_cfg,
                regime=regime,
                regime_policy=regime_policy,
                regime_context=regime_context,
            )
            signals           = prep.signals
            selected_setup_id = prep.selected_setup_id
            selected_score    = prep.selected_score
            _opens  = prep.opens
            _highs  = prep.highs
            _lows   = prep.lows
            _closes = prep.closes
        else:
            selected_setup_id = None
            selected_score    = None
            multi_setup_mode  = False
            _opens  = self.opens
            _highs  = self.highs
            _lows   = self.lows
            _closes = self.closes

        # ── 3. Compiler les features sur les mêmes prix que les signaux
        if len(self._features) > 0:
            feature_specs = [
                FeatureSpec(name=e.name, fn=e.fn, params=e.params)
                for e in self._features
            ]
            compiled_features = compile_features(
                opens=_opens,
                highs=_highs,
                lows=_lows,
                closes=_closes,
                volumes=None,
                feature_specs=feature_specs,
            )
            features = compiled_features.matrix
        else:
            compiled_features = None
            features = None

        # ── 4. Compiler execution_context ─────────────────
        execution_context = None

        if use_exit_system:
            # Construire setup_exit_binding depuis les bindings
            setup_exit_binding = {}
            for b in self._bindings:
                sid = self._setup_name_to_id[b.setup_name]
                setup_exit_binding[sid] = {
                    "exit_profile_id":   self._profile_name_to_id[b.profile_name],
                    "exit_strategy_id":  self._strategy_name_to_id[b.strategy_name],
                }

            # Construire strategy_profile_binding depuis les bindings
            strategy_profile_binding = {}
            for b in self._bindings:
                strat_id = self._strategy_name_to_id[b.strategy_name]
                default_pid = self._profile_name_to_id[b.profile_name]
                allowed_pids = [self._profile_name_to_id[ap] for ap in b.allowed_profiles]

                if strat_id not in strategy_profile_binding:
                    strategy_profile_binding[strat_id] = {
                        "default_profile_id":  default_pid,
                        "allowed_profile_ids": allowed_pids,
                    }
                else:
                    # Si même strat liée à plusieurs setups — fusionner les allowed
                    existing = strategy_profile_binding[strat_id]["allowed_profile_ids"]
                    for pid in allowed_pids:
                        if pid not in existing:
                            existing.append(pid)

            # Inférer strategy_param_names depuis les clés des params
            all_param_keys: list[str] = []
            for spec in self._strategies.values():
                for k in spec.params.keys():
                    if k not in all_param_keys:
                        all_param_keys.append(k)
            strategy_param_names = tuple(all_param_keys)
            if inspection:
                print("setup_exit_binding:", setup_exit_binding)
                print("strategy_profile_binding:", strategy_profile_binding)
                print("n_setups:", len(self._setups))
                print("n_strategies:", len(self._strategies))
                print("profile_name_to_id:", self._profile_name_to_id)
                print("strategy_name_to_id:", self._strategy_name_to_id)
            execution_context = build_execution_context(
                cfg=self.cfg,
                exit_profile_specs=list(self._profiles.values()),
                setup_exit_binding=setup_exit_binding,
                strategy_profile_binding=strategy_profile_binding,
                n_setups=len(self._setups),
                exit_strategy_specs=list(self._strategies.values()),
                n_strategies=len(self._strategies),
                compiled_features=compiled_features,
                strategy_param_names=strategy_param_names,
            )

            if inspection:
                self._last_execution_context = execution_context
                for i, spec in enumerate(list(self._profiles.values())):
                    print(f"Profile [{i}] {spec.name}:")
                    print(f"  tp={spec.tp_pct}, sl={spec.sl_pct}, max_hold={spec.max_holding_bars}")
                    print(f"  be_trigger={spec.be_trigger_pct}, be_offset={spec.be_offset_pct}, be_delay={spec.be_delay_bars}")
                    print(f"  trailing={spec.trailing_trigger_pct}, runner_mult={spec.runner_trailing_mult}")

        # ── 5. Déléguer à NJITEngine.run() ────────────────
        return super().run(
            signals=signals,
            use_exit_system=use_exit_system,
            execution_context=execution_context,
            features=features,
            selected_setup_id=selected_setup_id,
            selected_score=selected_score,
            multi_setup_mode=multi_setup_mode,
            setup_specs=setup_specs if regime_policy is not None else None,
            regime=regime,
            regime_policy=regime_policy,
            regime_context=regime_context,
            entry_limit_prices=entry_limit_prices,
            limit_expiry_bars=limit_expiry_bars,
            tp_prices=tp_prices,
            sl_prices=sl_prices,
            check_filters_on_fill=check_filters_on_fill,
            exit_signals=exit_signals,
            signal_tags=signal_tags,
            exit_ema1=exit_ema1,
            exit_ema2=exit_ema2,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            be_trigger_pct=be_trigger_pct,
            be_offset_pct=be_offset_pct,
            be_delay_bars=be_delay_bars,
            trailing_trigger_pct=trailing_trigger_pct,
            runner_trailing_mult=runner_trailing_mult,
            max_holding_bars=max_holding_bars,
            session_1=session_1,
            session_2=session_2,
            session_3=session_3,
            track_mae_mfe=track_mae_mfe,
            hold_minutes=hold_minutes,
            bar_duration_min=bar_duration_min,
            commission_pct=commission_pct,
            commission_per_lot_usd=commission_per_lot_usd,
            contract_size=contract_size,
            spread_pct=spread_pct,
            spread_abs=spread_abs,
            slippage_pct=slippage_pct,
            max_tp=max_tp,
            tp_period_mode=tp_period_mode,
            tp_period_bars=tp_period_bars,
            max_sl=max_sl,
            sl_period_mode=sl_period_mode,
            sl_period_bars=sl_period_bars,
            plot=plot,
            return_df_after=return_df_after,
            crypto=crypto,
            period_freq=period_freq,
            **kwargs,
        )

    # ══════════════════════════════════════════════════════════
    # Helpers inspection
    # ══════════════════════════════════════════════════════════

    def summary(self) -> None:
        """Affiche un résumé de la configuration enregistrée."""
        print(f"StrategyEngine — {len(self._setups)} setup(s), "
              f"{len(self._features)} feature(s), "
              f"{len(self._profiles)} profile(s), "
              f"{len(self._strategies)} strateg(ies), "
              f"{len(self._bindings)} binding(s)")

        if self._setups:
            print("\nSetups:")
            for s in self._setups:
                print(f"  [{self._setup_name_to_id[s.name]}] {s.name}")

        if self._features:
            print("\nFeatures:")
            for f in self._features:
                print(f"  {f.name}")

        if self._profiles:
            print("\nProfiles:")
            for name, pid in self._profile_name_to_id.items():
                print(f"  [{pid}] {name}")

        if self._strategies:
            print("\nStrategies:")
            for name, sid in self._strategy_name_to_id.items():
                print(f"  [{sid}] {name}")

        if self._bindings:
            print("\nBindings:")
            for b in self._bindings:
                sid  = self._setup_name_to_id[b.setup_name]
                pid  = self._profile_name_to_id[b.profile_name]
                stid = self._strategy_name_to_id[b.strategy_name]
                allowed = [f"{self._profile_name_to_id[ap]}:{ap}" for ap in b.allowed_profiles]
                print(f"  setup[{sid}:{b.setup_name}] → "
                      f"profile[{pid}:{b.profile_name}] + "
                      f"strat[{stid}:{b.strategy_name}] "
                      f"(allowed: {allowed})")

    @property
    def profile_ids(self) -> dict[str, int]:
        return dict(self._profile_name_to_id)

    @property
    def strategy_ids(self) -> dict[str, int]:
        return dict(self._strategy_name_to_id)

    @property
    def setup_ids(self) -> dict[str, int]:
        return dict(self._setup_name_to_id)
