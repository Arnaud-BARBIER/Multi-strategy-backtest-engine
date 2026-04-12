"""
stateful_config.py
──────────────────
Configuration stateful globale par stratégie de sortie.
Attachée à ExitStrategySpec — agit sur TOUTES les positions de la strat.

Usage :
    from Backtest_Framework.stateful_config import StatefulConfig
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class StatefulConfig:
    """
    Règles stateful globales par stratégie de sortie.
    Toutes les valeurs à 0 / False désactivent la fonctionnalité.

    Ces règles sont évaluées automatiquement dans le moteur à chaque barre,
    sans écrire de code Numba.

    ── Consecutive SL counter ────────────────────────────────────────────────
    Pause automatique après N SL consécutifs.
    Utilise SG_CONSEC_SL et SG_COOLDOWN_UNTIL du stateful engine.

        max_consec_sl                 : nb de SL consécutifs avant pause (0 = désactivé)
        cooldown_bars_after_consec_sl : nb de barres de pause après le trigger

    Exemple :
        max_consec_sl=3, cooldown_bars_after_consec_sl=20
        → après 3 SL de suite, pause de 20 barres avant nouvelle entrée

    ── Exposition globale ────────────────────────────────────────────────────
    Limiter le nombre de positions simultanées par stratégie.
    Utilise SG_TOTAL_EXPOSURE du stateful engine.

        max_simultaneous_positions : nb max de positions ouvertes en même temps
                                     0 = pas de limite

    Exemple :
        max_simultaneous_positions=2
        → la strat ne peut pas avoir plus de 2 positions ouvertes à la fois

    ── Invalidation par régime ───────────────────────────────────────────────
    Invalider toutes les positions ouvertes si le régime change depuis l'entrée.
    Compare SP_REGIME_AT_ENTRY vs SG_CURRENT_REGIME pour chaque position.

        invalidate_on_regime_change : True = invalider si régime change

    Exemple :
        invalidate_on_regime_change=True
        → si une position a été ouverte en régime 0 et le régime passe à 1,
          SP_ENTRY_VALID = 0.0 → sortie immédiate

    ── Rolling winrate ───────────────────────────────────────────────────────
    Pause automatique si le rolling winrate tombe sous un seuil.
    Utilise SG_ROLLING_WINRATE du stateful engine (EMA sur ~20 trades).

        min_rolling_winrate          : seuil minimum (0.0 = désactivé)
        cooldown_bars_if_low_winrate : nb de barres de pause

    Exemple :
        min_rolling_winrate=0.35, cooldown_bars_if_low_winrate=50
        → si le winrate glissant tombe sous 35%, pause de 50 barres

    ── Exemple complet ───────────────────────────────────────────────────────

        from Backtest_Framework import ExitStrategySpec, StatefulConfig

        strat = ExitStrategySpec(
            strategy_id=0,
            stateful_config=StatefulConfig(
                max_consec_sl=3,
                cooldown_bars_after_consec_sl=20,
                max_simultaneous_positions=2,
                invalidate_on_regime_change=True,
                min_rolling_winrate=0.35,
                cooldown_bars_if_low_winrate=50,
            )
        )
    """

    # ── Consecutive SL cooldown ───────────────────────────────────────────
    max_consec_sl: int = 0
    cooldown_bars_after_consec_sl: int = 0

    # ── Exposition globale ────────────────────────────────────────────────
    max_simultaneous_positions: int = 0     # 0 = pas de limite

    # ── Invalidation par régime ───────────────────────────────────────────
    invalidate_on_regime_change: bool = False

    # ── Rolling winrate ───────────────────────────────────────────────────
    min_rolling_winrate: float = 0.0        # 0.0 = désactivé
    cooldown_bars_if_low_winrate: int = 0

    def __post_init__(self):
        if self.max_consec_sl < 0:
            raise ValueError("max_consec_sl doit être >= 0")
        if self.cooldown_bars_after_consec_sl < 0:
            raise ValueError("cooldown_bars_after_consec_sl doit être >= 0")
        if self.max_simultaneous_positions < 0:
            raise ValueError("max_simultaneous_positions doit être >= 0")
        if not (0.0 <= self.min_rolling_winrate <= 1.0):
            raise ValueError("min_rolling_winrate doit être entre 0.0 et 1.0")
        if self.cooldown_bars_if_low_winrate < 0:
            raise ValueError("cooldown_bars_if_low_winrate doit être >= 0")

    @property
    def is_active(self) -> bool:
        """True si au moins une règle stateful est activée."""
        return (
            self.max_consec_sl > 0
            or self.max_simultaneous_positions > 0
            or self.invalidate_on_regime_change
            or self.min_rolling_winrate > 0.0
        )

    def to_rt_array(self) -> list:
        """
        Compiler en vecteur runtime pour backtest_njit.
        Ordre correspondant aux constantes SCFG_* dans core_engine.py.
        """
        return [
            float(self.max_consec_sl),                  # SCFG_MAX_CONSEC_SL
            float(self.cooldown_bars_after_consec_sl),  # SCFG_COOLDOWN_BARS
            float(self.max_simultaneous_positions),     # SCFG_MAX_POSITIONS
            1.0 if self.invalidate_on_regime_change else 0.0,  # SCFG_INVALIDATE_ON_REGIME
            float(self.min_rolling_winrate),            # SCFG_MIN_WINRATE
            float(self.cooldown_bars_if_low_winrate),   # SCFG_WINRATE_COOLDOWN
        ]
