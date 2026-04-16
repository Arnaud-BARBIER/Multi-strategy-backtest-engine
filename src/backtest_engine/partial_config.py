"""
partial_config.py
─────────────────
Configuration des distributions d'ordres et des phases pour ExitProfileSpec.

Contient :
    DistributionFn  — fonction mathématique pour spacing et/ou sizing
    PartialConfig   — configuration des sorties partielles
    PyramidConfig   — configuration du pyramiding
    AveragingConfig — configuration de l'averaging
    PhaseSpec       — configuration d'une phase du trade

Usage :
    from Backtest_Framework.partial_config import (
        DistributionFn,
        PartialConfig, PyramidConfig, AveragingConfig,
        PhaseSpec,
    )
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import math


# ══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DistributionFn:
    """
    Fonction mathématique qui contrôle l'espacement et/ou le sizing des ordres.

    La variable x représente la distance depuis l'entrée selon distance_ref :
        "rr"       → distance en multiples du SL initial (RR)
        "mfe_pct"  → MFE en % depuis l'entrée
        "mae_pct"  → MAE en % (valeur absolue, toujours positive)
        "atr"      → distance en multiples de l'ATR
        "index"    → numéro de l'ordre (0, 1, 2...)

    f(x) → float représente selon apply_to :
        "sizing"  → fraction à utiliser pour cet ordre
        "spacing" → distance au prochain ordre (en unités de distance_ref)
        "both"    → contrôle les deux simultanément

    Logique dérivée :
        f'(x) < 0 → décélération (tailles ou espacements décroissants)
        f'(x) > 0 → accélération (tailles ou espacements croissants)
        f'(x) = 0 → constant (equal distribution)

    ── Modes disponibles ─────────────────────────────────────────────────────

    "linear"
        f(x) = max(0, start - slope * x)
        Décroissance linéaire. Contrôle la pente avec slope.
        Ex: start=1.0, slope=0.25 → [1.0, 0.75, 0.5, 0.25] à x=[0,1,2,3]

    "expo"
        f(x) = start * ratio^x
        Décroissance exponentielle. Chaque couche = ratio * la précédente.
        Ex: start=0.5, ratio=0.5 → [0.5, 0.25, 0.125] pour n_levels=3

    "log"
        f(x) = start / log(x + e)
        Décroissance logarithmique, plus douce que expo.

    "sqrt"
        f(x) = start / sqrt(x + 1)
        Entre linear et log.

    "equal"
        f(x) = start (constant)
        Même fraction à chaque niveau.

    "custom_points"
        Interpolation linéaire entre les points fournis.
        custom_points = [(x0, y0), (x1, y1), ...]
        x = distance, y = fraction/espacement
        L'espacement est déduit des x quand apply_to="both".

    "callable"
        f(x) défini par l'user via custom_fn.
        custom_fn : Callable[[float], float]
        Si f'(x) > 0 → accélération, si f'(x) < 0 → décélération.

    ── Exemples ──────────────────────────────────────────────────────────────

        # Expo sur sizing — fractions décroissantes
        DistributionFn(mode="expo", apply_to="sizing", ratio=0.5)
        # → fractions [0.5, 0.25, 0.125] à chaque niveau

        # Custom points — spacing ET sizing définis ensemble
        DistributionFn(
            mode="custom_points",
            apply_to="both",
            distance_ref="rr",
            custom_points=[(1.0, 0.5), (2.5, 0.3), (4.0, 0.2)]
        )
        # → sortie à RR 1.0 (50%), RR 2.5 (30%), RR 4.0 (20%)

        # Callable — accélération linéaire (averaging agressif)
        DistributionFn(
            mode="callable",
            apply_to="sizing",
            distance_ref="mae_pct",
            custom_fn=lambda x: 0.2 * x + 0.1
        )
        # → plus la MAE est grande, plus la couche est importante
    """

    mode: str = "expo"
    apply_to: str = "sizing"        # "sizing", "spacing", "both"
    distance_ref: str = "rr"        # "rr", "mfe_pct", "mae_pct", "atr", "index"

    # Paramètres presets
    start: float = 1.0
    ratio: float = 0.5              # expo : f(x) = start * ratio^x
    slope: float = 0.25             # linear : f(x) = start - slope * x

    # Mode custom_points
    custom_points: list = None      # [(x0, y0), (x1, y1), ...]

    # Mode callable
    custom_fn: Callable = None      # f(x) → float

    def __post_init__(self):
        valid_modes = {"linear", "expo", "log", "sqrt", "equal",
                       "custom_points", "callable"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode doit être parmi {valid_modes}, reçu: {self.mode}")

        valid_apply = {"sizing", "spacing", "both"}
        if self.apply_to not in valid_apply:
            raise ValueError(f"apply_to doit être parmi {valid_apply}")

        valid_refs = {"rr", "mfe_pct", "mae_pct", "atr", "index"}
        if self.distance_ref not in valid_refs:
            raise ValueError(f"distance_ref doit être parmi {valid_refs}")

        if self.mode == "custom_points" and self.custom_points is None:
            raise ValueError("custom_points requis quand mode='custom_points'")

        if self.mode == "callable" and self.custom_fn is None:
            raise ValueError("custom_fn requis quand mode='callable'")

    def evaluate(self, x: float) -> float:
        """
        Évaluer f(x) pour un x donné.
        Utilisé lors de la compilation des matrices runtime.
        """
        if self.mode == "linear":
            return max(0.0, self.start - self.slope * x)
        elif self.mode == "expo":
            return self.start * (self.ratio ** x)
        elif self.mode == "log":
            return self.start / math.log(x + math.e)
        elif self.mode == "sqrt":
            return self.start / math.sqrt(x + 1.0)
        elif self.mode == "equal":
            return self.start
        elif self.mode == "custom_points":
            return self._interpolate(x)
        elif self.mode == "callable":
            return float(self.custom_fn(x))
        return self.start

    def _interpolate(self, x: float) -> float:
        """Interpolation linéaire entre les custom_points."""
        pts = sorted(self.custom_points, key=lambda p: p[0])
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return pts[-1][1]

    def get_spacings(self, n_levels: int) -> list[float]:
        """
        Calculer les positions de spacing pour N niveaux.
        Utilisé quand apply_to="spacing" ou "both".
        Retourne une liste de distances [d0, d1, d2, ...].
        """
        if self.mode == "custom_points" and self.apply_to in ("spacing", "both"):
            return [p[0] for p in sorted(self.custom_points, key=lambda p: p[0])]
        # Pour les autres modes, espacements réguliers calculés
        spacings = []
        for i in range(n_levels):
            spacings.append(self.evaluate(float(i)))
        return spacings

    def get_fractions(self, n_levels: int) -> list[float]:
        """
        Calculer les fractions pour N niveaux.
        Utilisé quand apply_to="sizing" ou "both".
        """
        if self.mode == "custom_points" and self.apply_to in ("sizing", "both"):
            pts = sorted(self.custom_points, key=lambda p: p[0])
            return [p[1] for p in pts[:n_levels]]
        fractions = []
        for i in range(n_levels):
            fractions.append(self.evaluate(float(i)))
        return fractions

    def to_rt_encoding(self) -> tuple[int, float, float]:
        """
        Encoder en (dist_mode_int, param1, param2) pour les matrices runtime.
        Les custom_points et callable sont pré-calculés à la compilation.
        """
        mode_map = {
            "linear": 0,
            "expo": 1,
            "log": 2,
            "sqrt": 3,
            "equal": 4,
            "custom_points": 5,
            "callable": 6,
        }
        dist_mode = mode_map[self.mode]

        if self.mode == "expo":
            return dist_mode, self.ratio, self.start
        elif self.mode == "linear":
            return dist_mode, self.slope, self.start
        else:
            return dist_mode, self.start, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PARTIAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PartialConfig:
    """
    Configuration des sorties partielles automatiques.

    Le moteur évalue cette config à chaque barre pour chaque position.
    Quand le trigger du niveau courant est atteint → ExitPartial automatique.

    Paramètres :
        n_levels                   : nombre de sorties partielles
        spacing                    : trigger ou liste de triggers
                                     None → déduit de sizing si apply_to="both"
                                     OnRR(1.0) → sortie à RR 1, 2, 3...
                                     [OnRR(1.0), OnRR(2.5), OnRR(4.0)] → custom
        sizing                     : DistributionFn pour les fractions
                                     None → fractions égales (1/n_levels)
        move_sl_to_be_after_first  : déplacer le SL au BE après le 1er partial
        ref                        : référence pour les fractions
                                     "remaining" (défaut) → % de ce qui reste
                                     "original"           → % de la taille initiale

    Exemples :

        # Simple — 2 sorties à RR 1 et 2, fractions expo
        PartialConfig(
            n_levels=2,
            spacing=OnRR(1.0),
            sizing=DistributionFn(mode="expo", ratio=0.5),
        )
        # → à RR 1 sortir 50%, à RR 2 sortir 25% du remaining

        # Custom — spacings et fractions définis ensemble
        PartialConfig(
            n_levels=3,
            sizing=DistributionFn(
                mode="custom_points",
                apply_to="both",
                distance_ref="rr",
                custom_points=[(1.0, 0.5), (2.5, 0.3), (4.0, 0.2)]
            ),
        )
        # → à RR 1.0 → 50%, à RR 2.5 → 30%, à RR 4.0 → 20%

        # Feature-driven — sortie partielle sur RSI overbought
        PartialConfig(
            n_levels=1,
            spacing=OnFeature("rsi_14", "gt", 70.0),
            sizing=DistributionFn(mode="equal", start=0.5),
        )
    """

    n_levels: int = 2
    spacing: Any = None             # trigger, list de triggers, ou None
    sizing: Any = None              # DistributionFn ou None
    move_sl_to_be_after_first: bool = True
    ref: str = "remaining"          # "remaining", "original"

    def __post_init__(self):
        if self.n_levels < 1:
            raise ValueError("n_levels doit être >= 1")
        if self.ref not in ("remaining", "original"):
            raise ValueError("ref doit être 'remaining' ou 'original'")

    def get_fractions(self) -> list[float]:
        """Calculer les fractions pour tous les niveaux."""
        if self.sizing is None:
            # Fractions égales
            f = 1.0 / self.n_levels
            return [f] * self.n_levels
        return self.sizing.get_fractions(self.n_levels)

    def get_spacings(self) -> list:
        """Retourner la liste de triggers d'espacement."""
        if self.spacing is None:
            return []
        if isinstance(self.spacing, list):
            return self.spacing
        # Même trigger répété n_levels fois
        return [self.spacing] * self.n_levels


# ══════════════════════════════════════════════════════════════════════════════
# PYRAMID CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PyramidConfig:
    """
    Configuration du pyramiding — ajouter des positions dans le sens du trade.

    Le moteur évalue cette config à chaque barre.
    Quand le trigger du niveau courant est atteint → AddPosition automatique.

    Paramètres :
        n_levels            : nombre de couches de pyramiding max
        trigger             : trigger ou liste de triggers par niveau
                              OnRR(1.0) → add à chaque RR entier
                              OnAll([OnRR(1.0), OnFeature(...)]) → avec confirmation
                              [trigger0, trigger1, trigger2] → triggers différents
        sizing              : DistributionFn pour les tailles
                              None → même taille que la position initiale (1.0)
        move_sl_to_be       : déplacer le SL au BE après chaque add
        sl_mode             : type de SL pour les nouvelles positions
                              "breakeven" (défaut), "original", "feature", "atr_mult"
        sl_feature          : nom de la feature si sl_mode="feature"
        sl_atr_mult         : multiplicateur ATR si sl_mode="atr_mult"
        group_sl_mode       : 0=indépendant, 1=partagé (si SL → ferme tout)
        size_scales_with_mfe: True → taille inversement proportionnelle à MFE
                              (plus on est loin en profit, plus petite la couche)

    Exemple :

        # Pyramiding classique — expo décroissante, SL au BE
        PyramidConfig(
            n_levels=3,
            trigger=OnAll([OnRR(1.0), OnFeature("rsi_14", "lt", 65.0)]),
            sizing=DistributionFn(mode="expo", ratio=0.5),
            move_sl_to_be=True,
            group_sl_mode=1,
        )
        # → 3 couches : 50%, 25%, 12.5% de la position initiale
        # → trigger : RR 1 atteint ET RSI < 65
        # → si SL touché → ferme tout le groupe
    """

    n_levels: int = 2
    trigger: Any = None             # trigger ou list par niveau
    sizing: Any = None              # DistributionFn ou None
    move_sl_to_be: bool = True
    sl_mode: str = "breakeven"      # "breakeven", "original", "feature", "atr_mult"
    sl_feature: str = ""
    sl_atr_mult: float = 0.0
    group_sl_mode: int = 1
    size_scales_with_mfe: bool = False

    def __post_init__(self):
        if self.n_levels < 1:
            raise ValueError("n_levels doit être >= 1")
        valid_sl = {"breakeven", "original", "feature", "atr_mult"}
        if self.sl_mode not in valid_sl:
            raise ValueError(f"sl_mode doit être parmi {valid_sl}")

    def get_fractions(self) -> list[float]:
        """Calculer les fractions pour tous les niveaux."""
        if self.sizing is None:
            return [1.0] * self.n_levels
        return self.sizing.get_fractions(self.n_levels)

    def get_triggers(self) -> list:
        """Retourner la liste de triggers par niveau."""
        if self.trigger is None:
            return [None] * self.n_levels
        if isinstance(self.trigger, list):
            t = self.trigger
            # Compléter si liste trop courte
            while len(t) < self.n_levels:
                t = t + [t[-1]]
            return t[:self.n_levels]
        return [self.trigger] * self.n_levels


# ══════════════════════════════════════════════════════════════════════════════
# AVERAGING CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AveragingConfig:
    """
    Configuration de l'averaging — renforcer une position adverse.
    ⚠️ Feature avancée — risque accru. Toujours utiliser max_avg_down_pct.

    Le moteur évalue cette config à chaque barre.
    Quand le trigger est atteint → AddPosition automatique dans la direction adverse.

    Paramètres :
        n_levels            : nombre de couches d'averaging max
        trigger             : trigger ou liste de triggers
                              Typiquement OnMAEPct(-0.005) + confirmation feature
        sizing              : DistributionFn pour les tailles
                              None → même taille que la position initiale
        sl_mode             : type de SL — "original" fortement recommandé
        tp_mode             : type de TP — "same" fortement recommandé
        max_avg_down_pct    : MAE max avant d'arrêter d'ajouter (valeur négative)
                              0.0 = pas de limite (déconseillé)
        size_scales_with_mae: True → taille proportionnelle à la MAE
                              (plus la MAE est grande, plus grande la couche)

    Exemple :

        # Averaging prudent — 2 niveaux avec confirmation RSI
        AveragingConfig(
            n_levels=2,
            trigger=OnAll([
                OnMAEPct(-0.003),
                OnFeature("rsi_14", "lt", 30.0),
            ]),
            sizing=DistributionFn(mode="expo", ratio=0.5),
            sl_mode="original",
            tp_mode="same",
            max_avg_down_pct=-0.01,   # stopper si MAE > 1%
        )
    """

    n_levels: int = 2
    trigger: Any = None
    sizing: Any = None              # DistributionFn ou None
    sl_mode: str = "original"       # garder SL original pour averaging
    tp_mode: str = "same"           # garder TP original
    max_avg_down_pct: float = 0.0   # 0.0 = pas de limite
    size_scales_with_mae: bool = False

    def __post_init__(self):
        if self.n_levels < 1:
            raise ValueError("n_levels doit être >= 1")
        if self.max_avg_down_pct > 0.0:
            raise ValueError("max_avg_down_pct doit être <= 0 (valeur adverse)")

    def get_fractions(self) -> list[float]:
        """Calculer les fractions pour tous les niveaux."""
        if self.sizing is None:
            return [1.0] * self.n_levels
        return self.sizing.get_fractions(self.n_levels)

    def get_triggers(self) -> list:
        """Retourner la liste de triggers par niveau."""
        if self.trigger is None:
            return [None] * self.n_levels
        if isinstance(self.trigger, list):
            t = self.trigger
            while len(t) < self.n_levels:
                t = t + [t[-1]]
            return t[:self.n_levels]
        return [self.trigger] * self.n_levels


# ══════════════════════════════════════════════════════════════════════════════
# PHASE SPEC
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseSpec:
    """
    Configuration d'une phase du trade dans le multi-phase.

    Chaque phase peut redéfinir les paramètres du profil de sortie
    ET définir des règles custom (PositionRule) actives seulement dans cette phase.

    Quand SetPhase(N) est exécuté dans une rule → SP_PHASE = N
    Le moteur applique automatiquement les params de la PhaseSpec correspondante.

    Paramètres :
        phase                : numéro de la phase (0, 1, 2, 3...)
        tp_pct               : nouveau TP en % (None = garder le TP courant)
        sl_pct               : nouveau SL en % (None = garder le SL courant)
        be_trigger_pct       : nouveau trigger BE (None = garder)
        trailing_trigger_pct : nouveau trigger trailing (None = garder)
        max_holding_bars     : nouveau max hold (None = garder)
        rules                : list[PositionRule] actives seulement dans cette phase

    Exemple multi-phase complet :

        phases=[
            PhaseSpec(
                phase=0,                    # Phase 0 : trade initial
                rules=[
                    PositionRule(
                        trigger=OnRR(1.0),
                        actions=[ExitPartial(0.5), MoveSLtoBE(), SetPhase(1)]
                    ),
                ]
            ),
            PhaseSpec(
                phase=1,                    # Phase 1 : runner après TP1
                trailing_trigger_pct=0.003, # trailing plus serré
                rules=[
                    PositionRule(
                        trigger=OnRR(2.0),
                        actions=[ExitPartial(0.5, ref="remaining"), SetPhase(2)]
                    ),
                    PositionRule(
                        trigger=OnFeature("rsi_14", "gt", 80.0),
                        actions=[Invalidate()]
                    ),
                ]
            ),
            PhaseSpec(
                phase=2,                    # Phase 2 : closing mode
                max_holding_bars=10,        # sortir après 10 barres max
                rules=[
                    PositionRule(
                        trigger=OnBars(3),
                        actions=[MoveSLto("low_5")]
                    ),
                ]
            ),
        ]
    """

    phase: int
    tp_pct: float = None            # None = garder le TP courant
    sl_pct: float = None            # None = garder le SL courant
    be_trigger_pct: float = None    # None = garder
    trailing_trigger_pct: float = None
    max_holding_bars: int = None
    rules: list = field(default_factory=list)  # list[PositionRule]

    def __post_init__(self):
        if self.phase < 0:
            raise ValueError("phase doit être >= 0")

    def has_param_overrides(self) -> bool:
        """True si cette phase redéfinit au moins un paramètre."""
        return any([
            self.tp_pct is not None,
            self.sl_pct is not None,
            self.be_trigger_pct is not None,
            self.trailing_trigger_pct is not None,
            self.max_holding_bars is not None,
        ])

