"""
position_rules.py
─────────────────
Triggers, actions et règles de position pour ExitProfileSpec.
 
Usage :
    from Backtest_Framework.position_rules import (
        OnRR, OnMFEPct, OnFeature, OnAll,
        ExitPartial, MoveSLtoBE, AddPosition, SetPhase, Invalidate,
        PositionRule,
    )
"""
 
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TRIGGERS
# ══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class OnRR:
    """
    Déclenche quand MFE >= value * distance_SL_initiale.
 
    Exemple :
        OnRR(1.0)   → déclenche quand le trade est en RR 1:1
        OnRR(2.5)   → déclenche quand le trade est en RR 2.5:1
    """
    value: float
 
 
@dataclass
class OnMFEPct:
    """
    Déclenche quand MFE >= value (en fraction).
 
    Exemple :
        OnMFEPct(0.01)   → déclenche quand MFE >= 1%
        OnMFEPct(0.005)  → déclenche quand MFE >= 0.5%
    """
    value: float
 
 
@dataclass
class OnMAEPct:
    """
    Déclenche quand MAE <= value (adverse, valeur négative).
 
    Exemple :
        OnMAEPct(-0.005)  → déclenche quand la position est adverse de 0.5%
        OnMAEPct(-0.01)   → déclenche quand la position est adverse de 1%
    """
    value: float
 
 
@dataclass
class OnATRMult:
    """
    Déclenche quand MFE >= value * ATR courant.
 
    Exemple :
        OnATRMult(2.0)  → déclenche quand MFE >= 2 * ATR
    """
    value: float
 
 
@dataclass
class OnBars:
    """
    Déclenche quand le nombre de barres depuis l'entrée >= value.
 
    Exemple :
        OnBars(5)   → déclenche après 5 barres en position
        OnBars(20)  → déclenche après 20 barres en position
    """
    value: int
 
 
@dataclass
class OnBarsAfterLastTP:
    """
    Déclenche quand le nombre de barres depuis le dernier partial TP >= value.
    Utilise SP_BARS_SINCE_TP du stateful engine.
 
    Exemple :
        OnBarsAfterLastTP(3)  → déclenche 3 barres après le dernier partial
    """
    value: int
 
 
@dataclass
class OnFeature:
    """
    Déclenche quand une feature compilée satisfait une condition.
 
    Paramètres :
        name     : nom de la feature dans CompiledFeatures
        operator : "gt", "lt", "gte", "lte", "cross_above", "cross_below"
        value    : float ou nom d'une autre feature (str)
 
    Exemples :
        OnFeature("rsi_14", "lt", 30.0)
            → déclenche quand RSI < 30
 
        OnFeature("rsi_14", "gt", 70.0)
            → déclenche quand RSI > 70
 
        OnFeature("ema_20", "cross_above", "ema_50")
            → déclenche quand ema_20 croise au-dessus de ema_50
 
        OnFeature("close", "cross_below", "ema_200")
            → déclenche quand le close croise sous ema_200
 
        OnFeature("volume", "gt", "volume_ma_20")
            → déclenche quand le volume est au-dessus de sa moyenne
    """
    name: str
    operator: str   # "gt", "lt", "gte", "lte", "cross_above", "cross_below"
    value: Any      # float ou str (nom d'une autre feature)
 
 
@dataclass
class OnPhase:
    """
    Déclenche seulement si la position est dans la phase N.
    Utilise SP_PHASE du stateful engine.
 
    Exemple :
        OnPhase(1)  → actif seulement en phase 1
        OnPhase(2)  → actif seulement en phase 2
    """
    value: int
 
 
@dataclass
class OnAll:
    """
    Déclenche seulement si TOUS les triggers de la liste sont vrais (ET logique).
 
    Exemple :
        OnAll([OnRR(1.0), OnFeature("rsi_14", "lt", 65.0)])
            → déclenche si RR >= 1.0 ET RSI < 65
    """
    triggers: list
 
 
@dataclass
class OnAny:
    """
    Déclenche si AU MOINS UN trigger de la liste est vrai (OU logique).
 
    Exemple :
        OnAny([OnRR(2.0), OnBars(20)])
            → déclenche si RR >= 2.0 OU 20 barres écoulées
    """
    triggers: list
 
 
# ══════════════════════════════════════════════════════════════════════════════
# ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class ExitPartial:
    """
    Sortir une fraction de la position.
 
    Paramètres :
        fraction     : fraction fixe à sortir (0.5 = 50%)
        ref          : référence pour la fraction
                       "remaining" (défaut) → % de ce qui reste
                       "original"           → % de la taille d'entrée
        price        : prix d'exécution
                       "market"  (défaut) → ouverture de la barre courante
                       "tp"               → au prix du TP courant
                       "feature"          → au prix d'une feature compilée
        price_feature: nom de la feature si price="feature"
        fraction_fn  : DistributionFn optionnel — si fourni, écrase fraction
                       et calcule la taille dynamiquement selon la distance
 
    Exemples :
        ExitPartial(0.5)
            → sortir 50% de ce qui reste au marché
 
        ExitPartial(0.5, ref="original")
            → sortir 50% de la taille d'entrée
 
        ExitPartial(0.3, price="feature", price_feature="resistance_20")
            → sortir 30% au niveau de résistance calculé
 
        ExitPartial(fraction_fn=DistributionFn(mode="expo", ratio=0.5))
            → taille calculée dynamiquement par la fonction
    """
    fraction: float = 0.5
    ref: str = "remaining"          # "remaining", "original"
    price: str = "market"           # "market", "tp", "feature"
    price_feature: str = ""
    fraction_fn: Any = None         # DistributionFn optionnel
 
 
@dataclass
class MoveSLtoBE:
    """
    Déplacer le stop-loss au prix d'entrée (break-even).
 
    Paramètres :
        offset_pct : offset en % au-dessus du BE pour les longs,
                     en dessous pour les shorts (0.0 = BE exact)
 
    Exemples :
        MoveSLtoBE()
            → SL au prix d'entrée exact
 
        MoveSLtoBE(offset_pct=0.001)
            → SL à entry + 0.1% (long) ou entry - 0.1% (short)
    """
    offset_pct: float = 0.0
 
 
@dataclass
class MoveSLto:
    """
    Déplacer le stop-loss à un niveau calculé par une feature.
 
    Paramètres :
        feature    : nom de la feature compilée (ex: "prev_low_5", "ema_20")
        offset_pct : offset additionnel en % (positif = au-dessus, négatif = en dessous)
 
    Exemples :
        MoveSLto("prev_low_5")
            → SL au plus bas des 5 dernières barres
 
        MoveSLto("ema_20", offset_pct=-0.001)
            → SL à ema_20 - 0.1%
    """
    feature: str
    offset_pct: float = 0.0
 
 
@dataclass
class SetTP:
    """
    Modifier le take-profit de la position.
    Un seul des trois paramètres doit être fourni.
 
    Paramètres :
        rr        : nouveau TP exprimé en RR (ex: 2.0 = RR 2:1)
        feature   : nom d'une feature compilée comme niveau de TP
        atr_mult  : nouveau TP à N * ATR depuis le prix d'entrée
 
    Exemples :
        SetTP(rr=2.0)
            → TP à entry + 2 * SL_distance (long)
 
        SetTP(feature="resistance_20")
            → TP au niveau de résistance calculé
 
        SetTP(atr_mult=3.0)
            → TP à entry + 3 * ATR (long)
    """
    rr: float = 0.0
    feature: str = ""
    atr_mult: float = 0.0
 
 
@dataclass
class AddPosition:
    """
    Ajouter une position liée au groupe courant.
    Utilisé pour le pyramiding (dans le sens du trade) ou l'averaging (sens adverse).
 
    Paramètres :
        size_fraction : taille de la nouvelle position
        size_ref      : référence pour la taille
                        "original"  (défaut) → fraction de la taille d'entrée
                        "remaining"          → fraction de ce qui reste
        size_fn       : DistributionFn optionnel — calcule la taille dynamiquement
        sl            : type de SL pour la nouvelle position
                        "breakeven" (défaut) → SL au BE de la position parent
                        "original"           → SL original de la position parent
                        "feature"            → SL à un niveau de feature
                        "atr_mult"           → SL à N * ATR
        sl_feature    : nom de la feature si sl="feature"
        sl_atr_mult   : multiplicateur ATR si sl="atr_mult"
        tp            : type de TP pour la nouvelle position
                        "same"    (défaut) → même TP que la position parent
                        "feature"          → TP à un niveau de feature
                        "rr"               → TP à RR N
        tp_feature    : nom de la feature si tp="feature"
        tp_rr         : valeur RR si tp="rr"
        group_sl_mode : mode de SL partagé pour le groupe
                        0 → indépendant (chaque position a son propre SL)
                        1 → partagé (si SL touché → ferme tout le groupe)
 
    Exemples pyramiding :
        AddPosition(0.5, sl="breakeven", group_sl_mode=1)
            → ajouter 50% de la taille originale, SL au BE, groupe partagé
 
        AddPosition(0.25, sl="atr_mult", sl_atr_mult=2.0)
            → ajouter 25%, SL à 2*ATR sous le prix courant
 
    Exemples averaging :
        AddPosition(0.5, sl="original", tp="same")
            → renforcer de 50%, garder SL et TP originaux
    """
    size_fraction: float = 0.5
    size_ref: str = "original"      # "original", "remaining"
    size_fn: Any = None             # DistributionFn optionnel
    sl: str = "breakeven"           # "original", "breakeven", "feature", "atr_mult"
    sl_feature: str = ""
    sl_atr_mult: float = 0.0
    tp: str = "same"                # "same", "feature", "rr"
    tp_feature: str = ""
    tp_rr: float = 0.0
    group_sl_mode: int = 1
 
 
@dataclass
class SetPhase:
    """
    Changer la phase de la position (stocké dans SP_PHASE).
    Permet de changer de comportement au fil du trade.
 
    Exemple :
        SetPhase(1)  → passer en phase 1 (runner actif)
        SetPhase(2)  → passer en phase 2 (closing mode)
    """
    value: int
 
 
@dataclass
class Invalidate:
    """
    Forcer la sortie immédiate de la position.
    Écrit SP_ENTRY_VALID = 0.0 → le moteur sort au prochain check.
 
    Usage typique :
        PositionRule(
            trigger=OnFeature("ema_20", "cross_below", "ema_50"),
            actions=[Invalidate()]
        )
    """
    pass
 
 
# ══════════════════════════════════════════════════════════════════════════════
# RÈGLE DE POSITION
# ══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class PositionRule:
    """
    Une règle = un trigger + une liste d'actions.
    Évaluée à chaque barre pour chaque position ouverte.
 
    Paramètres :
        trigger      : trigger ou combinaison de triggers (OnRR, OnFeature, OnAll...)
        actions      : liste d'actions à exécuter quand le trigger est vrai
        phase_filter : -1 = s'applique à toutes les phases (défaut)
                       N  = s'applique seulement quand SP_PHASE == N
        max_times    : -1 = pas de limite (défaut)
                       N  = la règle ne peut se déclencher que N fois par trade
 
    Exemples :
 
        # Partial TP à RR 1, SL au BE, passer en phase 1
        PositionRule(
            trigger=OnRR(1.0),
            actions=[ExitPartial(0.5), MoveSLtoBE(), SetPhase(1)]
        )
 
        # En phase 1 seulement — sortie si RSI overbought
        PositionRule(
            trigger=OnFeature("rsi_14", "gt", 75.0),
            phase_filter=1,
            actions=[Invalidate()]
        )
 
        # Pyramiding avec confirmation — max 1 fois par trade
        PositionRule(
            trigger=OnAll([
                OnRR(1.0),
                OnFeature("ema_20", "cross_above", "ema_50"),
                OnPhase(0),
            ]),
            actions=[AddPosition(0.5, sl="breakeven"), SetPhase(1)],
            max_times=1,
        )
 
        # Averaging en phase 0 seulement — max 2 fois
        PositionRule(
            trigger=OnAll([
                OnMAEPct(-0.003),
                OnFeature("rsi_14", "lt", 30.0),
                OnPhase(0),
            ]),
            actions=[AddPosition(0.5, sl="original", tp="same"), SetPhase(1)],
            max_times=2,
        )
    """
    trigger: Any
    actions: list
    phase_filter: int = -1      # -1 = toutes phases
    max_times: int = -1         # -1 = pas de limite
 
 
