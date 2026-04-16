# Backtest_Framework/event_log.py
from __future__ import annotations
import numpy as np
import pandas as pd


def build_event_log(
    trades_df: pd.DataFrame,
    phase_events_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construit un event log depuis trades_df.
    Chaque ligne = un événement analysable avec TradeContextEngine.
    
    event_type  : PARTIAL_TP_0, PARTIAL_TP_1, ADD_POSITION, PHASE_CHANGE, TRADE_CLOSE
    event_idx   : barre de l'événement — point de référence pour before/after
    entry_idx   : barre d'entrée du trade parent
    group_id    : groupe commun (pyramiding/averaging partagent le même)
    level       : niveau de l'événement (0=premier, 1=second...)
    """
    required = ["entry_idx", "exit_idx", "side", "return", "reason", "group_id"]
    missing = [c for c in required if c not in trades_df.columns]
    if missing:
        raise ValueError(f"trades_df missing columns: {missing}")

    events = []

    # ── Premier entry_idx de chaque groupe = trade parent ─────────
    first_entry = trades_df.groupby("group_id")["entry_idx"].min()
    trades_df = trades_df.copy()
    trades_df["parent_entry_idx"] = trades_df["group_id"].map(first_entry)

    # ── Partiels ──────────────────────────────────────────────────
    partials = trades_df[trades_df["reason"] == "PARTIAL_TP"].copy()
    if len(partials) > 0:
        partials["level"]      = partials.groupby("group_id").cumcount()
        partials["event_type"] = "PARTIAL_TP_" + partials["level"].astype(str)
        partials["event_idx"]  = partials["exit_idx"]
        events.append(partials)

    # ── Adds pyramiding/averaging ─────────────────────────────────
    adds = trades_df[
        (trades_df["entry_idx"] != trades_df["parent_entry_idx"]) &
        (trades_df["reason"] != "PARTIAL_TP")
    ].copy()
    if len(adds) > 0:
        adds["level"]      = adds.groupby("group_id").cumcount()
        adds["event_type"] = "ADD_POSITION"
        adds["event_idx"]  = adds["entry_idx"]
        events.append(adds)

    # ── Phases ────────────────────────────────────────────────────
    if phase_events_df is not None and len(phase_events_df) > 0:
        ph = phase_events_df.copy()
        ph["level"]      = ph["phase_to"]
        ph["event_type"] = "PHASE_CHANGE"
        ph["event_idx"]  = ph["event_idx"]
        # Aligner les colonnes manquantes
        for col in ["exit_idx", "return", "remaining_size"]:
            if col not in ph.columns:
                ph[col] = np.nan
        events.append(ph)

    # ── Clôtures finales ──────────────────────────────────────────
    finals = trades_df[
        (trades_df["reason"] != "PARTIAL_TP") &
        (trades_df["entry_idx"] == trades_df["parent_entry_idx"])
    ].copy()
    if len(finals) > 0:
        finals["level"]      = 0
        finals["event_type"] = "TRADE_CLOSE"
        finals["event_idx"]  = finals["exit_idx"]
        events.append(finals)

    if not events:
        return pd.DataFrame()

    event_log = (
        pd.concat(events, ignore_index=True)
        .sort_values(["group_id", "event_idx"])
        .reset_index(drop=True)
    )

    return event_log