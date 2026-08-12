"""
Versioned roster-status normalization (Phase 2D).

The nflverse weekly-roster `status` field is the authoritative coarse roster
designation. Every non-null `status` is DELIBERATELY mapped here or the build
fails loudly (unseen status is never silently bucketed). Raw status, raw team,
raw position, and source season/week are preserved by the caller.

Design honesty:
  * booleans are NULLABLE. A code sets only the flags it directly establishes;
    everything it does not establish stays None (never inferred).
  * the coarse `status` does NOT separate injured reserve from other reserve
    designations, so `is_ir` is left None for generic RESERVE. The single
    well-known detail code R01 (Reserve/Injured) is the one deliberate
    refinement used to set is_ir=True; all other detail codes are preserved as
    raw evidence only and never drive a boolean.
  * no health/availability/active inference. Missing status -> all-null booleans.

This module produces per-source-row normalization only. Weekly-roster
transaction duplicates (the 2010-2015 era) are resolved by the caller
(player_team_week): compatible membership is preserved, contradictory status is
quarantined — never arbitrarily selected.
"""
from __future__ import annotations

import pandas as pd

ROSTER_MAP_VERSION = "rosterstatus_v0.1"

# status -> (normalized_label, on_roster, active_roster, practice_squad, ir, pup, suspended)
# True / False = directly established by the code; None = not established (null).
_T, _F, _N = True, False, None
_STATUS_MAP = {
    "ACT": ("ACTIVE", _T, _T, _F, _F, _F, _F),
    "INA": ("INACTIVE", _T, _F, _F, _F, _F, _F),
    "DEV": ("PRACTICE_SQUAD", _T, _F, _T, _F, _F, _F),
    "PUP": ("RESERVE_PUP", _T, _F, _F, _F, _T, _F),
    "SUS": ("SUSPENDED", _T, _F, _F, _F, _F, _T),
    "RES": ("RESERVE", _T, _F, _F, _N, _F, _F),   # generic reserve; IR-ness unknown from coarse code
    "RSN": ("RESERVE_NON_FOOTBALL", _T, _F, _F, _F, _F, _F),
    "RSR": ("RESERVE", _T, _F, _F, _N, _F, _F),
    "EXE": ("EXEMPT", _T, _F, _F, _F, _F, _F),
    "E01": ("EXEMPT", _T, _F, _F, _F, _F, _F),
    "E14": ("EXEMPT", _T, _F, _F, _F, _F, _F),
    "CUT": ("CUT", _F, _F, _F, _F, _F, _F),
    "RET": ("RETIRED", _F, _F, _F, _F, _F, _F),
    "NWT": ("NOT_WITH_TEAM", _F, _F, _F, _F, _F, _F),
    # transaction markers — membership timing is ambiguous without effective time
    "TRC": ("TRANSACTION_TRADE", _N, _N, _F, _F, _F, _F),
    "TRD": ("TRANSACTION_TRADE", _N, _N, _F, _F, _F, _F),
    "TRT": ("TRANSACTION_TRADE", _N, _N, _F, _F, _F, _F),
    # free-agent / draft designations — not team membership
    "RFA": ("FREE_AGENT", _N, _F, _F, _F, _F, _F),
    "UFA": ("FREE_AGENT", _N, _F, _F, _F, _F, _F),
    "UDF": ("UNDRAFTED_FREE_AGENT", _N, _F, _F, _F, _F, _F),
}

_FLAG_FIELDS = ["is_on_roster", "is_active_roster", "is_practice_squad",
                "is_ir", "is_pup", "is_suspended"]

# The one deliberate detail-code refinement: R01 == Reserve/Injured -> IR.
_IR_DETAIL_CODE = "R01"


def known_statuses() -> set:
    return set(_STATUS_MAP)


def normalize_status(status_raw, status_detail_raw=None) -> dict:
    """Map a raw roster status to a normalized label + nullable boolean flags.

    Missing (null) status -> normalized None and all-null booleans.
    Unseen non-null status -> ValueError (fail loudly; never bucketed).
    """
    if status_raw is None or (isinstance(status_raw, float) and pd.isna(status_raw)):
        return {"roster_status_normalized": None,
                **{f: None for f in _FLAG_FIELDS},
                "roster_map_version": ROSTER_MAP_VERSION}
    s = str(status_raw).strip()
    if s == "" or s.lower() == "nan":
        return {"roster_status_normalized": None,
                **{f: None for f in _FLAG_FIELDS},
                "roster_map_version": ROSTER_MAP_VERSION}
    if s not in _STATUS_MAP:
        raise ValueError(f"Unseen roster status {status_raw!r} ({ROSTER_MAP_VERSION}); "
                         f"refusing to bucket (no silent default)")
    label, on_r, act, ps, ir, pup, sus = _STATUS_MAP[s]
    # deliberate IR refinement using the single well-known detail code
    if ir is None and status_detail_raw is not None and not (
            isinstance(status_detail_raw, float) and pd.isna(status_detail_raw)):
        if str(status_detail_raw).strip() == _IR_DETAIL_CODE:
            ir = True
    return {
        "roster_status_normalized": label,
        "is_on_roster": on_r, "is_active_roster": act, "is_practice_squad": ps,
        "is_ir": ir, "is_pup": pup, "is_suspended": sus,
        "roster_map_version": ROSTER_MAP_VERSION,
    }
