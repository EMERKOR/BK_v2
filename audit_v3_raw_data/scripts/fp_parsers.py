"""
FantasyPoints-specific parsers for the Ball Knower v3 raw-data audit.

These parsers are DELIBERATELY dataset-specific. There is NO universal CSV
parser and NO universal candidate-key list in this module. Each FantasyPoints
family has its own physical layout:

  * a "super-header" row 0 (group labels like "Team Details", "Player Details",
    "Man/Zone", "Snap Share", "Weeks" ...),
  * a REAL header row 1 (Rank, Name, Season, ... and either coverage columns or
    wide W1..W18 weekly columns),
  * football-observation rows,
  * a trailing glossary/footer block (key -> definition, one per line), often
    separated from the data by a physically blank line.

Row classification (correction pass, 2026-08): every non-blank post-header row
is classified as exactly one of:
  * FOOTBALL           -> the Season cell is a 4-digit year,
  * RECOGNIZED_GLOSSARY -> a key->definition line whose first cell is a real
                          header token and whose columns beyond the definition
                          are empty,
  * UNCLASSIFIED       -> anything else (malformed / unexpected).
Any UNCLASSIFIED row makes `contract_ok=False`. The audit treats that as a
CONTRACT FAIL rather than silently counting the row as glossary.

Nothing here mutates or rewrites the source files.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Team normalization for the team-grain FantasyPoints families (coverage,
# fp_allowed). These files carry the FULL team name in the `Name` column.
# Map full name -> nflverse team code. Injective by construction; the audit
# asserts no two names collapse to the same code and that every football row
# resolves to a code (an unmapped name is a loud failure, not a silent drop).
# --------------------------------------------------------------------------
NFLVERSE_TEAM_FROM_FULLNAME: dict[str, str] = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def normalize_team_fullname(name: str) -> str | None:
    """Full FantasyPoints team name -> nflverse code, or None if unmapped."""
    if name is None:
        return None
    return NFLVERSE_TEAM_FROM_FULLNAME.get(str(name).strip())


@dataclass
class FPParseResult:
    path: str
    physical_data_rows: int          # non-empty physical rows excluding the 2 header rows
    blank_rows: int                  # physically empty rows
    header_rows: int                 # super-header + real header (=2 when both present)
    football_rows: int               # real football observations (Season = 4-digit year)
    glossary_rows: int               # recognized key->definition rows
    unclassified_rows: int           # malformed / unexpected rows (contract-fatal)
    contract_ok: bool = True         # False if any unclassified row present or header bad
    real_header: list = field(default_factory=list)
    week_columns: list = field(default_factory=list)   # W1..Wn wide columns present
    season_values: list = field(default_factory=list)  # distinct Season values among football rows
    unclassified_examples: list = field(default_factory=list)  # (physical_row_index, row)
    note: str = ""


def _read_physical_rows(path: str | Path) -> list[list[str]]:
    """Read every physical row verbatim with the csv module (handles quoted commas).

    utf-8-sig strips the BOM that every FantasyPoints export carries.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


def _is_blank(row: list[str]) -> bool:
    return all((c is None or str(c).strip() == "") for c in row)


def _is_year(cell: str) -> bool:
    c = (cell or "").strip()
    return len(c) == 4 and c.isdigit()


def parse_fp_table(path: str | Path) -> FPParseResult:
    """Parse a FantasyPoints two-header table with strict row classification.

    The real header is row index 1 (0-based) for every observed FantasyPoints
    export; row 0 is the merged group-label band. We assert row 1 contains the
    anchor field 'Season', otherwise we refuse to guess (contract_ok=False).

    Classification of each non-blank row after the header:
      football           : Season cell is a 4-digit year
      recognized glossary : first cell is a header token AND every cell at
                            index >= 2 is empty (i.e. a `key,"definition"` pair)
      unclassified        : neither of the above  -> contract_ok=False
    """
    rows = _read_physical_rows(path)
    blank = sum(1 for r in rows if _is_blank(r))

    if len(rows) < 2:
        return FPParseResult(str(path), 0, blank, 0, 0, 0, 0, contract_ok=False,
                             note="fewer than 2 physical rows")

    real_header = rows[1]
    if "Season" not in real_header:
        return FPParseResult(
            str(path), 0, blank, 0, 0, 0, 0, contract_ok=False,
            real_header=real_header,
            note="CONTRACT FAIL: 'Season' not found in physical row index 1",
        )

    season_idx = real_header.index("Season")
    header_tokens = {str(c).strip() for c in real_header if c is not None and str(c).strip() != ""}
    week_cols = [c for c in real_header if c and c.upper().startswith("W")
                 and c.upper()[1:].isdigit()]

    football_rows = 0
    glossary_rows = 0
    unclassified = 0
    unclassified_examples: list = []
    season_values: set[str] = set()

    for phys_idx, r in enumerate(rows[2:], start=2):
        if _is_blank(r):
            continue
        season_cell = r[season_idx].strip() if len(r) > season_idx else ""
        col0 = (r[0] or "").strip() if len(r) > 0 else ""
        if _is_year(season_cell):
            football_rows += 1
            season_values.add(season_cell)
            continue
        # candidate glossary: key->definition, real header token, nothing past col1
        tail_empty = all((c is None or str(c).strip() == "") for c in r[2:])
        if col0 in header_tokens and tail_empty:
            glossary_rows += 1
            continue
        # anything else is malformed / unexpected
        unclassified += 1
        if len(unclassified_examples) < 5:
            unclassified_examples.append((phys_idx, r))

    physical_data_rows = sum(1 for r in rows[2:] if not _is_blank(r))

    return FPParseResult(
        path=str(path),
        physical_data_rows=physical_data_rows,
        blank_rows=blank,
        header_rows=2,
        football_rows=football_rows,
        glossary_rows=glossary_rows,
        unclassified_rows=unclassified,
        contract_ok=(unclassified == 0),
        real_header=real_header,
        week_columns=week_cols,
        season_values=sorted(season_values),
        unclassified_examples=unclassified_examples,
        note=("OK" if unclassified == 0 else
              f"CONTRACT FAIL: {unclassified} unclassified row(s)"),
    )


def football_frame(path: str | Path):
    """Return only the football rows as a pandas DataFrame.

    Uses the correct parser (row-1 header) and keeps rows whose `Season` is a
    4-digit year — the same predicate as `parse_fp_table`. Callers cross-check
    len(df) against `parse_fp_table(...).football_rows` for consistency.
    """
    import pandas as pd
    df = pd.read_csv(path, skiprows=1, encoding="utf-8-sig")
    if "Season" not in df.columns:
        return df.iloc[0:0]
    # pandas may read Season as float (e.g. 2024.0) because glossary rows make
    # the column NaN-typed; match a 4-digit year via numeric comparison.
    yr = pd.to_numeric(df["Season"], errors="coerce")
    mask = yr.notna() & (yr == yr.round(0)) & (yr >= 1900) & (yr <= 2100)
    return df[mask].copy()


def plain_read_csv_row_count(path: str | Path) -> int:
    """Row count a naive `pandas.read_csv(path)` (no skiprows) would report.

    This models the OLD/WRONG parser used by roster.py for the wide FP files:
    the super-header row becomes the column header and pandas counts every
    remaining physical line (including glossary lines) as a data row.
    """
    import pandas as pd
    df = pd.read_csv(path, encoding="utf-8-sig")
    return len(df)


def skiprows1_read_count(path: str | Path) -> int:
    """Row count the coverage.py parser reports BEFORE the Season-notna filter."""
    import pandas as pd
    df = pd.read_csv(path, skiprows=1, encoding="utf-8-sig")
    return len(df)
