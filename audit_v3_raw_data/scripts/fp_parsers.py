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

The parsers below identify the real header, split football rows from glossary
rows, and return raw counts so the audit can report physical vs football vs
glossary vs header rows separately (contract global rule 8).

Nothing here mutates or rewrites the source files.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FPParseResult:
    path: str
    physical_data_rows: int          # non-empty physical rows excluding the 2 header rows
    blank_rows: int                  # physically empty rows
    header_rows: int                 # super-header + real header (=2 when both present)
    football_rows: int               # real football observations (Season populated & numeric-ish)
    glossary_rows: int               # trailing key->definition rows
    real_header: list = field(default_factory=list)
    week_columns: list = field(default_factory=list)   # W1..Wn wide columns present
    season_values: list = field(default_factory=list)  # distinct Season values among football rows
    note: str = ""


def _read_physical_rows(path: str | Path) -> list[list[str]]:
    """Read every physical row verbatim with the csv module (handles quoted commas).

    utf-8-sig strips the BOM that every FantasyPoints export carries.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


def _is_blank(row: list[str]) -> bool:
    return all((c is None or str(c).strip() == "") for c in row)


def parse_fp_table(path: str | Path) -> FPParseResult:
    """Parse a FantasyPoints two-header table (coverage, fp_allowed, wide-weekly).

    Contract rule: the real header is row index 1 (0-based) for every observed
    FantasyPoints export; row 0 is the merged group-label band. We assert that
    row 1 contains the anchor field 'Season', otherwise we refuse to guess.

    Football rows are rows AFTER the real header whose 'Season' cell is a 4-digit
    year. Glossary rows are trailing rows whose first cell is a known column
    token and whose 'Season' position is not a year (they are key->definition
    pairs). We classify every non-blank post-header row as either football or
    glossary and never silently drop.
    """
    rows = _read_physical_rows(path)
    blank = sum(1 for r in rows if _is_blank(r))

    if len(rows) < 2:
        return FPParseResult(str(path), 0, blank, 0, 0, 0, note="fewer than 2 physical rows")

    super_header = rows[0]
    real_header = rows[1]

    if "Season" not in real_header:
        # Fail loudly per contract (no silent header guessing).
        return FPParseResult(
            str(path), 0, blank, 0, 0, 0, real_header=real_header,
            note="CONTRACT FAIL: 'Season' not found in physical row index 1",
        )

    season_idx = real_header.index("Season")
    week_cols = [c for c in real_header if c and c.upper().startswith("W")
                 and c.upper()[1:].isdigit()]

    football_rows = 0
    glossary_rows = 0
    season_values: set[str] = set()

    for r in rows[2:]:
        if _is_blank(r):
            continue
        cell = r[season_idx].strip() if len(r) > season_idx else ""
        if len(cell) == 4 and cell.isdigit():
            football_rows += 1
            season_values.add(cell)
        else:
            # trailing glossary / footer definition line (or a short 2-col pair)
            glossary_rows += 1

    physical_data_rows = sum(1 for r in rows[2:] if not _is_blank(r))

    return FPParseResult(
        path=str(path),
        physical_data_rows=physical_data_rows,
        blank_rows=blank,
        header_rows=2,
        football_rows=football_rows,
        glossary_rows=glossary_rows,
        real_header=real_header,
        week_columns=week_cols,
        season_values=sorted(season_values),
    )


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
    """Row count the coverage.py parser reports BEFORE the Season-notna filter.

    coverage.py does: read_csv(skiprows=1) then df[df['Season'].notna()].
    This returns the pre-filter length (glossary rows still present as NaN-Season
    rows) so we can show the delta the Season filter removes.
    """
    import pandas as pd
    df = pd.read_csv(path, skiprows=1, encoding="utf-8-sig")
    return len(df)
