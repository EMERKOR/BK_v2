"""Timestamped sportsbook quote contract for Ball Knower v3 Phase 3A.

This module does not fetch odds. It defines and validates the factual quote shape
that future providers must map into. It deliberately stays separate from
``canonical_market`` because the existing historical nflverse market rows do not
carry genuine pricing timestamps.

One row represents one observed sportsbook offer at one provider snapshot.
No row may be upgraded to OPEN / DECISION / CLOSE unless the label is explicitly
supported by the ingestion process that created it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
import re
from typing import Optional

import pandas as pd

QUOTE_SCHEMA_VERSION = "market_quote_v0.1"

MARKETS = ("SPREAD", "TOTAL", "MONEYLINE")
TIMING_LABELS = ("OPEN", "DECISION", "CLOSE", "OTHER")
STATUSES = ("OPEN", "SUSPENDED", "CLOSED", "UNKNOWN")
SIDES = ("HOME", "AWAY", "OVER", "UNDER")
PERIODS = ("FULL_GAME",)

QUOTE_COLUMNS = [
    "game_id", "sportsbook", "provider", "market", "side",
    "provider_snapshot_time", "line", "price_american",
    "bookmaker_last_update_time", "market_last_update_time", "ingested_at",
    "timing_label", "status",
    "period", "provider_event_id", "book_event_id", "raw_payload_id",
    "raw_payload_sha256", "event_match_method", "event_match_version",
    "quote_schema_version",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _aware_utc(ts, *, field: str, required: bool = True):
    if ts is None or pd.isna(ts):
        if required:
            raise ValueError(f"{field} requires a timezone-aware timestamp")
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware; got {ts!r}")
    return t.tz_convert("UTC")


def _required_text(value, *, field: str) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"{field} is required and must be non-blank")
    return str(value)


def _finite_line(value, *, required: bool):
    if value is None or value is pd.NA:
        if required:
            raise ValueError("line is required for SPREAD and TOTAL")
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        if pd.isna(value) and not required:
            return None
        raise ValueError(f"line must be a finite number; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        if pd.isna(value) and not required:
            return None
        raise ValueError(f"line must be finite; got {value!r}")
    return number


def _american_price(value) -> int:
    if value is None or pd.isna(value):
        raise ValueError("price_american is required for a sportsbook offer")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"price_american must be an integer; got {value!r}")
    if isinstance(value, Real) and not float(value).is_integer():
        raise ValueError(f"price_american must be an integer; got {value!r}")
    price = int(value)
    if -100 < price < 100:
        raise ValueError("American price must be <= -100 or >= +100")
    return price


@dataclass(frozen=True)
class MarketQuote:
    game_id: str
    sportsbook: str
    provider: str
    market: str
    side: str
    provider_snapshot_time: object
    line: Optional[float] = None
    price_american: Optional[int] = None
    bookmaker_last_update_time: object = None
    market_last_update_time: object = None
    ingested_at: object = None
    timing_label: Optional[str] = None
    status: str = "UNKNOWN"
    period: str = "FULL_GAME"
    provider_event_id: Optional[str] = None
    book_event_id: Optional[str] = None
    raw_payload_id: Optional[str] = None
    raw_payload_sha256: Optional[str] = None
    event_match_method: Optional[str] = None
    event_match_version: Optional[str] = None
    quote_schema_version: str = QUOTE_SCHEMA_VERSION

    def validate(self) -> "MarketQuote":
        for field in (
            "game_id", "sportsbook", "provider", "provider_event_id",
            "raw_payload_id", "event_match_method", "event_match_version",
        ):
            _required_text(getattr(self, field), field=field)
        if not _SHA256_RE.fullmatch(str(self.raw_payload_sha256 or "")):
            raise ValueError("raw_payload_sha256 must be a lowercase 64-character SHA-256")
        if self.market not in MARKETS:
            raise ValueError(f"market {self.market!r} must be one of {MARKETS}")
        if self.side not in SIDES:
            raise ValueError(f"side {self.side!r} must be one of {SIDES}")
        if self.timing_label is not None and self.timing_label not in TIMING_LABELS:
            raise ValueError(f"timing_label {self.timing_label!r} must be one of {TIMING_LABELS}")
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} must be one of {STATUSES}")
        if self.period not in PERIODS:
            raise ValueError(f"period {self.period!r} must be one of {PERIODS}")
        if self.quote_schema_version != QUOTE_SCHEMA_VERSION:
            raise ValueError(
                f"quote_schema_version {self.quote_schema_version!r} != {QUOTE_SCHEMA_VERSION!r}")

        snap = _aware_utc(self.provider_snapshot_time, field="provider_snapshot_time")
        book = _aware_utc(
            self.bookmaker_last_update_time,
            field="bookmaker_last_update_time",
            required=False,
        )
        market_update = _aware_utc(
            self.market_last_update_time,
            field="market_last_update_time",
            required=False,
        )
        ingest = _aware_utc(self.ingested_at, field="ingested_at", required=False)

        # A bookmaker update cannot be observed after the provider snapshot that
        # contains it. An ingestion time, when recorded, cannot precede the
        # provider snapshot it ingested.
        if book is not None and book > snap:
            raise ValueError("bookmaker_last_update_time cannot be after provider_snapshot_time")
        if market_update is not None and market_update > snap:
            raise ValueError("market_last_update_time cannot be after provider_snapshot_time")
        if ingest is not None and ingest < snap:
            raise ValueError("ingested_at cannot be before provider_snapshot_time")

        if self.market == "SPREAD" and self.side not in ("HOME", "AWAY"):
            raise ValueError("SPREAD side must be HOME or AWAY")
        if self.market == "TOTAL" and self.side not in ("OVER", "UNDER"):
            raise ValueError("TOTAL side must be OVER or UNDER")
        if self.market == "MONEYLINE" and self.side not in ("HOME", "AWAY"):
            raise ValueError("MONEYLINE side must be HOME or AWAY")
        _finite_line(self.line, required=self.market in ("SPREAD", "TOTAL"))
        if self.market == "MONEYLINE" and self.line is not None and not pd.isna(self.line):
            raise ValueError("MONEYLINE line must be null")
        _american_price(self.price_american)
        return self

    def to_record(self) -> dict:
        self.validate()
        d = asdict(self)
        for field in (
            "provider_snapshot_time", "bookmaker_last_update_time",
            "market_last_update_time", "ingested_at",
        ):
            if d[field] is not None and not pd.isna(d[field]):
                d[field] = _aware_utc(d[field], field=field).isoformat()
            else:
                d[field] = None
        d["line"] = _finite_line(d["line"], required=self.market in ("SPREAD", "TOTAL"))
        d["price_american"] = _american_price(d["price_american"])
        return d


def validate_quote_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a normalized quote frame and return a UTC-normalized copy.

    Required columns are intentionally strict. Missing provider/book timestamps
    remain null only where allowed; provider_snapshot_time is always required.
    Duplicate observations at the exact quote grain fail loudly.
    """
    required = QUOTE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"market quote frame missing required columns: {missing}")

    out = df.copy()
    normalized = []
    for rec in out[required].to_dict("records"):
        normalized.append(MarketQuote(**rec).to_record())
    out = pd.DataFrame(normalized, columns=required)

    # Keep temporal semantics in the returned frame. Serializing timestamps to
    # strings made comparisons/version behavior depend on the pandas release.
    for field in (
        "provider_snapshot_time", "bookmaker_last_update_time",
        "market_last_update_time", "ingested_at",
    ):
        out[field] = pd.to_datetime(out[field], utc=True, errors="raise")
    out["line"] = pd.to_numeric(out["line"], errors="raise")
    out["price_american"] = pd.array(out["price_american"], dtype="Int64")

    key = [
        "game_id", "sportsbook", "provider", "market", "side", "period",
        "provider_snapshot_time",
    ]
    dup = out.duplicated(key, keep=False)
    if dup.any():
        sample = out.loc[dup, key].head(10).to_dict("records")
        raise ValueError(f"duplicate market quote grain detected: {sample}")
    return out.sort_values(key).reset_index(drop=True)
