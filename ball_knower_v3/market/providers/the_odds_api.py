"""The Odds API historical NFL featured-market adapter.

This module parses SAVED historical API response payloads into Ball Knower's
`market_quote_v0.1` schema. It deliberately performs no HTTP requests and
contains no API-key handling. Raw payload acquisition/storage is an external
collection step so archived responses can be hashed and reproduced.

Supported featured markets:
- h2h -> MONEYLINE
- spreads -> SPREAD
- totals -> TOTAL

Expected historical response shape:
{
  "timestamp": "...Z",
  "previous_timestamp": "...Z",
  "data": [events...]
}

The provider snapshot timestamp is response.timestamp. Bookmaker/market
`last_update` is preserved as bookmaker_last_update_time when present.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..quotes import MARKET_QUOTE_VERSION, validate_quotes

PROVIDER = "the_odds_api"
SPORT_KEY = "americanfootball_nfl"
SUPPORTED_MARKETS = {"h2h": "MONEYLINE", "spreads": "SPREAD", "totals": "TOTAL"}


def _utc(value, field: str):
    if value is None:
        return pd.NaT
    t = pd.Timestamp(value)
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware; got {value!r}")
    return t.tz_convert("UTC")


def _payload_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _american_price(value):
    if value is None:
        return pd.NA
    return float(value)


def _spread_side(outcome_name: str, home_team: str, away_team: str) -> str:
    if outcome_name == home_team:
        return "HOME"
    if outcome_name == away_team:
        return "AWAY"
    raise ValueError(f"spread outcome {outcome_name!r} is neither home nor away team")


def _moneyline_side(outcome_name: str, home_team: str, away_team: str) -> str:
    return _spread_side(outcome_name, home_team, away_team)


def _total_side(outcome_name: str) -> str:
    name = str(outcome_name).strip().upper()
    if name == "OVER":
        return "OVER"
    if name == "UNDER":
        return "UNDER"
    raise ValueError(f"unexpected totals outcome {outcome_name!r}")


def parse_historical_payload(payload: dict, *, ingested_at,
                             source_payload_id: str | None = None,
                             line_timing_label=None) -> pd.DataFrame:
    """Parse one archived historical response into validated quote rows.

    `line_timing_label` is normally left null. It may be supplied only when the
    calling collection process has independently proven OPEN/DECISION/CLOSE
    semantics; the quote validator will reject unknown labels.
    """
    if not isinstance(payload, dict):
        raise ValueError("historical payload must be a dict")
    if "timestamp" not in payload or "data" not in payload:
        raise ValueError("historical payload requires timestamp and data")

    provider_snapshot_time = _utc(payload["timestamp"], "payload.timestamp")
    ingested = _utc(ingested_at, "ingested_at")
    payload_id = source_payload_id or _payload_sha256(payload)

    rows = []
    for event in payload.get("data", []):
        if event.get("sport_key") not in (None, SPORT_KEY):
            continue
        provider_event_id = event.get("id")
        commence_time = _utc(event.get("commence_time"), "event.commence_time")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        if not provider_event_id or not home_team or not away_team:
            raise ValueError("event missing id/home_team/away_team")

        for bookmaker in event.get("bookmakers", []):
            book = bookmaker.get("key")
            if not book:
                raise ValueError("bookmaker missing key")
            book_update = bookmaker.get("last_update")

            for market in bookmaker.get("markets", []):
                raw_key = market.get("key")
                if raw_key not in SUPPORTED_MARKETS:
                    continue
                market_name = SUPPORTED_MARKETS[raw_key]
                market_update = market.get("last_update") or book_update
                last_update = _utc(market_update, "market.last_update") if market_update else pd.NaT

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    if market_name == "SPREAD":
                        side = _spread_side(name, home_team, away_team)
                        line = outcome.get("point")
                    elif market_name == "TOTAL":
                        side = _total_side(name)
                        line = outcome.get("point")
                    else:
                        side = _moneyline_side(name, home_team, away_team)
                        line = pd.NA

                    rows.append({
                        "market_quote_version": MARKET_QUOTE_VERSION,
                        "provider": PROVIDER,
                        "provider_event_id": str(provider_event_id),
                        "book_event_id": pd.NA,
                        "game_id": pd.NA,
                        "sport_key": event.get("sport_key") or SPORT_KEY,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "book": str(book),
                        "market": market_name,
                        "period": "FULL_GAME",
                        "side": side,
                        "line": line,
                        "price": _american_price(outcome.get("price")),
                        "odds_format": "AMERICAN",
                        "provider_snapshot_time": provider_snapshot_time,
                        "bookmaker_last_update_time": last_update,
                        "ingested_at": ingested,
                        "line_timing_label": line_timing_label,
                        "status": "OPEN",
                        "source_payload_id": str(payload_id),
                        "source_market_key": raw_key,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return validate_quotes(df)


def parse_historical_file(path, *, ingested_at, line_timing_label=None) -> pd.DataFrame:
    """Read one archived JSON response and parse it reproducibly."""
    p = Path(path)
    payload = json.loads(p.read_text())
    source_payload_id = hashlib.sha256(p.read_bytes()).hexdigest()
    return parse_historical_payload(
        payload,
        ingested_at=ingested_at,
        source_payload_id=source_payload_id,
        line_timing_label=line_timing_label,
    )
