"""The Odds API historical NFL featured-market adapter.

Parses SAVED historical API response payloads into Ball Knower's
`market_quote_v0.1` schema. No HTTP requests or API keys live here.

Supported featured markets:
- h2h -> MONEYLINE
- spreads -> SPREAD
- totals -> TOTAL

Historical responses contain a provider snapshot timestamp. Bookmaker/market
`last_update` is preserved separately. Provider event IDs must be mapped to BK
`game_id` explicitly; this adapter never guesses event identity from team names.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..quotes import QUOTE_SCHEMA_VERSION, validate_quote_frame

PROVIDER = "the_odds_api"
SPORT_KEY = "americanfootball_nfl"
SUPPORTED_MARKETS = {"h2h": "MONEYLINE", "spreads": "SPREAD", "totals": "TOTAL"}


def _utc(value, field: str):
    if value is None:
        return None
    t = pd.Timestamp(value)
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware; got {value!r}")
    return t.tz_convert("UTC")


def _payload_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _side(outcome_name: str, home_team: str, away_team: str) -> str:
    if outcome_name == home_team:
        return "HOME"
    if outcome_name == away_team:
        return "AWAY"
    raise ValueError(f"outcome {outcome_name!r} is neither home nor away team")


def _total_side(outcome_name: str) -> str:
    name = str(outcome_name).strip().upper()
    if name in ("OVER", "UNDER"):
        return name
    raise ValueError(f"unexpected totals outcome {outcome_name!r}")


def parse_historical_payload(payload: dict, *, ingested_at, event_game_map,
                             raw_payload_id: str | None = None,
                             timing_label=None) -> pd.DataFrame:
    """Parse one archived historical response into validated quote rows.

    `event_game_map` maps provider event id -> canonical BK game_id. Missing
    mappings fail loudly. `timing_label` should normally remain null; use it only
    when the collection procedure independently proves OPEN/DECISION/CLOSE.
    """
    if not isinstance(payload, dict):
        raise ValueError("historical payload must be a dict")
    if "timestamp" not in payload or "data" not in payload:
        raise ValueError("historical payload requires timestamp and data")
    if not isinstance(event_game_map, dict):
        raise ValueError("event_game_map must be a dict of provider_event_id -> BK game_id")

    provider_snapshot_time = _utc(payload["timestamp"], "payload.timestamp")
    ingested = _utc(ingested_at, "ingested_at")
    payload_id = raw_payload_id or _payload_sha256(payload)

    rows = []
    for event in payload.get("data", []):
        if event.get("sport_key") not in (None, SPORT_KEY):
            continue
        provider_event_id = str(event.get("id") or "")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        if not provider_event_id or not home_team or not away_team:
            raise ValueError("event missing id/home_team/away_team")
        if provider_event_id not in event_game_map:
            raise ValueError(
                f"provider event {provider_event_id} has no explicit BK game_id mapping; refusing to guess")
        game_id = str(event_game_map[provider_event_id])
        if not game_id:
            raise ValueError(f"provider event {provider_event_id} maps to empty BK game_id")

        for bookmaker in event.get("bookmakers", []):
            sportsbook = bookmaker.get("key")
            if not sportsbook:
                raise ValueError("bookmaker missing key")
            book_update = bookmaker.get("last_update")

            for market in bookmaker.get("markets", []):
                raw_key = market.get("key")
                if raw_key not in SUPPORTED_MARKETS:
                    continue
                market_name = SUPPORTED_MARKETS[raw_key]
                market_update = market.get("last_update") or book_update
                last_update = _utc(market_update, "market.last_update") if market_update else None

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    side = _total_side(name) if market_name == "TOTAL" else _side(name, home_team, away_team)
                    line = outcome.get("point") if market_name in ("SPREAD", "TOTAL") else None
                    price = outcome.get("price")
                    price = int(price) if price is not None else None

                    rows.append({
                        "game_id": game_id,
                        "sportsbook": str(sportsbook),
                        "provider": PROVIDER,
                        "market": market_name,
                        "side": side,
                        "provider_snapshot_time": provider_snapshot_time,
                        "line": line,
                        "price_american": price,
                        "bookmaker_last_update_time": last_update,
                        "ingested_at": ingested,
                        "timing_label": timing_label,
                        "status": "OPEN",
                        "period": "FULL_GAME",
                        "provider_event_id": provider_event_id,
                        "book_event_id": None,
                        "raw_payload_id": str(payload_id),
                        "quote_schema_version": QUOTE_SCHEMA_VERSION,
                    })

    required = [
        "game_id", "sportsbook", "provider", "market", "side",
        "provider_snapshot_time", "line", "price_american",
        "bookmaker_last_update_time", "ingested_at", "timing_label", "status",
        "period", "provider_event_id", "book_event_id", "raw_payload_id",
        "quote_schema_version",
    ]
    if not rows:
        return pd.DataFrame(columns=required)
    return validate_quote_frame(pd.DataFrame(rows, columns=required))


def parse_historical_file(path, *, ingested_at, event_game_map, timing_label=None) -> pd.DataFrame:
    """Read one archived JSON response and parse it reproducibly."""
    p = Path(path)
    payload = json.loads(p.read_text())
    return parse_historical_payload(
        payload,
        ingested_at=ingested_at,
        event_game_map=event_game_map,
        raw_payload_id=hashlib.sha256(p.read_bytes()).hexdigest(),
        timing_label=timing_label,
    )
