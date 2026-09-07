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

from ..event_mapping import MAPPING_COLUMNS, validate_event_mapping_frame
from ..quotes import QUOTE_COLUMNS, QUOTE_SCHEMA_VERSION, validate_quote_frame

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
    try:
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"historical payload is not canonical JSON: {exc}") from exc
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


def _mapping_index(event_game_map) -> dict[str, dict]:
    """Normalize explicit mapping inputs without performing event matching.

    Production ingestion should pass a versioned mapping frame. A plain dict is
    retained as a small explicit-map convenience for callers/tests and is tagged
    as such; in neither form does this adapter derive a BK game id.
    """
    if isinstance(event_game_map, pd.DataFrame):
        records = validate_event_mapping_frame(event_game_map)[MAPPING_COLUMNS].to_dict("records")
        verified_frame = True
    elif isinstance(event_game_map, dict):
        verified_frame = False
        records = []
        for event_id, value in event_game_map.items():
            if isinstance(value, dict):
                game_id = value.get("game_id")
                method = value.get("match_method")
                version = value.get("match_version")
            else:
                game_id = value
                method = "EXPLICIT_PROVIDER_EVENT_MAP"
                version = "explicit_provider_event_map_v0.1"
            records.append({
                "provider_event_id": event_id,
                "game_id": game_id,
                "match_method": method,
                "match_version": version,
            })
    else:
        raise ValueError("event_game_map must be a dict or versioned mapping frame")

    index = {}
    for record in records:
        event_id = str(record.get("provider_event_id") or "").strip()
        game_id = str(record.get("game_id") or "").strip()
        method = str(record.get("match_method") or "").strip()
        version = str(record.get("match_version") or "").strip()
        if not event_id or not game_id or not method or not version:
            raise ValueError("event_game_map records require event id, game id, method and version")
        if event_id in index:
            raise ValueError(f"ambiguous duplicate mapping for provider event {event_id}")
        index[event_id] = {
            "game_id": game_id,
            "event_match_method": method,
            "event_match_version": version,
            "provider_commence_time": (
                record.get("provider_commence_time") if verified_frame else None
            ),
            "provider_home_team": record.get("provider_home_team") if verified_frame else None,
            "provider_away_team": record.get("provider_away_team") if verified_frame else None,
        }
    return index


def _require_list(value, *, field: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _validated_outcomes(market_name: str, market: dict, home_team: str,
                        away_team: str) -> list[dict]:
    outcomes = _require_list(market.get("outcomes"), field="market.outcomes")
    if len(outcomes) != 2 or not all(isinstance(outcome, dict) for outcome in outcomes):
        raise ValueError(f"{market_name} must contain exactly two outcome objects")
    if market_name == "TOTAL":
        sides = [_total_side(outcome.get("name")) for outcome in outcomes]
        if set(sides) != {"OVER", "UNDER"}:
            raise ValueError("TOTAL outcomes must contain exactly OVER and UNDER")
        points = [outcome.get("point") for outcome in outcomes]
        if points[0] != points[1]:
            raise ValueError("TOTAL over/under outcomes must carry the same line")
    else:
        sides = [_side(outcome.get("name"), home_team, away_team) for outcome in outcomes]
        if set(sides) != {"HOME", "AWAY"}:
            raise ValueError(f"{market_name} outcomes must contain exactly HOME and AWAY")
        if market_name == "SPREAD":
            points = [outcome.get("point") for outcome in outcomes]
            if any(point is None for point in points) or float(points[0]) != -float(points[1]):
                raise ValueError("SPREAD home/away lines must be exact opposites")
    return outcomes


def parse_historical_payload(payload: dict, *, ingested_at, event_game_map,
                             raw_payload_id: str | None = None,
                             raw_payload_sha256: str | None = None,
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
    data = _require_list(payload["data"], field="payload.data")
    mapping = _mapping_index(event_game_map)
    if timing_label is not None:
        raise ValueError(
            "The Odds API archive adapter cannot independently prove a timing_label; "
            "OPEN/DECISION/CLOSE must remain null")

    provider_snapshot_time = _utc(payload["timestamp"], "payload.timestamp")
    ingested = _utc(ingested_at, "ingested_at")
    computed_sha = _payload_sha256(payload)
    payload_sha = raw_payload_sha256 or computed_sha
    if raw_payload_sha256 is not None and raw_payload_sha256 != computed_sha:
        # A file's raw-byte hash differs from its parsed canonical-JSON hash by
        # design. File callers provide a stable id/hash pair and are checked
        # before reaching this function; dict callers may not forge a hash.
        if raw_payload_id is None:
            raise ValueError("raw_payload_sha256 does not match the supplied payload")
    payload_id = raw_payload_id or f"sha256:{payload_sha}"

    rows = []
    for event in data:
        if not isinstance(event, dict):
            raise ValueError("payload.data entries must be event objects")
        if event.get("sport_key") not in (None, SPORT_KEY):
            continue
        provider_event_id = str(event.get("id") or "")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        if not provider_event_id or not home_team or not away_team:
            raise ValueError("event missing id/home_team/away_team")
        if provider_event_id not in mapping:
            raise ValueError(
                f"provider event {provider_event_id} has no explicit BK game_id mapping; refusing to guess")
        mapped = mapping[provider_event_id]
        game_id = mapped["game_id"]
        if mapped["provider_commence_time"] is not None:
            event_commence = _utc(event.get("commence_time"), "event.commence_time")
            mapped_commence = _utc(
                mapped["provider_commence_time"], "event_mapping.provider_commence_time",
            )
            if (
                event_commence != mapped_commence
                or home_team != mapped["provider_home_team"]
                or away_team != mapped["provider_away_team"]
            ):
                raise ValueError(
                    f"provider event {provider_event_id} identity differs from the "
                    "versioned event mapping; refusing to reuse the mapping")

        for bookmaker in _require_list(event.get("bookmakers"), field="event.bookmakers"):
            if not isinstance(bookmaker, dict):
                raise ValueError("event.bookmakers entries must be objects")
            sportsbook = bookmaker.get("key")
            if not sportsbook:
                raise ValueError("bookmaker missing key")
            book_update_raw = bookmaker.get("last_update")
            book_update = _utc(book_update_raw, "bookmaker.last_update") if book_update_raw else None

            for market in _require_list(bookmaker.get("markets"), field="bookmaker.markets"):
                if not isinstance(market, dict):
                    raise ValueError("bookmaker.markets entries must be objects")
                raw_key = market.get("key")
                if raw_key not in SUPPORTED_MARKETS:
                    continue
                market_name = SUPPORTED_MARKETS[raw_key]
                market_update_raw = market.get("last_update")
                market_update = (
                    _utc(market_update_raw, "market.last_update")
                    if market_update_raw else None
                )

                for outcome in _validated_outcomes(market_name, market, home_team, away_team):
                    name = outcome.get("name")
                    side = _total_side(name) if market_name == "TOTAL" else _side(name, home_team, away_team)
                    line = outcome.get("point") if market_name in ("SPREAD", "TOTAL") else None
                    price = outcome.get("price")

                    rows.append({
                        "game_id": game_id,
                        "sportsbook": str(sportsbook),
                        "provider": PROVIDER,
                        "market": market_name,
                        "side": side,
                        "provider_snapshot_time": provider_snapshot_time,
                        "line": line,
                        "price_american": price,
                        "bookmaker_last_update_time": book_update,
                        "market_last_update_time": market_update,
                        "ingested_at": ingested,
                        "timing_label": timing_label,
                        # The historical endpoint does not prove executable/open
                        # status or expose suspension state.
                        "status": "UNKNOWN",
                        "period": "FULL_GAME",
                        "provider_event_id": provider_event_id,
                        "book_event_id": None,
                        "raw_payload_id": str(payload_id),
                        "raw_payload_sha256": str(payload_sha),
                        "event_match_method": mapped["event_match_method"],
                        "event_match_version": mapped["event_match_version"],
                        "quote_schema_version": QUOTE_SCHEMA_VERSION,
                    })

    if not rows:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    return validate_quote_frame(pd.DataFrame(rows, columns=QUOTE_COLUMNS))


def parse_historical_file(path, *, ingested_at, event_game_map, timing_label=None) -> pd.DataFrame:
    """Read one archived JSON response and parse it reproducibly."""
    p = Path(path)
    raw = p.read_bytes()
    payload_sha = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid archived JSON payload {p}: {exc}") from exc
    return parse_historical_payload(
        payload,
        ingested_at=ingested_at,
        event_game_map=event_game_map,
        raw_payload_id=f"sha256:{payload_sha}",
        raw_payload_sha256=payload_sha,
        timing_label=timing_label,
    )
