"""Versioned The Odds API event-to-Ball Knower game identity mapping.

The provider does not know Ball Knower ``game_id`` values. This module builds an
explicit, auditable mapping from factual provider event attributes: kickoff,
home team and away team. It never chooses a nearest candidate: zero or multiple
matches within the declared tolerance are hard failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from ..canonical import common

PROVIDER = "the_odds_api"
EVENT_MAPPING_VERSION = "the_odds_api_event_mapping_v0.1"
MATCH_METHOD = "NORMALIZED_TEAMS_AND_KICKOFF_TOLERANCE"
DEFAULT_KICKOFF_TOLERANCE = pd.Timedelta(minutes=5)

MAPPING_COLUMNS = [
    "provider", "provider_event_id", "game_id", "provider_commence_time",
    "provider_home_team", "provider_away_team", "normalized_home_team",
    "normalized_away_team", "canonical_kickoff", "match_method",
    "match_version", "kickoff_tolerance_seconds",
]

# The Odds API uses full display names. Historical franchise names map first to
# an existing BK source code and then through the canonical relocation mapping.
_TEAM_NAME_TO_SOURCE_CODE = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "oakland raiders": "OAK",
    "los angeles chargers": "LAC",
    "san diego chargers": "SD",
    "los angeles rams": "LAR",
    "st. louis rams": "STL",
    "st louis rams": "STL",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
    "washington football team": "WAS",
    "washington redskins": "WAS",
}


def _aware_utc(value, field: str) -> pd.Timestamp:
    if value is None or pd.isna(value):
        raise ValueError(f"{field} requires a timezone-aware timestamp")
    out = pd.Timestamp(value)
    if out.tzinfo is None or out.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware; got {value!r}")
    return out.tz_convert("UTC")


def normalize_provider_team_name(name: str) -> str:
    if name is None or not str(name).strip():
        raise ValueError("provider team name is required")
    key = " ".join(str(name).strip().casefold().split())
    source_code = _TEAM_NAME_TO_SOURCE_CODE.get(key)
    if source_code is None:
        raise ValueError(f"unknown The Odds API NFL team name {name!r}; refusing to guess")
    return common.normalize_team(source_code)


def _tolerance(value) -> pd.Timedelta:
    tolerance = pd.Timedelta(value)
    if pd.isna(tolerance) or tolerance < pd.Timedelta(0):
        raise ValueError("kickoff_tolerance must be a non-negative duration")
    return tolerance


def _events_from_payloads(payloads) -> list[dict]:
    events = []
    for payload in payloads:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("each historical payload must be a dict with a data list")
        for event in payload["data"]:
            if not isinstance(event, dict):
                raise ValueError("payload.data entries must be event objects")
            if event.get("sport_key") not in (None, "americanfootball_nfl"):
                continue
            event_id = str(event.get("id") or "").strip()
            if not event_id:
                raise ValueError("provider event missing id")
            events.append({
                "provider_event_id": event_id,
                "provider_commence_time": _aware_utc(
                    event.get("commence_time"), "event.commence_time"),
                "provider_home_team": event.get("home_team"),
                "provider_away_team": event.get("away_team"),
            })
    return events


def build_event_mapping(payloads, canonical_games: pd.DataFrame, *,
                        kickoff_tolerance=DEFAULT_KICKOFF_TOLERANCE) -> pd.DataFrame:
    """Match provider events to canonical games using teams and kickoff only."""
    required = ["game_id", "kickoff", "home_team", "away_team"]
    missing = sorted(set(required) - set(canonical_games.columns))
    if missing:
        raise ValueError(f"canonical games missing required columns: {missing}")
    games = canonical_games[required].copy()
    if games["game_id"].isna().any() or not games["game_id"].astype(str).is_unique:
        raise ValueError("canonical games must have unique, non-null game_id values")
    games["kickoff_utc"] = games["kickoff"].map(
        lambda value: _aware_utc(value, "canonical_games.kickoff"))
    if games[["home_team", "away_team"]].isna().any().any():
        raise ValueError("canonical games used for matching require non-null home and away teams")
    if not set(games["home_team"]).issubset(common.BK_CANONICAL_TEAMS):
        raise ValueError("canonical games contain non-canonical home teams")
    if not set(games["away_team"]).issubset(common.BK_CANONICAL_TEAMS):
        raise ValueError("canonical games contain non-canonical away teams")

    tolerance = _tolerance(kickoff_tolerance)
    observed = _events_from_payloads(payloads)
    if not observed:
        return pd.DataFrame(columns=MAPPING_COLUMNS)

    # A provider id may recur across archived snapshots, but its factual event
    # identity must not drift silently. Reschedules require an explicit/manual
    # mapping rather than selecting one observed kickoff.
    by_id = {}
    for event in observed:
        identity = (
            event["provider_commence_time"], event["provider_home_team"],
            event["provider_away_team"],
        )
        previous = by_id.setdefault(event["provider_event_id"], identity)
        if previous != identity:
            raise ValueError(
                f"provider event {event['provider_event_id']} has conflicting archived identity; "
                "create an explicit reviewed mapping")

    rows = []
    for event_id, (commence, home_name, away_name) in sorted(by_id.items()):
        home = normalize_provider_team_name(home_name)
        away = normalize_provider_team_name(away_name)
        if home == away:
            raise ValueError(f"provider event {event_id} has the same normalized home and away team")
        mask = (
            games["home_team"].eq(home)
            & games["away_team"].eq(away)
            & (games["kickoff_utc"].sub(commence).abs() <= tolerance)
        )
        candidates = games.loc[mask]
        if len(candidates) == 0:
            raise ValueError(
                f"provider event {event_id} has no canonical match within {tolerance} "
                f"for {away} at {home}; refusing to guess")
        if len(candidates) > 1:
            ids = sorted(candidates["game_id"].astype(str).tolist())
            raise ValueError(
                f"provider event {event_id} has multiple canonical matches {ids}; "
                "refusing to choose the closest")
        game = candidates.iloc[0]
        rows.append({
            "provider": PROVIDER,
            "provider_event_id": event_id,
            "game_id": str(game["game_id"]),
            "provider_commence_time": commence,
            "provider_home_team": str(home_name),
            "provider_away_team": str(away_name),
            "normalized_home_team": home,
            "normalized_away_team": away,
            "canonical_kickoff": game["kickoff_utc"],
            "match_method": MATCH_METHOD,
            "match_version": EVENT_MAPPING_VERSION,
            "kickoff_tolerance_seconds": int(tolerance.total_seconds()),
        })
    return validate_event_mapping_frame(pd.DataFrame(rows, columns=MAPPING_COLUMNS))


def validate_event_mapping_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MAPPING_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"event mapping missing required columns: {missing}")
    out = frame[MAPPING_COLUMNS].copy()
    for field in ("provider_event_id", "game_id", "match_method", "match_version"):
        if out[field].isna().any() or out[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"event mapping {field} values must be non-blank")
    if not out["provider"].eq(PROVIDER).all():
        raise ValueError(f"event mapping provider must be {PROVIDER!r}")
    if not out["match_version"].eq(EVENT_MAPPING_VERSION).all():
        raise ValueError(f"event mapping match_version must be {EVENT_MAPPING_VERSION!r}")
    for field in ("provider_commence_time", "canonical_kickoff"):
        out[field] = out[field].map(lambda value: _aware_utc(value, field))
    for field in ("normalized_home_team", "normalized_away_team"):
        if out[field].isna().any() or not set(out[field]).issubset(common.BK_CANONICAL_TEAMS):
            raise ValueError(f"event mapping {field} must contain BK canonical team codes")
    if out["normalized_home_team"].eq(out["normalized_away_team"]).any():
        raise ValueError("event mapping home and away teams must differ")
    if out["provider_event_id"].duplicated(keep=False).any():
        raise ValueError("event mapping has duplicate/ambiguous provider_event_id values")
    tolerances = pd.to_numeric(out["kickoff_tolerance_seconds"], errors="coerce")
    if tolerances.isna().any() or (tolerances < 0).any() or not (tolerances % 1 == 0).all():
        raise ValueError("kickoff_tolerance_seconds must be non-negative integers")
    out["kickoff_tolerance_seconds"] = tolerances.astype("int64")
    return out.sort_values("provider_event_id").reset_index(drop=True)


def _mapping_identity(frame: pd.DataFrame) -> tuple[str, list[dict]]:
    validated = validate_event_mapping_frame(frame)
    records = validated.to_dict("records")
    for record in records:
        for field in ("provider_commence_time", "canonical_kickoff"):
            record[field] = _aware_utc(record[field], field).isoformat()
        record["kickoff_tolerance_seconds"] = int(record["kickoff_tolerance_seconds"])
    blob = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return "eventmap_" + hashlib.sha256(blob).hexdigest()[:24], records


def write_event_mapping(frame: pd.DataFrame, path) -> dict:
    """Persist a deterministic, self-identifying JSON mapping artifact."""
    mapping_id, records = _mapping_identity(frame)
    artifact = {
        "event_mapping_version": EVENT_MAPPING_VERSION,
        "provider": PROVIDER,
        "event_mapping_id": mapping_id,
        "records": records,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(destination.parent), prefix=".eventmap_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(artifact, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return artifact


def load_event_mapping(path) -> pd.DataFrame:
    artifact = json.loads(Path(path).read_text())
    if not isinstance(artifact, dict) or artifact.get("event_mapping_version") != EVENT_MAPPING_VERSION:
        raise ValueError("unexpected or missing event_mapping_version")
    if artifact.get("provider") != PROVIDER or not isinstance(artifact.get("records"), list):
        raise ValueError("invalid event mapping artifact")
    frame = validate_event_mapping_frame(pd.DataFrame(artifact["records"], columns=MAPPING_COLUMNS))
    mapping_id, _ = _mapping_identity(frame)
    if artifact.get("event_mapping_id") != mapping_id:
        raise ValueError("event mapping artifact identity mismatch (mutated or corrupt)")
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a versioned The Odds API event-to-BK game mapping",
    )
    parser.add_argument("snapshots", nargs="+", help="saved historical JSON payloads")
    parser.add_argument("--canonical-games", default=str(common.OUT_DIR / "games.parquet"))
    parser.add_argument("--output", required=True, help="destination mapping JSON")
    parser.add_argument("--kickoff-tolerance-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    payloads = [json.loads(Path(path).read_bytes()) for path in args.snapshots]
    games = pd.read_parquet(args.canonical_games)
    mapping = build_event_mapping(
        payloads, games,
        kickoff_tolerance=pd.Timedelta(seconds=args.kickoff_tolerance_seconds),
    )
    artifact = write_event_mapping(mapping, args.output)
    print(f"{artifact['event_mapping_id']}: {len(mapping)} events -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
