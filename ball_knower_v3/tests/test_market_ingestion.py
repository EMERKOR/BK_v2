import hashlib
import json

import pandas as pd
import pytest

from ball_knower_v3.market.event_mapping import build_event_mapping, write_event_mapping
from ball_knower_v3.market import ingest
from ball_knower_v3.market.ingest import ingest_historical_files


def payload():
    return {
        "timestamp": "2025-10-01T16:00:00Z",
        "data": [{
            "id": "evt1",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2025-10-05T17:00:00Z",
            "home_team": "Buffalo Bills",
            "away_team": "New England Patriots",
            "bookmakers": [{
                "key": "draftkings",
                "last_update": "2025-10-01T15:59:00Z",
                "markets": [{
                    "key": "totals",
                    "last_update": "2025-10-01T15:59:00Z",
                    "outcomes": [
                        {"name": "Over", "price": -105, "point": 47.5},
                        {"name": "Under", "price": -115, "point": 47.5},
                    ],
                }],
            }],
        }],
    }


def games():
    return pd.DataFrame({
        "game_id": ["2025_05_NE_BUF"],
        "kickoff": pd.to_datetime(["2025-10-05T17:00:00Z"], utc=True),
        "home_team": ["BUF"],
        "away_team": ["NE"],
    })


def test_offline_ingestion_traces_raw_and_mapping_artifacts(tmp_path):
    raw = tmp_path / "snapshot.json"
    raw.write_text(json.dumps(payload(), indent=2) + "\n")
    mapping_path = tmp_path / "mapping.json"
    write_event_mapping(build_event_mapping([payload()], games()), mapping_path)
    output = tmp_path / "quotes.jsonl"
    manifest_path = tmp_path / "manifest.json"

    quotes, manifest = ingest_historical_files(
        [raw],
        event_mapping_path=mapping_path,
        ingested_at="2025-10-01T16:02:00Z",
        output_path=output,
        manifest_path=manifest_path,
        raw_root=tmp_path,
    )

    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert len(quotes) == 2
    assert set(quotes["raw_payload_sha256"]) == {raw_sha}
    assert manifest["raw_payloads"][0]["path"] == "snapshot.json"
    assert manifest["raw_payloads"][0]["sha256"] == raw_sha
    assert manifest["event_mapping"]["sha256"] == hashlib.sha256(
        mapping_path.read_bytes()).hexdigest()
    assert manifest["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(manifest_path.read_text())["quote_rows"] == 2


def test_manifest_prefers_repository_relative_mapping_and_output_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.common, "REPO", tmp_path)
    raw = tmp_path / "raw" / "snapshot.json"
    raw.parent.mkdir()
    raw.write_text(json.dumps(payload()))
    mapping_path = tmp_path / "metadata" / "mapping.json"
    mapping_path.parent.mkdir()
    write_event_mapping(build_event_mapping([payload()], games()), mapping_path)
    output = tmp_path / "normalized" / "quotes.jsonl"

    _, manifest = ingest_historical_files(
        [raw],
        event_mapping_path=mapping_path,
        ingested_at="2025-10-01T16:02:00Z",
        output_path=output,
    )

    assert manifest["raw_payloads"][0]["path"] == "raw/snapshot.json"
    assert manifest["event_mapping"]["path"] == "metadata/mapping.json"
    assert manifest["output"]["path"] == "normalized/quotes.jsonl"


def test_normalized_output_is_byte_reproducible(tmp_path):
    raw = tmp_path / "snapshot.json"
    raw.write_text(json.dumps(payload()))
    mapping_path = tmp_path / "mapping.json"
    write_event_mapping(build_event_mapping([payload()], games()), mapping_path)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    kwargs = dict(
        raw_paths=[raw], event_mapping_path=mapping_path,
        ingested_at="2025-10-01T16:02:00Z", raw_root=tmp_path,
    )
    ingest_historical_files(output_path=first, **kwargs)
    ingest_historical_files(output_path=second, **kwargs)
    assert first.read_bytes() == second.read_bytes()


def test_ingestion_refuses_event_identity_drift_after_mapping(tmp_path):
    original = payload()
    mapping_path = tmp_path / "mapping.json"
    write_event_mapping(build_event_mapping([original], games()), mapping_path)
    changed = payload()
    changed["data"][0]["home_team"] = "New York Jets"
    raw = tmp_path / "changed.json"
    raw.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity differs"):
        ingest_historical_files(
            [raw], event_mapping_path=mapping_path,
            ingested_at="2025-10-01T16:02:00Z", raw_root=tmp_path,
        )
