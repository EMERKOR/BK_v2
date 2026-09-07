"""Offline ingestion for archived The Odds API historical snapshots.

No network calls, secrets or API clients are present. The pipeline consumes
saved JSON plus a separately reviewed/versioned event mapping, emits normalized
quotes, and writes a manifest that binds every output row to raw and mapping
artifact hashes.
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
from .event_mapping import EVENT_MAPPING_VERSION, load_event_mapping
from .providers.the_odds_api import parse_historical_file
from .quotes import QUOTE_COLUMNS, validate_quote_frame

INGESTION_MANIFEST_VERSION = "market_ingestion_manifest_v0.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stored_path(path: Path, root: Path | None = None) -> str:
    resolved = path.resolve()
    if root is not None:
        try:
            return str(resolved.relative_to(root.resolve()))
        except ValueError as exc:
            raise ValueError(f"archived payload {path} is outside raw_root {root}") from exc
    try:
        return str(resolved.relative_to(common.REPO.resolve()))
    except ValueError:
        # External paths remain absolute so provenance is never made ambiguous.
        return str(resolved)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".market_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_quote_artifact(frame: pd.DataFrame, path) -> dict:
    """Write deterministic UTF-8 JSON Lines (one canonical quote per line)."""
    destination = Path(path)
    validated = validate_quote_frame(frame)
    records = []
    for record in validated[QUOTE_COLUMNS].to_dict("records"):
        for field in (
            "provider_snapshot_time", "bookmaker_last_update_time",
            "market_last_update_time", "ingested_at",
        ):
            value = record[field]
            record[field] = None if pd.isna(value) else pd.Timestamp(value).isoformat()
        record["line"] = None if pd.isna(record["line"]) else float(record["line"])
        record["price_american"] = int(record["price_american"])
        records.append(record)
    payload = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for record in records
    )
    _write_bytes_atomic(destination, payload)
    return {
        "path": str(destination),
        "rows": len(records),
        "sha256": _sha256(destination),
        "format": "jsonl",
    }


def ingest_historical_files(raw_paths, *, event_mapping_path, ingested_at,
                            output_path=None, manifest_path=None,
                            raw_root=None) -> tuple[pd.DataFrame, dict]:
    """Ingest archived files in stable path order using one mapping artifact."""
    supplied_paths = [Path(path).resolve() for path in raw_paths]
    if not supplied_paths:
        raise ValueError("at least one archived historical JSON file is required")
    if len(set(supplied_paths)) != len(supplied_paths):
        raise ValueError("duplicate archived historical JSON paths are not allowed")
    paths = sorted(supplied_paths, key=str)
    for path in paths:
        if not path.is_file():
            raise ValueError(f"archived historical payload does not exist: {path}")

    mapping_path = Path(event_mapping_path).resolve()
    mapping = load_event_mapping(mapping_path)
    root = Path(raw_root).resolve() if raw_root is not None else None
    frames = []
    inputs = []
    for path in paths:
        raw_sha = _sha256(path)
        parsed = parse_historical_file(
            path, ingested_at=ingested_at, event_game_map=mapping,
        )
        frames.append(parsed)
        inputs.append({
            "path": _stored_path(path, root),
            "sha256": raw_sha,
            "rows": int(len(parsed)),
            "provider_snapshot_times": sorted(
                {pd.Timestamp(value).isoformat() for value in parsed["provider_snapshot_time"]}
            ),
        })

    quotes = validate_quote_frame(pd.concat(frames, ignore_index=True))
    if quotes.empty:
        raise ValueError("archived payloads contained no supported NFL featured-market quotes")
    mapping_sha = _sha256(mapping_path)
    manifest = {
        "ingestion_manifest_version": INGESTION_MANIFEST_VERSION,
        "quote_schema_version": str(quotes["quote_schema_version"].iloc[0]),
        "provider": "the_odds_api",
        "ingested_at": pd.Timestamp(quotes["ingested_at"].iloc[0]).isoformat(),
        "event_mapping": {
            "path": _stored_path(mapping_path),
            "sha256": mapping_sha,
            "version": EVENT_MAPPING_VERSION,
        },
        "raw_payloads": inputs,
        "quote_rows": int(len(quotes)),
    }

    if output_path is not None:
        manifest["output"] = write_quote_artifact(quotes, output_path)
        manifest["output"]["path"] = _stored_path(Path(output_path))
    if manifest_path is not None:
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        _write_bytes_atomic(Path(manifest_path), manifest_bytes)
    return quotes, manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize saved The Odds API snapshots without network access",
    )
    parser.add_argument("snapshots", nargs="+", help="saved historical JSON payloads")
    parser.add_argument("--event-mapping", required=True, help="versioned mapping JSON")
    parser.add_argument("--ingested-at", required=True, help="timezone-aware import timestamp")
    parser.add_argument("--output", required=True, help="normalized quote JSONL")
    parser.add_argument("--manifest", required=True, help="ingestion manifest JSON")
    parser.add_argument("--raw-root", help="root used for portable raw payload paths")
    args = parser.parse_args(argv)
    quotes, _ = ingest_historical_files(
        args.snapshots,
        event_mapping_path=args.event_mapping,
        ingested_at=args.ingested_at,
        output_path=args.output,
        manifest_path=args.manifest,
        raw_root=args.raw_root,
    )
    print(f"normalized {len(quotes)} quote rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
