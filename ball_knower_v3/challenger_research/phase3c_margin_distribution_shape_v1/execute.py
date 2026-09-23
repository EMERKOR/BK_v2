"""Execution guard for the frozen, specification-only challenger.

The experiment is not authorized by the implementation PR.  A future caller
must explicitly acknowledge Stage A, persist and review it, and separately
acknowledge Stage B.  This guard deliberately does not run either stage now.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .runner import DEFAULT_CONFIG_PATH, config_sha256, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--acknowledge-stage-a", action="store_true")
    parser.add_argument("--acknowledge-stage-b-after-stage-a-pass", action="store_true")
    parser.add_argument("--stage-a-pass-receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = load_config(DEFAULT_CONFIG_PATH)
    if args.verify_only:
        print(f"{payload['experiment_id']} config_sha256={config_sha256()}")
        return 0
    if not args.acknowledge_stage_a:
        raise SystemExit("Stage A is not authorized without --acknowledge-stage-a")
    if args.acknowledge_stage_b_after_stage_a_pass:
        receipt = args.stage_a_pass_receipt
        if receipt is None or not receipt.is_file():
            raise SystemExit("Stage B requires a persisted Stage A pass receipt")
    raise SystemExit(
        "Execution orchestration is intentionally disabled in the specification PR; "
        "authorize and implement artifact persistence in a separate execution change"
    )


if __name__ == "__main__":
    raise SystemExit(main())
