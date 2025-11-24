from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from ball_knower.calibration.calibration_v2 import fit_calibration_v2


def _parse_weeks_arg(weeks_str: str) -> List[int]:
    """
    Parse a weeks string like "1,2,3-5" into a sorted list of ints.
    """
    weeks: List[int] = []
    for part in weeks_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid week range: {part}")
            weeks.extend(range(start, end + 1))
        else:
            weeks.append(int(part))
    return sorted(sorted(set(weeks)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Ball Knower v2 seasonal calibration (drift + isotonic curves) "
            "from a historical calibration dataset."
        )
    )
    parser.add_argument("--season", type=int, required=True, help="NFL season")
    parser.add_argument(
        "--calibration-data-path",
        type=str,
        required=True,
        help=(
            "Path to a CSV or Parquet file containing calibration data. "
            "Must include columns: season, week, home_score, away_score, "
            "predicted_home_score, predicted_away_score, predicted_spread, "
            "predicted_total."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Optional dataset_v2 version label (e.g., v2_0).",
    )
    parser.add_argument(
        "--calibration-weeks",
        type=str,
        default=None,
        help='Optional weeks string like "1-8" or "1,2,3-5". '
        "If omitted, use all weeks available for the season.",
    )
    parser.add_argument(
        "--base-calibration-dir",
        type=str,
        default=None,
        help="Optional override for the root calibration directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.calibration_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Calibration data file not found: {data_path}")

    if data_path.suffix.lower() in {".parquet", ".pq"}:
        calibration_data = pd.read_parquet(data_path)
    else:
        calibration_data = pd.read_csv(data_path)

    if args.calibration_weeks is not None:
        weeks = _parse_weeks_arg(args.calibration_weeks)
    else:
        weeks = None

    base_calibration_dir = (
        Path(args.base_calibration_dir)
        if args.base_calibration_dir is not None
        else None
    )

    drift_coeffs, curves = fit_calibration_v2(
        season=args.season,
        calibration_data=calibration_data,
        calibration_weeks=weeks,
        dataset_version=args.dataset_version,
        base_calibration_dir=base_calibration_dir,
    )

    print(
        f"Fitted calibration for season={args.season} using "
        f"{drift_coeffs['n_games']} games across weeks={drift_coeffs['calibration_weeks']}."
    )
    print("Drift coefficients:")
    print(drift_coeffs)
    print("Calibration curves keys:", list(curves.keys()))


if __name__ == "__main__":
    main()
