# ball_knower/backtesting/config_v2.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import pathlib

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


@dataclass
class SeasonsConfig:
    train: List[int] = field(default_factory=list)
    test: List[int] = field(default_factory=list)


@dataclass
class BankrollConfig:
    initial_units: float = 100.0
    staking: str = "flat"  # "flat" or "fractional_kelly"
    kelly_fraction: float = 0.25
    max_stake_per_bet_units: float = 5.0


@dataclass
class BettingPolicyConfig:
    decision_point: str = "closing"  # reserved for future (open vs close)
    min_edge_points_spread: float = 1.0
    min_edge_points_total: float = 1.0
    max_spread_to_bet: float = 10.5

    # enable/disable bet categories
    enable_home_faves: bool = True
    enable_home_dogs: bool = True
    enable_road_faves: bool = True
    enable_road_dogs: bool = True

    bet_spreads: bool = True
    bet_totals: bool = True
    bet_moneylines: bool = False  # can be wired up later

    # probability-aware policies (optional, for later)
    min_prob_edge_spread: Optional[float] = None
    min_prob_edge_total: Optional[float] = None


@dataclass
class OutputConfig:
    base_dir: str = "data/backtests/v2"
    save_bet_log: bool = True
    save_game_summary: bool = True
    save_metrics: bool = True


@dataclass
class BacktestConfig:
    experiment_id: str
    dataset_version: str
    model_version: str
    seasons: SeasonsConfig
    weeks_test: List[int]
    markets: List[str]
    bankroll: BankrollConfig
    betting_policy: BettingPolicyConfig
    output: OutputConfig

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BacktestConfig":
        seasons = SeasonsConfig(**data["seasons"])
        bankroll = BankrollConfig(**data["bankroll"])
        betting_policy = BettingPolicyConfig(**data["betting_policy"])
        output = OutputConfig(**data["output"])

        return BacktestConfig(
            experiment_id=data["experiment_id"],
            dataset_version=data["dataset_version"],
            model_version=data["model_version"],
            seasons=seasons,
            weeks_test=data["weeks"]["test"],
            markets=data["markets"],
            bankroll=bankroll,
            betting_policy=betting_policy,
            output=output,
        )


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed, but a YAML config file was provided. "
            "Install `pyyaml` or use JSON."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_backtest_config(path: str | pathlib.Path) -> BacktestConfig:
    """
    Load a BacktestConfig from a JSON or YAML file.

    Expected top-level keys:
    - experiment_id, dataset_version, model_version
    - seasons: {train: [...], test: [...]}
    - weeks:   {test: [...]}
    - markets: [...]
    - bankroll: {...}
    - betting_policy: {...}
    - output: {...}
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() in {".json"}:
        raw = _load_json(p)
    elif p.suffix.lower() in {".yaml", ".yml"}:
        raw = _load_yaml(p)
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}")

    return BacktestConfig.from_dict(raw)
