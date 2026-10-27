"""Scenario definition, sampling from configured ranges, and the base volume rate."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import yaml

from idsai.config import REPO_ROOT

DEFAULT_SCENARIOS = REPO_ROOT / "configs" / "scenarios.yaml"

Family = Literal["pump_burst", "pump_stealth", "wash_pingpong"]
Region = Literal["train", "test"]

TECHNIQUES: dict[str, tuple[str, str]] = {
    "pump_burst": ("MM-T001", "MM-T001.001"),
    "pump_stealth": ("MM-T001", "MM-T001.002"),
    "wash_pingpong": ("MM-T005", "MM-T005.001"),
}


@dataclass(frozen=True)
class Scenario:
    family: Family
    technique_id: str
    sub_technique: str
    t_start_us: int
    duration_s: float
    impact: float
    vol_mult: float
    rush_size: int
    dump_tau_s: float
    seed: int
    region: Region


def load_scenario_ranges(path: str | Path | None = None) -> dict:
    """Read the scenario ranges YAML (default: `configs/scenarios.yaml` at the repo root)."""
    p = Path(path) if path is not None else DEFAULT_SCENARIOS
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sample_scenario(
    family: Family, region: Region, t_start_us: int, rng: np.random.Generator, ranges: dict
) -> Scenario:
    """Draw one scenario uniformly from `ranges[family][region]`.

    `rush_size` is an integer (both ends inclusive). Fields the family does not configure take
    neutral values: impact 0, dump_tau_s 0, rush_size 1.
    """
    if family not in TECHNIQUES:
        raise ValueError(f"unknown scenario family {family!r}")
    rg = ranges[family][region]

    def uniform(key: str, neutral: float) -> float:
        if key not in rg:
            return neutral
        lo, hi = rg[key]
        return float(rng.uniform(lo, hi))

    duration_s = uniform("duration_s", 0.0)
    impact = uniform("impact", 0.0)
    vol_mult = uniform("vol_mult", 1.0)
    if "rush_size" in rg:
        lo, hi = rg["rush_size"]
        rush_size = int(rng.integers(int(lo), int(hi) + 1))
    else:
        rush_size = 1
    dump_tau_s = uniform("dump_tau_s", 0.0)
    seed = int(rng.integers(0, 2**31 - 1))
    technique_id, sub_technique = TECHNIQUES[family]
    return Scenario(
        family=family,
        technique_id=technique_id,
        sub_technique=sub_technique,
        t_start_us=int(t_start_us),
        duration_s=duration_s,
        impact=impact,
        vol_mult=vol_mult,
        rush_size=rush_size,
        dump_tau_s=dump_tau_s,
        seed=seed,
        region=region,
    )


def base_quote_rate(prev_day_trades: pl.DataFrame) -> float:
    """Typical quote volume per second: `max(total_quote / 86400, median_trade_quote / 60)`.

    The median term keeps sparse pairs (a trade every few minutes) from getting a near-zero
    injection. Raises ValueError when there is no volume (the caller skips that symbol-day).
    """
    if prev_day_trades.height == 0:
        raise ValueError("no trades in the previous day: cannot derive a base quote rate")
    quote = prev_day_trades["quote_qty"].to_numpy().astype(np.float64)
    rate = max(float(quote.sum()) / 86400.0, float(np.median(quote)) / 60.0)
    if not rate > 0.0:
        raise ValueError("previous-day trades carry no quote volume")
    return rate
