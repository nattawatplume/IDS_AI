"""Red-Team simulator: semi-synthetic pump and wash-like injection (replay / sim only)."""

from idsai.detection.simulator.guard import EthicsError, assert_injection_allowed
from idsai.detection.simulator.inject import inject
from idsai.detection.simulator.scenarios import (
    Scenario,
    base_quote_rate,
    load_scenario_ranges,
    sample_scenario,
)

__all__ = [
    "EthicsError",
    "Scenario",
    "assert_injection_allowed",
    "base_quote_rate",
    "inject",
    "load_scenario_ranges",
    "sample_scenario",
]
