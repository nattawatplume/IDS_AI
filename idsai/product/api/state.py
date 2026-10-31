"""State shared by the API routes, and the one health report REST and WebSocket both serve.

Everything here runs on the event loop that owns the store, the bus, the replayer and the
pipeline: none of them is thread-safe.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from idsai.ai.auditor import NO_DATA_TH, Auditor, Check, HealthReport
from idsai.bus import Bus
from idsai.config import Settings
from idsai.data.replay import Replayer
from idsai.data.store import Store
from idsai.pipeline.stream import MODES, StreamPipeline


@dataclass
class AppState:
    """What the routes read. `pipeline` and `replayer` may be None (no stream, no time machine).

    `symbols` is the set the API knows (default: the live symbols); `injected` keeps the Red-Team
    scenarios of this app in memory, never in the store.
    """

    cfg: Settings
    store: Store
    mode: str
    pipeline: StreamPipeline | None
    replayer: Replayer | None
    auditor: Auditor | None = None
    symbols: list[str] | None = None
    injected: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.auditor is None:
            self.auditor = Auditor(self.cfg)
        if self.symbols is None:
            self.symbols = list(self.cfg.live.symbols)

    @property
    def bus(self) -> Bus | None:
        return None if self.pipeline is None else self.pipeline.bus


def clean(x: Any) -> Any:
    """`x` as strict JSON: non-finite floats become None, containers are walked, numpy scalars
    unwrapped, anything else that is not a JSON type becomes its string."""
    if x is None or isinstance(x, bool | int | str):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple | set | frozenset):
        return [clean(v) for v in x]
    item = getattr(x, "item", None)
    if callable(item):  # numpy scalar
        return clean(item())
    return str(x)


def _no_data(name: str, what: str) -> Check:
    return Check(name=name, status="warn", value=None, detail_th=f"{NO_DATA_TH}: {what}")


def health(state: AppState) -> dict[str, Any]:
    """Auditor checks over what the app really has; a check without data is `warn`, never `ok`."""
    pipe = state.pipeline
    inputs: dict[str, dict[str, Any]] = {}
    missing: list[Check] = []
    samples = list(pipe.latency_ms) if pipe is not None else []
    if samples:
        inputs["latency"] = {"samples_ms": samples}
    else:
        missing.append(_no_data("latency", "ยังไม่มีตัวอย่างเวลาแฝง (latency)"))
    if pipe is not None and state.symbols:
        # the slowest symbol decides whether the baselines are ready
        inputs["warmup"] = {"valid_samples": min(pipe.baseline_samples(s) for s in state.symbols)}
    else:
        missing.append(_no_data("warmup", "ยังไม่มี baseline"))
    report = state.auditor.report(**inputs)
    full = HealthReport(checks=[*report.checks, *missing], ts_us=report.ts_us)
    return clean(
        {
            "overall": full.overall,
            "ts_us": full.ts_us,
            "mode": state.mode,
            "checks": [c.model_dump() for c in full.checks],
            "late_trades": None if pipe is None else pipe.late_trades(),
            "dropped": {} if state.bus is None else dict(state.bus.dropped),
        }
    )
