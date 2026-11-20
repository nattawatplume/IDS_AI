"""scripts/bench_latency.py: stats, every sample kept, the replay wiring, output, no app DB."""

import importlib.util
import json
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import idsai.ai.learning
import idsai.pipeline.stream
from idsai.config import load_settings
from idsai.contracts.frames import TRADES
from idsai.data.store import Store

S = 1_000_000
DAY_US = 86_400 * S
DAY = date(2021, 1, 18)
SYMBOL = "AAABTC"

_path = Path(__file__).resolve().parents[1] / "scripts" / "bench_latency.py"
_spec = importlib.util.spec_from_file_location("bench_latency", _path)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


class FakePipeline:
    """Records the replay it is given; one latency sample per tick after `preroll_done`."""

    instances: list["FakePipeline"] = []

    def __init__(self, cfg, store, detectors, chief, bus, mode):
        self.mode = mode
        self.latency_ms: deque[float] = deque(maxlen=2)  # bounded, like the real one
        self.events: list[tuple[str, int]] = []
        FakePipeline.instances.append(self)

    async def run(self, source) -> None:
        published = False
        async for kind, payload in source:
            if kind == "preroll_done":
                published = True
                self.events.append((kind, payload))
            elif kind == "tick" and published:
                self.latency_ms.append(float(len(self.latency_ms)))


def trades_for(days: list[date], every_s: int = 600) -> pl.DataFrame:
    ts = np.concatenate(
        [bench.day_start_us(d) + np.arange(0, 86_400, every_s, dtype=np.int64) * S for d in days]
    )
    n = ts.size
    return pl.DataFrame(
        {
            "ts_us": ts,
            "recv_us": ts,
            "symbol": [SYMBOL] * n,
            "price": np.ones(n),
            "qty": np.ones(n),
            "quote_qty": np.ones(n),
            "taker_side": ["buy"] * n,
            "trade_id": np.arange(1, n + 1, dtype=np.int64),
            "n_fills": np.ones(n, dtype=np.int32),
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )


@pytest.fixture
def fake_pipeline(monkeypatch):
    FakePipeline.instances = []
    monkeypatch.setattr(idsai.pipeline.stream, "StreamPipeline", FakePipeline)
    return FakePipeline


def test_summarize_percentiles_and_empty():
    s = bench.summarize([float(x) for x in range(1, 101)])
    assert s == {
        "n": 100,
        "mean_ms": 50.5,
        "median_ms": 50.5,
        "p90_ms": 90.1,
        "p99_ms": 99.01,
        "max_ms": 100.0,
    }
    e = bench.summarize([])
    assert e["n"] == 0 and e["p99_ms"] is None and e["max_ms"] is None


def test_collect_keeps_every_sample_and_replays_like_serve(fake_pipeline):
    cfg = load_settings()
    trades = trades_for([DAY - timedelta(days=1), DAY])
    samples, wall_s = bench.collect_latency(cfg, None, [], None, trades, DAY)
    (p,) = fake_pipeline.instances
    assert p.mode == "replay"
    # the replay starts at D 00:00 after a 24 h pre-roll from D-1, as `serve --mode replay`
    assert p.events == [("preroll_done", bench.day_start_us(DAY))]
    # one tick per second of D after the start; far more than the bounded deque's 2
    assert len(samples) > 80_000
    assert samples[:3] == [0.0, 1.0, 2.0]
    assert wall_s >= 0


def test_format_text_is_deterministic_and_says_what_it_measures():
    r = bench.build_result(
        "AAABTC", DAY, Path("data/demo"), bench.summarize([1.0, 2.0, 3.0]), 1.23456, 7
    )
    assert r["meets_target"] is True and r["target_p99_ms"] == 2000.0
    text = bench.format_text(r)
    assert text == bench.format_text(r)
    assert "p99    2.980 ms" in text
    assert "compute latency on this machine" in text and "excludes network" in text
    assert "cpu_count" in text and "python" in text
    none = bench.build_result("AAABTC", DAY, Path("m"), bench.summarize([]), 0.0, 0)
    assert none["meets_target"] is None and "p99    n/a" in bench.format_text(none)


def _cfg_in(data_dir: Path):
    base = load_settings()
    return base.model_copy(update={"paths": base.paths.model_copy(update={"data_dir": data_dir})})


def test_main_writes_json_and_never_touches_an_app_db(tmp_path, monkeypatch, fake_pipeline):
    data_dir = tmp_path / "data"
    with Store(data_dir, tmp_path / "setup.duckdb") as st:
        st.write_trades(trades_for([DAY - timedelta(days=1), DAY]))
    monkeypatch.setattr(bench, "load_settings", lambda path=None: _cfg_in(data_dir))
    monkeypatch.setattr(idsai.ai.learning, "load_bundle", lambda root, cfg: ([], None, {}))

    out = tmp_path / "out" / "latency.json"
    assert bench.main(["--symbol", SYMBOL, "--day", DAY.isoformat(), "--out", str(out)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["symbol"] == SYMBOL and r["day"] == "2021-01-18" and r["speed"] == 0
    assert r["latency"]["n"] > 80_000 and r["n_trades"] == 2 * 144
    assert r["models"] == (data_dir / "demo").as_posix()
    assert set(r["machine"]) >= {"platform", "python", "cpu_count"}
    assert not list(data_dir.rglob("*.duckdb"))  # alerts went to a deleted temp DB


def test_main_needs_the_previous_day(tmp_path, monkeypatch, fake_pipeline, capsys):
    data_dir = tmp_path / "data"
    with Store(data_dir, tmp_path / "setup.duckdb") as st:
        st.write_trades(trades_for([DAY]))
    monkeypatch.setattr(bench, "load_settings", lambda path=None: _cfg_in(data_dir))
    out = tmp_path / "latency.json"
    assert bench.main(["--symbol", SYMBOL, "--day", DAY.isoformat(), "--out", str(out)]) == 1
    assert "2021-01-17" in capsys.readouterr().out and not out.exists()
