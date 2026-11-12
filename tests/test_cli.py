"""CLI: parser, exit codes, label alignment, downloads, demo pack, bundles and serve wiring.

No test touches the network: downloads go through a fake `Downloader`, labels through a stub of
`build_labels`, and `run_server` is replaced, so no socket is opened.
"""

import asyncio
import json
import math
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import yaml
from starlette.testclient import TestClient

import idsai.eval.experiments as ex
from idsai import cli
from idsai.ai.learning import BundleError, load_bundle
from idsai.bus import Bus
from idsai.config import load_settings
from idsai.contracts.frames import LABELS, TRADES
from idsai.data.replay import Replayer
from idsai.data.store import Store
from idsai.eval.experiments import fit_and_calibrate
from idsai.eval.labels import label_id_for
from idsai.eval.splits import SymDay
from idsai.pipeline.stream import StreamPipeline
from idsai.product.api import AppState, create_app
from tests.synth import S, day_us, make_trades

SYM = "AAABTC"
PUMP = date(2025, 3, 20)  # the demo label's day
PUMP_T0 = day_us(PUMP) + 12 * 3600 * S
FAKE_RANGE = (PUMP - timedelta(days=30), PUMP + timedelta(days=30))
COMMANDS = (
    "labels", "download", "train", "eval", "report", "demo-pack", "serve", "live-prep", "live-check"
)


def write_cfg(tmp: Path, **extra) -> Path:
    data = {
        "paths": {
            "data_dir": (tmp / "data").as_posix(),
            "models_dir": (tmp / "models").as_posix(),
            "app_db": (tmp / "data" / "app.duckdb").as_posix(),
            "research_db": (tmp / "data" / "research.duckdb").as_posix(),
            "report_json": (tmp / "data" / "report.json").as_posix(),
        },
        **extra,
    }
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def label_rows(*rows: tuple[str, str, int]) -> pl.DataFrame:
    """`(source, symbol, t_start_us)` -> REAL telegram labels."""
    return pl.DataFrame(
        [
            (label_id_for(sym, t), src, sym, t, t, "MM-T001", "telegram_pump", None, "REAL")
            for src, sym, t in rows
        ],
        schema=LABELS,
        orient="row",
    )


class FakeDownloader:
    """Synthetic BTC-pair trades for every day inside `lo..hi`; None (not in the archive) else."""

    def __init__(self, lo: date = FAKE_RANGE[0], hi: date = FAKE_RANGE[1], rate: float = 0.2):
        self.lo, self.hi, self.rate = lo, hi, rate
        self.calls: list[tuple[str, date]] = []
        self.closed = False

    def fetch(self, symbol: str, day: date) -> pl.DataFrame | None:
        self.calls.append((symbol, day))
        if not self.lo <= day <= self.hi:
            return None
        k = day.toordinal()
        return make_trades(day, 1, symbol=symbol, seed=k % 1000, rate=self.rate, first_id=k * 10**6)

    def close(self) -> None:
        self.closed = True


# ------------------------------------------------------------------ parser


def test_help_lists_every_subcommand(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for c in COMMANDS:
        assert c in out


def test_serve_binds_loopback_port_8800_by_default():
    args = cli.build_parser().parse_args(["serve"])
    assert cli.resolve_bind(args, load_settings()) == ("127.0.0.1", 8800)
    assert args.mode == "replay" and args.speed == 60.0
    args = cli.build_parser().parse_args(["serve", "--port", "9000", "--host", "0.0.0.0"])
    assert cli.resolve_bind(args, load_settings()) == ("0.0.0.0", 9000)


def test_usage_errors_exit_1(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["no-such-command"])
    assert e.value.code == 1
    assert cli.main(["download", "--symbols", "BTCUSDT"]) == 1  # neither --days nor --day
    assert "--days" in capsys.readouterr().err


def test_live_budget_is_per_symbol_day():
    assert cli.live_budget_per_symbol_day(load_settings()) == pytest.approx(2.0 * 24 / 3)


# ------------------------------------------------------------------ train


def test_train_live_with_empty_symbols_exits_1(tmp_path, capsys):
    cfg = write_cfg(tmp_path, live={"symbols": []})
    assert cli.main(["--config", str(cfg), "train", "--profile", "live"]) == 1
    assert "live.symbols" in capsys.readouterr().err


def test_train_live_without_data_names_the_download(tmp_path, capsys):
    cfg = write_cfg(tmp_path, live={"symbols": ["BTCUSDT"]})
    assert cli.main(["--config", str(cfg), "train", "--profile", "live"]) == 1
    assert "idsai download --symbols BTCUSDT" in capsys.readouterr().err


def test_live_days_split_train_and_calibration(tmp_path):
    with Store(tmp_path / "d", tmp_path / "d" / "a.duckdb") as store:
        for k in range(9):
            store.write_trades(make_trades(date(2025, 3, 1) + timedelta(days=k), 1, rate=0.001))
        train, cal = cli.live_days(store, ["TESTBTC"], 7)
    days = [date(2025, 3, 1) + timedelta(days=k) for k in range(2, 9)]  # the last 7 with a D-1
    assert [sd.day for sd in train] == days[:5] and [sd.day for sd in cal] == days[5:]


def test_fit_and_calibrate_refuses_overlapping_days():
    sd = SymDay(SYM, PUMP)
    with pytest.raises(ValueError, match="overlap"):
        fit_and_calibrate(load_settings(), None, 0, [sd], [sd], 2.0)


# ------------------------------------------------------------------ labels --align


def spiked(day: date, peak_us: int, symbol: str = SYM) -> pl.DataFrame:
    """Two days of quiet trades plus a volume spike in the minute of `peak_us`."""
    base = make_trades(day - timedelta(days=1), 2, symbol=symbol, seed=3, rate=0.05)
    n = 50
    ts = peak_us + np.arange(n) * 100_000
    spike = pl.DataFrame(
        {
            "ts_us": ts,
            "recv_us": ts,
            "symbol": [symbol] * n,
            "price": [5e-4] * n,
            "qty": [1e6] * n,
            "quote_qty": [500.0] * n,
            "taker_side": ["buy"] * n,
            "trade_id": np.arange(10**9, 10**9 + n),
            "n_fills": np.ones(n, dtype=np.int32),
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )
    return pl.concat([base, spike])


class StubStore:
    """What `run_alignment` reads: `has_day` and `read_trades`."""

    def __init__(self, trades: pl.DataFrame | None):
        self.trades = trades

    def has_day(self, symbol: str, day: date) -> bool:
        return self.trades is not None and day in {
            date(1970, 1, 1) + timedelta(days=int(t) // (86_400 * S))
            for t in self.trades["ts_us"].to_list()
        }

    def read_trades(self, symbol: str, start_us: int, end_us: int) -> pl.DataFrame:
        t = self.trades.filter(pl.col("symbol") == symbol)
        return t.filter((pl.col("ts_us") >= start_us) & (pl.col("ts_us") < end_us))


@pytest.mark.parametrize(
    # lags are whole minutes (bucket start - label): 120 s is the first failing value
    ("lag_s", "code"), [(0, 0), (60, 0), (-60, 0), (120, 1), (-120, 1), (3600, 1), (-600, 1)]
)
def test_align_exit_code_follows_the_median_lag(lag_s, code, capsys):
    labels = label_rows(("sapienza", SYM, PUMP_T0))
    got, lines = cli.run_alignment(StubStore(spiked(PUMP, PUMP_T0 + lag_s * S)), labels)
    assert got == code
    text = "\n".join(lines)
    assert "ฮิสโทแกรม" in text and f"{lag_s:+d}" in text


def test_align_fails_closed_when_nothing_can_be_measured():
    labels = label_rows(("sapienza", SYM, PUMP_T0))
    code, lines = cli.run_alignment(StubStore(None), labels)  # no data at all
    assert code == 1 and "วัดไม่ได้" in lines[0]
    empty = pl.DataFrame(schema={"has_data": pl.Boolean, "lag_s": pl.Float64})
    code, lines = cli.alignment_verdict(empty)
    assert code == 1 and "วัดไม่ได้" in lines[0]


def test_labels_align_command_end_to_end(tmp_path, monkeypatch, capsys):
    cfg = write_cfg(tmp_path)
    s = load_settings(cfg)
    other = PUMP_T0 + 5 * 86_400 * S  # a label whose day is not in the store: not measured
    labels = label_rows(("sapienza", SYM, PUMP_T0), ("hu", SYM, other))
    monkeypatch.setattr(cli, "build_labels", lambda c: (labels, ["sapienza", "hu"]))
    with Store(s.paths.data_dir, s.paths.app_db) as store:
        store.write_trades(spiked(PUMP, PUMP_T0 + 30 * S))
        store.write_labels(label_rows(("sapienza", SYM, PUMP_T0 - 3600 * S)))  # a stale offset
    assert cli.main(["--config", str(cfg), "labels", "--align"]) == 0
    out = capsys.readouterr().out
    assert "วัดได้ 1 จาก 1 ป้าย" in out
    with Store(s.paths.data_dir, s.paths.app_db) as store:
        assert sorted(store.read_labels()["t_start_us"].to_list()) == [PUMP_T0, other]
    assert cli.main(["--config", str(cfg), "labels"]) == 0  # without --align: no check


# ------------------------------------------------------------------ download


def test_ingest_skips_days_already_stored(tmp_path):
    dl = FakeDownloader()
    days = [(SYM, PUMP), (SYM, PUMP - timedelta(days=1)), (SYM, date(2020, 1, 1))]
    with Store(tmp_path / "d", tmp_path / "d" / "a.duckdb") as store:
        c = cli.ingest(store, dl, days)
        assert (c.downloaded, c.skipped, c.missing) == (2, 0, 1)
        again = cli.ingest(store, dl, days)
    assert (again.downloaded, again.skipped, again.missing) == (0, 2, 1)
    assert len(dl.calls) == 4  # only the missing day was asked again


def test_download_last_days_moves_back_when_yesterday_is_unpublished(tmp_path, monkeypatch):
    cfg = write_cfg(tmp_path)
    dl = FakeDownloader(hi=PUMP - timedelta(days=1))  # "today" is PUMP + 1: yesterday is missing
    monkeypatch.setattr(cli, "make_downloader", lambda c: dl)
    monkeypatch.setattr(cli, "_today", lambda: PUMP + timedelta(days=1))
    assert cli.main(["--config", str(cfg), "download", "--symbols", SYM, "--days", "2"]) == 0
    s = load_settings(cfg)
    with Store(s.paths.data_dir, s.paths.app_db) as store:
        have = store.days(SYM)
    assert have == [PUMP - timedelta(days=k) for k in (3, 2, 1)]  # 2 days + the first one's D-1
    assert dl.closed


def test_download_one_day_takes_its_previous_day(tmp_path, monkeypatch):
    cfg = write_cfg(tmp_path)
    dl = FakeDownloader()
    monkeypatch.setattr(cli, "make_downloader", lambda c: dl)
    assert cli.main(["--config", str(cfg), "download", "--symbols", SYM, "--day", str(PUMP)]) == 0
    assert sorted(dl.calls) == [(SYM, PUMP - timedelta(days=1)), (SYM, PUMP)]


def test_research_symdays_cover_events_and_spaced_normal_candidates():
    cfg = load_settings(
        overrides={
            "splits": {
                "train": ["2025-01-01", "2025-01-31"],
                "cal": ["2025-02-01", "2025-02-28"],
                "accept": ["2025-03-01", "2025-03-31"],
                "test": ["2025-04-01", "2025-04-30"],
                "normal_days_per_symbol": 2,
                "min_gap_days": 2,
                "drift_v1": None,
                "drift_v2": None,
            }
        }
    )
    late = day_us(date(2025, 4, 10)) + 86_400 * S - 5 * 60 * S  # 23:55: needs D+1
    cal_label = day_us(date(2025, 2, 10)) + 3600 * S
    want = set(cli.research_symdays(cfg, label_rows(("hu", SYM, late), ("hu", SYM, cal_label))))
    assert {(SYM, date(2025, 4, 9)), (SYM, date(2025, 4, 10)), (SYM, date(2025, 4, 11))} <= want
    assert (SYM, date(2025, 2, 10)) not in want  # the cal period has no events
    events = {date(2025, 4, 9), date(2025, 4, 10), date(2025, 4, 11)}
    labels = [date(2025, 4, 10), date(2025, 2, 10)]

    def gap(d: date) -> int:
        return min(abs((d - ld).days) for ld in labels)

    normal = {d for _, d in want} - events
    # a normal day keeps the gap from every label, or is the D-1 of one that does
    assert all(gap(d) >= 2 or gap(d + timedelta(days=1)) >= 2 for d in normal)
    anchors = sorted(d for d in normal if gap(d) >= 2 and d - timedelta(days=1) in normal)
    for period in ("train", "cal", "accept", "test"):
        lo, hi = getattr(cfg.splits, period)
        mine = [d for d in anchors if lo <= d <= hi]
        assert 1 <= len(mine) <= 2 * 2  # normal_days_per_symbol x RESEARCH_CANDIDATE_FACTOR


# ------------------------------------------------------------------ host allowlist


@pytest.fixture
def bare_state(tmp_path):
    cfg = load_settings(
        overrides={"paths": {"data_dir": str(tmp_path / "d"), "app_db": str(tmp_path / "a.db")}}
    )
    with Store(cfg.paths.data_dir, cfg.paths.app_db) as store:
        yield AppState(cfg, store, "replay", None, None, symbols=[SYM])


def test_host_allowlist_rejects_a_foreign_host_and_accepts_loopback(bare_state):
    app = create_app(bare_state, allowed_hosts=cli.allowed_hosts("127.0.0.1"))
    with TestClient(app, base_url="http://127.0.0.1:8800") as c:
        assert c.get("/api/symbols").status_code == 200
        for host in ("localhost:8800", "[::1]:8800", "127.0.0.1"):
            assert c.get("/api/symbols", headers={"host": host}).status_code == 200
        bad = c.get("/api/symbols", headers={"host": "evil.example"})
        assert bad.status_code == 400 and "--host <IP" in bad.text  # a Thai hint, not a bare 400
        assert c.get("/api/symbols", headers={"host": "evil.example:8800"}).status_code == 400
        with pytest.raises(Exception):  # noqa: B017 - the handshake is refused before accept
            with c.websocket_connect("/ws/stream", headers={"host": "evil.example"}):
                pass


def test_allowlist_adds_the_bound_host_and_default_app_checks_none(bare_state):
    assert "192.168.1.5" in cli.allowed_hosts("192.168.1.5")
    assert "[::1]" in cli.allowed_hosts("::1")
    with TestClient(create_app(bare_state)) as c:  # existing behaviour: no host check
        assert c.get("/api/symbols", headers={"host": "evil.example"}).status_code == 200


# ------------------------------------------------------------------ serve errors


def test_serve_replay_without_previous_day_exits_1_in_thai(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    s = load_settings(cfg)
    with Store(s.paths.data_dir, s.paths.app_db) as store:
        store.write_trades(make_trades(PUMP, 1, symbol=SYM, rate=0.01))  # D only, no D-1
    argv = ["--config", str(cfg), "serve", "--mode", "replay", "--symbol", SYM, "--day", str(PUMP)]
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "ไม่มีข้อมูล" in err and f"idsai download --symbols {SYM} --day {PUMP}" in err


def test_serve_without_demo_pack_or_live_bundle_exits_1(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert cli.main(["--config", str(cfg), "serve"]) == 1
    assert "idsai demo-pack" in capsys.readouterr().err
    assert cli.main(["--config", str(cfg), "serve", "--mode", "live"]) == 1
    assert "idsai train --profile live" in capsys.readouterr().err
    assert cli.main(["--config", str(cfg), "serve", "--symbol", SYM]) == 1  # --day missing


def test_report_without_runs_exits_1(tmp_path, capsys):
    assert cli.main(["--config", str(write_cfg(tmp_path)), "report"]) == 1
    assert "idsai eval" in capsys.readouterr().err


# ------------------------------------------------------------------ demo pack + bundle


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("demo")
    cfg_path = write_cfg(
        tmp,
        ae={"epochs": 2, "hidden": 8, "bottleneck": 4, "patience": 1},
        chief={"budget_per_symbol_day": 10.0},
    )
    cfg = load_settings(cfg_path)
    # one Sapienza pump, one older Sapienza label (blocks days 8..12 before the pump) and a Hu
    # label of another symbol (not a demo pump: only Sapienza labels are)
    labels = label_rows(
        ("sapienza", SYM, PUMP_T0),
        ("sapienza", SYM, PUMP_T0 - 10 * 86_400 * S),
        ("hu", "BBBBTC", PUMP_T0 + 86_400 * S),
    )
    dl = FakeDownloader()
    seen = {}
    real = ex.fit_and_calibrate

    def spy(cfg, store, seed, train_days, cal_days, budget, *a, **k):
        seen.update(train=train_days, cal=cal_days, budget=budget)
        seen["fitted"] = real(cfg, store, seed, train_days, cal_days, budget, *a, **k)
        return seen["fitted"]

    mp = pytest.MonkeyPatch()
    mp.setattr(cli, "build_labels", lambda c: (labels, ["sapienza", "hu"]))
    mp.setattr(cli, "make_downloader", lambda c: dl)
    mp.setattr(ex, "fit_and_calibrate", spy)
    try:
        code = cli.main(
            ["--config", str(cfg_path), "demo-pack", "--pumps", "1", "--normal-days", "3"]
        )
    finally:
        mp.undo()
    return SimpleNamespace(
        code=code,
        cfg=cfg,
        cfg_path=cfg_path,
        dl=dl,
        dir=cli.demo_dir(cfg),
        seen=seen,
        labels=labels,
    )


def test_demo_pack_builds_a_manifest_and_bundle(demo):
    assert demo.code == 0
    man = json.loads((demo.dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["symbols"] == [SYM]
    assert man["pump_days"] == [
        {
            "symbol": SYM,
            "day": str(PUMP),
            "label_id": label_id_for(SYM, PUMP_T0),
            "source": "sapienza",
            "t_start_us": PUMP_T0,
        }
    ]
    train = [date.fromisoformat(d) for _, d in man["normal_days"]["train"]]
    cal = [date.fromisoformat(d) for _, d in man["normal_days"]["cal"]]
    assert len(train) == 2 and len(cal) == 1 and max(train) < min(cal)  # cal = the most recent
    labels = [PUMP, PUMP - timedelta(days=10)]
    for d in train + cal:
        assert min(abs((d - ld).days) for ld in labels) >= 2
    assert demo.dl.closed
    doc = json.loads((demo.dir / "bundle.json").read_text(encoding="utf-8"))
    assert doc["profile"] == "demo" and doc["symbols"] == [SYM]
    assert doc["registry_versions"] == {"ai.autoencoder_trade": "v1"}
    assert doc["model_versions"]["ai.autoencoder_trade"].startswith("demo-")
    assert doc["cal_days"] == man["normal_days"]["cal"]
    assert math.isfinite(doc["tau"])


def test_demo_pack_writes_its_labels_to_the_store(demo):
    # training_mask reads labels from the store: without them nothing would be masked
    with Store(demo.cfg.paths.data_dir, demo.cfg.paths.app_db) as store:
        stored = store.read_labels()
    assert sorted(stored["label_id"].to_list()) == sorted(demo.labels["label_id"].to_list())
    assert "train_excluded_frac" in demo.seen["fitted"].info


def test_chief_is_calibrated_on_the_held_out_days_only(demo):
    train, cal = demo.seen["train"], demo.seen["cal"]
    assert len(train) == 2 and len(cal) == 1 and not set(train) & set(cal)
    info = demo.seen["fitted"].info
    assert info["cal_days"] == 1  # one held-out symbol-day, not the two training days
    assert info["cal_rows"] == 8640  # one day of 10 s rows
    assert demo.seen["fitted"].chief.n_cal <= 8640
    assert demo.seen["budget"] == 10.0  # chief.budget_per_symbol_day of the config


def _series(cfg, dets, chief) -> list[dict]:
    d0 = day_us(PUMP)
    with Store(cfg.paths.data_dir, cfg.paths.app_db) as store:
        trades = store.read_trades(SYM, d0 - 8 * 3600 * S, d0 + 1800 * S)
        rp = Replayer(trades, speed=0, start_us=d0, preroll_from_us=d0 - 8 * 3600 * S)
        pipe = StreamPipeline(cfg, store, dets, chief, Bus(), "replay")
        asyncio.run(pipe.run(rp))
    return pipe.series(SYM)


def test_bundle_round_trip_scores_like_the_fitted_models(demo):
    dets, chief, doc = load_bundle(demo.dir, demo.cfg)
    assert chief.tau == doc["tau"] and chief.model_versions == doc["model_versions"]
    loaded = _series(demo.cfg, dets, chief)
    fitted = demo.seen["fitted"]
    direct = _series(demo.cfg, fitted.detectors, fitted.chief)
    assert loaded and loaded[0]["t_end_us"] >= day_us(PUMP)
    assert loaded == direct  # every row, F and lane identical
    # the loaded AE scores finite rows: some are rarer than its calibration minimum (p < 1)
    assert any(p["lane_ai"] > 0 for p in loaded)


def test_prepare_replay_prerolls_the_previous_day_and_starts_before_the_label(demo):
    args = cli.build_parser().parse_args(["serve"])
    with Store(demo.cfg.paths.data_dir, demo.cfg.paths.app_db) as store:
        state, background = cli.prepare_replay(args, demo.cfg, store)
        background().close()
    rp = state.replayer
    assert rp._pos == day_us(PUMP) - 86_400 * S  # pre-roll origin: D-1 00:00
    assert rp._start == PUMP_T0 - 1800 * S  # 30 min before the label
    assert rp._speed == 60.0
    assert state.budget_per_symbol_day == 10.0  # the bundle's budget (None if not passed)


def test_prepare_replay_drops_alerts_of_earlier_sessions_from_the_start_on(demo):
    from tests.product.test_api import make_alert

    start = PUMP_T0 - 1800 * S  # where the demo replay starts
    before = make_alert("r-before", ts=start - 600 * S, symbol=SYM)  # window ends 600 s before
    at = make_alert("r-at", ts=start, symbol=SYM)  # window ends at the start
    ahead = make_alert("r-ahead", ts=start + 3600 * S, symbol=SYM)
    sim = make_alert("s-ahead", ts=start + 3600 * S, symbol=SYM, prov="sim")
    args = cli.build_parser().parse_args(["serve"])
    with Store(demo.cfg.paths.data_dir, demo.cfg.paths.app_db) as store:
        for a in (before, at, ahead, sim):
            store.upsert_alert(a)
        store.insert_feedback("r-ahead", "tp", "analyst")
        _, background = cli.prepare_replay(args, demo.cfg, store)
        background().close()
        left = {a.alert_id for a in store.list_alerts(1000)}
        assert {"r-before", "s-ahead"} <= left and not {"r-at", "r-ahead"} & left
        assert [f["verdict"] for f in store.list_feedback("r-ahead")] == ["tp"]  # kept
        store.delete_alerts("replay", ["r-before"])
        store.delete_alerts("sim", ["s-ahead"])


def test_bundle_refuses_another_config_or_a_missing_dir(demo, tmp_path):
    other = load_settings(demo.cfg_path)
    other = other.model_copy(update={"chief": other.chief.model_copy(update={"suppress_for_s": 1})})
    with pytest.raises(BundleError, match="config_hash"):
        load_bundle(demo.dir, other)
    with pytest.raises(BundleError, match="ไม่พบชุดโมเดล"):
        load_bundle(tmp_path / "nothing", demo.cfg)


def test_serve_replay_wires_the_demo_pack(demo, monkeypatch, capsys):
    seen = {}

    def fake_run_server(app, host, port, background):
        seen.update(host=host, port=port)
        coro = background()
        coro.close()  # never started: the replay is not played here
        with TestClient(app, base_url="http://127.0.0.1:8800") as c:
            seen["symbols"] = c.get("/api/symbols").json()
            seen["health"] = c.get("/api/health").json()["mode"]
            seen["foreign"] = c.get("/api/symbols", headers={"host": "evil.example"}).status_code

    monkeypatch.setattr(cli, "run_server", fake_run_server)
    assert cli.main(["--config", str(demo.cfg_path), "serve"]) == 0
    assert seen == {
        "host": "127.0.0.1",
        "port": 8800,
        "symbols": [SYM],
        "health": "replay",
        "foreign": 400,
    }
    out = capsys.readouterr().out
    assert "11:30 UTC" in out  # 30 min before the 12:00 label


def test_serve_on_another_host_is_refused_without_login(demo, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_server", lambda app, host, port, bg: bg().close())
    argv = ["--config", str(demo.cfg_path), "serve", "--host", "0.0.0.0", "--port", "9001"]
    assert cli.main(argv) == 1
    assert "ล็อกอิน" in capsys.readouterr().err


def test_run_server_cancels_the_pipeline_and_bounds_the_shutdown(monkeypatch):
    import uvicorn
    from fastapi import FastAPI

    seen = {}
    init = uvicorn.Server.__init__

    def spy_init(self, config):
        seen.update(server=self, grace=config.timeout_graceful_shutdown)
        init(self, config)

    monkeypatch.setattr(uvicorn.Server, "__init__", spy_init)

    async def background():
        try:
            await asyncio.sleep(0.3)
            seen["server"].should_exit = True  # what Ctrl+C does
            await asyncio.sleep(60)  # a pipeline that would never end on its own
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise

    cli.run_server(FastAPI(), "127.0.0.1", 0, background)  # loopback, port 0: no fixed socket
    assert seen.get("cancelled") is True and seen["grace"] == 5


def test_train_live_uses_the_live_budget_and_dir_and_warns_on_stale_data(
    tmp_path, monkeypatch, capsys
):
    cfg = write_cfg(tmp_path, live={"symbols": ["TESTBTC"], "train_days": 4})
    s = load_settings(cfg)
    with Store(s.paths.data_dir, s.paths.app_db) as store:
        for k in range(5):
            store.write_trades(make_trades(date(2025, 3, 1) + timedelta(days=k), 1, rate=0.001))
    seen = {}

    def fake_train_and_save(cfg, store, out, profile, seed, train, cal, budget, extra=None):
        seen.update(out=out, profile=profile, budget=budget, train=train, cal=cal)

    monkeypatch.setattr(cli, "train_and_save", fake_train_and_save)
    monkeypatch.setattr(cli, "_today", lambda: date(2025, 3, 6))  # newest day is yesterday
    assert cli.main(["--config", str(cfg), "train", "--profile", "live"]) == 0
    assert seen["budget"] == pytest.approx(16.0) and seen["profile"] == "live"
    assert seen["out"] == s.paths.models_dir / "live"
    assert len(seen["train"]) == 2 and len(seen["cal"]) == 2
    assert "เก่ากว่า" not in capsys.readouterr().err
    monkeypatch.setattr(cli, "_today", lambda: date(2025, 3, 20))
    assert cli.main(["--config", str(cfg), "train", "--profile", "live"]) == 0
    assert "เก่ากว่า" in capsys.readouterr().err


@pytest.mark.parametrize("speed", ["0", "-1", "nan", "inf", "fast"])
def test_bad_speed_exits_1_in_thai(speed, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["serve", "--speed", speed])
    assert e.value.code == 1 and "ความเร็ว" in capsys.readouterr().err


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]"])
def test_loopback_hosts_do_not_warn(host, demo, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_server", lambda app, h, port, bg: bg().close())
    assert cli.main(["--config", str(demo.cfg_path), "serve", "--host", host]) == 0
    assert "ไม่มีการล็อกอิน" not in capsys.readouterr().err


def test_wildcard_bind_explains_which_hosts_are_accepted(demo, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_server", lambda app, h, port, bg: bg().close())
    s = load_settings(demo.cfg_path)
    with Store(s.paths.data_dir, s.paths.app_db) as store:
        if not store.has_enabled_admin():
            store.add_user("root", "", "admin", "x", 1)
    assert cli.main(["--config", str(demo.cfg_path), "serve", "--host", "0.0.0.0", "--auth"]) == 0
    assert "--host <IP" in capsys.readouterr().err


def test_wildcard_bind_note_lists_public_hosts(demo, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_server", lambda app, h, port, bg: bg().close())
    s = load_settings(demo.cfg_path)
    with Store(s.paths.data_dir, s.paths.app_db) as store:
        if not store.has_enabled_admin():
            store.add_user("root", "", "admin", "x", 1)
    argv = ["serve", "--host", "0.0.0.0", "--auth", "--public-host", "x.hf.space"]
    assert cli.main(["--config", str(demo.cfg_path), *argv]) == 0
    note = [ln for ln in capsys.readouterr().err.splitlines() if "หมายเหตุ" in ln]
    assert note and "x.hf.space" in note[0]


def test_serve_with_auth_gates_the_api_and_purges_expired_sessions(demo, monkeypatch):
    from idsai.product.auth import hash_password

    pw = "boss-password-1"
    s = load_settings(demo.cfg_path)
    with Store(s.paths.data_dir, s.paths.app_db) as store:
        if store.get_user_by_name("boss") is None:
            store.add_user("boss", "Boss", "admin", hash_password(pw), 1)
        store.create_session(store.get_user_by_name("boss")["user_id"], "stale", 1, 2, False)
    seen = {}

    def fake_run_server(app, host, port, background):
        background().close()
        with TestClient(app, base_url="http://127.0.0.1:8800") as c:
            seen["anon"] = c.get("/api/symbols").status_code
            seen["health"] = c.get("/api/health").status_code
            seen["login"] = c.post(
                "/api/auth/login", json={"username": "boss", "password": pw}
            ).status_code
            seen["me"] = c.get("/api/me").json()["role"]

    monkeypatch.setattr(cli, "run_server", fake_run_server)
    assert cli.main(["--config", str(demo.cfg_path), "serve", "--auth"]) == 0
    assert seen == {"anon": 401, "health": 200, "login": 200, "me": "admin"}
    with Store(s.paths.data_dir, s.paths.app_db) as store:
        assert store.purge_sessions(10**18) == 1  # the stale one was gone at startup


def test_help_imports_no_models_and_names_every_experiment():
    import subprocess
    import sys

    code = (
        "import sys, idsai.cli as c; c.build_parser(); "
        "print([m for m in ('torch', 'sklearn', 'idsai.eval.experiments', "
        "'idsai.pipeline.stream', 'duckdb') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
    assert cli.EXPERIMENT_NAMES == ex.EXPERIMENTS


# ------------------------------------------------------------------ live: prep, source, check

LIVE_SYMS = ["SOLUSDT", "DOGEUSDT"]


def live_store(tmp_path, days: list[date], symbols=None):
    store = Store(tmp_path / "data", tmp_path / "data" / "a.duckdb")
    for s in symbols or LIVE_SYMS:
        for d in days:
            store.write_trades(make_trades(d, 1, symbol=s, seed=d.toordinal() % 100, rate=0.05))
    return store


def run_events(source) -> list:
    async def main():
        return [ev async for ev in source]

    return asyncio.run(main())


def test_default_live_universe_is_three_pump_prone_coins():
    s = load_settings()
    assert s.live.symbols == ["SOLUSDT", "DOGEUSDT", "PEPEUSDT"] and s.live.universe_size == 3


def test_live_prep_ingest_takes_the_two_latest_published_days(tmp_path, monkeypatch):
    cfg = write_cfg(tmp_path)
    dl = FakeDownloader(hi=PUMP - timedelta(days=1))
    monkeypatch.setattr(cli, "make_downloader", lambda c: dl)
    now = day_us(PUMP) + 3600 * S
    with Store(tmp_path / "d", tmp_path / "d" / "a.duckdb") as store:
        cli.live_prep_ingest(load_settings(cfg), store, LIVE_SYMS, now)
        for s in LIVE_SYMS:
            assert store.days(s) == [PUMP - timedelta(days=2), PUMP - timedelta(days=1)]
    assert dl.closed


def test_live_prep_command_uses_the_live_symbols_and_the_tmp_store(tmp_path, monkeypatch):
    cfg = write_cfg(tmp_path, live={"symbols": ["SOLUSDT"]})
    dl = FakeDownloader()
    monkeypatch.setattr(cli, "make_downloader", lambda c: dl)
    monkeypatch.setattr(cli, "_now_us", lambda: day_us(PUMP) + 3600 * S)
    assert cli.main(["--config", str(cfg), "live-prep"]) == 0
    s = load_settings(cfg)
    with Store(s.paths.data_dir, s.paths.app_db) as store:
        assert store.days("SOLUSDT") == [PUMP - timedelta(days=2), PUMP - timedelta(days=1)]
    assert {c[0] for c in dl.calls} == {"SOLUSDT"}


def test_live_prep_with_empty_symbols_exits_1(tmp_path, capsys):
    cfg = write_cfg(tmp_path, live={"symbols": []})
    assert cli.main(["--config", str(cfg), "live-prep"]) == 1
    assert "live.symbols" in capsys.readouterr().err


def test_live_source_is_the_last_12_h_of_the_newest_stored_day(tmp_path):
    d1 = PUMP - timedelta(days=1)
    now = day_us(PUMP) + 10 * 3600 * S
    with live_store(tmp_path, [d1 - timedelta(days=1), d1]) as store:
        src, end = cli.live_preroll_source(store, LIVE_SYMS, now)
        got = run_events(src)
        assert end == day_us(PUMP)  # midnight after the newest day
        start = end - 12 * 3600 * S
        frame = pl.concat([store.read_trades(s, start, end) for s in LIVE_SYMS])
        every = run_events(Replayer(frame, speed=0, start_us=start))
        want = [e for e in every if e[0] != "preroll_done"]
    assert got == want and got and got[-1][0] == "tick"


def test_live_source_of_a_day_still_running_ends_now(tmp_path):
    now = day_us(PUMP) + 10 * 3600 * S
    with live_store(tmp_path, [PUMP - timedelta(days=1), PUMP]) as store:
        src, end = cli.live_preroll_source(store, LIVE_SYMS, now)
        run_events(src)
    assert end == now


def test_live_source_is_none_without_fresh_enough_data(tmp_path):
    now = day_us(PUMP) + 10 * 3600 * S
    with live_store(tmp_path, []) as store:
        assert cli.live_preroll_source(store, LIVE_SYMS, now) is None
    with live_store(tmp_path, [PUMP - timedelta(days=5)]) as store:  # ends before now - 24 h
        assert cli.live_preroll_source(store, LIVE_SYMS, now) is None


def test_download_split_live_takes_the_live_symbols_and_train_days(tmp_path, monkeypatch):
    cfg = write_cfg(tmp_path, live={"symbols": LIVE_SYMS, "train_days": 2})
    dl = FakeDownloader()
    monkeypatch.setattr(cli, "make_downloader", lambda c: dl)
    monkeypatch.setattr(cli, "_today", lambda: PUMP + timedelta(days=1))
    assert cli.main(["--config", str(cfg), "download", "--split", "live"]) == 0
    want = {(s, PUMP - timedelta(days=k)) for s in LIVE_SYMS for k in range(3)}  # 2 days + D-1
    assert set(dl.calls) == want
    assert cli.main(["--config", str(cfg), "download", "--split", "live", "--days", "3"]) == 1


def test_download_split_live_with_empty_symbols_exits_1(tmp_path, capsys):
    cfg = write_cfg(tmp_path, live={"symbols": []})
    assert cli.main(["--config", str(cfg), "download", "--split", "live"]) == 1
    assert "live.symbols" in capsys.readouterr().err


class LiveFake:
    """A loopback Binance stand-in in a thread: sends 3 trades (or nothing), records receipts."""

    def __init__(self, send: bool = True):
        import threading

        self.send, self.received, self.url = send, [], None
        ready = threading.Event()

        def serve():
            import time

            import websockets

            async def handler(ws):
                if self.send:
                    now = int(time.time() * S)
                    for i in range(3):
                        t = now + i
                        body = {
                            "e": "trade", "E": t, "s": "SOLUSDT", "t": i, "p": "2.5", "q": "4",
                            "T": t, "m": False,
                        }
                        await ws.send(json.dumps({"stream": "solusdt@trade", "data": body}))
                try:
                    async for msg in ws:
                        self.received.append(msg)
                except Exception:
                    pass

            async def main():
                async with websockets.serve(handler, "127.0.0.1", 0) as srv:
                    port = srv.sockets[0].getsockname()[1]
                    self.url = f"ws://127.0.0.1:{port}/stream"
                    self.stop = asyncio.Event()
                    self.loop = asyncio.get_running_loop()
                    ready.set()
                    await self.stop.wait()

            asyncio.run(main())

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        assert ready.wait(10)

    def close(self):
        self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(5)


def test_live_check_exits_0_at_the_first_trade_and_sends_nothing(tmp_path, capsys):
    fake = LiveFake()
    try:
        cfg = write_cfg(tmp_path, live={"symbols": ["SOLUSDT"], "ws_url": fake.url})
        assert cli.main(["--config", str(cfg), "live-check", "--timeout", "10"]) == 0
    finally:
        fake.close()
    assert fake.received == []
    assert "รับเทรดแรก" in capsys.readouterr().out


def test_live_check_times_out_with_exit_1_and_a_thai_message(tmp_path, capsys):
    fake = LiveFake(send=False)
    try:
        cfg = write_cfg(tmp_path, live={"symbols": ["SOLUSDT"], "ws_url": fake.url})
        assert cli.main(["--config", str(cfg), "live-check", "--timeout", "1"]) == 1
    finally:
        fake.close()
    assert "ต่อสายข้อมูลสดไม่ได้" in capsys.readouterr().err


def test_live_check_with_a_refused_host_exits_1(tmp_path, capsys):
    cfg = write_cfg(tmp_path, live={"symbols": ["SOLUSDT"], "ws_url": "wss://example.com/stream"})
    assert cli.main(["--config", str(cfg), "live-check"]) == 1
    assert "ไม่ถูกต้อง" in capsys.readouterr().err


def test_live_check_timeout_must_be_positive():
    with pytest.raises(SystemExit):
        cli.main(["live-check", "--timeout", "0"])
