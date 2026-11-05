"""Command line: `idsai <command>` (argparse). Every message to the user is Thai.

Exit codes: 0 ok, 1 a user or data error (the message names the command to run next); `--help`
exits 0. Paths come from the config (`--config`, default `configs/default.yaml`) and are relative
to the working directory: run from the repository root.

Commands: `labels [--align]`, `download`, `train`, `eval`, `report`, `demo-pack`, `serve`.

Network: only `idsai.data.archive` (data.binance.vision, CHECKSUM verified) for downloads and the
LIVE sensor (data-stream.binance.vision). Nothing here sends orders or uses keys. Downloads go
through a `Downloader` (`make_downloader`), so tests inject a fake and never touch the network.

Model bundles (`idsai.ai.learning.save_bundle`): `train --profile research` -> `models/research/`,
`train --profile live` -> `models/live/`, `demo-pack` -> `data/demo/` plus `manifest.json`.
Detectors are fitted on normal days (labelled periods masked, `training_mask`) and the Chief is
calibrated on other normal days (`idsai.eval.experiments.fit_and_calibrate`).
"""

import argparse
import json
import math
import re
import sys
from collections.abc import Callable, Coroutine, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import polars as pl

from idsai.config import Settings, config_hash, load_settings

if TYPE_CHECKING:  # the store pulls in DuckDB: imported only where a command needs it
    from idsai.data.store import Store

S_US = 1_000_000
DAY_US = 86_400 * S_US
DEFAULT_HOST = "127.0.0.1"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")  # as Starlette parses the Host header
DEFAULT_SPEED = 60.0
GRACEFUL_SHUTDOWN_S = 5
SESSION_PURGE_S = 3600  # expired sessions are deleted at startup and then once an hour
DEMO_DIR = "demo"
MANIFEST = "manifest.json"
DEMO_PUMPS = 3
DEMO_NORMAL_DAYS = 20
DEMO_CAL_FRACTION = 0.3  # of each symbol's normal days, the most recent ones calibrate the Chief
DEMO_MAX_PUMP_TRIES = 30
DEMO_SEARCH_DAYS = 120
DEMO_LEAD_S = 1800  # the demo replay starts this long before the label
ALIGN_MAX_LAG_S = 120.0
ALIGN_BINS_S = (-10_800, -3600, -600, -120, 120, 600, 3600, 10_800)
LIVE_CAL_DAYS = 2
LIVE_STALE_S = 17.5 * 3600  # older pre-roll data leaves the baseline under 6 h of samples
LIVE_PUBLISHED_LOOKBACK = 5
RESEARCH_CANDIDATE_FACTOR = 2  # normal-day candidates downloaded per day make_split samples
RESEARCH_CANDIDATE_SPACING = 7  # days between two downloaded candidates of one symbol-period
# = idsai.eval.experiments.EXPERIMENTS (a test checks it); kept here so `--help` imports no models
EXPERIMENT_NAMES = (
    "real_t001",
    "semisyn",
    "ablation",
    "baselines",
    "ref_supervised",
    "retrain_drift",
)
LOOPBACK_BINDS = ("127.0.0.1", "localhost", "::1", "[::1]")
WILDCARD_BINDS = ("0.0.0.0", "::", "[::]")
LIVE_STALE_DAYS = 2  # train --profile live warns when a symbol's newest usable day is older
HU_FILE = Path("labels") / "hu_pd_logs.tsv"  # under data_dir; written by scripts/fetch_labels.py


class CliError(Exception):
    """A user or data error: the Thai message is printed and the exit code is 1."""


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _warn(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _today() -> date:
    return datetime.now(UTC).date()


def _now_us() -> int:
    return int(datetime.now(UTC).timestamp() * S_US)


def _day_start_us(day: date) -> int:
    return (day - date(1970, 1, 1)).days * DAY_US


def _day_of_us(t_us: int) -> date:
    return date(1970, 1, 1) + timedelta(days=int(t_us) // DAY_US)


def _app_store(cfg: Settings, db: Path | None = None) -> "Store":
    from idsai.data.store import Store

    return Store(cfg.paths.data_dir, db if db is not None else cfg.paths.app_db)


# ------------------------------------------------------------------ downloads


class Downloader(Protocol):
    def fetch(self, symbol: str, day: date) -> pl.DataFrame | None:
        """The TRADES frame of one UTC day, or None when the archive has no such day."""

    def close(self) -> None: ...


class ArchiveDownloader:
    """Binance Vision daily trades through `idsai.data.archive` (sha256 CHECKSUM verified).

    The verified zip is parsed, then deleted: the store keeps the trades.
    """

    def __init__(self, dest_dir: Path):
        self.dest_dir = Path(dest_dir)
        self._client = None

    def fetch(self, symbol: str, day: date) -> pl.DataFrame | None:
        import httpx

        from idsai.data.archive import download_day, read_trades_zip

        if self._client is None:
            self._client = httpx.Client(timeout=120.0, follow_redirects=False)
        path = download_day(symbol, day, self.dest_dir / symbol, self._client)
        if path is None:
            return None
        try:
            return read_trades_zip(path, symbol)
        finally:
            path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def make_downloader(cfg: Settings) -> Downloader:
    return ArchiveDownloader(cfg.paths.data_dir / "archive")


@dataclass
class Counts:
    downloaded: int = 0
    skipped: int = 0
    missing: int = 0
    missing_days: list[tuple[str, date]] = field(default_factory=list)

    def add(self, other: "Counts") -> None:
        self.downloaded += other.downloaded
        self.skipped += other.skipped
        self.missing += other.missing
        self.missing_days += other.missing_days

    def text(self) -> str:
        return (
            f"ดาวน์โหลด {self.downloaded} วัน, มีอยู่แล้ว (ข้าม) {self.skipped} วัน, "
            f"ไม่มีในคลังข้อมูล {self.missing} วัน"
        )


def ingest(store: "Store", dl: Downloader, symdays: Iterable[tuple[str, date]]) -> Counts:
    """Download every `(symbol, day)` the store lacks; days already stored are skipped (they were
    verified when ingested). A day the archive does not have is counted, not an error."""
    c = Counts()
    for symbol, day in sorted(set(symdays)):
        if store.has_day(symbol, day):
            c.skipped += 1
            continue
        df = dl.fetch(symbol, day)
        if df is None or df.height == 0:
            c.missing += 1
            c.missing_days.append((symbol, day))
            continue
        store.write_trades(df)
        c.downloaded += 1
    return c


def _with_prev(symdays: Iterable[tuple[str, date]]) -> list[tuple[str, date]]:
    out = []
    for s, d in symdays:
        out += [(s, d - timedelta(days=1)), (s, d)]
    return out


# ------------------------------------------------------------------ labels


def hu_path(cfg: Settings) -> Path:
    return cfg.paths.data_dir / HU_FILE


def build_labels(cfg: Settings) -> tuple[pl.DataFrame, list[str]]:
    """Sapienza always, Hu only when its downloaded file is present; offsets from splits.yaml;
    merged and de-duplicated. Returns `(labels, sources used)`."""
    from idsai.eval.labels import load_hu, load_sapienza, merge_dedup, sapienza_csv_path

    off = cfg.splits.label_offset_s
    frames = [load_sapienza(sapienza_csv_path(), off.get("sapienza", 0))]
    used = ["sapienza"]
    if hu_path(cfg).is_file():
        frames.append(load_hu(hu_path(cfg), off.get("hu", 0)))
        used.append("hu")
    return merge_dedup(frames), used


def _histogram(lags: np.ndarray) -> list[str]:
    """Text histogram of lags: left bins `[lo, hi)`, the on-time bin `[-120, 120]`, right bins
    `(lo, hi]`."""
    e = ALIGN_BINS_S
    bins = [(-math.inf, e[0]), *zip(e, e[1:], strict=False), (e[-1], math.inf)]
    lines = ["ฮิสโทแกรมของ lag (จุดที่ปริมาณซื้อขายสูงสุด ลบ เวลาในป้าย, วินาที):"]
    n = max(len(lags), 1)
    for lo, hi in bins:
        if hi <= -ALIGN_MAX_LAG_S:
            k, name = ((lags >= lo) & (lags < hi)).sum(), f"[{lo:>7.0f}, {hi:>7.0f})"
        elif lo >= ALIGN_MAX_LAG_S:
            k, name = ((lags > lo) & (lags <= hi)).sum(), f"({lo:>7.0f}, {hi:>7.0f}]"
        else:
            k, name = ((lags >= lo) & (lags <= hi)).sum(), f"[{lo:>7.0f}, {hi:>7.0f}]"
        lines.append(f"  {name} {int(k):>5d} {'#' * round(40 * int(k) / n)}")
    return lines


def alignment_verdict(al: pl.DataFrame) -> tuple[int, list[str]]:
    """Exit code and Thai lines for a `label_alignment` frame.

    Exit 1 when no label could be measured (fail closed) or the median |lag| is >= 120 s.
    """
    measured = al.filter(pl.col("has_data") & pl.col("lag_s").is_not_null())
    if measured.height == 0:
        return 1, [
            f"วัดไม่ได้: ไม่มีป้ายใดเลย (จาก {al.height}) ที่มีข้อมูลการซื้อขายในเครื่องรอบเวลาของป้าย",
            "จึงยังยืนยันไม่ได้ว่าเวลาในป้ายถูกต้อง ให้รัน `uv run idsai download --split research` "
            "แล้วรัน `uv run idsai labels --align` อีกครั้ง",
        ]
    lags = measured["lag_s"].to_numpy().astype(np.float64)
    med_abs = float(np.median(np.abs(lags)))
    med = float(np.median(lags))
    lines = _histogram(lags)
    lines += [
        f"วัดได้ {measured.height} จาก {al.height} ป้าย",
        f"ค่ามัธยฐานของ |lag| = {med_abs:.0f} วินาที (เกณฑ์: น้อยกว่า {ALIGN_MAX_LAG_S:.0f})",
        f"ค่ามัธยฐานของ lag = {med:+.0f} วินาที",
    ]
    if med_abs >= ALIGN_MAX_LAG_S:
        lines.append(
            "ไม่ผ่าน: เวลาในป้ายอาจคลาดจากจุดที่ปริมาณซื้อขายสูงสุด ให้ตรวจ label_offset_s ใน "
            f"configs/splits.yaml (ถ้าคลาดไปทางเดียวกันเกือบทั้งหมด ลองค่าใกล้ {med:+.0f}) "
            "แล้วรัน `uv run idsai labels --align` ใหม่"
        )
        return 1, lines
    lines.append("ผ่าน: เวลาในป้ายสอดคล้องกับจุดที่ปริมาณซื้อขายสูงสุดของข้อมูลจริง")
    return 0, lines


def run_alignment(store: Any, labels: pl.DataFrame) -> tuple[int, list[str]]:
    """`label_alignment` over the labels whose day is in the store, then `alignment_verdict`."""
    from idsai.eval.labels import label_alignment

    pairs = labels.select("symbol", "t_start_us").rows()
    keep = [store.has_day(sym, _day_of_us(t0)) for sym, t0 in pairs]
    present = labels.filter(pl.Series(keep, dtype=pl.Boolean))
    return alignment_verdict(label_alignment(store, present))


def cmd_labels(args: argparse.Namespace, cfg: Settings) -> int:
    labels, used = build_labels(cfg)
    with _app_store(cfg) as store:
        removed = store.replace_labels(labels)
        _say(
            f"เขียนป้ายเหตุการณ์จริง {labels.height} รายการ (แหล่ง: {', '.join(used)}) "
            f"ลง {cfg.paths.app_db} (แทนที่ของเดิม {removed} รายการ)"
        )
        if "hu" not in used:
            _say(
                f"ไม่พบไฟล์ป้ายของ Hu et al. ที่ {hu_path(cfg)} จึงใช้เฉพาะ Sapienza "
                "(ถ้าต้องการ: `uv run python scripts/fetch_labels.py`)"
            )
        if not args.align:
            return 0
        code, lines = run_alignment(store, labels)
    for line in lines:
        _say(line)
    return code


# ------------------------------------------------------------------ download


def research_symdays(cfg: Settings, labels: pl.DataFrame) -> list[tuple[str, date]]:
    """Every symbol-day the research split and eval read (see `download --help`)."""
    from idsai.eval.splits import NEXT_MINUTES, PERIODS, period_of

    sp = cfg.splits
    out: set[tuple[str, date]] = set()
    by_sym: dict[str, list[date]] = {}
    for sym, t0 in labels.select("symbol", "t_start_us").rows():
        d = _day_of_us(t0)
        by_sym.setdefault(sym, []).append(d)
        if period_of(d, sp) not in (None, "cal"):
            out |= {(sym, d - timedelta(days=1)), (sym, d)}
            if _day_start_us(d) + DAY_US - t0 <= NEXT_MINUTES * 60 * S_US:
                out.add((sym, d + timedelta(days=1)))
    k = sp.normal_days_per_symbol * RESEARCH_CANDIDATE_FACTOR
    for sym, lab in sorted(by_sym.items()):
        for period in PERIODS:
            lo, hi = getattr(sp, period)
            cand = [
                lo + timedelta(days=i)
                for i in range((hi - lo).days + 1)
                if all(abs((lo + timedelta(days=i) - ld).days) >= sp.min_gap_days for ld in lab)
            ]
            # nearest to a label of the symbol first: the coin is known to trade around its labels
            cand.sort(key=lambda d: (min(abs((d - ld).days) for ld in lab), d))
            picked: list[date] = []
            for d in cand:
                if all(abs((d - p).days) >= RESEARCH_CANDIDATE_SPACING for p in picked):
                    picked.append(d)
                if len(picked) == k:
                    break
            out |= set(_with_prev((sym, d) for d in picked))
    return sorted(out)


def _last_days(symbols: list[str], end: date, n: int) -> list[tuple[str, date]]:
    """The `n` days ending at `end` plus the D-1 of the first, for every symbol."""
    return [(s, end - timedelta(days=i)) for s in symbols for i in range(n + 1)]


def cmd_download(args: argparse.Namespace, cfg: Settings) -> int:
    dl = make_downloader(cfg)
    try:
        with _app_store(cfg) as store:
            if args.split == "research":
                labels = store.read_labels().filter(pl.col("provenance") == "REAL")
                if labels.height == 0:
                    raise CliError("ยังไม่มีป้ายเหตุการณ์ในเครื่อง ให้รัน `uv run idsai labels` ก่อน")
                want = research_symdays(cfg, labels)
                _say(f"ชุดข้อมูลวิจัยต้องใช้ {len(want)} สัญลักษณ์-วัน")
                c = ingest(store, dl, want)
            else:
                symbols = list(dict.fromkeys(args.symbols))
                if args.day is not None:
                    c = ingest(store, dl, _with_prev((s, args.day) for s in symbols))
                else:
                    end = _today() - timedelta(days=1)
                    c = ingest(store, dl, _last_days(symbols, end, args.days))
                    if all((s, end) in c.missing_days for s in symbols):
                        # yesterday is not published yet: the window moves back one day
                        first = end - timedelta(days=args.days)
                        c.add(ingest(store, dl, [(s, first - timedelta(days=1)) for s in symbols]))
    finally:
        dl.close()
    _say(c.text())
    for s, d in c.missing_days[:20]:
        _say(f"  ไม่มี: {s} {d}")
    if len(c.missing_days) > 20:
        _say(f"  ... และอีก {len(c.missing_days) - 20} วัน")
    return 0


# ------------------------------------------------------------------ train


def live_budget_per_symbol_day(cfg: Settings) -> float:
    """`live.budget_per_hour` over the whole universe -> alerts per symbol-day."""
    return cfg.live.budget_per_hour * 24.0 / cfg.live.universe_size


def _data_hash(store: "Store", days: Iterable[Any]) -> str:
    import hashlib

    parts = sorted(
        f"{sd.symbol}|{sd.day.isoformat()}|{store.partition_sha(sd.symbol, sd.day)}" for sd in days
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def _days_json(days: Iterable[Any]) -> list[list[str]]:
    return [[sd.symbol, sd.day.isoformat()] for sd in sorted(days)]


def train_and_save(
    cfg: Settings,
    store: "Store",
    out_dir: Path,
    profile: str,
    seed: int,
    train_days: list[Any],
    cal_days: list[Any],
    budget: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit + calibrate (`fit_and_calibrate`) and save the bundle; returns `bundle.json`."""
    import warnings

    from idsai.ai.chief import CalibrationError
    from idsai.ai.learning import save_bundle
    from idsai.eval.experiments import fit_and_calibrate, git_sha

    _say(f"ฝึกตัวตรวจจับบน {len(train_days)} สัญลักษณ์-วัน, สอบเทียบ Chief บน {len(cal_days)} วัน")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            fitted = fit_and_calibrate(
                cfg, store, seed, train_days, cal_days, budget, version_prefix=profile
            )
    except (ValueError, CalibrationError) as e:
        raise CliError(f"ฝึกหรือสอบเทียบไม่สำเร็จ: {e}") from e
    for w in caught:
        _warn(f"คำเตือน: {w.message}")
    every = sorted([*train_days, *cal_days])
    meta = {
        "profile": profile,
        "symbols": sorted({sd.symbol for sd in every}),
        "train_days": _days_json(train_days),
        "cal_days": _days_json(cal_days),
        "seed": int(seed),
        "git_sha": git_sha(),
        "config_hash": config_hash(cfg),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_hash": _data_hash(store, every),
        "train_range": [
            min(sd.day for sd in train_days).isoformat(),
            max(sd.day for sd in train_days).isoformat(),
        ],
        "budget_per_symbol_day": float(budget),
        "info": fitted.info,
        **(extra or {}),
    }
    doc = save_bundle(out_dir, list(cfg.detectors), fitted.detectors, fitted.chief, meta)
    _say(
        f"บันทึกชุดโมเดลที่ {out_dir} (tau = {fitted.chief.tau:.3f}, "
        f"งบการแจ้งเตือน {budget:g} ครั้ง/สัญลักษณ์-วัน)"
    )
    return doc


def live_days(store: "Store", symbols: list[str], n: int) -> tuple[list[Any], list[Any]]:
    """`(train, cal)`: per symbol the last `n` stored days that have their D-1; the last
    `LIVE_CAL_DAYS` of them calibrate the Chief, the earlier ones train the detectors."""
    from idsai.eval.splits import SymDay

    train, cal = [], []
    for s in symbols:
        have = set(store.days(s))
        usable = sorted(d for d in have if d - timedelta(days=1) in have)[-n:]
        cal += [SymDay(s, d) for d in usable[-LIVE_CAL_DAYS:]]
        train += [SymDay(s, d) for d in usable[:-LIVE_CAL_DAYS]]
    return train, cal


def cmd_train(args: argparse.Namespace, cfg: Settings) -> int:
    with _app_store(cfg) as store:
        if args.profile == "live":
            symbols = list(cfg.live.symbols)
            if not symbols:
                raise CliError("live.symbols ใน configs/default.yaml ว่าง: ใส่สัญลักษณ์อย่างน้อย 1 ตัว")
            n = cfg.live.train_days
            train, cal = live_days(store, symbols, n)
            for s in symbols:
                newest = max((sd.day for sd in [*train, *cal] if sd.symbol == s), default=None)
                if newest is None:
                    _warn(f"คำเตือน: {s} ไม่มีข้อมูลที่ใช้ได้ จึงไม่อยู่ในโมเดลโหมดสด")
                elif (_today() - newest).days > LIVE_STALE_DAYS:
                    _warn(
                        f"คำเตือน: ข้อมูลล่าสุดของ {s} คือวันที่ {newest} เก่ากว่า {LIVE_STALE_DAYS} วัน "
                        f"โมเดลโหมดสดอาจไม่ตรงกับตลาดตอนนี้ ให้รัน `uv run idsai download "
                        f"--symbols {s} --days {n}` ก่อน"
                    )
            if not train or not cal:
                raise CliError(
                    f"ข้อมูลไม่พอสำหรับโมเดลโหมดสด (ต้องมีอย่างน้อย {LIVE_CAL_DAYS + 1} วันต่อสัญลักษณ์ "
                    f"พร้อมวันก่อนหน้า) ให้รัน `uv run idsai download --symbols {' '.join(symbols)} "
                    f"--days {n}` ก่อน"
                )
            budget = live_budget_per_symbol_day(cfg)
            out = cfg.paths.models_dir / "live"
        else:
            from idsai.eval.splits import make_split

            real = store.read_labels().filter(pl.col("provenance") == "REAL")
            split = make_split(real, store, cfg)
            train, cal = split.train_normal, split.cal_normal
            if not train or not cal:
                raise CliError(
                    f"ยังไม่มีวันปกติพอ (ฝึก {len(train)} วัน, สอบเทียบ {len(cal)} วัน) ให้รัน "
                    "`uv run idsai labels` และ `uv run idsai download --split research` ก่อน"
                )
            budget = cfg.chief.budget_per_symbol_day
            out = cfg.paths.models_dir / "research"
        train_and_save(cfg, store, out, args.profile, args.seed, train, cal, budget)
    return 0


# ------------------------------------------------------------------ eval / report


def _seeds(text: str) -> list[int]:
    try:
        seeds = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"seeds ต้องเป็นจำนวนเต็มคั่นด้วยจุลภาค เช่น 0,1,2: {text!r}"
        ) from None
    if not seeds:
        raise argparse.ArgumentTypeError("ต้องมี seed อย่างน้อย 1 ค่า")
    return seeds


def cmd_eval(args: argparse.Namespace, cfg: Settings) -> int:
    from idsai.detection.simulator.guard import EthicsError
    from idsai.eval.experiments import run_experiment

    names = list(EXPERIMENT_NAMES) if args.experiment == "all" else [args.experiment]
    seeds = args.seeds if args.seeds is not None else list(cfg.eval.seeds)
    n_err = 0
    with _app_store(cfg) as app, _app_store(cfg, cfg.paths.research_db) as research:
        for name in names:
            for seed in seeds:
                try:
                    recs = run_experiment(name, cfg, app, seed, research=research)
                except EthicsError as e:
                    raise CliError(f"การทดลอง {name} ใช้ข้อมูลจำลองได้เฉพาะโหมด replay: {e}") from e
                errs = sum(r["error"] is not None for r in recs)
                n_err += errs
                _say(f"{name} seed {seed}: {len(recs)} รายการ ({errs} รายการวัดไม่ได้)")
    _say(f"บันทึกผลลง {cfg.paths.research_db} แล้ว รัน `uv run idsai report` เพื่อสร้างรายงาน")
    return 0


def cmd_report(args: argparse.Namespace, cfg: Settings) -> int:
    from idsai.eval.report import build_report

    db = cfg.paths.research_db
    if not db.is_file():
        raise CliError(f"ยังไม่มีผลการทดลองที่ {db} ให้รัน `uv run idsai eval` ก่อน")
    out_json = cfg.paths.report_json
    out_md = out_json.with_suffix(".md")
    report = build_report(db, out_md, out_json, cfg=cfg)  # opens the db read-only
    _say(f"เขียนรายงาน {out_md} และ {out_json} (จากการทดลองทั้งหมด {report.get('runs', 0)} รายการ)")
    return 0


# ------------------------------------------------------------------ demo pack


def demo_dir(cfg: Settings) -> Path:
    return cfg.paths.data_dir / DEMO_DIR


def _have_day(store: "Store", sym: str, day: date) -> bool:
    return store.has_day(sym, day) and store.has_day(sym, day - timedelta(days=1))


def build_demo_pack(
    cfg: Settings,
    store: "Store",
    dl: Downloader,
    labels: pl.DataFrame,
    n_pumps: int,
    n_normal: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Pick, download, train and save the demo pack; returns the manifest.

    Pump days: Sapienza labels from the most recent back, one per `(symbol, day)`, kept when the
    archive has the day and its D-1 (at most `DEMO_MAX_PUMP_TRIES` tried). Normal days: for each
    symbol of those pumps (quotas split evenly, sorted symbols first), days walking back from its
    most recent pump day, then forward from it, at least `min_gap_days` from every label of the
    symbol (any source), each with its D-1 (at most `DEMO_SEARCH_DAYS` each way). Per symbol the
    most recent `DEMO_CAL_FRACTION` of its normal days (at least 1 when it has 2 or more)
    calibrate the Chief; the rest train the detectors. Labelled periods are masked in training.
    """
    from idsai.eval.splits import SymDay

    gap = cfg.splits.min_gap_days
    sap = labels.filter(pl.col("source").str.contains("sapienza")).sort(
        ["t_start_us", "symbol"], descending=[True, False]
    )
    pumps: list[dict[str, Any]] = []
    tried: set[tuple[str, date]] = set()
    counts = Counts()
    for row in sap.iter_rows(named=True):
        key = (row["symbol"], _day_of_us(row["t_start_us"]))
        if key in tried:
            continue
        if len(tried) >= DEMO_MAX_PUMP_TRIES or len(pumps) >= n_pumps:
            break
        tried.add(key)
        counts.add(ingest(store, dl, _with_prev([key])))
        if _have_day(store, *key):
            pumps.append({**row, "day": key[1]})
            _say(f"วันที่มีป้าย pump: {key[0]} {key[1]}")
    if not pumps:
        raise CliError(
            f"ไม่พบวันที่มีป้าย pump ที่ดาวน์โหลดได้ครบ (ลองแล้ว {len(tried)} วัน) "
            "ตรวจการเชื่อมต่อกับ data.binance.vision แล้วรันใหม่"
        )
    symbols = sorted({p["symbol"] for p in pumps})
    lab_days: dict[str, list[date]] = {}
    for sym, t0 in labels.select("symbol", "t_start_us").rows():
        lab_days.setdefault(sym, []).append(_day_of_us(t0))
    pump_keys = {(p["symbol"], p["day"]) for p in pumps}
    train: list[SymDay] = []
    cal: list[SymDay] = []
    last_ok = _today() - timedelta(days=1)
    for i, sym in enumerate(symbols):
        quota = n_normal // len(symbols) + (i < n_normal % len(symbols))
        anchor = max(p["day"] for p in pumps if p["symbol"] == sym)
        steps = [-k for k in range(1, DEMO_SEARCH_DAYS + 1)]
        steps += [k for k in range(1, DEMO_SEARCH_DAYS + 1)]
        found: list[date] = []
        for k in steps:
            if len(found) >= quota:
                break
            day = anchor + timedelta(days=k)
            if day > last_ok or (sym, day) in pump_keys:
                continue
            if any(abs((day - ld).days) < gap for ld in lab_days.get(sym, [])):
                continue
            counts.add(ingest(store, dl, _with_prev([(sym, day)])))
            if _have_day(store, sym, day):
                found.append(day)
        found.sort()
        n_cal = max(1, round(len(found) * DEMO_CAL_FRACTION)) if len(found) >= 2 else 0
        train += [SymDay(sym, d) for d in found[: len(found) - n_cal]]
        cal += [SymDay(sym, d) for d in found[len(found) - n_cal :]]
    _say(counts.text())
    if not train or not cal:
        raise CliError(
            f"หาวันปกติได้ไม่พอ (ฝึก {len(train)} วัน, สอบเทียบ {len(cal)} วัน) "
            "ตรวจการเชื่อมต่อแล้วรัน `uv run idsai demo-pack` ใหม่"
        )
    manifest = {
        "symbols": symbols,
        "pump_days": [
            {
                "symbol": p["symbol"],
                "day": p["day"].isoformat(),
                "label_id": p["label_id"],
                "source": p["source"],
                "t_start_us": int(p["t_start_us"]),
            }
            for p in pumps
        ],
        "normal_days": {"train": _days_json(train), "cal": _days_json(cal)},
        "config_hash": config_hash(cfg),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    train_and_save(
        cfg,
        store,
        out_dir,
        "demo",
        seed,
        train,
        cal,
        cfg.chief.budget_per_symbol_day,
        extra={"pump_days": manifest["pump_days"]},
    )
    (out_dir / MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    _say(f"เขียน {out_dir / MANIFEST} แล้ว ต่อไปรัน `uv run idsai serve --mode replay`")
    return manifest


def cmd_demo_pack(args: argparse.Namespace, cfg: Settings) -> int:
    labels, used = build_labels(cfg)
    dl = make_downloader(cfg)
    try:
        with _app_store(cfg) as store:
            store.replace_labels(labels)
            _say(f"ใช้ป้ายเหตุการณ์ {labels.height} รายการ (แหล่ง: {', '.join(used)})")
            build_demo_pack(
                cfg, store, dl, labels, args.pumps, args.normal_days, args.seed, demo_dir(cfg)
            )
    finally:
        dl.close()
    return 0


# ------------------------------------------------------------------ serve


def resolve_bind(args: argparse.Namespace, cfg: Settings) -> tuple[str, int]:
    return args.host, args.port if args.port is not None else cfg.port


def allowed_hosts(host: str, public: Sequence[str] = ()) -> list[str]:
    """Loopback names, the bound host and any public host names (behind a TLS proxy): the `Host`
    header allowlist (DNS rebinding)."""
    bound = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return list(dict.fromkeys([*LOOPBACK_HOSTS, bound.lower(), *(h.lower() for h in public)]))


def run_server(
    app: Any, host: str, port: int, background: Callable[[], Coroutine[Any, Any, None]]
) -> None:
    """uvicorn on one event loop with `background()` (the pipeline) as a task on that loop; the
    task is cancelled when the server stops (Ctrl+C), within `GRACEFUL_SHUTDOWN_S`."""
    import asyncio
    import contextlib

    import uvicorn

    async def main() -> None:
        config = uvicorn.Config(
            app, host=host, port=port, timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(background())

        def done(t: asyncio.Task) -> None:
            if not t.cancelled() and t.exception() is not None:
                _warn(f"สายประมวลผลหยุดเพราะข้อผิดพลาด: {t.exception()!r}")

        task.add_done_callback(done)
        try:
            await server.serve()
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    asyncio.run(main())


def _missing_day_error(symbol: str, day: date) -> CliError:
    prev = day - timedelta(days=1)
    return CliError(
        f"ไม่มีข้อมูล {symbol} วันที่ {day} และ/หรือวันก่อนหน้า ({prev}) ในเครื่อง "
        "ต้องมีครบทั้งสองวัน เพราะระบบเทียบกับ 24 ชั่วโมงก่อนหน้าเสมอ "
        f"ให้รัน `uv run idsai download --symbols {symbol} --day {day}` ก่อน"
    )


def _replay_target(
    args: argparse.Namespace, cfg: Settings, store: "Store"
) -> tuple[str, date, int, Path]:
    """`(symbol, day, start_us, bundle dir)` of a replay, or CliError."""
    demo = demo_dir(cfg)
    if args.symbol is None and args.day is None:
        path = demo / MANIFEST
        if not path.is_file():
            raise CliError("ยังไม่มี demo pack ให้รัน `uv run idsai demo-pack` ก่อน")
        pump = json.loads(path.read_text(encoding="utf-8"))["pump_days"][0]
        symbol, day = pump["symbol"], date.fromisoformat(pump["day"])
        if not _have_day(store, symbol, day):
            raise CliError(
                f"ไม่มีข้อมูลของ demo pack ({symbol} {day} และวันก่อนหน้า) ในเครื่อง "
                "ให้รัน `uv run idsai demo-pack` อีกครั้ง"
            )
        start = max(_day_start_us(day), int(pump["t_start_us"]) - DEMO_LEAD_S * S_US)
        return symbol, day, start, Path(args.models) if args.models else demo
    if args.symbol is None or args.day is None:
        raise CliError("ต้องระบุทั้ง --symbol และ --day หรือไม่ระบุทั้งสองอย่าง (เล่น demo pack)")
    symbol, day = args.symbol, args.day
    if not _have_day(store, symbol, day):
        raise _missing_day_error(symbol, day)
    if args.models:
        models = Path(args.models)
    elif (demo / "bundle.json").is_file():
        models = demo
    else:
        raise CliError(
            "ไม่พบชุดโมเดล: ใช้ --models models/research (หลัง `uv run idsai train`) "
            "หรือรัน `uv run idsai demo-pack` ก่อน"
        )
    d0 = _day_start_us(day)
    t0s = [t for t in store.read_labels(symbol)["t_start_us"].to_list() if d0 <= t < d0 + DAY_US]
    start = max(d0, min(t0s) - DEMO_LEAD_S * S_US) if t0s else d0
    return symbol, day, start, models


def _load(models: Path, cfg: Settings) -> tuple[list[Any], Any, dict[str, Any]]:
    from idsai.ai.learning import BundleError, load_bundle

    try:
        return load_bundle(models, cfg)
    except BundleError as e:
        raise CliError(str(e)) from e


def prepare_replay(
    args: argparse.Namespace, cfg: Settings, store: "Store"
) -> tuple[Any, Callable[[], Coroutine[Any, Any, None]]]:
    """The app state and the pipeline coroutine of `serve --mode replay`."""
    from idsai.bus import Bus
    from idsai.data.replay import Replayer
    from idsai.pipeline.stream import StreamPipeline
    from idsai.product.api import AppState

    symbol, day, start, models = _replay_target(args, cfg, store)
    dets, chief, doc = _load(models, cfg)
    d0 = _day_start_us(day)
    trades = store.read_trades(symbol, d0 - DAY_US, d0 + DAY_US)
    replayer = Replayer(trades, speed=args.speed, start_us=start, preroll_from_us=d0 - DAY_US)
    pipeline = StreamPipeline(cfg, store, dets, chief, Bus(), "replay")
    state = AppState(
        cfg,
        store,
        "replay",
        pipeline,
        replayer,
        symbols=[symbol],
        budget_per_symbol_day=doc.get("budget_per_symbol_day"),
        auth_enabled=auth_enabled(args, cfg),
    )
    # alerts a previous session raised at or after the start would be shown as "future" ones and
    # be raised again: drop them as a seek to `start` does (feedback rows stay)
    stale = store.delete_alerts_from("replay", start)
    if stale:
        _say(
            f"ลบการแจ้งเตือน replay เดิม {len(stale)} รายการที่อยู่หลังจุดเริ่มเล่น "
            "(ผลการตรวจของผู้วิเคราะห์ยังเก็บไว้)"
        )
    when = datetime.fromtimestamp(start / S_US, UTC).strftime("%Y-%m-%d %H:%M UTC")
    _say(
        f"เล่นย้อนหลัง {symbol} วันที่ {day} เริ่มที่ {when} ความเร็ว {args.speed:g} เท่า "
        f"(อุ่นเครื่องจากข้อมูล 24 ชม.ก่อนหน้าก่อน) โมเดลจาก {models}"
    )
    return state, lambda: pipeline.run(replayer)


def _live_preroll(
    cfg: Settings, store: "Store", symbols: list[str], now_us: int
) -> pl.DataFrame | None:
    """The trailing 24 h of the two most recent published archive days of every symbol."""
    dl = make_downloader(cfg)
    try:
        for s in symbols:
            found = 0
            for k in range(1, LIVE_PUBLISHED_LOOKBACK + 1):
                day = _day_of_us(now_us) - timedelta(days=k)
                ingest(store, dl, [(s, day)])
                found += store.has_day(s, day)
                if found == 2:
                    break
    finally:
        dl.close()
    frames = [store.read_trades(s, now_us - DAY_US, now_us) for s in symbols]
    frames = [f for f in frames if f.height]
    return pl.concat(frames) if frames else None


def prepare_live(
    args: argparse.Namespace, cfg: Settings, store: "Store"
) -> tuple[Any, Callable[[], Coroutine[Any, Any, None]]]:
    """The app state and the pipeline coroutine of `serve --mode live`."""
    from idsai.bus import Bus
    from idsai.data.live import LiveTradeSensor, warmup_then_live
    from idsai.pipeline.stream import StreamPipeline
    from idsai.product.api import AppState

    models = Path(args.models) if args.models else cfg.paths.models_dir / "live"
    if not (models / "bundle.json").is_file():
        raise CliError(
            f"ยังไม่มีชุดโมเดลสำหรับโหมดสดที่ {models} ให้รัน `uv run idsai train --profile live` ก่อน"
        )
    dets, chief, doc = _load(models, cfg)
    symbols = list(doc["symbols"])
    now = _now_us()
    archive = _live_preroll(cfg, store, symbols, now)
    _say("ลำดับการอุ่นเครื่องของโหมดสด (นับจากตอนเชื่อมต่อ):")
    _say("  ประมาณ +30 นาที: แถวข้อมูลแรก (ต้องมีข้อมูลครบหน้าต่าง 1800 วินาทีก่อน)")
    _say(
        "  ค่า z ใช้ได้ตั้งแต่แถวแรกถ้าข้อมูลย้อนหลังครอบคลุมอย่างน้อย 6 ชม. ของ 24 ชม. ล่าสุด มิฉะนั้นประมาณ +6.6 ชม."
    )
    newest = None if archive is None else int(archive["ts_us"].max())
    if newest is None or now - newest > LIVE_STALE_S * S_US:
        age = (
            "ไม่มีข้อมูลย้อนหลังเลย"
            if newest is None
            else (f"ข้อมูลย้อนหลังล่าสุดจบเมื่อ {(now - newest) / 3600 / S_US:.1f} ชม. ก่อน")
        )
        _warn(f"คำเตือน: {age} (เกิน 17.5 ชม.) ค่า z จะยังใช้ไม่ได้จนประมาณ +6.6 ชม. หลังเชื่อมต่อ")
    sensor = LiveTradeSensor(symbols, cfg.live.ws_url)
    pipeline = StreamPipeline(cfg, store, dets, chief, Bus(), "live")
    state = AppState(
        cfg,
        store,
        "live",
        pipeline,
        None,
        symbols=symbols,
        budget_per_symbol_day=doc.get("budget_per_symbol_day"),
        auth_enabled=auth_enabled(args, cfg),
    )
    return state, lambda: pipeline.run(warmup_then_live(archive, sensor))


def auth_enabled(args: argparse.Namespace, cfg: Settings) -> bool:
    """`--auth` / `--no-auth` override `auth.enabled` in the config."""
    flag = getattr(args, "auth", None)
    return cfg.auth.enabled if flag is None else bool(flag)


def check_serve_auth(host: str, enabled: bool, store: "Store") -> None:
    if not enabled and host.lower() not in LOOPBACK_BINDS:
        raise CliError(
            f"ไม่เปิดให้เครื่องอื่นเข้าถึงที่ {host} เพราะยังไม่เปิดระบบล็อกอิน "
            "ใช้ --auth (หรือ auth.enabled: true ในไฟล์ค่าตั้ง) หรือเปิดเฉพาะเครื่องนี้ด้วย --host 127.0.0.1"
        )
    if enabled and not store.has_enabled_admin():
        raise CliError(
            "เปิดระบบล็อกอินแล้วแต่ยังไม่มีผู้ดูแลที่ใช้งานได้ สร้างด้วย: "
            "uv run idsai user add <ชื่อ> --role admin"
        )


def read_new_password(args: argparse.Namespace) -> str:
    """The password from one stdin line (`--password-stdin`) or two getpass prompts; never argv."""
    from idsai.product.auth import check_password

    if getattr(args, "password_stdin", False):
        pw = sys.stdin.readline().rstrip("\r\n")
    else:
        import getpass

        pw = getpass.getpass("รหัสผ่าน: ")
        if getpass.getpass("พิมพ์รหัสผ่านอีกครั้ง: ") != pw:
            raise CliError("รหัสผ่านสองครั้งไม่ตรงกัน")
    try:
        check_password(pw)
    except ValueError as e:
        raise CliError(str(e)) from None
    return pw


RESERVED_USERNAMES = ("cli", "local")  # the audit log's actor names for the CLI and auth-off mode


def _when(us: int | None) -> str:
    if us is None:
        return "-"
    return datetime.fromtimestamp(us / S_US, UTC).strftime("%Y-%m-%d %H:%M UTC")


def parse_seed_users(text: str) -> list[tuple[str, str, str]]:
    """Parse IDSAI_USERS (`username:role:password` per line; `#` comments, blank lines skipped).
    The password is everything after the second colon. Errors name the 1-based line, never the
    password. At least one admin is required."""
    from idsai.product.auth import ROLES, check_password, check_username

    out: list[tuple[str, str, str]] = []
    for no, raw in enumerate(text.replace(chr(13), "").split(chr(10)), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":", 2)
        if len(parts) != 3:
            raise CliError(f"บรรทัด {no}: ต้องเป็นรูปแบบ ชื่อ:สิทธิ์:รหัสผ่าน")
        name, role, pw = parts
        try:
            check_username(name)
            check_password(pw)
        except ValueError as e:
            raise CliError(f"บรรทัด {no}: {e}") from None
        if role not in ROLES:
            raise CliError(f"บรรทัด {no}: สิทธิ์ต้องเป็นหนึ่งใน {', '.join(ROLES)}")
        if name in RESERVED_USERNAMES:
            raise CliError(f"บรรทัด {no}: ชื่อ {name} สงวนไว้ใช้ในบันทึกกิจกรรม เลือกชื่ออื่น")
        if any(n == name for n, _, _ in out):
            raise CliError(f"บรรทัด {no}: ชื่อ {name} ซ้ำ")
        out.append((name, role, pw))
    if not any(role == "admin" for _, role, _ in out):
        raise CliError("ต้องมีบัญชี admin อย่างน้อยหนึ่งบัญชีใน IDSAI_USERS")
    return out


def cmd_user(args: argparse.Namespace, cfg: Settings) -> int:
    import duckdb

    try:
        return _cmd_user(args, cfg)
    except duckdb.IOException:  # DuckDB lets one process hold app.duckdb at a time
        raise CliError(
            f"เปิดฐานข้อมูล {cfg.paths.app_db} ไม่ได้ น่าจะมีโปรแกรมอื่นใช้อยู่ (เช่น `idsai serve` "
            "ที่กำลังรัน) ให้หยุด serve ก่อน แล้วลองสั่งใหม่"
        ) from None


def _cmd_user(args: argparse.Namespace, cfg: Settings) -> int:
    from idsai.product.auth import check_username, hash_password

    act = args.user_cmd
    with _app_store(cfg) as store:

        def audit(action: str, target: str, detail: dict | None = None) -> None:
            store.add_audit(_now_us(), None, "cli", action, target, detail)

        if act == "list":
            users = store.list_users()
            if not users:
                _say("ยังไม่มีผู้ใช้ (สร้างด้วย uv run idsai user add <ชื่อ> --role admin)")
            for u in users:
                flag = "  disabled" if u["disabled"] else ""
                shown = f"  ({u['display_name']})" if u["display_name"] else ""
                _say(
                    f"{u['username']}{shown}  {u['role']}{flag}  "
                    f"สร้างเมื่อ {_when(u['created_us'])}  เข้าล่าสุด {_when(u['last_seen_us'])}"
                )
            return 0
        if act == "seed-env":
            import os

            text = os.environ.get("IDSAI_USERS", "")
            if not text.strip():
                raise CliError("ไม่พบตัวแปร IDSAI_USERS (หรือว่างเปล่า)")
            accounts = parse_seed_users(text)  # validate everything before writing anything
            created = updated = 0
            for name, role, pw in accounts:
                if store.get_user_by_name(name) is None:
                    store.add_user(name, "", role, hash_password(pw), _now_us())
                    created += 1
                else:
                    store.set_role(name, role)
                    store.set_password(name, hash_password(pw))
                    store.set_disabled(name, False)
                    updated += 1
                audit("user_seed", name, {"role": role})
            _say(f"seeded {len(accounts)} accounts (created {created}, updated {updated})")
            return 0
        name = args.username
        if act == "add":
            try:
                check_username(name)
            except ValueError as e:
                raise CliError(str(e)) from None
            if name in RESERVED_USERNAMES:
                raise CliError(f"ชื่อ {name} สงวนไว้ใช้ในบันทึกกิจกรรม เลือกชื่ออื่น")
            if len(args.display_name) > 64:
                raise CliError("ชื่อที่แสดงต้องไม่เกิน 64 ตัวอักษร")
            if store.get_user_by_name(name) is not None:
                raise CliError(f"มีผู้ใช้ชื่อ {name} แล้ว")
            pw = read_new_password(args)
            store.add_user(name, args.display_name, args.role, hash_password(pw), _now_us())
            audit("user_add", name, {"role": args.role})
            _say(f"สร้างผู้ใช้ {name} ({args.role}) แล้ว")
            return 0
        if store.get_user_by_name(name) is None:  # the store no-ops silently on an unknown name
            raise CliError(f"ไม่พบผู้ใช้ {name}")
        if act == "role":
            store.set_role(name, args.role)
            audit("user_role", name, {"role": args.role})
            _say(f"ตั้งสิทธิ์ของ {name} เป็น {args.role} แล้ว")
        elif act in ("disable", "enable"):
            store.set_disabled(name, act == "disable")
            audit(f"user_{act}", name)
            _say(f"{'ปิด' if act == 'disable' else 'เปิด'}ใช้งาน {name} แล้ว")
        else:  # passwd
            store.set_password(name, hash_password(read_new_password(args)))
            audit("user_passwd", name)
            _say(f"เปลี่ยนรหัสผ่านของ {name} แล้ว (ออกจากระบบทุกอุปกรณ์)")
    return 0


def with_session_purge(
    store: "Store", background: Callable[[], Coroutine[Any, Any, None]]
) -> Callable[[], Coroutine[Any, Any, None]]:
    """`background` plus a task that deletes expired sessions every `SESSION_PURGE_S`; the purge
    keeps running after the pipeline ends (the server still does) and stops with the server."""
    import asyncio
    import contextlib

    async def purge() -> None:
        while True:
            await asyncio.sleep(SESSION_PURGE_S)
            store.purge_sessions(_now_us())

    async def run() -> None:
        task = asyncio.create_task(purge())
        try:
            await background()
            await task
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    return run


def cmd_serve(args: argparse.Namespace, cfg: Settings) -> int:
    from idsai.product.api import create_app

    host, port = resolve_bind(args, cfg)
    use_auth = auth_enabled(args, cfg)
    with _app_store(cfg) as store:
        check_serve_auth(host, use_auth, store)
        if args.loop and args.mode == "live":
            raise CliError("--loop ใช้ได้กับโหมด replay เท่านั้น (ไม่ใช้กับ --mode live)")
        if args.secure_cookies:
            cfg = cfg.model_copy(
                update={"auth": cfg.auth.model_copy(update={"cookie_secure": True})}
            )
        if host in WILDCARD_BINDS:
            _warn(
                f"หมายเหตุ: API รับเฉพาะคำขอที่เรียกด้วยชื่อ {', '.join(allowed_hosts(host))} "
                "ถ้าให้เครื่องอื่นเรียก ให้เปิดด้วย --host <IP ของเครื่องนี้> แล้วเรียกผ่าน IP นั้น"
            )
        if args.mode == "live":
            state, background = prepare_live(args, cfg, store)
        else:
            state, background = prepare_replay(args, cfg, store)
        store.purge_sessions(_now_us())
        app = create_app(state, allowed_hosts=allowed_hosts(host, args.public_host))
        _say(f"เปิดที่ http://{host if ':' not in host else f'[{host}]'}:{port} (กด Ctrl+C เพื่อหยุด)")
        run_server(app, host, port, with_session_purge(store, background))
    return 0


# ------------------------------------------------------------------ parser


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse would exit 2; a usage error is exit 1
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: ใช้คำสั่งไม่ถูกต้อง: {message}\n")


def _day_arg(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"วันที่ต้องเป็นรูปแบบ YYYY-MM-DD: {text!r}") from None


def _speed(text: str) -> float:
    try:
        x = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ความเร็วต้องเป็นตัวเลข: {text!r}") from None
    if not (math.isfinite(x) and x > 0):
        raise argparse.ArgumentTypeError(f"ความเร็วต้องเป็นตัวเลขจำกัดที่มากกว่า 0: {text!r}")
    return x


_HOST_RE = re.compile(r"^[a-z0-9.-]+$")


def _public_host(text: str) -> str:
    name = text.strip().lower()
    if not _HOST_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(
            f"ชื่อโฮสต์ต้องเป็นชื่ออย่างเดียว ไม่มี http:// พอร์ต หรือเส้นทาง: {text!r}"
        )
    return name


def _positive(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ต้องเป็นจำนวนเต็ม: {text!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"ต้องมากกว่า 0: {n}")
    return n


DOWNLOAD_HELP = """\
--split research: ทุกสัญลักษณ์-วันที่ชุดแบ่งข้อมูลวิจัย (make_split) และ eval ต้องอ่าน
  จากป้ายในเครื่อง (รัน `idsai labels` ก่อน):
  - วันที่มีป้ายในช่วง train / accept / test พร้อมวันก่อนหน้า (D-1) และวันถัดไป (D+1)
    เมื่อป้ายอยู่ภายใน 10 นาทีก่อนเที่ยงคืน
  - วันปกติที่เป็นตัวเลือก: ต่อสัญลักษณ์และช่วง (train/cal/accept/test)
    normal_days_per_symbol x 2 วันที่ห่างจากทุกป้ายของสัญลักษณ์นั้นอย่างน้อย min_gap_days
    เลือกวันที่ใกล้ป้ายที่สุดก่อน (รู้ว่าเหรียญมีการซื้อขายช่วงนั้น) ห่างกันอย่างน้อย 7 วัน
    แต่ละวันพร้อม D-1; make_split จะสุ่มจากวันที่ดาวน์โหลดได้จริง
--symbols S --days N: N วันล่าสุดที่เผยแพร่แล้ว (ถึงเมื่อวาน UTC) และ D-1 ของวันแรก
--symbols S --day D: วัน D และ D-1
วันที่คลังข้อมูลไม่มี (เช่นเหรียญยังไม่เข้าตลาด) นับว่า "ไม่มี" ไม่ใช่ข้อผิดพลาด
ดาวน์โหลดจาก data.binance.vision เท่านั้น และตรวจ CHECKSUM ทุกไฟล์
"""


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="idsai",
        description=(
            "Market IDS: เฝ้าดูตลาดคริปโตและแจ้งความผิดปกติทางสถิติ (เพื่อการศึกษา "
            "ไม่ส่งคำสั่งซื้อขาย ไม่ใช่คำแนะนำการลงทุน)"
        ),
    )
    p.add_argument(
        "--config", type=Path, default=None, help="ไฟล์ค่าตั้ง (ค่าเริ่มต้น configs/default.yaml)"
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>", parser_class=_Parser)

    s = sub.add_parser("labels", help="เขียนป้ายเหตุการณ์ pump จริง (Sapienza และ Hu ถ้ามี) ลงเครื่อง")
    s.add_argument(
        "--align",
        action="store_true",
        help="ตรวจว่าเวลาในป้ายตรงกับจุดที่ปริมาณซื้อขายสูงสุด (ไม่ผ่านถ้ามัธยฐาน |lag| >= 120 วินาที)",
    )

    s = sub.add_parser(
        "download",
        help="ดาวน์โหลดข้อมูลการซื้อขายย้อนหลังจาก Binance Vision",
        description=DOWNLOAD_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", choices=["research"], help="ทุกวันที่ชุดข้อมูลวิจัยต้องใช้")
    g.add_argument("--symbols", nargs="+", metavar="S", help="สัญลักษณ์ เช่น BTCUSDT ETHUSDT")
    w = s.add_mutually_exclusive_group()
    w.add_argument("--days", type=_positive, metavar="N", help="จำนวนวันล่าสุด (ใช้กับ --symbols)")
    w.add_argument("--day", type=_day_arg, metavar="D", help="วันเดียว YYYY-MM-DD (ใช้กับ --symbols)")

    s = sub.add_parser("train", help="ฝึกตัวตรวจจับและสอบเทียบ Chief แล้วบันทึกชุดโมเดล")
    s.add_argument("--seed", type=int, default=0, help="seed ของโมเดล (ค่าเริ่มต้น 0)")
    s.add_argument(
        "--profile",
        choices=["research", "live"],
        default="research",
        help="research: วันปกติช่วง train/cal ของงานวิจัย -> models/research; "
        "live: train_days วันล่าสุดของ live.symbols -> models/live",
    )

    s = sub.add_parser("eval", help="รันการทดลองและบันทึกผลลง data/research.duckdb")
    s.add_argument(
        "--experiment", choices=[*EXPERIMENT_NAMES, "all"], default="all", metavar="NAME|all"
    )
    s.add_argument("--seeds", type=_seeds, default=None, help="เช่น 0,1,2,3,4 (ค่าเริ่มต้นจากค่าตั้ง)")

    sub.add_parser("report", help="สร้างรายงานจากผลการทดลองทุกครั้ง (data/report.json และ .md)")

    s = sub.add_parser(
        "demo-pack",
        help="สร้างชุดเดโมเล็ก: วันที่มีป้าย pump 3 วัน + วันปกติ 20 วัน แล้วฝึกโมเดลลง data/demo/",
    )
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--pumps", type=_positive, default=DEMO_PUMPS, help="จำนวนวันที่มีป้าย pump")
    s.add_argument("--normal-days", type=_positive, default=DEMO_NORMAL_DAYS, help="จำนวนวันปกติ")

    s = sub.add_parser("serve", help="เปิด API และสายประมวลผล (replay หรือ live)")
    s.add_argument("--mode", choices=["replay", "live"], default="replay")
    s.add_argument("--symbol", default=None, help="replay: สัญลักษณ์ (ไม่ระบุ = demo pack)")
    s.add_argument("--day", type=_day_arg, default=None, help="replay: วัน YYYY-MM-DD")
    s.add_argument("--models", default=None, help="โฟลเดอร์ชุดโมเดล (ค่าเริ่มต้นตามโหมด)")
    s.add_argument("--host", default=DEFAULT_HOST, help="ค่าเริ่มต้น 127.0.0.1 (เฉพาะเครื่องนี้)")
    s.add_argument("--port", type=int, default=None, help="ค่าเริ่มต้น 8800")
    s.add_argument("--speed", type=_speed, default=DEFAULT_SPEED, help="replay: เท่าของเวลาจริง (60)")
    s.add_argument(
        "--public-host", action="append", default=[], type=_public_host, metavar="NAME",
        help="ชื่อโฮสต์สาธารณะหลัง proxy TLS (เช่น x.hf.space) เพิ่มใน allowlist ของ Host; ใช้ซ้ำได้",
    )
    s.add_argument(
        "--secure-cookies", action="store_true",
        help="ตั้งคุกกี้เซสชันเป็น Secure (แทน auth.cookie_secure)",
    )
    s.add_argument(
        "--loop", action="store_true", help="replay: เล่นวนซ้ำเมื่อจบ (ไม่ใช้กับ --mode live)"
    )
    g = s.add_mutually_exclusive_group()
    g.add_argument(
        "--auth", dest="auth", action="store_const", const=True, default=None,
        help="บังคับล็อกอิน (แทน auth.enabled)",
    )
    g.add_argument(
        "--no-auth", dest="auth", action="store_const", const=False,
        help="ไม่ล็อกอิน (แทน auth.enabled)",
    )

    from idsai.product.auth import ROLES

    s = sub.add_parser("user", help="จัดการบัญชีผู้ใช้ (รหัสผ่านถามแบบซ่อน ไม่รับทาง argv)")
    us = s.add_subparsers(dest="user_cmd", required=True, metavar="<action>", parser_class=_Parser)
    u = us.add_parser("add", help="สร้างผู้ใช้")
    u.add_argument("username")
    u.add_argument("--role", choices=ROLES, default="viewer")
    u.add_argument("--display-name", default="", help="ชื่อที่แสดง (ไม่เกิน 64 ตัว)")
    u.add_argument("--password-stdin", action="store_true", help="อ่านรหัสผ่านหนึ่งบรรทัดจาก stdin")
    us.add_parser("list", help="แสดงผู้ใช้ทั้งหมด")
    us.add_parser("seed-env", help="สร้าง/อัปเดตบัญชีจากตัวแปร IDSAI_USERS (ใช้ตอนบูตบนเซิร์ฟเวอร์)")
    u = us.add_parser("passwd", help="เปลี่ยนรหัสผ่าน (ออกจากระบบทุกอุปกรณ์)")
    u.add_argument("username")
    u.add_argument("--password-stdin", action="store_true", help="อ่านรหัสผ่านหนึ่งบรรทัดจาก stdin")
    u = us.add_parser("role", help="เปลี่ยนสิทธิ์")
    u.add_argument("username")
    u.add_argument("role", choices=ROLES)
    for act, hlp in (("disable", "ปิดใช้งานบัญชี"), ("enable", "เปิดใช้งานบัญชี")):
        u = us.add_parser(act, help=hlp)
        u.add_argument("username")
    return p


COMMANDS: dict[str, Callable[[argparse.Namespace, Settings], int]] = {
    "labels": cmd_labels,
    "download": cmd_download,
    "train": cmd_train,
    "eval": cmd_eval,
    "report": cmd_report,
    "demo-pack": cmd_demo_pack,
    "serve": cmd_serve,
    "user": cmd_user,
}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:  # Thai text must not crash a console or pipe with a legacy code page
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    if args.command == "download" and args.symbols and args.days is None and args.day is None:
        _warn("ใช้คำสั่งไม่ถูกต้อง: --symbols ต้องมี --days N หรือ --day D")
        return 1
    if args.command == "download" and args.split and (args.days or args.day):
        _warn("ใช้คำสั่งไม่ถูกต้อง: --split research ไม่ใช้ --days / --day")
        return 1
    try:
        cfg = load_settings(args.config)
    except (OSError, ValueError) as e:
        _warn(f"อ่านค่าตั้งไม่ได้: {e}")
        return 1
    try:
        return COMMANDS[args.command](args, cfg)
    except CliError as e:
        _warn(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
