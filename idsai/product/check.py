"""Phase 2: a coin's alert history (2.1) and the check-before-buying summary (2.2).

Product layer only: it reads the store and the pipeline's ring, never the detectors, and changes no
config. The label is one of ปกติ / เฝ้าระวัง / ผิดปกติมาก (or ยังไม่มีข้อมูลพอ during warm-up): a
reading of statistical anomaly, never a buy or sell signal. F is an anomaly score compared with the
alert threshold τ; neither F nor p is a probability.

"Now" is the pipeline's clock (`latest_us`: the replay clock in replay), else the wall clock, so a
replay's history and 24 h window are relative to the replayed time.
"""

import statistics
import time
from typing import Any

from idsai.contracts.alert import Alert
from idsai.data.store import COUNTED_PROVENANCE, Store

S_US = 1_000_000
MIN_US = 60 * S_US
HOUR_US = 60 * MIN_US
DAY_US = 24 * HOUR_US

# ---- the label rule's thresholds: fixed here, no config ----
HISTORY_DAYS = (7, 30)
ALERTS_WINDOW_US = DAY_US  # alerts_24h
OPEN_ALERT_WINDOW_US = 60 * MIN_US  # an open / updated alert this recent -> ผิดปกติมาก
WATCH_F_FRACTION = 0.5  # F_now >= 0.5 * tau -> เฝ้าระวัง
THIN_RATIO = 0.3  # liquidity ratio below this -> เฝ้าระวัง (thin market)
SURGE_RATIO = 5.0  # liquidity ratio above this -> เฝ้าระวัง (unusual surge)
LIQ_RECENT_S = 300  # qvol_5m: the last 5 minutes
LIQ_MIN_MINUTES = 60  # fewer whole one-minute buckets in the ring: liquidity is unknown (null)

LABEL_NORMAL = "ปกติ"
LABEL_WATCH = "เฝ้าระวัง"
LABEL_HIGH = "ผิดปกติมาก"
LABEL_WARMUP = "ยังไม่มีข้อมูลพอ"
LEVELS = {LABEL_NORMAL: "normal", LABEL_WATCH: "watch", LABEL_HIGH: "high", LABEL_WARMUP: "warmup"}
WARMUP_REASON_TH = "ระบบกำลังเก็บข้อมูลพื้นฐาน"
THIN_ZERO_TH = "สภาพคล่องบาง: ช่วงที่ผ่านมาส่วนใหญ่แทบไม่มีการซื้อขาย"
DISCLAIMER_TH = "ข้อมูลนี้ไม่ใช่คำแนะนำการลงทุน"


def now_us_of(pipeline: Any) -> int:
    """The pipeline's latest event time when it has one, else the wall clock."""
    latest = getattr(pipeline, "latest_us", None) if pipeline is not None else None
    return int(latest) if latest is not None else time.time_ns() // 1000


def coin_history(store: Store, symbol: str, now_us: int) -> dict[str, Any]:
    """`{symbol, now_us, d7, d30, by_phase_d30}`: alerts (live and replay, never sim) per window,
    each episode once; `by_phase_d30` splits the 30 d count into pump / sell / unknown."""
    d30 = store.alerts_between(symbol, now_us - 30 * DAY_US, now_us, COUNTED_PROVENANCE)
    since7 = now_us - 7 * DAY_US
    by_phase = {"pump": 0, "sell": 0, "unknown": 0}
    for a in d30:
        by_phase[a.phase if a.phase in ("pump", "sell") else "unknown"] += 1
    return {
        "symbol": symbol,
        "now_us": now_us,
        "d7": sum(1 for a in d30 if a.ts_event_us >= since7),
        "d30": len(d30),
        "by_phase_d30": by_phase,
    }


def liquidity(quote_vols: list[float], stride_s: int) -> dict[str, float | None]:
    """`{qvol_5m, qvol_median_24h, ratio}` from the ring's per-stride quote volumes (oldest first).

    `qvol_5m` is the mean quote volume per minute over the last 5 minutes; `qvol_median_24h` the
    median per-minute volume over whole minutes of the ring (up to 24 h). Each is None with fewer
    than `LIQ_MIN_MINUTES` whole one-minute buckets; `ratio` is None too when the median is not
    positive (`label_for` reads a zero median as a thin market).
    """
    vols = [v for v in quote_vols if v is not None]
    per_min = 60.0 / stride_s  # one point's volume times this is a per-minute rate
    k = max(1, 60 // stride_s)  # points per minute bucket, counted back from the newest
    n_full = len(vols) // k * k
    if n_full // k < LIQ_MIN_MINUTES:
        return {"qvol_5m": None, "qvol_median_24h": None, "ratio": None}
    recent = vols[-max(1, LIQ_RECENT_S // stride_s) :]
    qvol_5m = sum(recent) * per_min / len(recent)
    tail = vols[len(vols) - n_full :]
    minutes = [sum(tail[i : i + k]) * per_min / k for i in range(0, n_full, k)]
    median = float(statistics.median(minutes))
    ratio = qvol_5m / median if median > 0 else None
    return {"qvol_5m": qvol_5m, "qvol_median_24h": median, "ratio": ratio}


def label_for(
    score: dict[str, Any] | None,
    alerts_24h: list[Alert],
    liq: dict[str, float | None],
    now_us: int,
) -> tuple[str, list[str]]:
    """The fixed label rule; returns `(label, reasons_th)`.

    ผิดปกติมาก: F_now >= τ, or an open / updated alert whose episode last exceeded the threshold
    (`window[1]`) in the last 60 min. Else เฝ้าระวัง: an alert in 24 h, F_now >= 0.5 τ, a liquidity
    ratio < 0.3 or > 5, or a zero median volume (a market that mostly does not trade). Else ปกติ.
    No score yet: ยังไม่มีข้อมูลพอ.
    """
    f_now = None if score is None else score.get("F_now")
    tau = None if score is None else score.get("tau")
    if f_now is None or tau is None:
        return LABEL_WARMUP, [WARMUP_REASON_TH]
    high: list[str] = []
    if f_now >= tau:
        high.append(f"คะแนนความผิดปกติ (F) ตอนนี้ {f_now:.2f} ถึงเกณฑ์แจ้งเตือน τ = {tau:.2f}")
    recent_open = [
        a
        for a in alerts_24h
        if a.status in ("open", "updated") and a.window[1] >= now_us - OPEN_ALERT_WINDOW_US
    ]
    if recent_open:
        high.append("มีการแจ้งเตือนที่ยังเปิดอยู่ภายใน 60 นาทีล่าสุด")
    if high:
        return LABEL_HIGH, high
    watch: list[str] = []
    if alerts_24h:
        watch.append(f"ถูกเตือน {len(alerts_24h)} ครั้งใน 24 ชม. ล่าสุด")
    if f_now >= WATCH_F_FRACTION * tau:
        watch.append(f"คะแนนความผิดปกติ (F) {f_now:.2f} เกินครึ่งของเกณฑ์ τ = {tau:.2f}")
    ratio = liq.get("ratio")
    if ratio is not None and ratio < THIN_RATIO:
        watch.append(f"สภาพคล่องบาง: ปริมาณซื้อขาย 5 นาทีล่าสุดเป็น {ratio:.2f} เท่าของปกติ")
    if liq.get("qvol_median_24h") == 0:
        watch.append(THIN_ZERO_TH)
    if ratio is not None and ratio > SURGE_RATIO:
        watch.append(f"ปริมาณซื้อขายสูงผิดปกติ: {ratio:.1f} เท่าของปกติ")
    if watch:
        return LABEL_WATCH, watch
    return LABEL_NORMAL, ["ไม่พบความผิดปกติทางสถิติเด่นชัดในตอนนี้"]


def check(
    store: Store, symbol: str, pipeline: Any, tau: float | None, stride_s: int
) -> dict[str, Any]:
    """The check-before-buying JSON: score, alerts_24h, liquidity, history, label, reasons."""
    now_us = now_us_of(pipeline)
    last = None if pipeline is None else pipeline.last_point(symbol)
    score = None if last is None else {"F_now": last["F"], "tau": tau, "t_end_us": last["t_end_us"]}
    recent = store.alerts_between(symbol, now_us - ALERTS_WINDOW_US, now_us, COUNTED_PROVENANCE)
    newest = recent[0] if recent else None
    liq = liquidity([] if pipeline is None else pipeline.quote_vols(symbol), stride_s)
    label, reasons = label_for(score, recent, liq, now_us)
    return {
        "symbol": symbol,
        "now_us": now_us,
        "score": score,
        "alerts_24h": {
            "count": len(recent),
            "latest": None
            if newest is None
            else {
                "alert_id": newest.alert_id,
                "ts_event_us": newest.ts_event_us,
                "technique_id": newest.technique_id,
                "phase": newest.phase,
                "status": newest.status,
            },
        },
        "liquidity": liq,
        "history": coin_history(store, symbol, now_us),
        "label": label,
        "level": LEVELS[label],
        "reasons_th": reasons,
        "disclaimer_th": DISCLAIMER_TH,
    }
