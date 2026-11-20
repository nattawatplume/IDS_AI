"""M4 sell-phase labels of REAL pumps and the per-phase recall of an alert set.

For a REAL label measured by `pump_measure.measure_label_store` (status `measured`) the sell window
is `[t_peak, t_peak + clip(dump_tau_s, 60 s, 1 h)]`; a right-censored dump (`dump_tau_s` None: the
price never got back within the 1 h search) gets the full hour and is flagged `dump_censored`.
Labels with another status have no sell window and are left out of the sell-phase recall (their
statuses are counted).

Per label (`per_event`):
- `detected_pump_phase`: the existing match on the label's onset window (= `detected_any`).
- `detected_sell_phase`: an alert episode of any technique overlaps the sell window.
- `detected_sell_phase_correct`: such an episode names the sell phase (`Alert.phase == "sell"`,
  i.e. its signature is MM-T001.003 or MM-T006.*).
Both sell keys are None for a label without a sell window. These are recalls of statistical
anomalies (ความผิดปกติทางสถิติ), never a probability and never an accuracy.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any

import polars as pl

from idsai.contracts.alert import Alert
from idsai.eval.pump_measure import measure_label_store

S_US = 1_000_000
SELL_MIN_S = 60
SELL_MAX_S = 3600
SELL_KEYS = ("detected_sell_phase", "detected_sell_phase_correct")


def sell_window(m: dict[str, Any]) -> tuple[int, int] | None:
    """`(lo, hi)` µs of the sell window of one measurement, or None unless it was measured."""
    if m.get("status") != "measured":
        return None
    tau = m.get("dump_tau_s")
    span = SELL_MAX_S if tau is None else min(max(float(tau), SELL_MIN_S), SELL_MAX_S)
    t = int(m["t_peak_us"])
    return t, t + int(round(span * S_US))


def derive_sell_labels(store: Any, labels: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """`{label_id: {status, sell_window, dump_censored}}` for every label (deterministic; one
    measurement per label, in label_id order)."""
    out: dict[str, dict[str, Any]] = {}
    rows = labels.select("label_id", "symbol", "t_start_us").sort("label_id").iter_rows()
    for lid, sym, t0 in rows:
        if lid in out:
            continue
        m = measure_label_store(store, sym, int(t0))
        win = sell_window(m)
        out[lid] = {
            "status": m["status"],
            "sell_window": list(win) if win is not None else None,
            "dump_censored": win is not None and m.get("dump_tau_s") is None,
        }
    return out


class SellPhase:
    """Per-phase recall of alert episodes against the sell windows of `sell` (see module)."""

    def __init__(self, sell: dict[str, dict[str, Any]]):
        self.sell = sell

    def per_event(self, alerts: Sequence[Alert], labels: pl.DataFrame) -> dict[str, dict]:
        by_sym: dict[str, list[Alert]] = {}
        for a in alerts:
            by_sym.setdefault(a.symbol, []).append(a)
        out: dict[str, dict] = {}
        for lid, sym in labels.select("label_id", "symbol").iter_rows():
            info = self.sell.get(lid, {"status": "not_measured", "sell_window": None})
            win = info["sell_window"]
            pe: dict[str, Any] = {
                "sell_status": info["status"],
                "dump_censored": bool(info.get("dump_censored", False)),
            }
            if win is None:
                pe.update(dict.fromkeys(SELL_KEYS))
            else:
                lo, hi = win
                hits = [a for a in by_sym.get(sym, []) if a.window[0] <= hi and a.window[1] >= lo]
                pe["detected_sell_phase"] = bool(hits)
                pe["detected_sell_phase_correct"] = any(a.phase == "sell" for a in hits)
            out[lid] = pe
        return out

    @staticmethod
    def summarize(per_event: dict[str, dict]) -> dict[str, Any]:
        """`recall_pump_phase`, `recall_sell_phase`, `recall_sell_phase_correct`, the number of
        labels with a sell window and the count of each measurement status."""
        out: dict[str, Any] = {}
        pump = [bool(pe["detected_any"]) for pe in per_event.values() if "detected_any" in pe]
        out["recall_pump_phase"] = sum(pump) / len(pump) if pump else float("nan")
        for key in SELL_KEYS:
            vals = [pe[key] for pe in per_event.values() if pe.get(key) is not None]
            out[key.replace("detected_", "recall_")] = (
                sum(vals) / len(vals) if vals else float("nan")
            )
        out["n_sell_events"] = sum(
            1 for pe in per_event.values() if pe.get("detected_sell_phase") is not None
        )
        status = Counter(pe.get("sell_status", "not_measured") for pe in per_event.values())
        out["sell_status_counts"] = dict(sorted(status.items()))
        return out
