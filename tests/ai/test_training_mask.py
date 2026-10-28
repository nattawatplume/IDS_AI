import inspect

import polars as pl

from idsai.ai.learning import training_mask
from idsai.contracts.frames import LABELS, conform, empty

S = 1_000_000
H = 3600 * S


def labels(*rows) -> pl.DataFrame:
    if not rows:
        return empty(LABELS)
    return conform(
        pl.DataFrame(
            {
                "label_id": [f"L{i}" for i in range(len(rows))],
                "source": "test",
                "symbol": [r[0] for r in rows],
                "t_start_us": [r[1] for r in rows],
                "t_end_us": [r[2] for r in rows],
                "technique_id": "MM-T001",
                "family": "pump",
                "scenario_seed": pl.Series([None] * len(rows), dtype=pl.Int64),
                "provenance": "REAL",
            }
        ),
        LABELS,
    )


def windows(symbol_times: dict[str, list[int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [s for s, ts in symbol_times.items() for _ in ts],
            "t_end_us": [t for ts in symbol_times.values() for t in ts],
        },
        schema={"symbol": pl.String, "t_end_us": pl.Int64},
    )


def test_excludes_only_labeled_symbol_within_margin_inclusive():
    a, b = 10 * H, 11 * H
    ts = [a - H - 1, a - H, a, (a + b) // 2, b, b + H, b + H + 1]
    w = windows({"AAA": ts, "BBB": ts})
    keep, frac = training_mask(w, labels(("AAA", a, b)))
    assert keep.name == "keep" and keep.dtype == pl.Boolean and keep.len() == w.height
    assert keep.to_list() == [True, False, False, False, False, False, True] + [True] * 7
    assert frac == 5 / 14


def test_margin_parameter_and_nested_labels():
    a, b = 10 * H, 11 * H
    # a long label containing a short one: the short one must not hide the long one's tail
    labs = labels(("AAA", a, b), ("AAA", a + 10 * S, a + 20 * S))
    w = windows({"AAA": [a - 61 * S, a - 60 * S, b - S, b + 60 * S, b + 61 * S]})
    keep, frac = training_mask(w, labs, margin_s=60)
    assert keep.to_list() == [True, False, False, False, True]
    assert frac == 3 / 5


def test_unsorted_windows_and_no_labels():
    a, b = 10 * H, 11 * H
    w = windows({"AAA": [b + 2 * H, a, a - 2 * H]})
    keep, _ = training_mask(w, labels(("AAA", a, b)))
    assert keep.to_list() == [True, False, True]
    keep, frac = training_mask(w, labels())
    assert keep.to_list() == [True, True, True] and frac == 0.0


def test_empty_windows():
    keep, frac = training_mask(windows({}), labels(("AAA", 0, H)))
    assert keep.len() == 0 and keep.name == "keep" and keep.dtype == pl.Boolean
    assert frac == 0.0


def test_no_alert_parameter():
    params = inspect.signature(training_mask).parameters
    assert "alerts" not in params
    assert list(params) == ["windows", "labels", "margin_s"]
    assert params["margin_s"].default == 3600
