import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from idsai.config import (
    FeatureCfg,
    config_hash,
    file_sha,
    load_settings,
    quote_unit_for,
)


def test_defaults_match_spec():
    s = load_settings()
    assert s.features.windows_s == [10, 60, 300, 1800] and s.features.stride_s == 10
    assert s.features.min_baseline_samples == 360 and s.features.baseline_sample_s == 60
    assert s.chief.budget_per_symbol_day == 2.0 and s.chief.attribution_min == 3.0
    assert s.eval.split_seed == 0 and s.port == 8800


def test_remaining_defaults_match_spec():
    s = load_settings()
    assert s.features.baseline_s == 86400 and s.features.baseline_refresh_s == 300
    assert s.features.quote_unit == {"BTC": 1e-4, "USDT": 1.0}
    assert s.chief.suppress_for_s == 300 and s.chief.agreement_p == 0.01
    assert s.eval.before_s == 120 and s.eval.after_s == 300
    assert s.eval.seeds == [0, 1, 2, 3, 4] and s.eval.n_boot == 1000
    assert s.eval.min_effect == 0.02 and s.eval.hard_negative_k == 20
    assert s.eval.budgets_per_symbol_day == [0.1, 0.5, 2, 10]
    assert s.ae.hidden == 64 and s.ae.bottleneck == 16 and s.ae.train_stride_s == 60
    assert s.live.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "PEPEUSDT"]
    assert s.live.ws_url == "wss://data-stream.binance.vision/stream"
    assert s.live.budget_per_hour == 2.0 and s.live.universe_size == 5
    assert s.live.train_days == 7 and s.live.bar_grace_s == 2
    assert s.detectors == ["rules", "stat.robust_z", "stat.cusum", "ai.autoencoder_trade"]
    assert s.mode in ("replay", "live")
    assert s.paths.app_db == Path("data/app.duckdb")
    assert s.paths.research_db == Path("data/research.duckdb")
    assert s.paths.report_json == Path("data/report.json")


def test_splits_strictly_ordered():
    sp = load_settings().splits
    assert sp.train[1] < sp.cal[0] <= sp.cal[1] < sp.accept[0] <= sp.accept[1] < sp.test[0]
    assert sp.train == (date(2018, 1, 1), date(2019, 6, 30))
    assert sp.test == (date(2020, 7, 1), date(2022, 1, 31))
    assert sp.normal_days_per_symbol == 4 and sp.min_gap_days == 2
    assert sp.label_offset_s == {"sapienza": 0, "hu": 0}


def test_splits_overlap_rejected():
    with pytest.raises(ValidationError):
        load_settings(overrides={"splits": {"cal": ["2019-06-01", "2019-12-31"]}})


def test_load_settings_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_settings().port == 8800


def test_overrides_deep_merge():
    s = load_settings(overrides={"features": {"stride_s": 5}, "port": 9000})
    assert s.features.stride_s == 5 and s.port == 9000
    assert s.features.windows_s == [10, 60, 300, 1800]


def test_load_settings_explicit_path(tmp_path):
    p = tmp_path / "custom.yaml"
    p.write_text("port: 9001\n", encoding="utf-8")
    s = load_settings(p)
    assert s.port == 9001
    assert s.features.windows_s == [10, 60, 300, 1800]


def test_config_hash_stable_and_sensitive():
    a, b = load_settings(), load_settings()
    h = config_hash(a)
    assert h == config_hash(b)
    assert len(h) == 12 and int(h, 16) >= 0
    assert config_hash(load_settings(overrides={"port": 9000})) != h
    assert config_hash(load_settings(overrides={"chief": {"agreement_p": 0.05}})) != h


def test_file_sha(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello")
    assert file_sha(p) == "2cf24dba5fb0"  # first 12 hex of sha256(b"hello")


def test_stride_must_divide_windows_and_sample():
    with pytest.raises(ValidationError):
        load_settings(overrides={"features": {"stride_s": 7}})
    with pytest.raises(ValidationError):
        FeatureCfg(stride_s=20)  # does not divide window 10
    with pytest.raises(ValidationError):
        FeatureCfg(windows_s=[20, 60], stride_s=20, baseline_sample_s=50)


def test_sample_must_divide_refresh():
    with pytest.raises(ValidationError):
        FeatureCfg(baseline_sample_s=60, baseline_refresh_s=250)


def test_quote_unit_for():
    cfg = load_settings().features
    assert quote_unit_for("BRDBTC", cfg) == 1e-4
    assert quote_unit_for("PEPEUSDT", cfg) == 1.0
    with pytest.raises(KeyError):
        quote_unit_for("ETHEUR", cfg)


def test_settings_rejects_bad_mode():
    with pytest.raises(ValidationError):
        load_settings(overrides={"mode": "paper"})


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        load_settings(overrides={"chief": {"agrement_p": 0.05}})
    with pytest.raises(ValidationError):
        load_settings(overrides={"prot": 1})
    with pytest.raises(ValidationError):
        load_settings(overrides={"splits": {"tran": ["2018-01-01", "2018-02-01"]}})


def test_config_hash_platform_independent_paths():
    base = load_settings()
    # os-native join (backslashes on Windows) must hash like the posix spelling
    native = load_settings(overrides={"paths": {"app_db": Path("data") / "app.duckdb"}})
    assert config_hash(native) == config_hash(base)
    assert "\\" not in json.dumps(base.model_dump(mode="json")["paths"])
    assert "\\" not in json.dumps(native.model_dump(mode="json")["paths"])
    assert config_hash(load_settings(overrides={"paths": {"app_db": "data/other.duckdb"}})) != (
        config_hash(base)
    )
