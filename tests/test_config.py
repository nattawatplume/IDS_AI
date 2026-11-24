import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from idsai.config import (
    REPO_ROOT,
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
    assert s.live.symbols == ["BTCUSDT", "SOLUSDT", "DOGEUSDT", "PEPEUSDT"]
    assert s.live.ws_url == "wss://data-stream.binance.vision/stream"
    assert s.live.budget_per_hour == 2.0 and s.live.universe_size == 4
    assert s.live.train_days == 7 and s.live.bar_grace_s == 2
    assert s.detectors == ["rules", "stat.robust_z", "stat.cusum", "ai.iforest"]  # M2 live AI
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


def test_config_hash_ignores_the_auth_section():
    base = config_hash(load_settings())
    for auth in (
        {"enabled": True},
        {"cookie_secure": True},
        {"enabled": True, "cookie_secure": True},
    ):
        assert config_hash(load_settings(overrides={"auth": auth})) == base
    # other sections still change it
    assert config_hash(load_settings(overrides={"live": {"symbols": ["ZZZBTC"]}})) != base


def test_config_hash_matches_the_existing_demo_pack():
    manifest = Path(__file__).resolve().parents[1] / "data" / "demo" / "manifest.json"
    if not manifest.exists():
        pytest.skip("data/demo/manifest.json is absent: no demo pack to compare against")
    recorded = json.loads(manifest.read_text(encoding="utf-8"))["config_hash"]
    assert config_hash(load_settings()) == recorded


# ---------------------------------------------------------------- live AI fusion (ChiefCfg.ai_*)

# configs/default.yaml before the live AI layer (AE, fusion "fuse") and before M3 naming rules
OLD_DEFAULT = {
    "detectors": ["rules", "stat.robust_z", "stat.cusum", "ai.autoencoder_trade"],
    "chief": {"ai_mode": "fuse", "ai_shift": 0, "attr_rules": False},
}


def test_chief_ai_defaults_are_absent_from_the_dump():
    s = load_settings(overrides=OLD_DEFAULT)
    dumped = s.model_dump(mode="json")["chief"]
    assert not {"ai_mode", "ai_shift", "ai_regime_feature"} & set(dumped)
    # the pre-live-AI default.yaml and eval_prev_live.yaml keep their pinned hashes
    assert config_hash(s) == "6da45ffe7efd"
    assert config_hash(load_settings(REPO_ROOT / "configs/eval_prev_live.yaml")) == "53e3b662ab96"


def test_chief_ai_non_default_is_dumped_and_changes_the_hash():
    base = load_settings(overrides=OLD_DEFAULT)
    shift = load_settings(overrides={**OLD_DEFAULT, "chief": {"ai_mode": "shift", "ai_shift": 1}})
    dumped = shift.model_dump(mode="json")["chief"]
    assert dumped["ai_mode"] == "shift" and dumped["ai_shift"] == 1.0
    assert dumped["ai_regime_feature"] is None
    assert config_hash(shift) != config_hash(base)
    excl = load_settings(overrides={"chief": {"ai_mode": "exclude", "ai_shift": 0}})
    assert config_hash(excl) not in (config_hash(base), config_hash(shift))


def test_default_config_runs_iforest_with_shift1():
    s = load_settings()
    assert s.detectors == ["rules", "stat.robust_z", "stat.cusum", "ai.iforest"]
    assert (s.chief.ai_mode, s.chief.ai_shift, s.chief.ai_regime_feature) == ("shift", 1.0, None)


@pytest.mark.parametrize(
    "bad",
    [
        {"ai_mode": "nope"},
        {"ai_mode": "fuse", "ai_shift": 1.0},
        {"ai_mode": "shift", "ai_shift": -1.0},
        {"ai_mode": "shift", "ai_shift": float("inf")},
        {"ai_mode": "mondrian"},
        {"ai_mode": "shift", "ai_shift": 1.0, "ai_regime_feature": "z_quote_vol_300"},
    ],
)
def test_chief_ai_rejects_what_aifusion_rejects(bad):
    from idsai.ai.chief import AiFusion

    with pytest.raises(ValidationError):
        load_settings(overrides={"chief": bad})
    with pytest.raises(ValueError):
        AiFusion(bad["ai_mode"], bad.get("ai_shift", 0.0), bad.get("ai_regime_feature"))


def test_aifusion_from_cfg():
    from idsai.ai.chief import AiFusion

    assert AiFusion.from_cfg(load_settings().chief) == AiFusion("shift", 1.0)
    assert AiFusion.from_cfg(load_settings(overrides=OLD_DEFAULT).chief) == AiFusion()
    cfg = load_settings(
        overrides={"chief": {"ai_mode": "mondrian", "ai_shift": 0, "ai_regime_feature": "x"}}
    )
    assert AiFusion.from_cfg(cfg.chief) == AiFusion("mondrian", regime_feature="x")


# ---------------------------------------------------------------- M3 naming rules flag


def test_chief_attr_rules_off_is_absent_and_keeps_the_pinned_hashes():
    off = load_settings(overrides={"chief": {"attr_rules": False}})
    assert off.chief.attr_rules is False
    assert "attr_rules" not in off.model_dump(mode="json")["chief"]
    # the flag off reproduces default.yaml's hash from before M3 went live
    assert config_hash(off) == "7c3324364f13"
    prev = load_settings(REPO_ROOT / "configs/eval_prev_live.yaml")
    assert prev.chief.attr_rules is False
    assert config_hash(prev) == "53e3b662ab96"
    assert config_hash(load_settings(overrides=OLD_DEFAULT)) == "6da45ffe7efd"


def test_default_config_runs_the_m3_naming_rules():
    # M3 (docs/REPORT.md section 12): default.yaml turns the naming rules on
    default = load_settings()
    assert default.chief.attr_rules is True
    assert default.model_dump(mode="json")["chief"]["attr_rules"] is True
    assert config_hash(default) == "ee772544606c"


def test_chief_attr_rules_on_is_dumped_and_changes_the_hash():
    on = load_settings(overrides={"chief": {"attr_rules": True}})
    assert on.model_dump(mode="json")["chief"]["attr_rules"] is True
    assert config_hash(on) != config_hash(load_settings(overrides={"chief": {"attr_rules": False}}))
    # the AI flags keep their own drop rule beside it
    assert "ai_mode" in on.model_dump(mode="json")["chief"]
    both = load_settings(
        overrides={**OLD_DEFAULT, "chief": {**OLD_DEFAULT["chief"], "attr_rules": True}}
    )
    dumped = both.model_dump(mode="json")["chief"]
    assert dumped["attr_rules"] is True and "ai_mode" not in dumped
