"""Label leakage guard: detectors (`detection/`, `ai/`) never touch evaluation labels.

Exempt: `detection/simulator/` (writes simulated labels) and `ai/learning.py` (training mask).
"""

from pathlib import Path

import pytest

import idsai
from idsai.ai.auditor import leakage_violations

PKG = Path(idsai.__file__).resolve().parent


def test_real_repo_has_no_label_leakage():
    assert leakage_violations(PKG) == []


def test_real_repo_scan_covers_the_detector_modules():
    # sanity: the guard looks where the detectors are (an empty scan would pass vacuously)
    assert (PKG / "ai" / "chief.py").is_file() and (PKG / "detection" / "rules.py").is_file()


def _plant(root: Path, rel: str, source: str) -> Path:
    p = root / "idsai" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(source, encoding="utf-8")
    return root / "idsai"


@pytest.mark.parametrize(
    ("rel", "source", "lineno"),
    [
        ("ai/bad.py", "import os\nimport idsai.eval.metrics\n", 2),
        ("ai/bad.py", "import idsai.eval as ev\n", 1),
        ("ai/bad.py", "from idsai.eval.labels import load_labels\n", 1),
        ("ai/bad.py", "from idsai import eval\n", 1),
        ("ai/bad.py", "from ..eval import metrics\n", 1),
        ("ai/bad.py", "from .. import eval\n", 1),
        ("detection/features/bad.py", "from ...eval.splits import make_split\n", 1),
        ("ai/sub/__init__.py", "from ...eval import report\n", 1),
        ("ai/bad.py", "import importlib\nm = importlib.import_module('idsai.eval.labels')\n", 2),
        ("ai/bad.py", "from importlib import import_module\nimport_module('idsai.eval')\n", 2),
        ("ai/bad.py", "m = __import__('idsai.eval.metrics')\n", 1),
        ("ai/bad.py", "from idsai.contracts.frames import LABELS\n", 1),
        ("ai/bad.py", "from idsai.contracts.frames import LABELS as L\n", 1),
        ("ai/bad.py", "import idsai.contracts.frames as f\n\nschema = f.LABELS\n", 3),
        ("detection/bad.py", "def f():\n    return LABELS\n", 2),
    ],
)
def test_planted_violation_is_caught(tmp_path, rel, source, lineno):
    found = leakage_violations(_plant(tmp_path, rel, source))
    assert found, source
    assert all(v.startswith(f"idsai/{rel}:") for v in found)
    assert any(v.startswith(f"idsai/{rel}:{lineno}: ") for v in found), found


@pytest.mark.parametrize(
    "source",
    [
        "from .chief import Chief\nfrom ..contracts.frames import TRADES\n",
        "import idsai.evaluation_helpers\nfrom idsai.config import Settings\n",
        "model.eval()\nx = 'LABELS'\ny = 'idsai.eval'\n",  # strings and .eval() are fine
        "import importlib\nm = importlib.import_module('idsai.ai.chief')\n",
    ],
)
def test_clean_module_is_not_flagged(tmp_path, source):
    assert leakage_violations(_plant(tmp_path, "ai/ok.py", source)) == []


@pytest.mark.parametrize(
    "rel",
    [
        "detection/simulator/inject.py",
        "detection/simulator/deep/labels.py",
        "ai/learning.py",
        "eval/report.py",  # outside the scanned trees
        "pipeline/batch.py",
    ],
)
def test_exempt_paths_are_not_flagged(tmp_path, rel):
    src = "from idsai.contracts.frames import LABELS\nimport idsai.eval.metrics\n"
    assert leakage_violations(_plant(tmp_path, rel, src)) == []


def test_nested_learning_py_is_not_exempt(tmp_path):
    root = _plant(tmp_path, "ai/sub/learning.py", "import idsai.eval\n")
    assert leakage_violations(root) == ["idsai/ai/sub/learning.py:1: imports idsai.eval"]


def test_unparsable_module_is_reported(tmp_path):
    found = leakage_violations(_plant(tmp_path, "ai/broken.py", "def (:\n"))
    assert len(found) == 1 and found[0].startswith("idsai/ai/broken.py:1: ")


def test_violations_are_sorted_and_name_each_file(tmp_path):
    _plant(tmp_path, "detection/b.py", "import idsai.eval\n")
    root = _plant(tmp_path, "ai/a.py", "from idsai.contracts.frames import LABELS\n")
    found = leakage_violations(root)
    assert found == sorted(found) and len(found) == 2
    assert found[0].startswith("idsai/ai/a.py:1: ") and "LABELS" in found[0]
    assert found[1] == "idsai/detection/b.py:1: imports idsai.eval"


@pytest.mark.parametrize(
    ("source", "lineno", "reason"),
    [
        (
            "import importlib\nm = importlib.import_module(name='idsai.eval.labels')\n",
            2,
            "imports idsai.eval.labels via import_module",
        ),
        (
            "import importlib\nm = importlib.import_module('.eval', 'idsai')\n",
            2,
            "imports idsai.eval via import_module",
        ),
        (
            "import importlib\nm = importlib.import_module('.metrics', package='idsai.eval')\n",
            2,
            "imports idsai.eval.metrics via import_module",
        ),
        (  # non-constant package: resolved against the module's own package (idsai.ai)
            "import importlib\nm = importlib.import_module('..eval', __package__)\n",
            2,
            "imports idsai.eval via import_module",
        ),
        ("m = __import__(name='idsai.eval')\n", 1, "imports idsai.eval via __import__"),
        (
            "m = __import__('idsai', globals(), locals(), ['contracts', 'eval'])\n",
            1,
            "imports idsai.eval via __import__",
        ),
        ("m = __import__('idsai', fromlist=('eval',))\n", 1, "imports idsai.eval via __import__"),
        (
            "from importlib import import_module as im\nm = im('idsai.eval')\n",
            2,
            "imports idsai.eval via import_module",
        ),
        ("import idsai.contracts.frames as f\nx = getattr(f, 'LABELS')\n", 2, "via getattr"),
        ("import idsai\n\nx = idsai.eval.metrics.paired_bootstrap\n", 3, "attribute access"),
        ("import idsai.ai.chief\nx = idsai.eval.labels\n", 2, "attribute access"),
        ("import idsai as pkg\nx = pkg.eval.labels\n", 2, "attribute access"),
    ],
)
def test_dynamic_and_attribute_access_is_caught(tmp_path, source, lineno, reason):
    found = leakage_violations(_plant(tmp_path, "ai/bad.py", source))
    assert any(v.startswith(f"idsai/ai/bad.py:{lineno}: ") and reason in v for v in found), found


@pytest.mark.parametrize(
    "source",
    [
        "import importlib\nm = importlib.import_module('.chief', 'idsai.ai')\n",
        "x = getattr(obj, 'TRADES')\n",
        "import numpy as np\ny = np.eval\n",  # `.eval` on something that is not the package
        "model.eval()\n",
        "m = __import__('idsai', fromlist=['contracts'])\n",
        "from importlib import import_module as im\nm = im('idsai.contracts')\n",
    ],
)
def test_new_catches_do_not_flag_clean_code(tmp_path, source):
    assert leakage_violations(_plant(tmp_path, "ai/ok.py", source)) == []


def test_non_utf8_module_is_reported_not_crashing(tmp_path):
    p = tmp_path / "idsai" / "ai" / "latin.py"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x = '\xe9t\xe9'\n")
    found = leakage_violations(tmp_path / "idsai")
    assert len(found) == 1 and found[0].startswith("idsai/ai/latin.py:0: cannot decode")


def test_utf8_bom_module_is_parsed(tmp_path):
    p = tmp_path / "idsai" / "ai" / "bom.py"
    p.parent.mkdir(parents=True)
    p.write_bytes("﻿import idsai.eval\n".encode())
    assert leakage_violations(tmp_path / "idsai") == ["idsai/ai/bom.py:1: imports idsai.eval"]


@pytest.mark.parametrize("rel", ["detection/simulator.py", "detection/simulator_x/mod.py"])
def test_simulator_lookalikes_are_scanned(tmp_path, rel):
    found = leakage_violations(_plant(tmp_path, rel, "import idsai.eval\n"))
    assert found == [f"idsai/{rel}:1: imports idsai.eval"]


@pytest.mark.parametrize(
    ("source", "hit"),
    [
        ("from idsai.ai.learning import training_mask\n", "idsai.ai.learning"),
        ("from idsai.ai import learning\n", "idsai.ai.learning"),
        ("from . import learning\n", "idsai.ai.learning"),
        ("from idsai.data.store import Store\n", "idsai.data.store"),
        ("import idsai.pipeline.batch\n", "idsai.pipeline.batch"),
        ("from ..pipeline import batch\n", "idsai.pipeline"),
        ("import importlib\nm = importlib.import_module('idsai.data.store')\n", "idsai.data.store"),
    ],
)
def test_label_reading_modules_are_caught(tmp_path, source, hit):
    # unscanned modules that read labels: importing them from a detector is a leakage path
    found = leakage_violations(_plant(tmp_path, "ai/bad.py", source))
    assert found and all(hit in v for v in found), found


@pytest.mark.parametrize(
    "source",
    [
        "from idsai.ai.chief import Chief\n",
        "from idsai.contracts.frames import TRADES\n",
        "import idsai.datagen\n",
        "from idsai.config import load_settings\n",
    ],
)
def test_label_reader_prefix_is_exact(tmp_path, source):
    assert leakage_violations(_plant(tmp_path, "ai/ok.py", source)) == []
