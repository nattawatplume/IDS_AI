"""Detector registry: name -> factory, plus `build_detectors`.

Later tasks may add detectors with `register(name, factory)`. A factory takes
`(cfg: Settings, seed: int)` and returns one `Detector` or a list of them (`"rules"` expands to
every rule-layer signature). `"ai.autoencoder_trade"` is registered here with a lazy factory
(torch is imported only when it is built), so the default `Settings.detectors` always resolve
and `idsai.ai.autoencoder` never has to import this module.
"""

from collections.abc import Callable

from idsai.ai.baselines import IsolationForestDetector, PCADetector, RandomDetector
from idsai.config import Settings
from idsai.contracts.detector import Detector
from idsai.detection.features.registry import model_features
from idsai.detection.rules import KK_SID, SignatureDetector, load_signatures
from idsai.detection.stats import CusumDetector, RobustZDetector

Factory = Callable[[Settings, int], Detector | list[Detector]]

_FACTORIES: dict[str, Factory] = {}


def register(name: str, factory: Factory) -> None:
    """Add (or replace) a named detector factory."""
    _FACTORIES[name] = factory


def valid_names() -> list[str]:
    return sorted(_FACTORIES)


def build_detectors(names: list[str], cfg: Settings, seed: int = 0) -> list[Detector]:
    """Instantiate detectors by registry name; unknown names raise `KeyError` listing valid ones."""
    out: list[Detector] = []
    for name in names:
        if name not in _FACTORIES:
            raise KeyError(f"unknown detector {name!r}; valid names: {valid_names()}")
        made = _FACTORIES[name](cfg, seed)
        out.extend(made if isinstance(made, list) else [made])
    return out


def _rules(cfg: Settings, seed: int) -> list[Detector]:
    return [SignatureDetector(s) for s in load_signatures() if s.layer == "rule"]


def _kk(cfg: Settings, seed: int) -> Detector:
    (sig,) = [s for s in load_signatures() if s.sid == KK_SID]
    return SignatureDetector(sig)


def _robust_z(cfg: Settings, seed: int) -> Detector:
    return RobustZDetector([c for c in model_features(cfg.features) if c.startswith("z_")])


def _iforest(cfg: Settings, seed: int) -> Detector:
    return IsolationForestDetector(model_features(cfg.features), seed)


def _pca(cfg: Settings, seed: int) -> Detector:
    return PCADetector(model_features(cfg.features))


def _autoencoder(cfg: Settings, seed: int) -> Detector:
    from idsai.ai.autoencoder import AutoencoderDetector  # lazy: keeps torch off import path

    return AutoencoderDetector(model_features(cfg.features), cfg.ae, seed)


register("rules", _rules)
register("baseline.kk", _kk)
register("stat.robust_z", _robust_z)
register("stat.cusum", lambda cfg, seed: CusumDetector())
register("baseline.iforest", _iforest)
register("baseline.pca", _pca)
register("baseline.random", lambda cfg, seed: RandomDetector(seed))
register("ai.autoencoder_trade", _autoencoder)
