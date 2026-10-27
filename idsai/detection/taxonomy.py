"""Loader for `taxonomy.yaml` (technique ids, Thai names / definitions, references)."""

from pathlib import Path
from typing import Any

import yaml

TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy.yaml"


def load_taxonomy(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return `{technique_id: entry}`; each entry has `sub_techniques` as `[{id, ...}]`."""
    p = Path(path) if path is not None else TAXONOMY_PATH
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def technique_ids(taxonomy: dict[str, dict[str, Any]] | None = None) -> set[str]:
    """Every known id, parents and sub-techniques alike."""
    tax = taxonomy if taxonomy is not None else load_taxonomy()
    ids = set(tax)
    for entry in tax.values():
        ids.update(s["id"] for s in entry.get("sub_techniques", []))
    return ids
