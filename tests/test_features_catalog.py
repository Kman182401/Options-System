"""docs/FEATURES.md catalog stays exactly in sync with the engine (Task 4)."""

from __future__ import annotations

import re
from pathlib import Path

from options_system.features.compute import feature_names
from options_system.features.config import FeatureConfig

_CATALOG = Path(__file__).resolve().parents[1] / "docs" / "FEATURES.md"
# catalog rows start: | `feature_name` | ...
_ROW = re.compile(r"^\|\s*`([a-z0-9_]+)`\s*\|")


def _catalog_names() -> set[str]:
    names: set[str] = set()
    for line in _CATALOG.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m:
            names.add(m.group(1))
    return names


def test_every_feature_is_catalogued_and_no_orphans():
    emitted = set(feature_names(FeatureConfig.load()))
    catalogued = _catalog_names()
    missing = emitted - catalogued  # emitted but undocumented
    orphan = catalogued - emitted  # documented but not emitted
    assert not missing, f"features missing from docs/FEATURES.md: {sorted(missing)}"
    assert not orphan, f"orphan rows in docs/FEATURES.md: {sorted(orphan)}"
