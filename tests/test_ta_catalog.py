"""docs/TA_FEATURES.md catalog stays exactly in sync with the TA engine."""

from __future__ import annotations

import re
from pathlib import Path

from options_system.ta.compute import ta_feature_names
from options_system.ta.config import TaConfig

_CATALOG = Path(__file__).resolve().parents[1] / "docs" / "TA_FEATURES.md"
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
    emitted = set(ta_feature_names(TaConfig.load()))
    catalogued = _catalog_names()
    missing = emitted - catalogued  # emitted but undocumented
    orphan = catalogued - emitted  # documented but not emitted
    assert not missing, f"features missing from docs/TA_FEATURES.md: {sorted(missing)}"
    assert not orphan, f"orphan rows in docs/TA_FEATURES.md: {sorted(orphan)}"
