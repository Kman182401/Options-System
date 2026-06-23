"""Phase-24 unit tests — config inheritance, coverage gating, and the per-block decision logic.

Covers the NEW Phase-24 logic in isolation (the frozen contract loads to the right knobs, the
baseline/augmented feature toggles, the cheap coverage fraction, and the decision rule). The
end-to-end incremental verdict is exercised by the real ``run_freedata`` run.
"""

from __future__ import annotations

import numpy as np

from options_system.volatility.config_freedata import Phase24Config
from options_system.volatility.run_freedata import _coverage_fraction, decide_fd


# --------------------------------------------------------------------------- #
# Frozen contract + inheritance
# --------------------------------------------------------------------------- #
def test_phase24_config_inherits_phase23_core_and_knobs():
    p24 = Phase24Config.load()
    assert p24.freedata_version == "fd1"
    # core inherited verbatim from the confirmed Phase-23 contract (h=1)
    assert p24.core.horizons.primary == 1
    assert p24.core.symbols == ["MES", "MNQ"]
    assert p24.core.walk_forward.oos_start == "2022-01-01"
    # arms
    keys = {a.key: a.block for a in p24.arms}
    assert keys == {"x1": "marketdata", "s3": "gkg"}
    # coverage + gate thresholds
    assert p24.coverage.min_oos_fraction == 0.80
    assert p24.gates.g3_temporal_stability.min_folds_beating_baseline == 13


def test_baseline_and_augmented_feature_toggles():
    p24 = Phase24Config.load()
    base = p24.baseline_core()
    assert base.features.with_marketdata is False
    assert base.features.with_gkg is False

    x1 = next(a for a in p24.arms if a.key == "x1")
    s3 = next(a for a in p24.arms if a.key == "s3")
    aug_x1 = p24.augmented_core(x1)
    aug_s3 = p24.augmented_core(s3)
    # x1 turns on ONLY marketdata; s3 turns on ONLY gkg
    assert aug_x1.features.with_marketdata is True and aug_x1.features.with_gkg is False
    assert aug_s3.features.with_gkg is True and aug_s3.features.with_marketdata is False
    # the baseline is untouched (no shared mutation of the frozen core)
    assert base.features.with_marketdata is False


# --------------------------------------------------------------------------- #
# Coverage fraction
# --------------------------------------------------------------------------- #
def test_coverage_fraction_full_and_partial():
    oos = np.array(["2022-03-01", "2022-09-01", "2024-01-01"], dtype="datetime64[D]").astype(
        "datetime64[ns]"
    )
    full = _coverage_fraction(oos, np.datetime64("2018-01-01"), np.datetime64("2026-06-15"))
    assert full == 1.0
    # a lake reaching only 2022-08-31 covers just the first of the three OOS days
    partial = _coverage_fraction(oos, np.datetime64("2019-01-01"), np.datetime64("2022-08-31"))
    assert partial == 1 / 3
    none = _coverage_fraction(oos, np.datetime64("2010-01-01"), np.datetime64("2011-01-01"))
    assert none == 0.0


# --------------------------------------------------------------------------- #
# Per-block decision rule
# --------------------------------------------------------------------------- #
def _arm(status: str, candidate: bool) -> dict:
    return {"status": status, "candidate": candidate}


def _results(x1_mes, x1_mnq, s3_status="deferred_coverage"):
    s3 = _arm(s3_status, False)
    return {
        "MES": {"arms": {"x1": x1_mes, "s3": dict(s3)}},
        "MNQ": {"arms": {"x1": x1_mnq, "s3": dict(s3)}},
    }


def test_decide_both_pass_adds_value():
    p24 = Phase24Config.load()
    res = _results(_arm("run", True), _arm("run", True))
    dec = decide_fd(res, p24)
    assert dec["per_block"]["x1"]["verdict"] == "adds_incremental_value"
    assert dec["per_block"]["s3"]["verdict"] == "deferred_coverage"


def test_decide_one_symbol_is_fragile():
    p24 = Phase24Config.load()
    res = _results(_arm("run", True), _arm("run", False))
    dec = decide_fd(res, p24)
    assert dec["per_block"]["x1"]["verdict"] == "fragile_single_symbol"


def test_decide_neither_passes_is_no_value():
    p24 = Phase24Config.load()
    res = _results(_arm("run", False), _arm("run", False))
    dec = decide_fd(res, p24)
    assert dec["per_block"]["x1"]["verdict"] == "no_incremental_value"
