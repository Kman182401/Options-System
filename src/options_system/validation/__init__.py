"""Leakage-safe validation framework — the system's truth-detector.

This package is model-agnostic evaluation infrastructure. It answers one
question honestly: *would a model's apparent skill survive out-of-sample, or is
it an artifact of leakage and selection bias?* Nothing here trains the real
signal model, runs an economic backtest, or trades — it only judges estimators.

Modules:

* :mod:`~options_system.validation.config` — typed, versioned configuration.
* :mod:`~options_system.validation._purge` — the core purge + embargo primitive
  (single source of truth shared by every splitter).
* :mod:`~options_system.validation.purged_kfold` — purged + embargoed K-fold.
* :mod:`~options_system.validation.cpcv` — Combinatorial Purged CV with
  out-of-sample path reconstruction (AFML ch. 12).
* :mod:`~options_system.validation.walk_forward` — anchored/rolling sequential
  validation.
* :mod:`~options_system.validation.stats` — overfitting statistics: PBO (CSCV),
  Probabilistic / Deflated Sharpe Ratio, minimum track-record length.
* :mod:`~options_system.validation.evaluate` — the evaluation harness that ties
  an sklearn-compatible estimator to the leak-free (features@t0, y, t1, weight)
  matrix.

The leakage contract: every splitter PURGES training samples whose ``[t0, t1]``
label window overlaps a test fold and applies a forward EMBARGO, using the
``t1`` resolution times written by the labeling layer. Scoring honours the
Phase-3 uniqueness sample weights and reports effective sample size. See
``docs/VALIDATION.md``.
"""

from __future__ import annotations
