"""Microstructure / order-flow (OFI) feature layer.

A deliberately separate, additive experiment from the price ``feature_version=v1``
layer: it ingests CME Level-2 (MBP-10) book + trade events for the deep e-mini
parents (ES, NQ), reduces them to information-driven **dollar bars**, and computes
a causal, leakage-tested order-flow feature family stamped
``microstructure_feature_version="m1"``.

Modules:

* :mod:`options_system.microstructure.config` — typed, declarative configuration.
* :mod:`options_system.microstructure.ofi` — pure order-flow math (OFI, imbalance,
  micro-price) with no I/O, fully unit-testable.
* :mod:`options_system.microstructure.bars` — streaming dollar-bar reducer +
  feature assembly. O(1) memory; deterministic; roll/session-boundary aware.
* :mod:`options_system.microstructure.ingest` — Databento MBP-10 streaming
  ingestion, storage to the Parquet lake, and light MLflow stats. Cost-guarded.

See ``docs/MICROSTRUCTURE.md`` for the plain-English tour.
"""
