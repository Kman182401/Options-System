"""Fast, leak-free training-matrix assembly for the signal model.

Phase 4's :func:`options_system.validation.evaluate.load_matrix` materialises the
**entire** feature lake (~2.46M rows/symbol) into Python before an as-of join —
minutes per call. We only ever need features at the ~11k label ``t0`` timestamps.

This module pushes the as-of attach **down into DuckDB**: the small (~11k-row)
label table is the left side, the feature parquet is scanned columnar, and the two
are joined with DuckDB's ``ASOF`` operator (``feature.ts_event <= t0``). Only the
~11k joined rows ever cross into Python, so assembly is **seconds, not minutes**.

The leak-free guarantee is identical to Phase 4 (a label's features never come
from its own future), and the produced ``(X, y, t0, t1, ret, weight,
uniqueness)`` arrays match Phase 4's ``load_matrix`` **exactly** (same
post-processing: drop null/non-finite rows, ``t0``-sorted). On top of that the
labeling layer's 3-class label ``{-1, 0, +1}`` is reduced to a **directional**
target ``y_dir ∈ {-1, +1}`` (see :func:`derive_direction`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from glob import glob as _glob
from pathlib import Path

import numpy as np
import polars as pl

from config.settings import Settings

from ..common.logging import get_logger
from ..data.store import DuckStore
from ..features.build import partition_glob as feature_partition_glob
from ..features.compute import feature_names
from ..features.config import FeatureConfig
from ..labeling.build import read_labels
from ..labeling.config import LabelConfig

logger = get_logger(__name__)

_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)

# Columns the matrix needs present + finite before a row is admitted (matches
# validation.evaluate.load_matrix exactly so the two assemble identical matrices).
_REQUIRED_LABEL_COLS = ("label", "ret", "weight", "avg_uniqueness", "t1")


@dataclass(frozen=True)
class TrainingMatrix:
    """Leak-free, ``t0``-sorted design matrix for the directional signal model.

    Superset of ``validation.evaluate.Matrix``: it additionally carries the
    **directional** target ``y_dir`` and the version stamps that key the cache.
    """

    symbol: str
    X: np.ndarray  # (n, p) features as-of t0
    y: np.ndarray  # (n,) triple-barrier labels in {-1, 0, +1}
    y_dir: np.ndarray  # (n,) directional target in {-1, +1} (up/down)
    t0: np.ndarray  # (n,) event time
    t1: np.ndarray  # (n,) barrier-resolution time (drives purge/embargo)
    ret: np.ndarray  # (n,) realized log-return to t1 (return proxy)
    weight: np.ndarray  # (n,) persisted sample weight (≈ mean 1.0)
    uniqueness: np.ndarray  # (n,) average uniqueness (effective-N building block)
    feature_cols: list[str]
    feature_version: str
    label_version: str
    timeout_handling: str

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


# --------------------------------------------------------------------------- #
# Directional target
# --------------------------------------------------------------------------- #
def derive_direction(
    label: np.ndarray, ret: np.ndarray, handling: str
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the 3-class label ``{-1, 0, +1}`` to a directional target ``{-1, +1}``.

    Returns ``(y_dir, keep_mask)``. The tiny vertical-timeout class (``label == 0``)
    is handled per ``handling``:

    * ``"sign_return"`` — fold timeouts into direction by the **sign of the realised
      return** at ``t1`` (no rows discarded). An exactly-zero timeout return
      (measure-zero) maps to ``-1``.
    * ``"drop"`` — exclude timeout rows entirely (``keep_mask`` is ``False`` there).

    For the directional (``±1``) labels the sign of ``label`` is used directly.
    """
    label = np.asarray(label).astype(int)
    ret = np.asarray(ret).astype(float)
    if handling == "drop":
        keep = label != 0
        y_dir = np.where(label > 0, 1, -1).astype(int)
        return y_dir, keep
    if handling == "sign_return":
        # ±1 labels keep their sign; timeouts take the sign of their realised return.
        y_dir = np.where(label > 0, 1, np.where(label < 0, -1, np.where(ret > 0.0, 1, -1)))
        return y_dir.astype(int), np.ones(label.shape[0], dtype=bool)
    raise ValueError(f"unknown timeout_handling={handling!r}; use 'sign_return' or 'drop'")


# --------------------------------------------------------------------------- #
# Targeted (fast) as-of feature attach
# --------------------------------------------------------------------------- #
def _asof_attach(
    store: DuckStore, labels: pl.DataFrame, symbol: str, feat_cols: list[str]
) -> pl.DataFrame:
    """Attach the as-of feature row at each label ``t0`` (latest feature ``ts_event <= t0``).

    Two cheap steps instead of one expensive DuckDB ASOF over the whole lake:

    1. DuckDB reads **only** ``ts_event`` + ``feat_cols`` from the feature parquet
       (column projection is pushed into the scan), deduped latest-ingest-wins and
       sorted — ~2.5M rows materialise in a couple of seconds.
    2. polars ``join_asof`` (a merge over the two pre-sorted frames) attaches the
       backward (``<= t0``) match in a fraction of a second.

    A single DuckDB ``ASOF JOIN`` instead has to sort/materialise the 2.5M-row right
    side inside the join and takes *minutes*; this split is the fast, exact
    equivalent. Leak-free: ``strategy="backward"`` only ever matches a feature whose
    ``ts_event <= t0``.
    """
    glob_str = feature_partition_glob(symbol)
    if not _glob(glob_str):
        raise ValueError(f"no feature parquet for {symbol}; build features first")
    max_t0 = labels["t0"].max()
    feat_select = ", ".join(f'"{c}"' for c in feat_cols)
    feats = store.con.execute(
        f"""
        SELECT ts_event, {feat_select}
        FROM read_parquet('{glob_str}', hive_partitioning=false)
        WHERE ts_event <= ?
        QUALIFY row_number() OVER (PARTITION BY ts_event ORDER BY ts_ingest DESC) = 1
        ORDER BY ts_event
        """,
        [max_t0],
    ).pl()
    left = labels.sort("t0")
    joined = left.join_asof(feats, left_on="t0", right_on="ts_event", strategy="backward")
    # join_asof keeps the right key column; drop it so the frame mirrors load_matrix.
    return joined.drop("ts_event") if "ts_event" in joined.columns else joined


def _finalize(joined: pl.DataFrame, feat_cols: list[str]) -> pl.DataFrame:
    """Drop null/non-finite rows exactly as ``validation.evaluate.load_matrix`` does."""
    required = [*_REQUIRED_LABEL_COLS, *feat_cols]
    m = joined.sort("t0").drop_nulls(subset=required)
    # Degenerate windows (e.g. a z-score over zero rolling std) can emit ±inf/NaN,
    # which estimators reject — drop them. These are warmup/degenerate edges, not signal.
    m = m.filter(pl.all_horizontal(pl.col(c).is_finite() for c in feat_cols))
    return m


def _cache_path(symbol: str, fv: str, lv: str, handling: str) -> Path:
    d = Settings().data_dir / "models" / "cache"
    return d / f"matrix_{symbol}_{fv}_{lv}_{handling}.parquet"


def load_training_matrix(
    symbol: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    timeout_handling: str = "sign_return",
    store: DuckStore | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> TrainingMatrix:
    """Assemble the leak-free directional training matrix for ``symbol`` — fast.

    Reads labels (~11k rows) and attaches features with a pushed-down DuckDB ASOF
    join, then reduces the 3-class label to a directional ``y_dir`` per
    ``timeout_handling``. The arrays match Phase 4's ``load_matrix`` exactly.

    An optional on-disk cache (keyed by ``feature_version`` + ``label_version`` +
    ``timeout_handling``) short-circuits full-history loads; it is bypassed for
    bounded ``[start, end]`` windows.
    """
    fcfg = FeatureConfig.load()
    lcfg = LabelConfig.load()
    fv, lv = fcfg.feature_version, lcfg.label_version
    feat_cols = list(feature_names(fcfg))
    full_history = start is None and end is None
    cache = _cache_path(symbol, fv, lv, timeout_handling)

    if full_history and use_cache and not rebuild_cache and cache.exists():
        frame = pl.read_parquet(cache)
        return _matrix_from_frame(frame, symbol, feat_cols, fv, lv, timeout_handling)

    own = store is None
    store = store or DuckStore()
    try:
        labels = read_labels(symbol, start or _WIDE_START, end or _WIDE_END, store=store)
        if labels.is_empty():
            raise ValueError(
                f"no labels for {symbol} in window — build labels first "
                "(python -m options_system.labeling.build)"
            )
        joined = _asof_attach(store, labels, symbol, feat_cols)
        m = _finalize(joined, feat_cols)
        if m.is_empty():
            raise ValueError(
                f"all rows for {symbol} dropped (null/non-finite) after feature attach"
            )
        frame = m.select([*feat_cols, "label", "ret", "weight", "avg_uniqueness", "t0", "t1"])
        if full_history and use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(cache, compression="zstd")
        return _matrix_from_frame(frame, symbol, feat_cols, fv, lv, timeout_handling)
    finally:
        if own:
            store.close()


def _matrix_from_frame(
    frame: pl.DataFrame,
    symbol: str,
    feat_cols: list[str],
    fv: str,
    lv: str,
    timeout_handling: str,
) -> TrainingMatrix:
    """Build arrays + the directional target from a finalized (clean) frame."""
    y = frame["label"].to_numpy().astype(int)
    ret = frame["ret"].to_numpy().astype(float)
    y_dir, keep = derive_direction(y, ret, timeout_handling)
    X = frame.select(feat_cols).to_numpy()
    t0 = frame["t0"].to_numpy()
    t1 = frame["t1"].to_numpy()
    weight = frame["weight"].to_numpy().astype(float)
    uniqueness = frame["avg_uniqueness"].to_numpy().astype(float)
    if not keep.all():  # 'drop' handling: excise timeout rows from every array
        X, y, y_dir = X[keep], y[keep], y_dir[keep]
        t0, t1, ret = t0[keep], t1[keep], ret[keep]
        weight, uniqueness = weight[keep], uniqueness[keep]
    return TrainingMatrix(
        symbol=symbol,
        X=X,
        y=y,
        y_dir=y_dir,
        t0=t0,
        t1=t1,
        ret=ret,
        weight=weight,
        uniqueness=uniqueness,
        feature_cols=feat_cols,
        feature_version=fv,
        label_version=lv,
        timeout_handling=timeout_handling,
    )
