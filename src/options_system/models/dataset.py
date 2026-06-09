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

Two optional, additive feature layers can be appended at each label ``t0``, each
behind its own flag and each purely additive (they add columns, never drop rows —
the null/non-finite **row** gate stays on the price features only):

* ``with_macro`` (Phase 6) — leak-safe macro/economic-event features.
* ``with_ta`` (Phase 10, **opt-in, default off**) — the isolated v2 technical-
  analysis lake (``data/ta_features/``). TA is a price-derived transformation, not
  new information; this flag wires it in as an explicit controlled experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from ..features.macro_features import compute_macro_features, macro_feature_names
from ..labeling.build import read_labels
from ..labeling.config import LabelConfig
from ..macro.config import MacroConfig
from ..macro.ingest import partition_glob as macro_partition_glob
from ..macro.ingest import read_macro_events
from ..ta.build import partition_glob as ta_partition_glob
from ..ta.compute import ta_feature_names
from ..ta.config import TaConfig

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
    X: np.ndarray  # (n, p) features as-of t0 (price, then macro, then TA columns)
    y: np.ndarray  # (n,) triple-barrier labels in {-1, 0, +1}
    y_dir: np.ndarray  # (n,) directional target in {-1, +1} (up/down)
    t0: np.ndarray  # (n,) event time
    t1: np.ndarray  # (n,) barrier-resolution time (drives purge/embargo)
    ret: np.ndarray  # (n,) realized log-return to t1 (return proxy)
    weight: np.ndarray  # (n,) persisted sample weight (≈ mean 1.0)
    uniqueness: np.ndarray  # (n,) average uniqueness (effective-N building block)
    feature_cols: list[str]  # ALL model inputs in X order: price_cols + macro_cols + ta_cols
    feature_version: str
    label_version: str
    timeout_handling: str
    # the macro-feature subset of feature_cols ([] when price-only) + its version stamp
    macro_cols: list[str] = field(default_factory=list)
    macro_feature_version: str | None = None
    # the TA-feature subset of feature_cols ([] when TA off) + its version stamp
    ta_cols: list[str] = field(default_factory=list)
    ta_feature_version: str | None = None

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def with_macro(self) -> bool:
        return bool(self.macro_cols)

    @property
    def with_ta(self) -> bool:
        return bool(self.ta_cols)


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


def _attach_ta(
    store: DuckStore, frame: pl.DataFrame, symbol: str, ta_cols: list[str]
) -> pl.DataFrame:
    """Append TA features to an already-finalized ``frame`` (which carries ``t0``).

    Same pushed-down, backward as-of attach as :func:`_asof_attach`, but joined onto
    the finalized matrix (after the price-only row gate) so TA never changes the row
    count. TA columns are NaN during their warmup, which LightGBM handles natively.

    Defensive sanitation: any ``±inf`` a degenerate TA window could emit is converted
    to ``null`` here, so an infinity can never reach the estimator (LightGBM rejects
    ``inf``). The compute engine already guards its denominators, so this is belt-and-
    braces — and because TA columns are deliberately kept out of the row gate, a null
    is the correct, row-preserving representation of an undefined value.
    """
    glob_str = ta_partition_glob(symbol)
    max_t0 = frame["t0"].max()
    feat_select = ", ".join(f'"{c}"' for c in ta_cols)
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
    out = frame.sort("t0").join_asof(feats, left_on="t0", right_on="ts_event", strategy="backward")
    out = out.drop("ts_event") if "ts_event" in out.columns else out
    return out.with_columns(
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in ta_cols
    )


def _finalize(joined: pl.DataFrame, feat_cols: list[str]) -> pl.DataFrame:
    """Drop null/non-finite rows exactly as ``validation.evaluate.load_matrix`` does."""
    required = [*_REQUIRED_LABEL_COLS, *feat_cols]
    m = joined.sort("t0").drop_nulls(subset=required)
    # Degenerate windows (e.g. a z-score over zero rolling std) can emit ±inf/NaN,
    # which estimators reject — drop them. These are warmup/degenerate edges, not signal.
    m = m.filter(pl.all_horizontal(pl.col(c).is_finite() for c in feat_cols))
    return m


def _cache_path(symbol: str, fv: str, lv: str, handling: str, macro_tag: str, ta_tag: str) -> Path:
    d = Settings().data_dir / "models" / "cache"
    return d / f"matrix_{symbol}_{fv}_{lv}_{handling}_{macro_tag}_{ta_tag}.parquet"


def _macro_available() -> bool:
    """True if the macro_events table has been ingested (a cheap glob, no store)."""
    return bool(_glob(macro_partition_glob()))


def _ta_available(symbol: str) -> bool:
    """True if the v2 TA feature lake exists for ``symbol`` (a cheap glob, no store)."""
    return bool(_glob(ta_partition_glob(symbol)))


def load_training_matrix(
    symbol: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    timeout_handling: str = "sign_return",
    store: DuckStore | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
    with_macro: bool = True,
    with_ta: bool = False,
) -> TrainingMatrix:
    """Assemble the leak-free directional training matrix for ``symbol`` — fast.

    Reads labels (~11k rows) and attaches price features with a pushed-down DuckDB
    ASOF join, then reduces the 3-class label to a directional ``y_dir`` per
    ``timeout_handling``. The price arrays match Phase 4's ``load_matrix`` exactly.

    When ``with_macro`` is set and the ``macro_events`` table has been ingested
    (Phase 6), leak-safe macro/economic-event features are appended at each label
    ``t0``. Macro columns are NaN where undefined, which LightGBM handles natively;
    the null/non-finite **row** gate stays on the price features only, so the row
    count is identical to the price-only matrix. If ``with_macro`` is requested but
    no events exist, the matrix is built price-only with a warning.

    When ``with_ta`` is set (**opt-in, default off**, Phase 10), the isolated v2
    technical-analysis lake (``data/ta_features/``) is appended after the price and
    macro columns, with the same backward as-of rule (``ta.ts_event <= t0``) and
    latest-ingest-wins semantics. Unlike macro, TA is **never** silently skipped: if
    ``with_ta`` is requested but the TA lake is missing for ``symbol`` a clear
    :class:`ValueError` is raised telling the operator to build it first. TA is a
    price-derived transformation, not new information; it is wired in as an explicit
    controlled experiment and is not part of the default input set.

    ``with_macro`` and ``with_ta`` are independent. An on-disk cache (keyed by
    ``feature_version`` + ``label_version`` + ``timeout_handling`` + macro tag + TA
    tag) short-circuits full-history loads; it is bypassed for bounded windows. The
    TA tag (``nota`` vs ``ta-<version>``) keeps TA and no-TA matrices in separate
    cache files, so they can never collide.
    """
    fcfg = FeatureConfig.load()
    lcfg = LabelConfig.load()
    fv, lv = fcfg.feature_version, lcfg.label_version
    price_cols = list(feature_names(fcfg))

    use_macro = with_macro and _macro_available()
    if with_macro and not use_macro:
        logger.warning(
            f"[{symbol}] with_macro requested but macro_events table is empty — "
            "building price-only matrix (run python -m options_system.macro.ingest)"
        )
    mcfg = MacroConfig.load() if use_macro else None
    macro_cols = macro_feature_names(mcfg) if mcfg is not None else []
    macro_fv = mcfg.features.macro_feature_version if mcfg is not None else None
    macro_tag = f"macro-{macro_fv}" if use_macro else "nomacro"

    # TA is opt-in and never silently skipped: a missing lake is a hard error.
    if with_ta and not _ta_available(symbol):
        raise ValueError(
            f"[{symbol}] with_ta=True but no TA feature lake at "
            f"data/ta_features/symbol={symbol} — build it first:\n"
            "  uv run python -m options_system.ta.build --symbols MES MNQ"
        )
    tacfg = TaConfig.load() if with_ta else None
    ta_cols = ta_feature_names(tacfg) if tacfg is not None else []
    ta_fv = tacfg.ta_feature_version if tacfg is not None else None
    ta_tag = f"ta-{ta_fv}" if with_ta else "nota"

    full_history = start is None and end is None
    cache = _cache_path(symbol, fv, lv, timeout_handling, macro_tag, ta_tag)

    if full_history and use_cache and not rebuild_cache and cache.exists():
        frame = pl.read_parquet(cache)
        return _matrix_from_frame(
            frame,
            symbol,
            price_cols,
            macro_cols,
            ta_cols,
            fv,
            lv,
            timeout_handling,
            macro_fv,
            ta_fv,
        )

    own = store is None
    store = store or DuckStore()
    try:
        labels = read_labels(symbol, start or _WIDE_START, end or _WIDE_END, store=store)
        if labels.is_empty():
            raise ValueError(
                f"no labels for {symbol} in window — build labels first "
                "(python -m options_system.labeling.build)"
            )
        joined = _asof_attach(store, labels, symbol, price_cols)
        m = _finalize(joined, price_cols)  # row gate on PRICE features only
        if m.is_empty():
            raise ValueError(
                f"all rows for {symbol} dropped (null/non-finite) after feature attach"
            )
        if mcfg is not None:  # == use_macro; this form lets the type-checker narrow mcfg
            # compute_macro_features preserves the input t0 order → align by hstack.
            events = read_macro_events(store=store)
            macro_df = compute_macro_features(m["t0"], events, mcfg)
            m = m.hstack(macro_df.drop("t0"))
        if with_ta:  # append TA columns after price + macro (additive, row-preserving)
            m = _attach_ta(store, m, symbol, ta_cols)
        keep_cols = [
            *price_cols,
            *macro_cols,
            *ta_cols,
            "label",
            "ret",
            "weight",
            "avg_uniqueness",
            "t0",
            "t1",
        ]
        frame = m.select(keep_cols)
        if full_history and use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(cache, compression="zstd")
        return _matrix_from_frame(
            frame,
            symbol,
            price_cols,
            macro_cols,
            ta_cols,
            fv,
            lv,
            timeout_handling,
            macro_fv,
            ta_fv,
        )
    finally:
        if own:
            store.close()


def _matrix_from_frame(
    frame: pl.DataFrame,
    symbol: str,
    price_cols: list[str],
    macro_cols: list[str],
    ta_cols: list[str],
    fv: str,
    lv: str,
    timeout_handling: str,
    macro_fv: str | None,
    ta_fv: str | None,
) -> TrainingMatrix:
    """Build arrays + the directional target from a finalized (clean) frame.

    ``X`` columns are ``price_cols`` then ``macro_cols`` then ``ta_cols``;
    ``feature_cols`` records that exact order so SHAP / evaluation name the columns
    correctly. New layers are always appended at the END, never interleaved.
    """
    feat_cols = [*price_cols, *macro_cols, *ta_cols]
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
        macro_cols=macro_cols,
        macro_feature_version=macro_fv,
        ta_cols=ta_cols,
        ta_feature_version=ta_fv,
    )
