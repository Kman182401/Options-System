"""Leak-free micro training-matrix assembly for the 3-class signal model.

The micro labels (``data/micro_labels/``) already carry ``t0``, ``t1``, the 3-class
``label`` ∈ {-1, 0, +1}, the realized return to the barrier ``ret_t1``, and the
uniqueness / sample weights. The m1 order-flow features live ON the dollar bars
(``data/micro_bars/``). So assembly is a single backward as-of attach: for each
label, take the feature row of the bar at ``ts_event <= t0`` (in practice an exact
match, because a label's ``t0`` IS a bar's ``ts_event`` — the bar where the CUSUM
event fired). ``strategy="backward"`` guarantees a feature is never read from a
label's own future.

Only the causal m1 feature columns enter ``X`` (``microstructure.bars.feature_names``);
post-label / outcome columns (``label``, ``barrier_touched``, ``ret_t1``, ``t1``,
``sigma`` …) are NEVER features. NaN feature values are KEPT (LightGBM handles them
natively — e.g. ``ofi_top_lag1`` is null on the first bar of a session); rows with a
non-finite ``±inf`` feature, or a null/non-finite label/return/weight, are dropped.

No Databento, no IBKR, no network — reads only the local lakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from config.settings import Settings

from ..common.logging import get_logger
from ..data.store import DuckStore
from ..microstructure.bars import feature_names
from ..microstructure.config import MicrostructureConfig
from ..microstructure.ingest import read_micro_bars
from ..microstructure.label_config import MicroLabelConfig
from ..microstructure.labels import read_micro_labels
from ..microstructure.model_config import MicroModelConfig

logger = get_logger(__name__)

_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)

# Label columns required present + finite before a row is admitted (the features
# are allowed to be NaN; only ±inf features are rejected — see _finalize).
_REQUIRED_LABEL_COLS = ("label", "ret_t1", "sample_weight", "uniqueness_weight", "t1")


@dataclass(frozen=True)
class MicroTrainingMatrix:
    """Leak-free, ``t0``-sorted design matrix for the 3-class micro signal model."""

    symbol: str
    X: np.ndarray  # (n, p) m1 features as-of t0
    y: np.ndarray  # (n,) 3-class label in {-1, 0, +1}
    t0: np.ndarray  # (n,) event time
    t1: np.ndarray  # (n,) barrier-resolution time (drives purge/embargo)
    ret_t1: np.ndarray  # (n,) realized log-return to the barrier (gross signal-return proxy)
    sample_weight: np.ndarray  # (n,) persisted normalized sample weight (≈ mean 1.0)
    uniqueness_weight: np.ndarray  # (n,) average uniqueness (effective-N building block)
    feature_cols: list[str]
    microstructure_feature_version: str
    micro_label_version: str
    micro_model_version: str

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def effective_n(self) -> float:
        """Effective sample size = Σ average-uniqueness (≈ independent observations)."""
        return float(self.uniqueness_weight.sum())


def _cache_path(symbol: str, fv: str, lv: str, mmv: str, window_tag: str) -> Path:
    """Cache file under ``data/micro_models/cache`` — versioned, never collides with
    the daily model cache (``data/models/cache``)."""
    d = Settings().data_dir / "micro_models" / "cache"
    return d / f"micromatrix_{symbol}_{fv}_{lv}_{mmv}_{window_tag}.parquet"


def _attach_features(
    labels: pl.DataFrame, bars: pl.DataFrame, feature_cols: list[str]
) -> pl.DataFrame:
    """Backward as-of attach: each label's features come from the bar at ``ts_event <= t0``.

    Both frames are pre-sorted (labels by ``t0``, bars by ``ts_event``), so this is a
    cheap merge. ``strategy="backward"`` is the leak guarantee — a feature row is
    never taken from after the label's ``t0``.
    """
    feats = bars.select(["ts_event", *feature_cols]).sort("ts_event")
    left = labels.sort("t0")
    joined = left.join_asof(feats, left_on="t0", right_on="ts_event", strategy="backward")
    return joined.drop("ts_event") if "ts_event" in joined.columns else joined


def _finalize(joined: pl.DataFrame, feature_cols: list[str]) -> tuple[pl.DataFrame, dict[str, int]]:
    """Drop unusable rows; keep NaN features, reject ±inf features. Returns (frame, drops).

    A row is admitted iff: every required label column is present + finite, the as-of
    attach matched a bar (features not all-null), and no feature column is ``±inf``.
    NaN features are intentionally KEPT — LightGBM routes nulls down a default branch.
    """
    n0 = joined.height
    m = joined.sort("t0").drop_nulls(subset=list(_REQUIRED_LABEL_COLS))
    n_after_label = m.height
    # Drop a row only if a feature is non-finite-AND-not-NaN, i.e. ±inf. (is_finite()
    # is False for both NaN and ±inf; is_nan() is True only for NaN — so the reject
    # mask is "infinite": not finite and not nan.)
    inf_mask = pl.any_horizontal(
        (~pl.col(c).is_finite()) & (~pl.col(c).is_nan()) for c in feature_cols
    )
    m = m.filter(~inf_mask)
    n_after_inf = m.height
    # A label before the first bar would have all-null features (no as-of match) — drop.
    all_null = pl.all_horizontal(pl.col(c).is_null() for c in feature_cols)
    m = m.filter(~all_null)
    drops = {
        "dropped_null_label": n0 - n_after_label,
        "dropped_inf_feature": n_after_label - n_after_inf,
        "dropped_unmatched": n_after_inf - m.height,
    }
    return m, drops


def load_micro_matrix(
    symbol: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    store: DuckStore | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
    mmcfg: MicroModelConfig | None = None,
) -> MicroTrainingMatrix:
    """Assemble the leak-free 3-class micro training matrix for ``symbol``.

    Reads micro labels (``[start, end]`` on ``t0``) and attaches the m1 order-flow
    features as-of each label's ``t0`` from the dollar bars. ``start``/``end`` bound
    the labels selected (default: all available). An on-disk cache (keyed by
    micro-feature / label / model version + a window tag) short-circuits re-assembly;
    ``rebuild_cache`` forces a fresh build.
    """
    mmcfg = mmcfg or MicroModelConfig.load()
    mcfg = MicrostructureConfig.load()
    lcfg = MicroLabelConfig.load()
    fv = mcfg.microstructure_feature_version
    lv = lcfg.micro_label_version
    mmv = mmcfg.micro_model_version
    feature_cols = feature_names(mcfg)

    window_tag = (
        "full"
        if (start is None and end is None)
        else f"{(start or _WIDE_START):%Y%m%d}-{(end or _WIDE_END):%Y%m%d}"
    )
    cache = _cache_path(symbol, fv, lv, mmv, window_tag)
    if use_cache and not rebuild_cache and cache.exists():
        frame = pl.read_parquet(cache)
        return _matrix_from_frame(frame, symbol, feature_cols, fv, lv, mmv)

    own = store is None
    store = store or DuckStore()
    try:
        labels = read_micro_labels(symbol, start or _WIDE_START, end or _WIDE_END, store=store)
        if labels.is_empty() or labels.height == 0:
            raise ValueError(
                f"no micro labels for {symbol} in window — build labels first "
                "(python -m options_system.microstructure.labels)"
            )
        bars = read_micro_bars(symbol, _WIDE_START, _WIDE_END, store=store)
        if bars.is_empty() or bars.width == 0:
            raise ValueError(f"no micro bars for {symbol}; ingest microstructure bars first")
        joined = _attach_features(labels, bars, feature_cols)
        m, drops = _finalize(joined, feature_cols)
        if m.is_empty():
            raise ValueError(f"all rows for {symbol} dropped during micro-matrix assembly")
        if any(drops.values()):
            logger.info(f"[{symbol}] micro-matrix admitted {m.height} rows (drops={drops})")
        keep = [*feature_cols, "label", "ret_t1", "sample_weight", "uniqueness_weight", "t0", "t1"]
        frame = m.select(keep)
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(cache, compression="zstd")
        return _matrix_from_frame(frame, symbol, feature_cols, fv, lv, mmv)
    finally:
        if own:
            store.close()


def _matrix_from_frame(
    frame: pl.DataFrame, symbol: str, feature_cols: list[str], fv: str, lv: str, mmv: str
) -> MicroTrainingMatrix:
    """Build the typed matrix arrays from a finalized (clean) frame."""
    return MicroTrainingMatrix(
        symbol=symbol,
        X=frame.select(feature_cols).to_numpy(),
        y=frame["label"].to_numpy().astype(int),
        t0=frame["t0"].to_numpy(),
        t1=frame["t1"].to_numpy(),
        ret_t1=frame["ret_t1"].to_numpy().astype(float),
        sample_weight=frame["sample_weight"].to_numpy().astype(float),
        uniqueness_weight=frame["uniqueness_weight"].to_numpy().astype(float),
        feature_cols=list(feature_cols),
        microstructure_feature_version=fv,
        micro_label_version=lv,
        micro_model_version=mmv,
    )
