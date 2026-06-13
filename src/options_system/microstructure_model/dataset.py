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

Phase 19 adds an **opt-in** sentiment block (``with_sentiment``, default off — the
exact Phase-10 ``with_ta`` pattern). When on, the ``s2`` ``sent_*`` aggregate
features (``sentiment.join.attach_to_micro_labels``) are attached on ``t0`` — purely
point-in-time, every label row preserved, sentiment nulls KEPT (no imputation; "no
events" is information). The baseline (off) path is byte-identical to the Phase-14
matrix; the only delta between the two arms is the sentiment block. The row gate
(``_finalize``) stays on the m1 features ONLY, so both arms admit the identical rows.

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
from ..sentiment.config import SentimentConfig
from ..sentiment.features import read_sentiment_scores, sentiment_feature_names
from ..sentiment.join import attach_to_micro_labels

logger = get_logger(__name__)

_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)

# Label columns required present + finite before a row is admitted (the features
# are allowed to be NaN; only ±inf features are rejected — see _finalize).
_REQUIRED_LABEL_COLS = ("label", "ret_t1", "sample_weight", "uniqueness_weight", "t1")

# Phase 19 treatment arm = the mm1 OFI baseline PLUS the s2 sentiment block.
_SENTIMENT_MODEL_VERSION = "mm2"


@dataclass(frozen=True)
class MicroTrainingMatrix:
    """Leak-free, ``t0``-sorted design matrix for the 3-class micro signal model."""

    symbol: str
    X: np.ndarray  # (n, p) m1 features as-of t0 (+ s2 sentiment when with_sentiment)
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
    with_sentiment: bool = False

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def effective_n(self) -> float:
        """Effective sample size = Σ average-uniqueness (≈ independent observations)."""
        return float(self.uniqueness_weight.sum())


def _cache_path(symbol: str, fv: str, lv: str, mmv: str, sent_tag: str, window_tag: str) -> Path:
    """Cache file under ``data/micro_models/cache`` — versioned, never collides with
    the daily model cache (``data/models/cache``). The ``sent_tag`` (``base`` vs ``s2``)
    keeps the baseline and sentiment matrices in separate cache files so they can never
    collide even if the version stamp were mis-passed."""
    d = Settings().data_dir / "micro_models" / "cache"
    return d / f"micromatrix_{symbol}_{fv}_{lv}_{mmv}_{sent_tag}_{window_tag}.parquet"


def _attach_features(
    labels: pl.DataFrame, bars: pl.DataFrame, feature_cols: list[str]
) -> pl.DataFrame:
    """Backward as-of attach: each label's features come from the bar at ``ts_event <= t0``.

    Both frames are pre-sorted (labels by ``t0``, bars by ``ts_event``), so this is a
    cheap merge. ``strategy="backward"`` is the leak guarantee — a feature row is
    never taken from after the label's ``t0``. Any non-feature columns already on the
    label frame (e.g. attached ``sent_*`` columns) ride through as passengers.
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
    The row gate is computed on ``feature_cols`` (the m1 features) ONLY; any attached
    ``sent_*`` columns ride through untouched, so a sentiment null never drops a row.
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


def _attach_sentiment(
    labels: pl.DataFrame, scored_events: pl.DataFrame, scfg: SentimentConfig
) -> pl.DataFrame:
    """Attach the s2 sentiment block onto micro labels on ``t0`` (point-in-time).

    Thin wrapper over the Phase-17 join (``attach_to_micro_labels``): keys ONLY on
    ``t0``, never reads ``t1`` / ``ret_t1`` / the label outcome, and preserves every
    label row. Returns the labels with the ``sent_*`` columns appended; nulls (no prior
    events) are KEPT (LightGBM-native NaN), never imputed.
    """
    attached, _coverage = attach_to_micro_labels(labels, scored_events, scfg)
    return attached


def _require_scored_sentiment(scored_events: pl.DataFrame, symbol: str) -> None:
    """Fail closed if the treatment arm has no scored sentiment to attach.

    The Phase-19 contract forbids silently producing empty/zero sentiment and calling
    it a result: an absent / empty scored lake is an explicit, instructive error.
    """
    if scored_events.height == 0:
        raise ValueError(
            f"[{symbol}] with_sentiment=True but the scored sentiment lake "
            "(data/sentiment_scores/) is empty — score it first:\n"
            "  env -u CUDA_VISIBLE_DEVICES uv run python "
            "-m options_system.sentiment.score_backfill"
        )


def load_micro_matrix(
    symbol: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    store: DuckStore | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
    mmcfg: MicroModelConfig | None = None,
    with_sentiment: bool = False,
    scfg: SentimentConfig | None = None,
    scored_events: pl.DataFrame | None = None,
    version_stamp: str | None = None,
) -> MicroTrainingMatrix:
    """Assemble the leak-free 3-class micro training matrix for ``symbol``.

    Reads micro labels (``[start, end]`` on ``t0``) and attaches the m1 order-flow
    features as-of each label's ``t0`` from the dollar bars. ``start``/``end`` bound
    the labels selected (default: all available). An on-disk cache (keyed by
    micro-feature / label / model version + a sentiment tag + a window tag)
    short-circuits re-assembly; ``rebuild_cache`` forces a fresh build.

    When ``with_sentiment`` is set (**opt-in, default off**, Phase 19) the ``s2``
    ``sent_*`` aggregate features are attached on ``t0`` and appended to ``X`` after
    the m1 features; the model-version stamp becomes ``mm2`` (overridable via
    ``version_stamp``). The scored sentiment is read from the lake unless an explicit
    ``scored_events`` frame is injected (a test seam); an empty scored lake fails
    closed. ``with_sentiment=False`` is byte-identical to the Phase-14 matrix.
    """
    mmcfg = mmcfg or MicroModelConfig.load()
    mcfg = MicrostructureConfig.load()
    lcfg = MicroLabelConfig.load()
    fv = mcfg.microstructure_feature_version
    lv = lcfg.micro_label_version
    base_feature_cols = feature_names(mcfg)

    if with_sentiment:
        scfg = scfg or SentimentConfig.load()
        sent_cols = sentiment_feature_names(scfg)
        mmv = version_stamp or _SENTIMENT_MODEL_VERSION
    else:
        scfg = None
        sent_cols = []
        mmv = version_stamp or mmcfg.micro_model_version
    model_feature_cols = [*base_feature_cols, *sent_cols]
    sent_tag = "s2" if with_sentiment else "base"

    window_tag = (
        "full"
        if (start is None and end is None)
        else f"{(start or _WIDE_START):%Y%m%d}-{(end or _WIDE_END):%Y%m%d}"
    )
    cache = _cache_path(symbol, fv, lv, mmv, sent_tag, window_tag)
    if use_cache and not rebuild_cache and cache.exists():
        frame = pl.read_parquet(cache)
        return _matrix_from_frame(
            frame, symbol, model_feature_cols, fv, lv, mmv, with_sentiment=with_sentiment
        )

    own = store is None
    store = store or DuckStore()
    try:
        labels = read_micro_labels(symbol, start or _WIDE_START, end or _WIDE_END, store=store)
        if labels.is_empty() or labels.height == 0:
            raise ValueError(
                f"no micro labels for {symbol} in window — build labels first "
                "(python -m options_system.microstructure.labels)"
            )
        if with_sentiment:
            assert scfg is not None  # set above when with_sentiment
            scored = scored_events if scored_events is not None else read_sentiment_scores()
            _require_scored_sentiment(scored, symbol)
            labels = _attach_sentiment(labels, scored, scfg)
        bars = read_micro_bars(symbol, _WIDE_START, _WIDE_END, store=store)
        if bars.is_empty() or bars.width == 0:
            raise ValueError(f"no micro bars for {symbol}; ingest microstructure bars first")
        joined = _attach_features(labels, bars, base_feature_cols)
        m, drops = _finalize(joined, base_feature_cols)
        if m.is_empty():
            raise ValueError(f"all rows for {symbol} dropped during micro-matrix assembly")
        if any(drops.values()):
            logger.info(f"[{symbol}] micro-matrix admitted {m.height} rows (drops={drops})")
        keep = [
            *model_feature_cols,
            "label",
            "ret_t1",
            "sample_weight",
            "uniqueness_weight",
            "t0",
            "t1",
        ]
        frame = m.select(keep)
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(cache, compression="zstd")
        return _matrix_from_frame(
            frame, symbol, model_feature_cols, fv, lv, mmv, with_sentiment=with_sentiment
        )
    finally:
        if own:
            store.close()


def _matrix_from_frame(
    frame: pl.DataFrame,
    symbol: str,
    feature_cols: list[str],
    fv: str,
    lv: str,
    mmv: str,
    *,
    with_sentiment: bool = False,
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
        with_sentiment=with_sentiment,
    )
