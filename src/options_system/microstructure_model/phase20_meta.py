"""Phase-20 meta-labeling edge verdict — orchestrator.

For each symbol (ES, NQ separately) this runs the pre-registered meta-labeling A/B on
the **full Phase-14 window**::

    uv run python -m options_system.microstructure_model.phase20_meta --symbols ES NQ

A fixed, deterministic **primary** picks the side of every bet (``sign(ofi_top)`` as-of
``t0`` — causal, NOT fitted, so there is no primary-model leakage and no nested
out-of-fold construction). A fitted **binary meta-model** over the m1 order-flow block,
the ``s2`` sentiment block, and the primary's ``|ofi_top|`` magnitude predicts
``P(meta_label = 1)`` — the probability the primary called the correct barrier side —
and we **act when ``P > tau`` (tau = 0.5, fixed)**, else stay flat.

Two arms, per symbol:

- **B0 — always-act reference:** take the primary side on every event (no gate). The
  primary's unconditional performance; expected to fail (the Phase-14 OFI null).
- **M — meta-gated (PRIMARY VERDICT arm):** take the side only when the meta-model says
  act. Judged against the FIVE inherited mm1 gates **plus** the disclosed binary
  **meta-skill** gate (the substitute for the 3-class-only macro-F1 gate).

The meta-model is the SOLE fitted component; it runs entirely inside the existing
purged+embargoed K-fold / CPCV / PBO / PSR-DSR machinery (purge + embargo on each
label's ``t1``). The inherited LightGBM params, the 8-config search grid, the selection
metric, the seed, and the five gross gates are read UNCHANGED from
``config/micro_model.yaml`` (mm1); tau, the meta-skill floor, the window, the primary
feature and the symbols come from ``config/phase20.yaml``. The frozen contract is
``docs/PHASE20_PREREGISTRATION.md``; this run must not deviate from it.

It produces a **signal edge verdict only** (gross signal-return proxy, ``side · ret_t1``,
NO commissions, NO slippage). It is NOT a strategy and NOT an economic backtest; it
authorizes NO live trading. Per symbol, never pooled. Reads only the local lakes — no
Databento, no IBKR, no network, no spend.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import Settings

from ..common.logging import get_logger
from ..microstructure.model_config import MicroModelConfig, VerdictCfg
from ..sentiment.config import SentimentConfig
from ..validation import stats as st
from ..validation._purge import embargo_bars_from_pct
from ..validation.config import ValidationConfig
from ..validation.cpcv import CombinatorialPurgedCV
from ..validation.purged_kfold import PurgedKFold
from .dataset import MicroTrainingMatrix, load_micro_matrix
from .evaluate import VERDICT_EDGE, VERDICT_NONE, _distribution_summary, action_rate
from .lgbm import effective_sample_weight, fold_local_class_weights
from .meta_labeling import gated_position, meta_label, meta_set_mask, primary_side
from .meta_lgbm import META_CLASSES, build_meta_estimator, fit_meta_fold
from .phase20_config import Phase20Config

logger = get_logger(__name__)

# Treatment-arm model-version stamp (the s2-aware binary meta-gate); B0 is the bare primary.
_META_MODEL_VERSION = "p20-M-meta-s2"
_B0_MODEL_VERSION = "p20-B0-primary"
_MIN_SUPPORTED_ROWS = 50  # below this, skip the supported-region SHAP sub-analysis


def _runs_dir() -> Path:
    return Path(Settings().data_dir) / "phase20_meta" / "runs"


# --------------------------------------------------------------------------- #
# Meta-set matrix
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetaMatrix:
    """The leak-free meta-set: rows that have a primary side, with the binary target."""

    symbol: str
    X: np.ndarray  # (n, p) m1 OFI + s2 sentiment + abs(ofi_top)
    y: np.ndarray  # (n,) binary meta_label in {0, 1}
    side: np.ndarray  # (n,) primary side in {-1, +1}
    t0: np.ndarray
    t1: np.ndarray
    ret_t1: np.ndarray
    sample_weight: np.ndarray
    uniqueness_weight: np.ndarray
    feature_cols: list[str]
    n_excluded: int  # rows with no primary side (ofi_top == 0 / non-finite), dropped
    micro_model_version: str
    microstructure_feature_version: str
    micro_label_version: str

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def effective_n(self) -> float:
        return float(self.uniqueness_weight.sum())

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


def build_meta_matrix(mtm: MicroTrainingMatrix, p20cfg: Phase20Config) -> MetaMatrix:
    """Build the meta-set from a ``with_sentiment`` micro matrix (leak-safe, t0-sorted).

    The primary side is ``sign(<primary_feature>)`` as-of ``t0`` (the as-of attach makes
    it causal). Rows with no side (``ofi_top == 0`` / non-finite) are excluded. The meta
    feature matrix is the m1 OFI block + the s2 sentiment block + the primary's
    ``|ofi_top|`` magnitude; the target is :func:`meta_label`.
    """
    if p20cfg.primary_feature not in mtm.feature_cols:
        raise ValueError(
            f"primary_feature {p20cfg.primary_feature!r} is not an m1 feature column "
            f"({mtm.feature_cols[:5]}...); STOP — do not substitute a different feature "
            "(see docs/PHASE20_PREREGISTRATION.md)."
        )
    ofi_idx = mtm.feature_cols.index(p20cfg.primary_feature)
    side_all = primary_side(mtm.X[:, ofi_idx])
    mask = meta_set_mask(side_all)

    X_base = mtm.X[mask]
    side = side_all[mask]
    abs_ofi = np.abs(X_base[:, ofi_idx]).reshape(-1, 1)
    X_meta = np.hstack([X_base, abs_ofi])
    feature_cols = [*mtm.feature_cols, "abs_ofi_top"]
    y_meta = meta_label(mtm.y[mask], side)

    return MetaMatrix(
        symbol=mtm.symbol,
        X=X_meta,
        y=y_meta,
        side=side.astype(int),
        t0=mtm.t0[mask],
        t1=mtm.t1[mask],
        ret_t1=mtm.ret_t1[mask],
        sample_weight=mtm.sample_weight[mask],
        uniqueness_weight=mtm.uniqueness_weight[mask],
        feature_cols=feature_cols,
        n_excluded=int((~mask).sum()),
        micro_model_version=_META_MODEL_VERSION,
        microstructure_feature_version=mtm.microstructure_feature_version,
        micro_label_version=mtm.micro_label_version,
    )


def _subset(mm: MetaMatrix, mask: np.ndarray) -> MetaMatrix:
    """Row-subset a MetaMatrix (preserves t0 order). Used for the supported-region SHAP."""
    return MetaMatrix(
        symbol=mm.symbol,
        X=mm.X[mask],
        y=mm.y[mask],
        side=mm.side[mask],
        t0=mm.t0[mask],
        t1=mm.t1[mask],
        ret_t1=mm.ret_t1[mask],
        sample_weight=mm.sample_weight[mask],
        uniqueness_weight=mm.uniqueness_weight[mask],
        feature_cols=mm.feature_cols,
        n_excluded=mm.n_excluded,
        micro_model_version=mm.micro_model_version,
        microstructure_feature_version=mm.microstructure_feature_version,
        micro_label_version=mm.micro_label_version,
    )


# --------------------------------------------------------------------------- #
# Pure metric helpers (unit-tested directly)
# --------------------------------------------------------------------------- #
def gated_gross_metrics(
    pos: np.ndarray, ret_t1: np.ndarray, w: np.ndarray, gross_sr_trials: np.ndarray
) -> dict[str, Any]:
    """Gross signal-return proxy stats for a gated position (NO costs): Sharpe / PSR / DSR.

    ``gross = position · ret_t1`` (``side · ret`` when acting, ``0`` when flat). DSR
    deflates by the trial distribution of per-config gross Sharpes — selection bias
    priced in. Reuses the audited PSR/DSR/return-moment primitives in ``validation.stats``.
    """
    pos = np.asarray(pos, dtype=float)
    ret = np.asarray(ret_t1, dtype=float)
    w = np.asarray(w, dtype=float)
    gross = pos * ret
    n_g, sr_g, sk_g, ku_g = st.return_moments(gross)
    psr_g = st.probabilistic_sharpe_ratio(sr_g, 0.0, n_g, sk_g, ku_g) if n_g >= 2 else None
    dsr_g = (
        st.deflated_sharpe_ratio(sr_g, np.asarray(gross_sr_trials, dtype=float), n_g, sk_g, ku_g)
        if n_g >= 2 and np.asarray(gross_sr_trials).size >= 1
        else None
    )
    sw = float(w.sum())
    mean_gross = float((gross * w).sum() / sw) if sw > 0 else float("nan")
    return {
        "n_scored": int(pos.size),
        "gross_sharpe": round(sr_g, 5),
        "gross_psr": None if psr_g is None else round(psr_g, 5),
        "gross_dsr": None if dsr_g is None else round(dsr_g, 5),
        "mean_gross_return": round(mean_gross, 8),
    }


def meta_skill_metrics(y: np.ndarray, p_meta: np.ndarray, tau: float) -> dict[str, Any]:
    """Meta-skill stats: OOF balanced accuracy + acted-on vs always-act hit-rate.

    The meta-model predicts act (``P > tau`` ⇒ predicted ``meta_label = 1``) or flat.
    ``balanced_accuracy`` is over (true ``meta_label``, predicted act). ``always_act_hit_rate``
    is the unconditional precision of the primary (``mean(meta_label)`` — B0's hit-rate);
    ``acted_hit_rate`` is ``mean(meta_label | acted)`` — the precision the gate buys.
    """
    from sklearn.metrics import balanced_accuracy_score

    y = np.asarray(y).astype(int)
    act = np.asarray(p_meta, dtype=float) > tau
    pred = act.astype(int)
    bal = float(balanced_accuracy_score(y, pred)) if y.size else float("nan")
    always_hit = float(np.mean(y)) if y.size else float("nan")
    acted_hit = float(np.mean(y[act])) if act.any() else float("nan")
    beats = bool(np.isfinite(acted_hit) and np.isfinite(always_hit) and acted_hit > always_hit)
    return {
        "balanced_accuracy": round(bal, 4),
        "acted_hit_rate": None if not np.isfinite(acted_hit) else round(acted_hit, 4),
        "always_act_hit_rate": None if not np.isfinite(always_hit) else round(always_hit, 4),
        "acted_hit_beats_always": beats,
        "n_acted": int(act.sum()),
    }


def decide_meta_verdict(
    *,
    pbo: float | None,
    gross_dsr: float | None,
    mean_gross_return: float | None,
    action_rate_value: float,
    cpcv_median_gross_sharpe: float | None,
    balanced_accuracy: float | None,
    acted_hit_rate: float | None,
    always_act_hit_rate: float | None,
    v: VerdictCfg,
    meta_skill_min_balanced_accuracy: float,
) -> tuple[str, dict[str, bool]]:
    """Apply the FIVE inherited mm1 gates + the disclosed binary meta-skill gate.

    The five gross gates read their thresholds from ``v`` (mm1's ``VerdictCfg``,
    unchanged); the macro-F1 gate is undefined for a binary meta-model and is replaced by
    the meta-skill gate (balanced accuracy ≥ the configured floor AND acted-on hit-rate
    strictly above the always-act hit-rate). ALL six must pass for an 'edge candidate'.
    A missing statistic (None) fails its gate.
    """
    checks = {
        "pbo_below_max": bool(pbo is not None and pbo < v.max_pbo),
        "gross_dsr_above_min": bool(gross_dsr is not None and gross_dsr > v.min_gross_dsr),
        "positive_gross_return": bool(
            (not v.require_positive_gross_return)
            or (mean_gross_return is not None and mean_gross_return > 0.0)
        ),
        "action_rate_above_min": bool(action_rate_value >= v.min_action_rate),
        "cpcv_median_gross_sharpe_positive": bool(
            (not v.require_cpcv_median_gross_sharpe_positive)
            or (cpcv_median_gross_sharpe is not None and cpcv_median_gross_sharpe > 0.0)
        ),
        "meta_skill": bool(
            balanced_accuracy is not None
            and balanced_accuracy >= meta_skill_min_balanced_accuracy
            and acted_hit_rate is not None
            and always_act_hit_rate is not None
            and acted_hit_rate > always_act_hit_rate
        ),
    }
    verdict = VERDICT_EDGE if all(checks.values()) else VERDICT_NONE
    return verdict, checks


# --------------------------------------------------------------------------- #
# In-CV binary meta search (trials counted for the DSR) + CPCV path distribution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetaSearchResult:
    """Outcome of the in-CV binary-meta grid search (inputs to deflated evaluation)."""

    selected_overrides: dict
    selected_index: int
    n_trials: int
    selection_metric: str
    configs: list[dict]
    per_config: list[dict]
    gross_sr: np.ndarray  # (n_trials,) per-config gated gross Sharpe (DSR trials)
    pbo_matrix: np.ndarray  # (n_metaset, n_trials) per-row gated gross return (PBO)
    selected_proba: np.ndarray  # (n_metaset,) pooled OOS P(meta=1) for the winner
    scored: np.ndarray  # (n_metaset,) bool


def kfold_oos_proba(
    mm: MetaMatrix, overrides: dict, mmcfg: MicroModelConfig, n_splits: int, embargo_pct: float
) -> tuple[np.ndarray, np.ndarray]:
    """Purged-KFold pooled OOS ``P(meta=1)`` for one config (every row scored once)."""
    n = mm.n
    embargo = embargo_bars_from_pct(embargo_pct, n)
    cv = PurgedKFold(n_splits, mm.t0, mm.t1, embargo_bars=embargo)
    proba = np.zeros(n, dtype=float)
    scored = np.zeros(n, dtype=bool)
    for train_idx, test_idx in cv.split():
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        est = build_meta_estimator(mmcfg, overrides)
        fit_meta_fold(
            est,
            mm.X,
            mm.y,
            mm.sample_weight,
            t0=mm.t0,
            t1=mm.t1,
            train_idx=train_idx,
            n=n,
            early_stopping=mmcfg.early_stopping,
            embargo_bars=embargo,
            class_weighting=mmcfg.class_weighting,
        )
        proba[test_idx] = est.predict_proba_pos(mm.X[test_idx])
        scored[test_idx] = True
    return proba, scored


def run_meta_search(
    mm: MetaMatrix, mmcfg: MicroModelConfig, vcfg: ValidationConfig, tau: float
) -> MetaSearchResult:
    """Run the inherited 8-config grid as binary meta-models; select on gated gross Sharpe.

    Selection uses the pooled OOS **gated** gross signal Sharpe (the inherited
    ``gross_signal_sharpe`` metric, now of the meta-gated signal). All trials' gross
    Sharpes and the per-row gated-return matrix feed the deflated evaluation (DSR + PBO).
    """
    if mmcfg.search.selection_metric != "gross_signal_sharpe":
        raise ValueError(
            f"Phase-20 meta selection requires gross_signal_sharpe, got "
            f"{mmcfg.search.selection_metric!r} — the binary gate has no macro-F1/excess metric."
        )
    configs = mmcfg.search.configs()
    n_trials = len(configs)
    n = mm.n
    pbo_matrix = np.zeros((n, n_trials), dtype=float)
    gross_sr = np.zeros(n_trials, dtype=float)
    per_config: list[dict] = []
    probas: list[np.ndarray] = []
    scored_final = np.zeros(n, dtype=bool)

    for j, overrides in enumerate(configs):
        proba, scored = kfold_oos_proba(
            mm, overrides, mmcfg, vcfg.kfold.n_splits, vcfg.kfold.embargo_pct
        )
        scored_final = scored
        pos = gated_position(mm.side, proba, tau)
        gross = pos * mm.ret_t1
        pbo_matrix[:, j] = gross
        s = scored
        gsr = st.sharpe_ratio(gross[s])
        action = float(np.mean(proba[s] > tau)) if s.any() else 0.0
        skill = meta_skill_metrics(mm.y[s], proba[s], tau)
        mean_gross = (
            float((gross[s] * mm.sample_weight[s]).sum() / mm.sample_weight[s].sum())
            if s.any() and mm.sample_weight[s].sum() > 0
            else float("nan")
        )
        gross_sr[j] = gsr
        probas.append(proba)
        per_config.append(
            {
                "overrides": overrides,
                "gross_signal_sharpe": round(gsr, 5),
                "action_rate": round(action, 4),
                "mean_gross_return": round(mean_gross, 8),
                "balanced_accuracy": skill["balanced_accuracy"],
                "acted_hit_rate": skill["acted_hit_rate"],
            }
        )
        logger.info(
            f"  trial {j + 1}/{n_trials} {overrides} -> gross_sr={gsr:.4f} "
            f"action={action:.3f} balAcc={skill['balanced_accuracy']:.3f}"
        )

    scores = np.array([c["gross_signal_sharpe"] for c in per_config], dtype=float)
    selected_index = int(np.argmax(scores))  # deterministic: first max wins
    return MetaSearchResult(
        selected_overrides=configs[selected_index],
        selected_index=selected_index,
        n_trials=n_trials,
        selection_metric=mmcfg.search.selection_metric,
        configs=configs,
        per_config=per_config,
        gross_sr=gross_sr,
        pbo_matrix=pbo_matrix,
        selected_proba=probas[selected_index],
        scored=scored_final,
    )


def meta_cpcv_distribution(
    mm: MetaMatrix,
    overrides: dict,
    mmcfg: MicroModelConfig,
    vcfg: ValidationConfig,
    tau: float,
) -> list[dict[str, float]]:
    """Per-OOS-path gated gross Sharpe for the selected config via CPCV (5 paths)."""
    n = mm.n
    embargo = embargo_bars_from_pct(vcfg.cpcv.embargo_pct, n)
    cv = CombinatorialPurgedCV(
        vcfg.cpcv.n_groups, vcfg.cpcv.test_groups, mm.t0, mm.t1, embargo_bars=embargo
    )
    n_paths, mapping = cv.assign_paths()
    buffers: list[dict[int, tuple[np.ndarray, np.ndarray]]] = [{} for _ in range(n_paths)]

    for split_index, (train_idx, _test) in enumerate(cv.split()):
        if train_idx.size == 0:
            continue
        est = build_meta_estimator(mmcfg, overrides)
        fit_meta_fold(
            est,
            mm.X,
            mm.y,
            mm.sample_weight,
            t0=mm.t0,
            t1=mm.t1,
            train_idx=train_idx,
            n=n,
            early_stopping=mmcfg.early_stopping,
            embargo_bars=embargo,
            class_weighting=mmcfg.class_weighting,
        )
        for g in cv.combos[split_index]:
            g_idx = cv.groups[g]
            buffers[mapping[(split_index, g)]][g] = (g_idx, est.predict_proba_pos(mm.X[g_idx]))

    paths: list[dict[str, float]] = []
    for path in buffers:
        if len(path) != cv.n_groups:
            continue
        order = sorted(path)
        idx = np.concatenate([path[g][0] for g in order])
        proba = np.concatenate([path[g][1] for g in order])
        pos = gated_position(mm.side[idx], proba, tau)
        gross = pos * mm.ret_t1[idx]
        paths.append(
            {
                "gross_sharpe": round(st.sharpe_ratio(gross), 5),
                "action_rate": round(float(np.mean(proba > tau)), 4),
                "effective_n": round(float(mm.uniqueness_weight[idx].sum()), 2),
            }
        )
    return paths


# --------------------------------------------------------------------------- #
# SHAP (interpretation only — never a performance number)
# --------------------------------------------------------------------------- #
def _full_meta_estimator(mm: MetaMatrix, mmcfg: MicroModelConfig, overrides: dict) -> Any:
    """Full-data binary fit with global class weights — SHAP background only."""
    est = build_meta_estimator(mmcfg, overrides)
    cw = fold_local_class_weights(
        mm.y,
        mm.sample_weight,
        META_CLASSES,
        use_sample_weight=mmcfg.class_weighting.use_sample_weight_in_balance,
    )
    est.fit(mm.X, mm.y, sample_weight=effective_sample_weight(mm.y, mm.sample_weight, cw))
    return est


def meta_shap(
    mm: MetaMatrix,
    est: Any,
    *,
    seed: int = 7,
    max_samples: int = 2000,
    out_png: Path | None = None,
) -> dict[str, Any]:
    """Global SHAP importances for the binary meta-gate (interpretation only).

    Adds ``sentiment_share`` (Σ share over ``sent_*`` features) so the report can state
    plainly whether the gate leans on the ``s2`` block. NEVER raises: any failure is
    surfaced as ``{"available": False, ...}`` and the run continues.
    """
    try:
        import shap

        from .interpret import _mean_abs_shap, _summary_plot

        feat_cols = mm.feature_cols
        p = len(feat_cols)
        rng = np.random.default_rng(seed)
        n = mm.n
        sel = (
            np.arange(n)
            if n <= max_samples
            else np.sort(rng.choice(n, size=max_samples, replace=False))
        )
        X_bg = mm.X[sel]
        explainer = shap.TreeExplainer(est._model)
        mean_abs = _mean_abs_shap(explainer.shap_values(X_bg), p)
        total = float(mean_abs.sum()) or 1.0
        order = np.argsort(mean_abs)[::-1]
        importances = [
            {
                "feature": feat_cols[i],
                "mean_abs_shap": round(float(mean_abs[i]), 6),
                "share": round(float(mean_abs[i] / total), 4),
            }
            for i in order
        ]
        sent_share = sum(
            float(mean_abs[i] / total) for i in range(p) if feat_cols[i].startswith("sent_")
        )
        top_share = importances[0]["share"] if importances else 0.0
        png_path = _summary_plot(explainer, X_bg, feat_cols, out_png) if out_png else None
        return {
            "available": True,
            "n_background": int(X_bg.shape[0]),
            "top_features": [d["feature"] for d in importances[:10]],
            "importances": importances,
            "top_feature_share": round(float(top_share), 4),
            "sentiment_share": round(float(sent_share), 4),
            "summary_plot": png_path,
        }
    except Exception as exc:  # noqa: BLE001 - interpretation is optional, never fail the run
        logger.warning(f"meta SHAP unavailable ({exc}); continuing without it")
        return {"available": False, "reason": str(exc)}


# --------------------------------------------------------------------------- #
# Arm evaluation
# --------------------------------------------------------------------------- #
def _eval_common(
    mm: MetaMatrix,
    *,
    arm: str,
    model_version: str,
    pos: np.ndarray,
    proba: np.ndarray,
    mask: np.ndarray,
    gross_sr_trials: np.ndarray,
    n_trials: int,
    selected_overrides: dict,
    pbo: dict[str, Any] | None,
    cpcv: dict[str, Any],
    n_features: int,
    mmcfg: MicroModelConfig,
    vcfg: ValidationConfig,
    p20cfg: Phase20Config,
) -> dict[str, Any]:
    """Assemble one arm's summary + the six-gate verdict from its pooled OOS position."""
    pos_s = pos[mask]
    ret_s = mm.ret_t1[mask]
    w_s = mm.sample_weight[mask]
    proba_s = proba[mask]
    y_s = mm.y[mask]

    sig = gated_gross_metrics(pos_s, ret_s, w_s, gross_sr_trials)
    arate = action_rate(pos_s)
    skill = meta_skill_metrics(y_s, proba_s, p20cfg.decision_threshold)
    cpcv_median = cpcv["gross_sharpe"]["median"]

    verdict, checks = decide_meta_verdict(
        pbo=pbo["pbo"] if pbo else None,
        gross_dsr=sig["gross_dsr"],
        mean_gross_return=sig["mean_gross_return"],
        action_rate_value=arate,
        cpcv_median_gross_sharpe=cpcv_median,
        balanced_accuracy=skill["balanced_accuracy"],
        acted_hit_rate=skill["acted_hit_rate"],
        always_act_hit_rate=skill["always_act_hit_rate"],
        v=mmcfg.verdict,
        meta_skill_min_balanced_accuracy=p20cfg.meta_skill_min_balanced_accuracy,
    )
    return {
        "symbol": mm.symbol,
        "arm": arm,
        "verdict": verdict,
        "verdict_checks": checks,
        "verdict_thresholds": mmcfg.verdict.model_dump(mode="json"),
        "meta_skill_min_balanced_accuracy": p20cfg.meta_skill_min_balanced_accuracy,
        "decision_threshold": p20cfg.decision_threshold,
        "micro_model_version": model_version,
        "microstructure_feature_version": mm.microstructure_feature_version,
        "micro_label_version": mm.micro_label_version,
        "validation_version": vcfg.validation_version,
        "target_mode": "binary_meta",
        "n_samples": int(mask.sum()),
        "n_metaset": mm.n,
        "n_excluded_no_side": mm.n_excluded,
        "effective_n": round(float(mm.uniqueness_weight[mask].sum()), 2),
        "n_features": n_features,
        "n_trials": n_trials,
        "selection_metric": mmcfg.search.selection_metric,
        "selected_overrides": selected_overrides,
        "action_rate": round(arate, 4),
        "signal_return": sig,
        "meta_skill": skill,
        "pbo": pbo,
        "cpcv": cpcv,
        "cost_disclaimer": "gross signal-return proxy only (side * ret_t1); not an executable "
        "backtest (no commissions or slippage)",
    }


def evaluate_meta_arm(
    mm: MetaMatrix,
    search: MetaSearchResult,
    mmcfg: MicroModelConfig,
    vcfg: ValidationConfig,
    p20cfg: Phase20Config,
) -> dict[str, Any]:
    """Arm M — the meta-gated primary-verdict arm (selected config, deflated, six gates)."""
    tau = p20cfg.decision_threshold
    proba = search.selected_proba
    pos = gated_position(mm.side, proba, tau)
    s = search.scored

    pbo: dict[str, Any] | None = None
    if search.n_trials >= 2:
        n_part = min(10, (search.pbo_matrix.shape[0] // 2) * 2)
        if n_part >= 2:
            res = st.probability_of_backtest_overfitting(search.pbo_matrix, n_partitions=n_part)
            pbo = {
                "pbo": round(float(res["pbo"]), 4),
                "n_combinations": int(res["n_combinations"]),
                "n_partitions": n_part,
                "n_configs": search.n_trials,
            }

    paths = meta_cpcv_distribution(mm, search.selected_overrides, mmcfg, vcfg, tau)
    cpcv = {
        "n_paths": len(paths),
        "gross_sharpe": _distribution_summary([p["gross_sharpe"] for p in paths]),
        "action_rate_mean": (
            None if not paths else round(float(np.mean([p["action_rate"] for p in paths])), 4)
        ),
        "effective_n_mean": (
            None if not paths else round(float(np.mean([p["effective_n"] for p in paths])), 2)
        ),
        "paths": paths,
    }

    summary = _eval_common(
        mm,
        arm="M",
        model_version=_META_MODEL_VERSION,
        pos=pos,
        proba=proba,
        mask=s,
        gross_sr_trials=search.gross_sr,
        n_trials=search.n_trials,
        selected_overrides=search.selected_overrides,
        pbo=pbo,
        cpcv=cpcv,
        n_features=mm.n_features,
        mmcfg=mmcfg,
        vcfg=vcfg,
        p20cfg=p20cfg,
    )
    summary["search_configs"] = search.per_config
    summary["feature_cols"] = list(mm.feature_cols)
    return summary


def evaluate_b0(
    mm: MetaMatrix, mmcfg: MicroModelConfig, vcfg: ValidationConfig, p20cfg: Phase20Config
) -> dict[str, Any]:
    """B0 — the always-act primary reference (no model, no gate). Expected to fail.

    There is no search and no model, so PBO is undefined (None ⇒ fails its gate) and the
    CPCV path distribution is degenerate (a fixed predictor ⇒ all paths identical), so it
    is reported as the single gross Sharpe of ``side · ret``. B0 also fails the meta-skill
    gate by construction (acting on everything ⇒ acted-hit-rate == always-act-hit-rate,
    and balanced accuracy of an all-act predictor is 0.5).
    """
    pos = mm.side.astype(float)  # always act
    proba = np.ones(mm.n, dtype=float)  # P=1 everywhere ⇒ act on all (tau-independent)
    mask = np.ones(mm.n, dtype=bool)
    b0_gross_sharpe = st.sharpe_ratio(pos * mm.ret_t1)
    cpcv = {
        "n_paths": 1,
        "gross_sharpe": _distribution_summary([round(b0_gross_sharpe, 5)]),
        "action_rate_mean": 1.0,
        "effective_n_mean": round(float(mm.uniqueness_weight.sum()), 2),
        "paths": [{"gross_sharpe": round(b0_gross_sharpe, 5), "action_rate": 1.0}],
        "note": "degenerate (model-free reference): all reconstructed paths are identical",
    }
    summary = _eval_common(
        mm,
        arm="B0",
        model_version=_B0_MODEL_VERSION,
        pos=pos,
        proba=proba,
        mask=mask,
        gross_sr_trials=np.array([b0_gross_sharpe], dtype=float),  # 1 trial ⇒ no deflation
        n_trials=1,
        selected_overrides={},
        pbo=None,
        cpcv=cpcv,
        n_features=0,
        mmcfg=mmcfg,
        vcfg=vcfg,
        p20cfg=p20cfg,
    )
    return summary


# --------------------------------------------------------------------------- #
# Attribution + decision (pure, exactly as pre-registered)
# --------------------------------------------------------------------------- #
def attribute(b0: dict[str, Any], m: dict[str, Any]) -> dict[str, Any]:
    """Frozen attribution. Primary verdict = does arm M clear all six gates?

    B0 is the reference (expected to fail). M's pass/fail IS the per-symbol verdict.
    """
    return {
        "b0_pass": b0["verdict"] == VERDICT_EDGE,
        "m_pass": m["verdict"] == VERDICT_EDGE,
    }


def decide(symbol_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Frozen decision rule over the per-symbol M-arm verdicts.

    M clears all six on a symbol → 'meta_labeling_edge_candidate' (authorizes ONLY a
    future Phase 21 economic backtest for that symbol; never live). A single-symbol
    candidate while the other fails is flagged **fragile**. M fails on both → meta-labeling
    is the next honest null.
    """
    per_symbol: dict[str, str] = {}
    candidates: list[str] = []
    for sym, res in symbol_results.items():
        m_pass = res["attribution"]["m_pass"]
        per_symbol[sym] = "meta_labeling_edge_candidate" if m_pass else "null"
        if m_pass:
            candidates.append(sym)

    n = len(symbol_results)
    fragile = bool(candidates) and len(candidates) < n
    if not candidates:
        overall = "no_significant_edge"
        note = (
            "meta-labeling is the next honest null; because it is the canonical remedy for "
            "this low-precision/imbalanced regime, its failure is strong evidence the binding "
            "constraint is sample size / edge existence at this horizon, not model framing — "
            "escalating the strategic fork (acquire more data/depth, redesign the horizon/"
            "market, or accept the result). The lever is not re-litigated."
        )
    elif fragile:
        overall = "meta_labeling_edge_candidate_fragile"
        note = (
            f"single-symbol candidate ({', '.join(candidates)}) with the other symbol "
            "failing — flagged FRAGILE; authorizes ONLY a Phase 21 economic backtest for the "
            "passing symbol, never live trading."
        )
    else:
        overall = "meta_labeling_edge_candidate"
        note = (
            "authorizes ONLY a future Phase 21 economic backtest (realistic costs/slippage) "
            "per passing symbol — never live trading."
        )
    return {
        "per_symbol": per_symbol,
        "candidates": candidates,
        "fragile": fragile,
        "overall": overall,
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _arm_view(summary: dict[str, Any]) -> dict[str, Any]:
    """A compact, comparison-friendly slice of a full arm summary (for the verdict file)."""
    sig = summary["signal_return"]
    skill = summary["meta_skill"]
    return {
        "arm": summary["arm"],
        "model_version": summary["micro_model_version"],
        "n_samples": summary["n_samples"],
        "n_features": summary["n_features"],
        "verdict": summary["verdict"],
        "verdict_checks": summary["verdict_checks"],
        "selected_overrides": summary["selected_overrides"],
        "metrics": {
            "pbo": summary["pbo"]["pbo"] if summary["pbo"] else None,
            "gross_dsr": sig["gross_dsr"],
            "gross_sharpe": sig["gross_sharpe"],
            "mean_gross_return": sig["mean_gross_return"],
            "action_rate": summary["action_rate"],
            "cpcv_median_gross_sharpe": summary["cpcv"]["gross_sharpe"]["median"],
            "balanced_accuracy": skill["balanced_accuracy"],
            "acted_hit_rate": skill["acted_hit_rate"],
            "always_act_hit_rate": skill["always_act_hit_rate"],
        },
        "shap_top": (summary.get("shap") or {}).get("top_features", [])[:8],
        "shap_sentiment_share": (summary.get("shap") or {}).get("sentiment_share"),
        "shap_supported_top": (summary.get("shap_supported") or {}).get("top_features", [])[:8],
        "shap_supported_sentiment_share": (summary.get("shap_supported") or {}).get(
            "sentiment_share"
        ),
    }


def save_arm_run(summary: dict[str, Any], *, runs_dir: Path | None = None) -> Path:
    """Persist one full arm evaluation under ``data/phase20_meta/runs/<symbol>_<arm>.json``."""
    d = runs_dir or _runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{summary['symbol']}_{summary['arm']}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return path


def save_verdict(verdict: dict[str, Any], *, runs_dir: Path | None = None) -> Path:
    """Persist the combined verdict under ``data/phase20_meta/runs/verdict.json``."""
    d = runs_dir or _runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "verdict.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    return path


def read_verdict(*, runs_dir: Path | None = None) -> dict[str, Any] | None:
    """Load the saved combined verdict, or ``None`` if no run has been saved."""
    path = (runs_dir or _runs_dir()) / "verdict.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Per-symbol + full run
# --------------------------------------------------------------------------- #
def run_phase20_symbol(
    symbol: str,
    p20cfg: Phase20Config,
    mmcfg: MicroModelConfig,
    vcfg: ValidationConfig,
    scfg: SentimentConfig,
    *,
    interpret: bool = True,
    rebuild_cache: bool = False,
    log_mlflow: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Run B0 + M for one symbol → attribution + the compact per-symbol result."""
    start, end = p20cfg.window.start_dt(), p20cfg.window.end_dt()
    logger.info(f"[{symbol}] Phase-20 meta-labeling on t0∈[{start.date()}..{end.date()}]")

    mtm = load_micro_matrix(
        symbol,
        start=start,
        end=end,
        mmcfg=mmcfg,
        with_sentiment=True,
        scfg=scfg,
        version_stamp="mm2",
        rebuild_cache=rebuild_cache,
    )
    mm = build_meta_matrix(mtm, p20cfg)
    logger.info(
        f"[{symbol}] meta-set n={mm.n} (excluded {mm.n_excluded} no-side rows) "
        f"effN={mm.effective_n:.1f} meta_label_rate={float(np.mean(mm.y)):.3f}"
    )

    search = run_meta_search(mm, mmcfg, vcfg, p20cfg.decision_threshold)
    m_summary = evaluate_meta_arm(mm, search, mmcfg, vcfg, p20cfg)
    b0_summary = evaluate_b0(mm, mmcfg, vcfg, p20cfg)

    if interpret:
        est = _full_meta_estimator(mm, mmcfg, search.selected_overrides)
        out_png = _runs_dir() / f"{symbol}_M_shap.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        m_summary["shap"] = meta_shap(mm, est, seed=mmcfg.seed, out_png=out_png)
        # Secondary sentiment sub-analysis: SHAP on the supported region only (diagnostic).
        sup_mask = mm.t0 >= np.datetime64(p20cfg.supported_region_start)
        if int(sup_mask.sum()) >= _MIN_SUPPORTED_ROWS:
            mm_sup = _subset(mm, sup_mask)
            est_sup = _full_meta_estimator(mm_sup, mmcfg, search.selected_overrides)
            sup_png = _runs_dir() / f"{symbol}_M_shap_supported.png"
            m_summary["shap_supported"] = meta_shap(
                mm_sup, est_sup, seed=mmcfg.seed, out_png=sup_png
            )
            m_summary["shap_supported_n"] = int(sup_mask.sum())

    attribution = attribute(b0_summary, m_summary)

    if save:
        save_arm_run(b0_summary)
        save_arm_run(m_summary)
    if log_mlflow:
        from .tracking import log_run

        for summary in (b0_summary, m_summary):
            log_run(summary, summary.get("shap"), experiment=p20cfg.mlflow_experiment)

    return {
        "symbol": symbol,
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "n_rows": mtm.n,
        "n_metaset": mm.n,
        "n_excluded_no_side": mm.n_excluded,
        "effective_n": round(mm.effective_n, 2),
        "meta_label_rate": round(float(np.mean(mm.y)), 4),
        "b0": _arm_view(b0_summary),
        "meta": _arm_view(m_summary),
        "attribution": attribution,
    }


def run_phase20(
    symbols: list[str] | None = None,
    *,
    interpret: bool = True,
    rebuild_cache: bool = False,
    log_mlflow: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Run the full Phase-20 meta-labeling verdict for all symbols → the combined verdict."""
    p20cfg = Phase20Config.load()
    mmcfg = MicroModelConfig.load()
    vcfg = ValidationConfig.load()
    scfg = SentimentConfig.load()

    if scfg.aggregation.feature_version != p20cfg.sentiment_feature_version:
        raise ValueError(
            f"sentiment aggregation feature_version={scfg.aggregation.feature_version} "
            f"!= pre-registered {p20cfg.sentiment_feature_version} — the s2 block changed; "
            "reconcile before running the frozen meta-labeling verdict."
        )

    # The pre-registered decision rule is defined over the EXACT pre-registered symbol set
    # (ES + NQ). A subset run (e.g. --symbols ES) cannot test the other symbol, so its
    # combined decision would be wrong — a lone passing symbol would read as a non-fragile
    # candidate instead of the fragile single-symbol case. Refuse to SAVE a canonical
    # verdict over anything but the full set; diagnostic (save=False) runs are flagged
    # partial and never claim a clean decision.
    syms = list(dict.fromkeys(symbols or p20cfg.symbols))  # dedupe, preserve order
    is_full = set(syms) == set(p20cfg.symbols)
    if save and not is_full:
        raise ValueError(
            f"Phase-20's combined verdict is pre-registered over {list(p20cfg.symbols)}; "
            f"refusing to save a canonical decision over {syms}. Re-run with the exact "
            "pre-registered symbol set, or pass save=False for a diagnostic (non-canonical) run."
        )

    symbol_results: dict[str, dict[str, Any]] = {}
    for symbol in syms:
        symbol_results[symbol] = run_phase20_symbol(
            symbol,
            p20cfg,
            mmcfg,
            vcfg,
            scfg,
            interpret=interpret,
            rebuild_cache=rebuild_cache,
            log_mlflow=log_mlflow,
            save=save,
        )

    decision = decide(symbol_results)
    verdict = {
        "phase": "20",
        "phase20_version": p20cfg.phase20_version,
        "preregistration": "docs/PHASE20_PREREGISTRATION.md",
        "primary_rule": f"sign({p20cfg.primary_feature}) at t0",
        "decision_threshold": p20cfg.decision_threshold,
        "meta_skill_min_balanced_accuracy": p20cfg.meta_skill_min_balanced_accuracy,
        "sentiment_feature_version": p20cfg.sentiment_feature_version,
        "window": {"start": p20cfg.window.start, "end": p20cfg.window.end},
        "supported_region_start": p20cfg.supported_region_start,
        "verdict_thresholds": mmcfg.verdict.model_dump(mode="json"),
        "symbols": symbol_results,
        "decision": decision,
        "partial": not is_full,  # True ⇒ diagnostic subset; the combined decision is NOT canonical
        "cost_disclaimer": "gross signal-return proxy only (side * ret_t1); not an executable "
        "backtest (no commissions or slippage); authorizes no strategy/backtest/live trading.",
    }
    if save:
        save_verdict(verdict)
    return verdict


def _print_verdict(verdict: dict[str, Any]) -> None:
    logger.info("=== Phase-20 meta-labeling verdict ===")
    for sym, res in verdict["symbols"].items():
        b0, m = res["b0"], res["meta"]
        logger.info(
            f"[{sym}] metaset={res['n_metaset']} (excl {res['n_excluded_no_side']}) "
            f"effN={res['effective_n']} | "
            f"B0={b0['verdict']} | M={m['verdict']} gates_failed="
            f"{[k for k, v in m['verdict_checks'].items() if not v] or 'none'} | "
            f"m_pass={res['attribution']['m_pass']}"
        )
        if m.get("shap_top"):
            logger.info(
                f"[{sym}] M SHAP top: {m['shap_top'][:6]} "
                f"sent_share={m.get('shap_sentiment_share')}"
            )
    d = verdict["decision"]
    logger.info(f"DECISION: {d['overall']} (candidates={d['candidates'] or 'none'}) — {d['note']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="microstructure_model.phase20_meta", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: config phase20.symbols")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--no-interpret", action="store_true", help="skip SHAP interpretation")
    p.add_argument("--rebuild-cache", action="store_true", help="rebuild the micro-matrix cache")
    args = p.parse_args(argv)

    verdict = run_phase20(
        symbols=args.symbols,
        interpret=not args.no_interpret,
        rebuild_cache=args.rebuild_cache,
        log_mlflow=not args.no_mlflow,
    )
    _print_verdict(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
