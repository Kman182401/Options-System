"""Phase-25 config loader — the frozen economic-value (vol-timing) contract.

Loads ``config/phase25_econ.yaml`` (frozen by ``docs/PHASE25_PREREGISTRATION.md``). The entire
**modeling core** (RV target, the fixed LightGBM, h = 1, the anchored walk-forward, the regime
split, the symbols, seed 7, DM) is **inherited verbatim** from the Phase-23 contract
(``config/phase23_vol_h1.yaml``) via :class:`Phase23Config` — so the economic layer reuses the
confirmed forecasts with zero drift and adds **zero fitted parameters**. This file parses only the
Phase-25-specific frozen knobs: the six arms, the two weight rules, the overlay knobs (``gamma``,
``w_cap``, ``w_floor``, ``sigma_target``), the drift/``mu_bar`` firewall, the per-symbol transaction
costs, the stationary-bootstrap parameters, the gate thresholds, and the QLIKE reproduction pins.

Descriptive ``rule``/``note``/``spec`` strings in the YAML are documentation of the frozen logic
(the code is the source of truth); the lenient parsers ``ignore`` them rather than re-encode them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .config_h1 import Phase23Config

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "phase25_econ.yaml"


class _Lenient(BaseModel):
    # ignore the YAML's descriptive rule/spec strings — the code owns the logic.
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


# --------------------------------------------------------------------------- #
# Overlay / position knobs
# --------------------------------------------------------------------------- #
class OverlayCfg(_Lenient):
    gamma: float = Field(gt=0.0)
    gamma_reported_only: list[float] = Field(default_factory=lambda: [2.0, 10.0])
    w_cap: float = Field(gt=0.0)
    w_floor: float = Field(ge=0.0)
    sigma_target_annual: float = Field(gt=0.0)
    ann_factor: int = Field(gt=0)

    @property
    def sigma_target_daily(self) -> float:
        """The daily RTH vol budget: annual target de-annualized by ``sqrt(ann_factor)``."""
        return self.sigma_target_annual / math.sqrt(self.ann_factor)


# --------------------------------------------------------------------------- #
# Drift / expected-return firewall
# --------------------------------------------------------------------------- #
class DriftCfg(_Lenient):
    mu_floor_per_day: float = Field(gt=0.0)
    vol_target_leg_drift: float = 0.0
    rf: float = 0.0


# --------------------------------------------------------------------------- #
# Transaction-cost model (per symbol, frozen)
# --------------------------------------------------------------------------- #
class SymbolCostCfg(_Lenient):
    multiplier_usd_per_point: float = Field(gt=0.0)
    tick_value_usd: float = Field(gt=0.0)
    half_spread_usd: float = Field(ge=0.0)
    per_side_usd: float = Field(gt=0.0)  # = commission + half-spread (frozen, per side)
    ref_index_level: float = Field(gt=0.0)
    ref_notional_usd: float = Field(gt=0.0)
    c_per_side: float = Field(gt=0.0)

    def c_side(self) -> float:
        """Per-side cost fraction = per-side $ / frozen reference notional.

        Derived from the frozen per-side cost (commission + half-spread) and the frozen reference
        notional — never from a realized OOS price, so no OOS price leaks into the cost fraction.
        Cross-checked against the YAML's stated ``c_per_side``.
        """
        derived = self.per_side_usd / self.ref_notional_usd
        if not math.isclose(derived, self.c_per_side, rel_tol=5e-3):
            raise ValueError(
                f"cost mis-specification: derived c_side {derived:.3e} != stated "
                f"c_per_side {self.c_per_side:.3e}"
            )
        return derived


class CostsCfg(_Lenient):
    commission_per_side_usd: float = Field(gt=0.0)
    spread_ticks: float = Field(ge=0.0)
    per_symbol: dict[str, SymbolCostCfg]
    stress_multipliers_gated: list[float] = Field(default_factory=lambda: [1.0, 3.0])
    reported_only_multipliers: list[float] = Field(default_factory=lambda: [0.5, 5.0])

    def base_and_stress(self) -> tuple[float, float]:
        """The two GATED cost levels (base, stress) — the gates must hold at both."""
        levels = sorted(set(self.stress_multipliers_gated))
        if levels != [1.0, 3.0]:
            raise ValueError(f"gated cost multipliers are frozen at [1.0, 3.0]; got {levels}")
        return 1.0, 3.0


# --------------------------------------------------------------------------- #
# Significance test (stationary bootstrap)
# --------------------------------------------------------------------------- #
class SignificanceCfg(_Lenient):
    one_sided: bool = True
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    n_resamples: int = Field(default=10000, gt=0)
    expected_block_length: int = Field(default=10, gt=0)


# --------------------------------------------------------------------------- #
# Reproduction guard (the fail-closed QLIKE pins)
# --------------------------------------------------------------------------- #
class ReproductionGuardCfg(_Lenient):
    fingerprint_required: bool = True
    expected_qlike: dict[str, dict[str, float]]
    persist_oos_frame: str = "data/volatility/runs_ev"


# --------------------------------------------------------------------------- #
# Gate thresholds
# --------------------------------------------------------------------------- #
class GatesCfg(_Lenient):
    min_effect_floor_bps_per_year: float = 25.0
    vte_stability_min_folds: int = 13
    vte_stability_n_folds: int = 18
    value_stability_min_folds: int = 13
    value_stability_n_folds: int = 18
    stress_multiplier: float = 3.0
    g8_max_post_match_mean_weight_dev_vs_static: float = 0.05
    g8_min_net_to_gross_fee_frac_at_3x: float = 0.50
    e9_max_abs_corr: float = 0.10


class Phase25Config(_Lenient):
    """The Phase-25 contract: the inherited Phase-23 core + the frozen economic-overlay knobs."""

    econvalue_version: str
    core: Any  # VolatilityConfig — the confirmed Phase-23 modeling core (h = 1)
    symbols: list[str]
    arms: dict[str, str]
    overlay: OverlayCfg
    drift: DriftCfg
    costs: CostsCfg
    significance: SignificanceCfg
    reproduction_guard: ReproductionGuardCfg
    gates: GatesCfg
    artifacts: dict[str, Any] = Field(default_factory=dict)
    ewma_lambda: float

    @classmethod
    def load(cls, path: str | Path | None = None) -> Phase25Config:
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        p23 = Phase23Config.load()  # inherit the frozen Phase-23 modeling core verbatim
        core = p23.core

        gates_raw = data.get("gates", {}) or {}
        g25 = _parse_gates(gates_raw, data.get("metrics", {}) or {})

        return cls(
            econvalue_version=data["econvalue_version"],
            core=core,
            symbols=list(data.get("symbols", core.symbols)),
            arms=dict(data.get("arms", {})),
            overlay=OverlayCfg.model_validate(data["overlay"]),
            drift=DriftCfg.model_validate(data["drift"]),
            costs=CostsCfg.model_validate(data["costs"]),
            significance=SignificanceCfg.model_validate(data.get("significance", {})),
            reproduction_guard=ReproductionGuardCfg.model_validate(data["reproduction_guard"]),
            gates=g25,
            artifacts=data.get("artifacts", {}) or {},
            ewma_lambda=float(p23.benchmarks.ewma.lam),
        )


def _parse_gates(gates_raw: dict[str, Any], metrics_raw: dict[str, Any]) -> GatesCfg:
    """Pull the numeric gate thresholds out of the (mostly descriptive) frozen YAML gate blocks."""
    g2 = gates_raw.get("g2_vte_temporal_stability", {}) or {}
    g7 = gates_raw.get("g7_econ_value_temporal_stability", {}) or {}
    g5 = gates_raw.get("g5_stressed_cost_robustness", {}) or {}
    g8 = gates_raw.get("g8_exposure_neutrality_costerosion", {}) or {}
    e9 = gates_raw.get("e9_directional_leakage_void", {}) or {}
    return GatesCfg(
        min_effect_floor_bps_per_year=float(metrics_raw.get("min_effect_floor_bps_per_year", 25)),
        vte_stability_min_folds=int(g2.get("min_folds", 13)),
        vte_stability_n_folds=int(g2.get("n_folds", 18)),
        value_stability_min_folds=int(g7.get("min_folds", 13)),
        value_stability_n_folds=int(g7.get("n_folds", 18)),
        stress_multiplier=float(g5.get("stress_multiplier", 3.0)),
        g8_max_post_match_mean_weight_dev_vs_static=float(
            g8.get("max_post_match_mean_weight_dev_vs_static", 0.05)
        ),
        g8_min_net_to_gross_fee_frac_at_3x=float(g8.get("min_net_to_gross_fee_frac_at_3x", 0.50)),
        e9_max_abs_corr=float(e9.get("max_abs_corr", 0.10)),
    )
