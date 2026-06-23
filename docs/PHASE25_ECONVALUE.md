# PHASE25_ECONVALUE.md — Economic-value verdict of the confirmed 1-day vol forecast (Phase 25)

## What it is
Phase 23 confirmed a 1-day realized-volatility **forecast skill** (accuracy) on both symbols. Phase 25
asks the bridge question: does that accuracy translate into **economically-meaningful, cost-robust,
direction-free value** (money) when it sizes a position — net of realistic *and* stressed costs? Frozen
contract: `docs/PHASE25_PREREGISTRATION.md` / `config/phase25_econ.yaml` (`econvalue_version=ev1`). The
modeling core is inherited **verbatim** from the Phase-23 contract — the forecasts are reused, not
refit; the economic layer adds **zero fitted parameters**. Offline / simulated / paper verdict only —
authorizes **no** live trading and **no** automated paper-execution deployment, regardless of outcome.

## The integrity problem it solves
The program has six confirmed directional nulls (no edge in predicting the *sign/size* of returns). The
confirmed skill is purely about the **second moment** (volatility). An economic-value study is
dangerous because a volatility-timing overlay can secretly become a directional / risk-premium bet
(e.g. if calm days happen to be up days), smuggling a dead directional lever back in and crediting it
to the variance forecast. The design makes that **structurally impossible** and **detects it
empirically** if it leaks anyway.

## Design (Fleming-Kirby-Ostdiek + a return-free leg, both required)
Six arms per symbol — `TREAT` (the Phase-23 LightGBM), `HAR`, `RW`, `EWMA`, `GARCH`, and `STATIC` (the
no-timing FKO unconditional portfolio) — differ **only** through their variance forecast `σ²`. Two
long-only weight rules run in parallel:
- **Mean-variance (money leg):** `w = clip(μ_bar / (γ·σ²), 0, w_cap)`.
- **Vol-target (return-free leg):** `wv = clip(σ_target / σ, w_floor, w_cap)`, drift = 0.

The directional firewall: `μ_bar` (per-fold causal training-mean return, forced ≥ 0) and `γ = 5` are
single fixed scalars **identical across all six arms**; the position is long-only (`w ≥ 0`, no sign
decision); every arm earns the **identical** next-session RTH return `r_{t+1}` (a one-session forward
shift — the weight set at the close of row `t` earns session `t+1`, never the same-session `r_t`);
the money leg is **leverage-matched** to `STATIC` so a win cannot be a larger-average-long bet; and the
return-free VTE leg consumes no return assumption at all (a directional bet cannot pass it). Costs are a
full daily round-trip `2·c_side·w` on the flat-RTH-only held position (common-mode across arms), with
the reference notional frozen at pre-registration so no realized OOS price leaks into the cost fraction.
Significance is a Politis-Romano stationary bootstrap (B = 10,000, L = 10, one-sided) on the per-day
net realized-utility differential, per symbol.

**Nine gates** (per symbol, all required, `E9` a void): G1 VTE intersection (return-free), G2 VTE
fold-stability (≥ 13/18), G3 fee vs `STATIC` > +25 bps/yr (boot p < 0.05), G4 fee > 0 vs **every**
benchmark (p < 0.05), G5 the same in sign at **3×** cost, G6 regime robustness (≥ 0 in calm **and**
turbulent), G7 value fold-stability (beat `STATIC` and `RW` in ≥ 13/18 folds), G8 post-match exposure
neutrality + cost-erosion, and **E9** — any `|corr(w_treat − w_k, r_{t+1})| ≥ 0.10` on the active
overweight **voids** the verdict regardless of G1–G8.

## Reproduction guard (fail-closed)
The Phase-23 forecasts are regenerated deterministically (same config, seed 7) and the implementation
asserts, before any economic computation, that they reproduce the frozen result: per-arm OOS QLIKE
(treatment byte-exact `0.246401`/`0.224083`, each benchmark to 3 dp), the GARCH convergence
diagnostics, the 18-fold partition, and the frame length (n = 1,139) — every dimension with a frozen
Phase-23 reference is pinned. A full-frame SHA-256 fingerprint (forecasts, fold ids, `t0`, `rv`, `y`,
the aligned `next_rth_log_return`, GARCH diagnostics) is persisted for audit and compared on re-runs.
Any mismatch fails closed.

## Result — measured 2026-06-23 (OOS n = 1,139 per symbol, γ = 5, base cost)

**Honest null on both symbols — and *not* a directional leak (E9 clean).** The confirmed accuracy edge
does **not** translate into cost-robust, direction-free economic value.

| Symbol | fee vs STATIC (bps/yr) | boot p | E9 max \|corr\| | gates passed | verdict |
|--------|-----------------------:|-------:|----------------:|:------------:|---------|
| **MES** | **−36.1** | 0.731 | 0.047 (har) | 1 / 8 | honest null |
| **MNQ** | **−1.1** | 0.508 | 0.040 (har) | 0 / 8 | honest null |

What the numbers say:
- **Vol-timing does not beat the constant-weight baseline net of costs.** The fee vs `STATIC` is
  negative on both symbols; `STATIC`'s causal unconditional variance already hits the vol target nearly
  perfectly (VTE 0.075 MES / 0.003 MNQ), and the treatment's day-to-day reweighting adds realized-vol
  noise rather than control (VTE_treat 0.147/0.149 — worse, so G1 fails).
- **The forecast *does* size better than other forecasters** — `TREAT` earns a positive, often
  significant fee vs `HAR` (+64/+75 bps) and `RW` (+70/+73 bps) — but the gate that matters (vs the
  just-being-invested `STATIC` baseline) and the all-benchmark intersection (incl. `GARCH`) fail.
- **No directional contamination.** All E9 active-exposure correlations are < 0.05 (well under 0.10),
  the per-arm `|corr(w, r_{t+1})|` are all < 0.06, and leverage matching held the mean exposures close
  (MES wbar 0.219 vs 0.201; MNQ 0.229 vs 0.219). The honest null is a genuine "accuracy ≠ money net of
  costs" result, not a smuggled directional bet.
- The HAC/DM asymptotic cross-check p-values track the stationary-bootstrap p-values closely
  (e.g. MES vs `STATIC` 0.680 vs 0.731; vs `HAR` 0.009 vs 0.007), validating the significance machinery.

This is the contract's explicitly **acceptable, likely-modal** outcome. The confirmed Phase-23 forecast
skill stands; the economic-value lever is re-scoped only by a deliberate operator decision, never
auto-re-litigated. **No strategy / backtest / risk / execution / live trading is authorized.**

## Files
`volatility/econ.py` (new — pure FKO/VTE/fee-solver/stationary-bootstrap/leverage-match/E9 primitives),
`volatility/config_econ.py` (new — frozen-knob loader inheriting the Phase-23 core),
`volatility/run_econ.py` (new — the orchestrator + fail-closed reproduction guard), tests in
`tests/test_phase25_econ.py`. Artifacts (gitignored): `data/volatility/runs_ev/`. Two Codex/Workflow
adversarial-review findings folded before finalizing (the structural reproduction pins and the
save-flag persistence split). Forecast/economic verdict in offline simulation only; no spend, no trading.
