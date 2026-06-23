# Phase 25 Pre-Registration — Economic-Value Study of the Confirmed 1-Day Volatility Forecast

**Committed before any Phase 25 simulation is run or any forecast reaches the economic layer.** The
commit date of the finalized contract is the pre-registration timestamp.

> **Provenance note (integrity).** A first draft of this contract was auto-committed to `master` by a
> workflow subagent (commit `5053ad8`) *before* the design was finalized — it encoded a weaker,
> different design (variance-targeting + Sharpe only) and pre-dated the operator's stringency
> decisions. This file **supersedes** that draft with the reviewed, synthesis-faithful, operator-
> confirmed contract. No Phase-25 simulation had run at either commit, so the anti-snooping property
> (contract frozen before any results exist) is fully preserved.

Nothing here authorizes strategy, live backtest, risk, execution, or live trading. Phase 25 produces
an **economic-value verdict in offline simulation only**: does the **confirmed Phase-23 h = 1
realized-volatility forecast** deliver **positive, cost-robust, direction-free economic value** — net
of realistic *and* stressed transaction costs — over a no-volatility-timing baseline and over every
econometric benchmark, per symbol? The system stays hard-locked to paper. Reads only the
deterministically-regenerated Phase-23 OOS forecasts and the local bars lake — no Databento, no IBKR, no
network, no spend (`OPTIONS_DATABENTO_SPEND_OK` stays unset). It writes no execution code, touches no
Risk Manager, and deploys nothing.

## Why this study, and the honest prior

Phase 23 **confirmed** a 1-day realized-volatility forecast skill on both symbols: the fixed LightGBM
beats HAR-RV, a random walk, EWMA(λ=0.94) and GARCH(1,1) on QLIKE (treat 0.246/0.224 vs HAR
0.308/0.267, RW 0.760/0.695, EWMA 0.360/0.295, GARCH 0.331/0.285), Diebold-Mariano significant,
regime-robust, in 16+/18 walk-forward folds. That is the program's first positive verdict — but it is
a **forecast-accuracy** result. Accuracy is not money. This phase asks the bridge question: **does
that accuracy edge translate into economically-meaningful value once it is used to size a position
and charged realistic costs?**

The honest prior is genuinely two-sided. A more accurate variance forecast *should* size a
volatility-timing overlay better (the Fleming-Kirby-Ostdiek result that volatility timing has
economic value), so a positive result is plausible. But Phase 24 already showed that a real
*accuracy* signal (the confirmed baseline) resisted improvement, and the QLIKE gaps over the cheap
benchmarks, while significant, are modest — a real-but-tiny accuracy edge can be **economically
negligible after costs**. An honest null here ("the forecast is more accurate but not worth money net
of frictions") is an explicit, acceptable, and likely-modal outcome. A fabricated or overstated
economic edge is the one unacceptable outcome.

**The integrity problem this document exists to solve.** The program has **six confirmed directional
nulls** (price, macro, TA, microstructure, sentiment, meta-labeling) — there is no edge in predicting
the *sign* or *size* of returns. The confirmed skill is purely about the **second moment**
(volatility). An economic-value study is dangerous precisely because a volatility-timing overlay can
secretly become a directional / risk-premium bet (e.g. if calm days happen to be up days), smuggling
one of the six dead directional levers back in through the side door and crediting it to the variance
forecast. The whole design below is built to make that **structurally impossible**, and to **detect
it empirically** if it leaks anyway.

## The single question Phase 25 answers

For each symbol independently: when the **confirmed Phase-23 forecast** (reused *verbatim* — no
retrain, no re-tune) sizes a **capped, long-only** volatility-timing position with a **fixed,
non-directional** expected-return assumption held identical across every arm, does it —

1. control realized portfolio volatility **more tightly** than a no-timing baseline and every
   benchmark forecast (a **return-assumption-free** test), **and**
2. earn a **positive, ≥ +25 bps/yr, bootstrap-significant** performance fee over the no-timing
   baseline and a positive significant fee over **every** econometric benchmark, **net of realistic
   and 3×-stressed costs**, regime- and fold-stable, **and**
3. show **no directional-leakage contamination**

— on **both** MES and MNQ, per symbol, never pooled?

## What is IDENTICAL to Phase 23 (so this measures the confirmed forecast, not a new model)

Reused **byte-for-byte** from the frozen Phase-23 contract (`config/phase23_vol_h1.yaml`, `vf2`):

- **The forecasts themselves.** The per-day OOS forecast arrays for all six sources — treatment
  LightGBM, HAR, RW, EWMA(0.94), GARCH(1,1) — and the per-row `fold_id` are obtained by
  **deterministically re-running the frozen Phase-23 pipeline** (`run_h1.anchored_oos_h1` under
  `config/phase23_vol_h1.yaml`, seed 7) — **no retrain, no re-tune, no new model.** (Phase-23 persists
  only summary JSON, not the per-day forecast arrays; but the pipeline is bit-reproducible — one fixed
  config, no search — so regeneration is equivalent to a frozen artifact.) Following the Phase-24
  precedent, the implementation **MUST assert the regenerated FULL OOS frame reproduces byte-for-byte**
  — every `fcast_*` series, the per-row `fold_id`, `session_date`/`t0`, `rv`/`y`, the aligned
  `next_rth_log_return`, **and all per-arm OOS QLIKEs** (treat `0.246401`/`0.224083`; HAR `0.308`/`0.267`;
  RW `0.760`/`0.695`; EWMA `0.360`/`0.295`; GARCH `0.331`/`0.285`) plus the GARCH convergence
  diagnostics — via a frame **fingerprint**, and **fail closed** on any mismatch (treatment QLIKE alone
  is insufficient: a drift in a benchmark forecast, the fold ids, the GARCH fallback, or the return
  series could change the Phase-25 verdict while leaving treatment QLIKE unchanged). The regenerated OOS
  frame is **persisted** under `data/volatility/runs_ev/` (gitignored) so the fingerprint is auditable.
  The economic layer adds **zero fitted parameters** — it is a deterministic transform of the frozen
  forecasts under frozen knobs.
- **The RV target, the single fixed no-search LightGBM, the gated horizon h = 1, the anchored
  expanding walk-forward (OOS 2022-01-01 → end, 18 folds), the regime split** (causal trailing-22-day
  vs expanding-median), **the symbol set `{MES, MNQ}`** (per symbol, never pooled), and **seed 7** —
  all inherited from `Phase23Config.load().core`, the confirmed model with zero drift.
- **The realized return series.** P&L is earned on `realized.daily_rth_log_return` — the **RTH-internal
  open-to-close** log return on **exactly the RTH window the 5-minute RV target uses** (the audited
  Phase-23 return primitive). **Alignment (correctness-critical):** the weight is set at the close of
  forecast-row session `t` and earns the **NEXT** session's return `r_{t+1}`, so Phase 25 builds an
  explicit `next_rth_log_return` that maps each forecast row `session_date = t` to the RTH return of
  session `t+1` (a one-session **forward shift**), **dropping the last row** (no `t+1` exists). It does
  **NOT** use `run_h1.aligned_returns` directly — that returns the *same-session* `r_t` (correct for the
  GARCH conditional-variance filter, **wrong** for the held-period P&L; using it would multiply a
  close-set weight by an already-realized return). An **alignment unit test must prove** that forecast
  row `t` is scored only against return session `t+1` (no same-session / look-ahead leak), per symbol.

## What is NEW in Phase 25 — the economic layer (FIXED a priori)

The anchor is the **Fleming-Kirby-Ostdiek (2001, 2003) "economic value of volatility timing"**
performance-fee framework, with a **return-assumption-free volatility-targeting leg**
(Moreira-Muir 2017 / Harvey et al. 2018 mechanics) and **stationary-bootstrap (Politis-Romano 1994)
/ West-Edison-Cho (1993)** realized-utility differential testing.

### The six arms (one per forecast source, run once per symbol)

`TREAT` (LightGBM), `HAR`, `RW`, `EWMA`, `GARCH`, and `STATIC` (no-timing). For arm *k* on OOS day
*t* with the day-*t*-close log-variance forecast `f^k_t`: `σ²^k_t = exp(f^k_t)` (RTH-daily-RV variance
scale), `σ^k_t = sqrt(σ²^k_t)`. Two weight rules run in parallel on the **identical** arms and inputs:

- **Mean-variance weight (drives the FKO utility leg):**
  `w^k_t = clip( μ_bar / (γ · σ²^k_t), 0, w_cap )`.
- **Vol-target weight (drives the return-free leg, drift = 0):**
  `wv^k_t = clip( σ_target_daily / σ^k_t, w_floor, w_cap )`.

The day-*t*-close forecast sets the position for session *t+1*: a market-on-open order (rested
overnight — fully causal; the close-forecast cannot trade its own close) **enters `w` at *t+1*'s RTH
open** and a market-on-close order **exits to flat at *t+1*'s RTH close**. The overlay is therefore
**flat overnight** — pure intraday RTH-only exposure, so the P&L lives on **exactly** the RTH window
the RV forecast targets, with **zero overnight contamination** of either return or risk (the Phase-23
RTH-scale discipline carried into the P&L). It earns `r_{t+1} = daily_rth_log_return(t+1)`. Gross daily
return `= w · r_{t+1}`. Because the position is **opened and closed every session**, the cost is a
**full daily round trip on the held position** — `cost_t = 2 · c_side · w` (enter + exit) — **not** a
day-to-day turnover `c·|Δw|` (there is no overnight carry to net against; that would describe a
continuously-held position, which would also bear overnight P&L the RTH forecast never sees). **net**
`= gross − cost_t`. Realized quadratic utility `U^k_t = R − ½·γ/(1+γ)·R²` with `R = 1 + net` (FKO
realized-utility form, `rf = 0` excess-return convention). **Every arm sees the identical `r_{t+1}`,
`μ_bar`, `γ`, `σ_target`, `w_cap`, `w_floor` and cost model — arms differ ONLY through `σ²^k_t`.**

The daily round-trip cost is large in absolute terms but **common-mode**: every arm — including
`STATIC` — round-trips its position every session, so the bulk of the cost **cancels in the gated
`TREAT`-vs-benchmark differentials**, leaving the timing benefit net of only the *differential* traded
notional `2·c_side·(w^TREAT − w^k)`. This makes the flat-RTH-only model a fair differential test, not
a strawman that costs every arm into a null. To make the null-is-not-an-artifact-of-execution claim
explicit, a **realistic continuous-carry variant** — position held across nights, P&L on the
close-to-close return (overnight included), cost on turnover `c·|Δw|` — is **reported, not gated**, so
an outcome that would pass under realistic carry is surfaced for an operator re-scope rather than
buried.

### Frozen knobs (FIXED a priori; never tuned after seeing results)

| Knob | Value | Note |
|------|-------|------|
| Risk aversion `γ` | **5.0** (fixed) | FKO central value. `{2, 10}` reported as non-gated sensitivities — **never swept** (sweeping = multiplicity fishing). |
| Leverage cap `w_cap` | **1.5** (long-only) | Modest, realizable micro-futures overlay; identical across all arms incl. `STATIC`. Short side disallowed (`w ≥ 0`). |
| Vol-target floor `w_floor` | **0.1** | Vol-target leg never goes flat (no participation/sign decision). The mean-variance leg floors at 0. |
| Annualized vol target `σ_target` | **0.10** | Return-free leg only; identical across arms so the target level cancels in the cross-arm VTE comparison. Daily `= 0.10/√252`. |
| Expected return `μ_bar` | per-fold **training-mean** of `daily_rth_log_return` (rows strictly before the fold's OOS), frozen across the fold's OOS, **identical across all arms**; forced `≥ 0` | Disclosed fallback if a fold's training mean `≤ 0`: `μ_floor = 0.00012/day` (~3%/yr equity-premium proxy), identical across arms. **Drives only the FKO leg; the return-free leg uses drift = 0.** |
| Risk-free `rf` | **0** | Excess-return convention; identical across arms so it cancels in every differential. |
| Annualization | **252** RTH sessions/yr | |

### The no-timing baseline (`STATIC`)

The FKO unconditional / no-volatility-timing portfolio: a **constant** weight from a **causal**
unconditional variance `σ²_bar` = the mean realized daily RV over each fold's anchored-expanding
**training** rows (strictly before the fold's OOS), frozen across the fold's OOS days. Mean-variance
leg `w_static = clip(μ_bar/(γ·σ²_bar), 0, w_cap)`; vol-target leg
`wv_static = clip(σ_target_daily/√σ²_bar, w_floor, w_cap)`. Its *weight* steps only at fold boundaries
(no daily timing), **but under the gated flat-RTH-only model it still round-trips that constant weight
every session and pays `cost_t = 2·c_side·w_static` daily, exactly like every other arm** — no free
execution (which would unfairly handicap `TREAT`). It is the **toughest** economic yardstick (timing
must out-earn the *static* allocation net of the *same* per-day round-trip cost) and the literature's
canonical no-timing control. (The near-zero-cost characterization applies **only** to the reported
continuous-carry variant, where a constant weight implies ≈ 0 turnover.)

### Cost model (FIXED a priori)

Charged on traded notional as a **full daily round trip** on the held position (flat-RTH-only enters at
the open and exits at the close each session): `cost_t = 2 · c_side · w`, with the **per-side** fraction
`c_side = commission_fraction + half_spread_fraction`. Frozen inputs: commission **$0.62/contract/side**
(conservative IBKR retail, vs the ~$0.25 floor); spread **1 tick**, half-spread = 0.5 tick.

- **MES** — tick $1.25 → half-spread $0.625; per-side $1.245; **constant frozen reference notional**
  $5 × **5000** = $25,000 → `c_side(MES) = 4.98e-5` (≈ 0.50 bps/side; round trip ≈ 1.0 bps).
- **MNQ** — tick $0.50 → half-spread $0.25; per-side $0.87; reference notional $2 × **17500** =
  $35,000 → `c_side(MNQ) = 2.49e-5` (≈ 0.25 bps/side; round trip ≈ 0.5 bps).

The reference index levels (5000 / 17500) are **frozen at pre-registration** (≈ end-of-training
late-2021 levels) and held constant over the whole OOS, so **no realized OOS price leaks into the cost
fraction** (a deliberately conservative slight overstatement of cost fraction in the trending-up
years). **GATED cost levels: 1× and 3× base** (3× ≈ 1.5 bps MES / 0.75 bps MNQ — wider spreads,
partial fills, micro-contract granularity rounding). A **0.5×** optimistic level and a **5×** level
are **reported, never used to claim a pass**. Daily mark-to-notional cost is a reported robustness
check only.

### Significance test (FIXED a priori)

**Stationary bootstrap (Politis-Romano 1994)** on the **per-day** differential series, computed **per
symbol** (never pooled), **one-sided**, **B = 10,000** resamples, expected block length **L = 10**
trading days — all frozen. Applied to the per-day net realized-utility differential
`U^TREAT_t − U^bench_t` for each gated FKO comparison. A West-Edison-Cho / DM-style HAC asymptotic
p-value is **reported as a cross-check, never as the gate**. The return-free VTE legs use the frozen
**13/18 binomial sign test** for temporal stability (identical to Phase-23 G4).

### The two metric legs (BOTH required — a conjunction)

- **Leg 1 — volatility-targeting error (VTE), return-free.** `VTE^k = | log( realized_oos_vol(wv^k ·
  r) / σ_target_daily ) |` — how tightly the inverse-forecast overlay holds realized portfolio
  volatility to target. **No expected-return assumption enters** (the weight is `σ_target/σ`, drift 0),
  so a directional bet **cannot** pass it. Lower is better. A **per-fold** VTE is also computed for the
  temporal-stability gate.
- **Leg 2 — FKO performance fee (bps/yr), the money leg. Leverage-matched.** Because `1/σ²` is convex,
  an unmatched timing arm holds *more on average* than `STATIC` (Jensen: `w̄_treat ≥ w̄_static`), so a
  raw fee could be a larger-average-long bet harvesting the common equity risk premium rather than
  volatility-forecast value. To isolate *timing* (cross-day re-allocation), **every arm's mean-variance
  weight series is first scaled by a single causal per-fold-training constant so its training-window mean
  weight equals `STATIC`'s** (the Fleming-Kirby-Ostdiek / Moreira-Muir leverage-matching control), then
  re-clipped to `[0, w_cap]` (disclosed). **All gated money-leg metrics (G3–G8) use these matched
  weights;** the raw, un-matched fee is **reported** as a diagnostic. For each pairwise comparison
  (`TREAT` vs an arm *k*), the constant per-period fee `φ` solving `Σ_t U(R^TREAT_t − φ) = Σ_t U(R^k_t)`
  on **net-of-cost** returns; the economically-admissible root (smaller `|φ|`) is taken (disclosed; any
  non-convergence is a **FAIL**, never a silent re-spec); annualized `Δ_bps = φ · 252 · 1e4`. A positive
  `Δ_bps` means the γ-investor pays to switch to the treatment.

## Verdict gates — per symbol, ALL must clear; `E9` must not void (FIXED a priori)

| Gate | Threshold | Defends against |
|------|-----------|-----------------|
| **G1 — VTE intersection (return-free)** | `VTE_treat < VTE_static` **AND** `VTE_treat < VTE_bench` for **each** of {HAR, RW, EWMA, GARCH}. | An accuracy edge that doesn't even control second-moment risk; impossible to pass with a directional bet (no return enters). |
| **G2 — VTE temporal stability** | `treat` per-fold VTE `< static` per-fold VTE in **≥ 13 of 18** folds (one-sided 5% binomial sign test). | A single-period VTE win (e.g. the 2022 vol spike) masquerading as durable second-moment control. |
| **G3 — economic value vs `STATIC`** | `Δ_bps(TREAT vs STATIC) > +25 bps/yr` at **base** cost, `γ = 5`, stationary-bootstrap one-sided `p < 0.05`. | A statistically-real but economically-trivial fee; the +25 floor forces material value above costs; `STATIC` removes the just-being-invested risk-premium confound. |
| **G4 — economic value vs EVERY benchmark** | `Δ_bps(TREAT vs B) > 0` with bootstrap `p < 0.05` for **each** `B ∈ {HAR, RW, EWMA, GARCH}` (intersection). | Strawman-benchmark threat — the accuracy edge must out-earn the hardest variance forecasters (RW, GARCH) in money, net of cost. |
| **G5 — stressed-cost robustness** | `G3` and `G4` hold **in sign** with bootstrap `p < 0.05` at the **3×** cost level (the +25 floor in G3 may relax to strictly `> 0` at 3×; significance must hold). | Cost-assumption fragility / a high-turnover mirage that survives only at optimistic frictions. |
| **G6 — regime robustness** | `Δ_bps(TREAT vs STATIC) ≥ 0` in **both** the calm and turbulent OOS sub-samples (sign-consistent, base cost, `γ = 5`, frozen Phase-23 regime labels). | Regime concentration — value in turbulent periods but destroyed in calm, i.e. a disguised volatility-*level* bet. |
| **G7 — temporal stability of value** | `TREAT` beats `STATIC` on net realized utility in **≥ 13 of 18** folds **AND** beats `RW` in **≥ 13 of 18** folds. | A fee earned entirely in one fold/regime masquerading as durable economic value. |
| **G8 — exposure neutrality & cost-erosion** | Leverage-matching held: post-match `|w̄_treat − w̄_static| ≤ 0.05 · w̄_static` (the gated fee is pure timing, not a bigger-average-position bet) **AND** the `TREAT`-vs-`STATIC` **net** fee at 3× cost retains `≥ 50%` of its **gross** (pre-cost) fee (costs don't eat the timing benefit). | A "win" that is really a larger-average-long risk-premium grab — closed jointly with E9 (matched mean exposure **and** near-zero active-exposure correlation) — or an edge mostly consumed by churn. |
| **E9 — directional-leakage VOID gate** | For **each** gated comparison arm `k ∈ {STATIC, HAR, RW, EWMA, GARCH}`: `|corr(w^TREAT_t − w^k_t, r_{t+1})| < 0.10` — the **active** exposure that drives that money differential (mean-variance weight leg). Per-arm absolute `|corr(w_t, r_{t+1})|` reported as a diagnostic. **If any active-exposure correlation ≥ 0.10, the verdict is VOIDED**, regardless of G1–G8. | The active overweight secretly being a directional / risk-premium-timing bet (a low absolute-weight correlation can still hide a directionally-correlated overweight) — the program's refusal to smuggle back the six directional nulls. |

Calibration (Mincer-Zarnowitz, restated from Phase-23), the `γ ∈ {2, 10}` fees, the 0.5×/5× cost
levels, Sharpe / Sortino / Calmar / certainty-equivalent / max-drawdown, turnover and gross-vs-net
drag, the realized-weight distribution, the `exp(r)−1` simple-return re-run, the daily
mark-to-notional and fixed-equity-premium-`μ` robustness variants, and `corr(w_t, r_{t+1})` per arm
are **reported, not gated**.

## How the differential is isolated from a directional bet (the integrity core)

Four **structural** locks make any economic differential attributable **only** to the variance
forecast, plus one **empirical** backstop:

1. `μ_bar` and `γ` are **single fixed causal scalars identical across all six arms**, frozen from
   training data; the return-free leg uses **drift = 0**. `μ_bar` carries **zero day-to-day or
   directional information** — the same constant enters every arm's weight numerator every day, so the
   cross-arm difference in position is a deterministic function of the two variance forecasts alone.
   *(Note: because the weight cap and the quadratic utility are non-linear, `μ_bar` does **not** cancel
   the differential exactly — it is **not** assumed to. Its value injects no timing signal, which is
   the isolation claim; the verdict's near-invariance to the `μ_bar` specification is **verified by the
   reported fixed-equity-premium variant**, not asserted.)*
2. **Long-only clamp** `[0, w_cap]` (and `w_floor > 0` on the vol-target leg): no day can short or flip
   sign — the weight only **resizes a fixed-direction long exposure** (pure second-moment timing).
   Moreover the gated money leg is **leverage-matched** (every arm rescaled to `STATIC`'s causal
   training-mean weight), so the treatment cannot win by holding a larger *average* long position (the
   Jensen `w̄_treat ≥ w̄_static` risk-premium confound) — only by *re-allocating* exposure across days.
3. The **identical realized `r_{t+1}`** feeds every arm, so the cross-arm P&L difference is exactly
   `(w^TREAT_t − w^bench_t) · r_{t+1}` — a function of the two variance forecasts and the common return.
4. The **return-free VTE leg** (G1/G2) consumes **no return assumption whatsoever**; a directional bet
   cannot pass it, and it is a **co-required** part of every PASS.
5. **Empirical backstop `E9`:** the verdict is **voided** if the **active exposure** behind any gated
   differential is directionally correlated — `|corr(w^TREAT_t − w^k_t, r_{t+1})| ≥ 0.10` for any
   `k ∈ {STATIC, HAR, RW, EWMA, GARCH}` — catching any residual directional channel in the *overweight*
   (not just the absolute weight) even if the structural locks were somehow circumvented.

A positive result therefore means the treatment's variance forecast **sized exposure better** (more
when variance was truly low, less when truly high) — variance-forecast value, **not** return timing.

## Decision rule (FIXED a priori)

- **Per symbol: PASS iff `G1 ∧ G2 ∧ G3 ∧ G4 ∧ G5 ∧ G6 ∧ G7 ∧ G8` all clear AND `E9` is not voided**,
  at the headline `γ = 5` and base cost (with the pre-registered stress / regime / fold robustness as
  specified).
- **Both symbols PASS** → **"confirmed economic value of the 1-day volatility forecast (offline /
  simulated / paper, net of realistic and stressed costs)."** This authorizes, at most, a deliberate
  operator decision to scope a **separate, future, pre-registered** paper-trading-prototype (or an
  options / variance-swap valuation) design that must pass its own gates. **It never authorizes live
  trading.**
- **Exactly one symbol PASS** → flagged **fragile**; nothing promoted.
- **Neither / any gate fails on both / any VOID** → **honest null**: the confirmed accuracy edge does
  **not** translate into economically-meaningful, cost-robust, direction-free value. Recorded in
  `docs/RESEARCH_VERDICTS.md` as an explicitly **acceptable, likely-modal** outcome; the economic-value
  lever is then re-scoped only by a deliberate operator decision, not auto-re-litigated.

## Anti-snooping commitments (FIXED)

- The framework (FKO performance fee + return-free VTE leg), the six arms, the two weight rules, every
  frozen knob (`γ = 5`, `w_cap = 1.5`, `w_floor = 0.1`, `σ_target = 0.10`, the `μ_bar` rule and its
  `≥ 0` fallback, `rf = 0`, the exact cost numbers and the constant frozen reference notional), the
  gated 1×/3× cost levels, the stationary-bootstrap parameters (`B = 10,000`, `L = 10`), the +25 bps
  minimum-effect floor, the nine gates and their thresholds (including 13/18), and the RTH-internal
  return alignment are all fixed **before** any Phase-25 run and are **not** changed after seeing
  results.
- **No `γ`-sweeping, no metric-fishing, no threshold-tuning, no cost-number shopping, no
  framing-swapping** to manufacture a pass. Each arm runs **once** per symbol. Every gate is a
  conjunction the treatment must **all** pass — there is no "beat at least one" path; the secondary
  `γ`s and cost levels are reported-not-gated and add **no** pass-paths.
- The Phase-23 forecasts are reused **verbatim** (deterministically regenerated under the frozen
  Phase-23 config + seed, fingerprint-verified, never refit); the economic layer adds **zero fitted
  parameters**. Disclosed-not-silent fallbacks only: the `μ_bar ≤ 0 → μ_floor` substitution and
  the fee-root non-convergence FAIL.
- The canonical `verdict.json` is defined over **exactly `{MES, MNQ}`, per symbol, never pooled**; a
  subset / superset run **refuses to save** the canonical decision (the Phase-20/23 guard).
- If the pipeline must change for a legitimate engineering reason, the `econvalue_version` is bumped
  and the change is documented in `docs/DECISIONS.md`.

## Versioning & artifacts

- `econvalue_version: "ev1"`. Frozen knobs: `config/phase25_econ.yaml` (this commit); the modeling
  core is **inherited verbatim** from `config/phase23_vol_h1.yaml` (`vf2`).
- Per-symbol summaries → `data/volatility/runs_ev/<symbol>.json`; combined verdict →
  `data/volatility/runs_ev/verdict.json` (gitignored; regenerate by re-running the module). MLflow
  experiment `volatility-econvalue-ev1`.

## What this document is NOT

Not an implementation and not a run. It commits zero simulations and zero results. It writes no
execution / risk / live-loop code, changes nothing about the paper-only lock, and deploys nothing. The
Phase-25 implementation will reference this frozen contract and must not deviate from it. Economic
value in this offline simulation — like forecast skill before it — is **not** a trading authorization.
