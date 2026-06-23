# Phase 25 Pre-Registration — Economic-Value Study of the Confirmed 1-Day RV Forecast (Volatility Targeting)

**Committed before any Phase 25 backtest is run or any forecast reaches a P&L simulator.** The commit
date of this file is the pre-registration timestamp.

Nothing here authorizes strategy, risk, execution, or **live trading** — regardless of outcome. Phase
25 produces an **offline / simulated / paper economic-value verdict only**: does the *forecast-accuracy*
edge confirmed in Phase 23 translate into **economic value (money) under realistic costs**, when the
forecast is used purely as a **volatility-targeting** signal (constant-vol scaling), per symbol? The
system stays hard-locked to paper. Reads only the local lakes and the Phase-23 OOS forecast artifacts —
no Databento, no IBKR, no network, no spend (`OPTIONS_DATABENTO_SPEND_OK` stays unset).

## Why this study, and the honest prior

Phase 23 **confirmed** a 1-day realized-volatility forecast skill on both symbols (a single fixed
no-search LightGBM beats HAR-RV, random-walk, EWMA(0.94) and GARCH(1,1), regime- and fold-robust). That
is an **accuracy** result. The natural and only-authorized next question (per `docs/RESEARCH_VERDICTS.md`,
the leading fork) is the **bridge from accuracy to money**: a more accurate variance forecast should let
a constant-volatility-targeting overlay hold its risk budget *more tightly* and earn a *better
risk-adjusted return per unit of friction* than the same overlay driven by the benchmark forecasts or by
no volatility timing at all. An honest null — "the accuracy edge is real but too small to beat costs, or
does not improve vol-targeting tightness enough to matter" — is an explicit, acceptable, **modal** success
outcome (the program has six directional nulls and one positive accuracy verdict, and is proud of all of
them). A fabricated or overstated economic edge is the **one** unacceptable outcome. This design is built
to fool itself as little as possible and to expose its own failure modes.

## The academic framework we anchor on (FIXED a priori)

**Volatility targeting / constant-volatility scaling**, in the **Moreira–Muir (2017) "Volatility-Managed
Portfolios"** family and the broader vol-targeting literature (Fleming–Kirby–Ostdiek 2001/2003 on the
economic value of volatility timing; Harvey et al. 2018 on the mechanics of vol targeting). We deliberately
choose constant-vol scaling over a mean-variance / utility framing because it is **simpler, more robust,
and — critically — does not require an expected-return input to express the core test**. The position is

  `w_t = sigma_target / sigma_forecast_t`  (capped),

a *long-only*, leverage-bounded scaling of a single futures contract's RTH return. The variance forecast
enters in exactly one place — the denominator `sigma_forecast_t`. Every arm shares the identical
numerator (`sigma_target`), the identical cap, the identical underlying return stream, and (where a drift
is needed at all) the identical fixed drift. **The only thing that differs between arms is the variance
forecast**, so any difference in the two reported quantities is attributable solely to variance-forecast
quality.

The forecast quality is allowed to show up **two ways**, and we gate the cleaner one harder:

- **(a) Volatility-targeting accuracy** — how tightly the realized vol of the scaled position tracks
  `sigma_target`. This is a **pure variance-forecast-quality metric that needs NO return assumption
  whatsoever** (it is a statement about second moments only). It is the **primary, return-assumption-free
  gate**.
- **(b) Net realized risk-adjusted return** — the realized **net** Sharpe (after the pre-registered cost
  model) of the vol-targeted position, treatment vs benchmark and vs a no-timing baseline. This is the
  literal "money" leg, but it unavoidably touches a return stream, so it is held to a **looser,
  no-harm-style gate** and several of its components are reported-not-gated.

## The single question Phase 25 answers

For each symbol in EXACTLY `{MES, MNQ}`, per symbol, never pooled: **when the confirmed Phase-23 1-day RV
forecast is used as the denominator of a capped constant-volatility-targeting overlay, does it (a) hold
realized portfolio volatility closer to the target than every benchmark forecast and than a no-timing
baseline (return-assumption-free), and (b) deliver a net Sharpe that is no worse than — and ideally better
than — the benchmark- and no-timing overlays under a realistic, pre-registered, stressed cost model?**

## What is REUSED VERBATIM from Phase 23 (so no new modeling, no snooping surface)

The entire Phase-23 modeling core is **reused byte-for-byte** — **no retraining, no re-tuning, no new
forecasts are produced.** Phase 25 consumes the *already-computed* Phase-23 per-day OOS forecasts:

- **The five per-day OOS forecasts** (log-variance scale), aligned 1:1 on OOS days, for: treatment
  LightGBM (`fcast_treat`), HAR-RV (`fcast_har`), random-walk (`fcast_rw`), EWMA(0.94) (`fcast_ewma`),
  GARCH(1,1) (`fcast_garch`). Loaded from the Phase-23 OOS frame (`anchored_oos_h1`), **not regenerated**.
- **The realized daily RV series** `rv_t` (strictly positive; the forecast target's realization).
- **`daily_rth_log_return()`** — the within-session RTH open-to-close log return per day, computed on
  **exactly the RTH window the RV target uses** (it **excludes the overnight/weekend gap**). This is the
  return the P&L earns; see "Return-series alignment" below for why this is the only correct choice.
- **The 18 anchored expanding walk-forward fold ids**, the **causal regime labels** (calm/turbulent via
  trailing-22-day mean RV vs the causal expanding-window median), and the **t0/t1 decision timestamps**.

The Phase-23 frozen knobs (`config/phase23_vol_h1.yaml`, `vf2`) are inherited unchanged. Phase 25 adds
**only** the economic-overlay knobs below.

## The mechanism — forecast → position → P&L (FIXED a priori, exact)

All quantities are on the **variance / RTH-internal-return** scale used by the Phase-23 target, so they
are mutually consistent (the Phase-23 gotcha: the GARCH benchmark had to use RTH-internal returns because
close-to-close returns include the overnight gap and mis-scale relative to the RTH-only RV — the *same*
alignment subtlety governs the P&L return here).

For each arm `a ∈ {treat, har, rw, ewma, garch, notiming}` and each OOS day `t`:

1. **Forecast → forecast vol.** The arm's day-`t` log-variance forecast `f^a_t` (the next-day RV forecast,
   on **daily** RV scale) becomes a daily volatility forecast
   `sigma^a_t = sqrt( exp( f^a_t ) )`. (For `notiming`, see the static baseline below.)
2. **Position (capped, long-only).** The day-`t` close sets the position held over session `t+1`:
   `w^a_{t+1} = clip( sigma_target / sigma^a_t , w_floor , w_cap )`,
   with `sigma_target`, `w_floor`, `w_cap` fixed a priori (below). `w` is a *unitless leverage on the RTH
   return*; it is **always ≥ w_floor > 0** (long-only — no sign decision is ever taken; see "Isolation
   from direction").
3. **Gross P&L.** The position earns the next session's RTH-internal return:
   `r^a_{t+1,gross} = w^a_{t+1} * ( ret_{t+1} - drift )`,
   where `ret_{t+1} = daily_rth_log_return` of session `t+1`, and `drift` is the **fixed, causal,
   non-predictive** constant defined in "Expected-return handling" (identical across all arms; default 0).
4. **Turnover & cost.** Day-over-day the position changes by `|w^a_{t+1} - w^a_t|`; rebalancing that change
   costs (per the cost model below)
   `cost^a_{t+1} = |w^a_{t+1} - w^a_t| * c`,  with `c` the per-unit-notional round-trip-fraction cost.
   The position is opened/closed in *notional* terms (the overlay scales exposure to one micro contract's
   index notional), so turnover is measured in units of that notional and `c` is a fraction of notional.
5. **Net P&L.** `r^a_{t+1,net} = r^a_{t+1,gross} - cost^a_{t+1}`.

The realized **net portfolio return series** `{ r^a_{t,net} }` over all OOS days is the object both gates
read. (A leading `t` with no prior `w` uses `w_0 = sigma_target / sigma^a` at the first OOS day for the
turnover of the *next* step; the first day's turnover is measured against `w_floor`'s neutral start — a
disclosed, arm-identical convention so it cannot favor any arm.)

## Expected-return handling — how NO directional prediction is smuggled in (FIXED a priori)

This is the load-bearing integrity control. The program has **six** honest nulls at forecasting return
**direction**; there is **no** confirmed directional edge. The economic study must measure the value of the
**variance** forecast in isolation.

- **`drift` is a single fixed, causal, NON-predictive constant, identical across every arm and every day.**
  It is NOT a function of any forecast, any feature, any date, or any direction signal.
- **Default value: `drift = 0`.** With zero drift the position is a pure long-only vol-scaled exposure;
  the *only* thing that moves P&L differentially across arms is the scaling `w`, which depends *only* on the
  variance forecast. This makes the variance-forecast attribution airtight.
- **Disclosed robustness variant (reported, not gated): `drift = mu_train`**, the **in-sample training-window
  mean** of `ret` (estimated per walk-forward fold on that fold's *training* days only, frozen for the fold's
  OOS days — never using OOS returns). This is causal and identical across arms; it exists only to show the
  vol-targeting-error gate is unchanged by the drift choice (it is, mechanically — see below). It is **not**
  a directional prediction: it is a single scalar reused for every day in the fold, so it can never time
  entries or sign exposure.
- **No per-day, per-regime, or forecast-conditioned expected return is ever used.** Any such term would be a
  directional bet and is forbidden by this contract.

## Return-series alignment — the RTH-internal vs close-to-close resolution (FIXED a priori)

The day-`t` forecast sets a position **held over session `t+1`**, and the P&L it earns must be on the **same
scale as the RV the forecast targets**. The Phase-23 RV target and `sigma_target` are **RTH-internal**
(open-to-close, no overnight gap). Therefore:

- **P&L uses `daily_rth_log_return` of session `t+1` — the RTH-internal open-to-close return — NOT a
  close-to-close return.** A close-to-close return would inject the overnight/weekend gap, which the variance
  forecast never sees and `sigma_target` never budgets for; using it would mis-scale the realized portfolio
  vol relative to the target and **corrupt the vol-targeting-error metric** (the exact mistake the Phase-23
  GARCH benchmark had to avoid). The holding period is thus the RTH session `t+1`; the overnight gap is
  explicitly **out of scope** for this overlay (and would, if anything, only add un-forecast noise).
- Annualization (for Sharpe and for `sigma_target`) uses **252 RTH sessions/year**, consistent with the
  daily RTH series. `sigma_target` is specified as an **annualized** number and converted to a daily RTH vol
  by `sigma_target_daily = sigma_target_annual / sqrt(252)`.

## The "no volatility timing" static baseline (FIXED a priori, causal)

`notiming` is the "does volatility timing help at all?" control: a constant-weight overlay that targets the
**same** `sigma_target` using the **unconditional** variance instead of a daily forecast.

- For each OOS day `t`, `sigma^{notiming}_t = sqrt( exp( bar_f_t ) )`, where `bar_f_t` is the **causal
  expanding-window mean of log-RV** using only RV up to and including day `t` (the same causal,
  no-look-ahead construction Phase 23 uses for its expanding-median regime split). This yields a slowly
  drifting, **non-forecasting** constant-ish weight `w^{notiming}_{t+1} = clip(sigma_target/sigma^{notiming}_t,...)`
  — it sizes to long-run vol, it does **not** time day-to-day vol.
- This is the literature's "no volatility timing" benchmark (constant unconditional-variance weight). Beating
  it is the test that *daily vol timing* (any forecast) adds value; the treatment must also beat the *other
  forecasts'* timing, which is the test that the *confirmed* forecast's superior accuracy adds value over
  inferior forecasts.

## Instruments & the transaction-cost model (FIXED a priori, exact numbers)

- **MES (Micro E-mini S&P 500):** \$5 / index point; tick 0.25 pt = **\$1.25 / tick**. Notional per contract
  ≈ `5 * S&P_index_level`.
- **MNQ (Micro E-mini Nasdaq-100):** \$2 / index point; tick 0.25 pt = **\$0.50 / tick**. Notional per
  contract ≈ `2 * Nasdaq_index_level`.

The overlay turns over notional as `w` changes day to day. Cost is charged as a **fraction of the notional
turned over**: `cost = |Δw| * c`, with `c` the per-unit-notional round-trip cost fraction =
`(commission_fraction + spread_fraction)`, computed per symbol from the realistic IBKR-paper retail
frictions:

- **Commission (round trip, both sides):** IBKR micro futures ≈ **\$0.50 / contract / side** (within the
  stated \$0.25–0.62 band; the conservative-mid value is fixed a priori) → **\$1.00 round trip / contract**.
- **Spread (round trip):** typical **1 tick bid-ask** = pay half-spread on entry and half on exit = **1 full
  tick round trip** = \$1.25 (MES) / \$0.50 (MNQ) per contract.
- **Per-symbol round-trip dollar cost / contract:** MES = `1.00 + 1.25` = **\$2.25**; MNQ = `1.00 + 0.50` =
  **\$1.50**.
- **As a fraction of notional** (the `c` used in `cost = |Δw|*c`): `c = round_trip_dollar_cost / notional`.
  Notional is computed from a **fixed, pre-registered reference index level** per symbol (the OOS-window
  *median* RTH close of that symbol, computed once from the local lake and frozen into the YAML — a constant,
  so it cannot be tuned per-result and cannot leak), giving:
  - **MES:** with reference S&P ≈ 4500 → notional ≈ `5*4500` = \$22,500 → `c_MES ≈ 2.25 / 22,500 ≈ 1.0e-4`
    (**1.0 bp of notional per unit turnover**, baseline).
  - **MNQ:** with reference Nasdaq ≈ 15,500 → notional ≈ `2*15,500` = \$31,000 → `c_MNQ ≈ 1.50 / 31,000 ≈
    0.48e-4` (**≈ 0.5 bp of notional per unit turnover**, baseline).
  - The exact reference levels are frozen at implementation time from the lake median and recorded in the
    YAML and the results doc; the *formula* `c = round_trip_$ / (multiplier * ref_index_level)` is fixed
    here. (Disclosed simplification: a constant reference notional avoids smuggling a price path into the
    cost; the alternative — daily mark-to-notional — is a reported robustness check, never a gate.)
- **Stressed cost levels (FIXED a priori): the gates must hold at `c` AND at `3*c` AND at `5*c`.** The 3×
  level absorbs wider spreads / worse fills / partial-fill slippage; the 5× level is a deliberate
  break-it-stress. (A `0.5×` optimistic level is reported but **never** used to claim a pass.) A net-Sharpe
  edge that survives only at `1×` is not robust and does not pass the money gate.

## Leverage / weight cap (FIXED a priori)

- **`sigma_target = 0.10` annualized (10% annual RTH vol)** — a conventional, modest vol-targeting budget,
  fixed a priori (not tuned to either symbol). Converted to daily: `0.10 / sqrt(252) ≈ 0.0063` daily RTH vol.
- **`w_cap = 2.0`** (max 2× the per-contract RTH exposure) — a hard pre-registered leverage cap; it binds
  when `sigma^a_t` is very low (calm days), preventing the overlay from manufacturing return by levering into
  quiet markets and capping tail risk. **`w_floor = 0.1`** (never fully flat; keeps the overlay long-only and
  the turnover/cost comparison meaningful). The cap/floor are **identical across every arm**, so they cannot
  advantage the treatment.
- Rationale: vol targeting with a leverage cap is the standard, robust Moreira–Muir / Harvey-et-al
  implementation; an uncapped overlay is known to produce spurious Sharpe by levering calm periods, exactly a
  self-fooling mode (see below).

## Metrics (FIXED a priori, exact formulas)

Let `{w_t}` be an arm's daily weights, `{r_t,net}` its net daily RTH portfolio returns over the `N` OOS days,
and `sigma_tgt_d = sigma_target/sqrt(252)` the daily target vol.

### Primary, return-assumption-FREE: Volatility-Targeting Error (VTE)

The realized vol of the **gross, drift-free** scaled position vs the target — a pure second-moment statement:
let `p_t = w_t * ret_t` (gross, drift-free, RTH-internal). Define
  **`VTE = | log( realized_vol(p) / sigma_tgt_d ) |`**,  where `realized_vol(p) = std(p_t)` over the OOS days
(annualization cancels inside the ratio, so VTE is scale-free and computed on the daily series). Lower is
better; 0 means the overlay held the target exactly. **A per-fold VTE** is also computed (std within each of
the 18 folds) for the temporal-stability gate. VTE uses **no drift and no cost** — it is purely "did
`1/sigma_forecast` deliver constant vol", the cleanest possible variance-forecast-quality readout.

### Money leg: net Sharpe

  **`Sharpe^a = sqrt(252) * mean(r^a_{t,net}) / std(r^a_{t,net})`** over the OOS days (annualized), computed
at each cost level `c`, `3c`, `5c`. Differential `ΔSharpe = Sharpe^treat − Sharpe^benchmark` per benchmark
and vs `notiming`. (With `drift=0` the *level* of Sharpe is driven by the underlying RTH drift common to all
arms; the **differential** across arms is driven only by the variance-forecast scaling — see isolation
argument. We gate the differential, not the level.)

## Primary economic-value metric (the single headline)

**The annualized net Sharpe-ratio differential of the treatment-driven overlay over the no-timing baseline,
at the baseline cost level**, `ΔSharpe_{treat vs notiming}@1c`, reported per symbol — together with the
**VTE ratio** `VTE_treat / VTE_notiming`. The VTE leg is the *cleaner* headline; the Sharpe differential is
the *money* headline. Both are reported; the verdict gates below weight VTE more heavily.

## Secondary metrics (reported, NOT gated)

- Realized annualized vol of each arm's position (level, not just VTE).
- Net Sharpe **level** of every arm at every cost level.
- Per-arm annualized turnover `mean(|Δw|) * 252` and total cost drag (bps/yr).
- Max drawdown and Sortino of each arm's net series.
- VTE and Sharpe in the calm vs turbulent regime sub-samples (the Phase-23 split).
- Mincer–Zarnowitz-style calibration of `sigma_forecast` vs realized (already in Phase 23; restated).
- The `drift = mu_train` robustness variant's Sharpe differentials.
- The daily-mark-to-notional cost robustness variant.

## Verdict gates — per symbol, at h = 1 (FIXED a priori)

| Gate | Threshold | Defends against |
|------|-----------|-----------------|
| **G1 — VTE vs no-timing (PRIMARY, return-free)** | Treatment VTE **strictly below** the `notiming` baseline's VTE, AND treatment VTE strictly below **each** benchmark forecast's VTE {HAR, RW, EWMA, GARCH}. (Intersection test; the daily vol forecast must hold the target tighter than no-timing AND tighter than every rival forecast.) | A directionless, return-free test that the *confirmed-better* forecast actually controls vol better. Cannot be gamed by any return assumption. |
| **G2 — VTE temporal stability** | Treatment per-fold VTE below the `notiming` per-fold VTE in **≥ 13 of the 18** walk-forward folds (one-sided 5% binomial sign-test threshold, identical to Phase 23/24). | A single-regime VTE win (e.g. only the 2022 vol spike). |
| **G3 — net Sharpe no-harm under stress** | Treatment net Sharpe **≥** `notiming` net Sharpe (`ΔSharpe ≥ 0`) at **all three** cost levels {`c`, `3c`, `5c`}, AND treatment net Sharpe ≥ **each** benchmark forecast's net Sharpe at the baseline cost `c`. (No-harm form: the money leg must not be *worse* than no-timing even under 5× costs.) | Claiming a money edge that exists only at unrealistically low costs, or that a tighter vol target was bought with worse risk-adjusted return. |
| **G4 — turnover/cost sanity** | Treatment annualized cost drag at `3c` **< 25%** of its gross annualized return magnitude (the overlay is not a cost-churn machine), AND treatment turnover is within **2×** of the `notiming` turnover (the edge is not just "trade far more"). | An apparent edge manufactured by excessive rebalancing that realistic costs would erase, or a turnover-driven artifact. |

Calibration, the regime sub-sample breakdowns, the `0.5×` optimistic cost level, the h=5/h=22 diagnostic
overlays, and the `drift=mu_train` variant are **reported, not gated.**

## Decision rule (FIXED a priori)

- **Per symbol: ECONOMIC VALUE iff `G1 ∧ G2 ∧ G3 ∧ G4` all clear at h = 1.** The primary, return-free VTE
  gates (G1, G2) are the spine; the money gates (G3, G4) are deliberately *no-harm* (≥, not strictly >) so a
  PASS cannot be manufactured by an over-levered return grab.
- **Both symbols PASS** → **"confirmed economic value of the 1-day RV forecast as a volatility-targeting
  signal (offline, simulated, paper, costed)."** This authorizes **only** (a) documenting the result and (b)
  designing a *separate, future* pre-registered paper-trading study (and, much later, the Phase-2 options
  framing). **It NEVER authorizes live trading, nor even an automated paper-execution deployment** — those
  require their own pre-registered risk/execution gates. Economic value in a costed simulation is still not a
  trading authorization.
- **Exactly one symbol passes** → flagged **fragile**; recorded, nothing promoted.
- **Either/both fail** → an **honest null**: the confirmed *accuracy* edge does **not** translate into costed
  economic value as a vol-targeting overlay. Recorded in `docs/RESEARCH_VERDICTS.md`; the lever is re-scoped
  only by a deliberate operator decision (e.g. a different economic framing), never auto-re-litigated.

## Isolation from direction — why the differential measures variance-forecast value, NOT a directional bet

- The position is **always long-only** (`w ≥ w_floor > 0`): **no arm ever takes a sign/direction decision.**
  There is no entry/exit timing, no sign forecast, no conditioning of *whether* to be in the market.
- The **only** input that differs across arms is the variance forecast in the denominator `sigma^a_t`. The
  numerator (`sigma_target`), the cap/floor, the underlying RTH return stream `ret_{t+1}`, and the drift are
  **byte-identical** across `{treat, har, rw, ewma, garch, notiming}`.
- With `drift = 0`, each arm's P&L is `w^a_{t+1} * ret_{t+1}`; the cross-arm difference is
  `(w^{treat}_{t+1} - w^{bench}_{t+1}) * ret_{t+1}`, a function of nothing but the two variance forecasts and
  the (common) return. Any economic differential is therefore **mechanically attributable solely to the
  variance forecast.**
- The **VTE gate uses no return assumption at all** — it is a pure statement about whether `1/sigma_forecast`
  produces constant second moments. It is impossible for a directional bet to pass G1/G2, because direction
  never enters them.
- Consequently a PASS here is evidence that the *variance forecast* has economic value, and is **not** a
  return-timing bet relabeled as vol timing.

## Self-fooling modes and baked-in mitigations

| Mode | Mitigation (baked into this contract) |
|------|----------------------------------------|
| **Smuggling a directional bet** (sizing/sign reacts to a return signal). | Long-only `w ≥ w_floor`; `drift` fixed/causal/non-predictive and arm-identical; VTE gate is return-free; differential is mechanically variance-only (see isolation section). |
| **Leverage-grab Sharpe** (uncapped overlay levers calm periods to fake Sharpe). | Hard `w_cap = 2.0`, identical across arms; G3 is *no-harm* (≥ notiming), not "beat by levering"; G4 caps turnover within 2× of no-timing. |
| **Cost too optimistic** (edge vanishes with realistic frictions). | Gates must hold at `c`, `3c`, **and** `5c`; the `0.5×` optimistic level is reported but never gates; cost numbers fixed a priori from IBKR-paper retail frictions. |
| **Snooping / tuning after seeing results** (pick the framing/target/threshold that passes). | Reuse Phase-23 forecasts **verbatim** (no retrain); every overlay knob frozen in YAML before any P&L is computed; each arm runs once per symbol; per-symbol over exactly {MES,MNQ}, never pooled. |
| **Wrong return scale** (close-to-close return mis-scales vs RTH RV → fake VTE). | P&L uses `daily_rth_log_return` (RTH-internal, gap-excluded) exactly as the RV target; 252-RTH annualization; the Phase-23 gotcha explicitly carried forward. |
| **Strawman comparison** (beat only the easy no-timing baseline). | G1/G3 are intersection tests: treatment must beat no-timing AND each of {HAR, RW, EWMA, GARCH}. |
| **Single-regime artifact** (whole edge from one period). | G2 per-fold sign test (≥13/18); regime sub-sample reporting. |
| **Turnover/cost-churn artifact** (apparent edge is rebalancing noise costs would kill). | G4 turnover & cost-drag sanity; net (post-cost) series used in every money metric. |
| **Multiple-comparisons inflation** (many arms × many cost levels → something passes by chance). | Verdict requires the *intersection* of all four gates per symbol on *both* symbols; the worst-case cost level governs G3; no metric-fishing — VTE is the pre-declared spine. |
| **Calibration drift faking VTE** (a biased-but-tight forecast). | MZ calibration reported; VTE measured against the *target*, and the no-timing baseline shares the same target, isolating tightness from level. |

## What a PASS does and does NOT authorize

- **DOES:** record "confirmed economic value of the 1-day RV forecast as a costed, simulated,
  paper-only volatility-targeting overlay, per symbol"; authorize *designing* a separate, future,
  pre-registered **paper-trading** validation (and eventually the Phase-2 options framing) of the same
  overlay.
- **DOES NOT:** authorize live trading. Does **not** authorize an automated paper-execution deployment, a
  risk-manager wiring, or any capital. Does **not** promote a "model" into the live engine (the live engine
  stays empty). Economic value in an offline simulation is **not** a trading authorization — that requires
  its own pre-registered risk/execution gates. The system stays hard-locked to paper.

## Anti-snooping commitments (FIXED)

- `sigma_target = 0.10`, `w_cap = 2.0`, `w_floor = 0.1`, the cost numbers (commission \$0.50/side, 1-tick
  spread, the per-symbol round-trip dollars, the reference-notional formula), the stressed multipliers
  {1×, 3×, 5×} (and the non-gating 0.5×), the VTE and Sharpe formulas, the four gates and their thresholds
  (including 13/18), the long-only `drift=0` default and the `mu_train` reported variant, the no-timing
  baseline construction, the 252-RTH annualization, and the reuse-verbatim of Phase-23 forecasts are **all
  fixed before any Phase-25 P&L is computed** and are **not** changed after seeing results.
- **No knob-tuning, no cost-level fishing, no framing-swapping, no target-vol fishing** to manufacture a
  pass. Each arm runs once per symbol; the canonical verdict is over exactly {MES, MNQ}.
- Disclosed-not-silent conventions only: the first-day turnover convention, the constant reference notional,
  and the `mu_train` robustness variant (all reported).
- If the pipeline must change for a legitimate engineering reason, `econ_version` is bumped and the change is
  documented in `docs/DECISIONS.md`.

## Versioning & artifacts

- `econ_version: "ev1"` (Phase 25). Frozen knobs: `config/phase25_econ.yaml` (this commit). The forecasting
  core is inherited verbatim from `config/phase23_vol_h1.yaml` (`vf2`).
- Per-symbol summaries → `data/volatility/runs_ev/<symbol>.json`; combined verdict →
  `data/volatility/runs_ev/verdict.json` (gitignored). MLflow experiment `volatility-econ-ev1`.

## What this document is NOT

Not an implementation and not a backtest run. It commits zero P&L and zero models. The Phase 25
implementation will reference this frozen contract and must not deviate from it.
