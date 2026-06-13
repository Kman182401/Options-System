# Research Verdicts — Options-System

**Status as of Phase 20 (2026-06-13): no promoted model. No strategy, backtest, or
live trading is authorized.** Five feature/model levers (price 5, macro 6, TA 10,
microstructure 14, sentiment 19) were tested through the *same* fixed validation
framework (purged K-fold + CPCV + PBO + trial-deflated DSR), and a sixth attempt
(Phase 20) attacked a **different axis** — meta-labeling (keep a fixed primary for the
side, fit a binary gate for whether to *act*) rather than adding another directional
feature. All six returned honest nulls. An honest null was an explicit success
criterion; a fabricated edge was the one unacceptable outcome.

This document locks those conclusions so future work does not re-litigate them, and
records that none of them authorizes building the economic backtest / strategy / risk
/ execution stack on the signal.

## The "edge" bar (unchanged across all phases)

A lever clears only if it passes **all** of: directional accuracy > 0.52, PBO < 0.5,
excess DSR > 0.5, positive mean excess-over-long return, and (microstructure layer)
a positive CPCV path-median. The bar never moves between phases — only the inputs do.

## Verdict table

| Phase | Lever / input set | Sample / effective N | Key gates failed | Verdict | Authorizes backtest? |
|------:|-------------------|----------------------|------------------|---------|----------------------|
| **5** | Price-only LightGBM — 45 price features on 1-min bars (`feature_version=v1`) | Full history 2019→2026; ~10,827 (MES) / 11,688 (MNQ) rows | excess DSR ≈0 (MES 0.006, MNQ 0.021); PBO high (MES 0.83, MNQ 0.70); excess Sharpe negative — **loses to buy-and-hold** on both | **No significant edge** (MES + MNQ) | **No** |
| **6** | + Macro/economic-event layer — FRED/ALFRED first-print releases + FOMC calendar (+27 cols) | Same rows as Phase 5 (`with_macro=False` reproduces price-only) | MES pooled OOS acc 0.498 (<0.52); excess-over-beta negative; excess DSR ≈0; PBO MES 0.88 / MNQ ≈0.52 | **No significant edge** (verdict unchanged) | **No** |
| **10** | + TA v2 layer — 7 oscillators wired opt-in (`with_ta`, default off) → 79 features | Same rows; controlled A/B vs price+macro | TA cleared **no** gates; excess-over-beta stays negative ±TA. MES excess DSR ≈0, PBO 0.88→0.47; MNQ excess DSR 0.014→0.022 (≪0.5), PBO 0.52→0.89 | **No significant edge** (MES + MNQ) | **No** |
| **14** | Microstructure 3-class LightGBM (`mm1`) — 17 `m1` order-flow features on MBP-1 dollar bars + `ml1` ~30-min triple-barrier labels | 94 RTH-session MBP-1 window; **effective N ES 1,090.0 / NQ 1,024.3** | ES fails 4/6 (PBO 0.897, gross DSR 0.030, negative gross return, negative CPCV median). NQ passes 5/6 but fails the **deciding CPCV gate** — all 5 OOS paths negative (median −0.023) | **No significant edge** (ES + NQ) | **No** |
| **19** | + `s2` sentiment block on the micro-model — opt-in `with_sentiment` A/B (`mm1` OFI-only vs `mm2` = `mm1` + 80 `sent_*` features), supported region only (pre-registered: `docs/PHASE19_PREREGISTRATION.md`) | Per symbol, `t0 ∈ [2026-03-10, 2026-06-06]`; rows ES 1,132 / NQ 1,036, **effective N 735.5 / 696.3** | Treatment fails on both. **ES 1/6** (PBO 0.476→0.734 — a selection overfit: gross DSR 0.84 + CPCV median +0.010 did **not** survive PBO). **NQ 3/6** (gross DSR 0.081, macro F1 0.177, CPCV median −0.034). SHAP led by `sent_1d_topic_rates_mean_score` on both — sentiment was used heavily, with no OOS edge | **No significant edge** (ES + NQ) | **No** |
| **20** | **Meta-labeling** (different axis): fixed causal primary `sign(ofi_top)`@`t0` + a binary meta-gate over OFI + `s2` + `\|ofi_top\|` deciding *whether to act* (τ=0.5). Arm M (gated) vs B0 (always-act). Five inherited mm1 gates + a disclosed binary **meta-skill** gate (pre-registered: `docs/PHASE20_PREREGISTRATION.md`) | Per symbol, full window `t0 ∈ [2026-01-26, 2026-06-06]`; meta-set ES 1,688 / NQ 1,519, **effective N 1,089.3 / 1,023.1** | Arm M fails on both. **ES 2/6** (gross DSR 0.060, CPCV median −0.010). **NQ 4/6** (PBO 0.722, gross DSR 0.247, mean gross −1.2e-5, CPCV median −0.027). **Meta-skill PASSED both** (acted-hit > always-hit: ES 0.134>0.122, NQ 0.130>0.103; balAcc 0.52/0.57) — the gate *did* add precision, just too little for a deflated edge. SHAP ≈70–75% sentiment | **No significant edge** (ES + NQ) | **No** |

## Why each null was still useful

- **Phase 5 (price):** Established the framework and the honest baseline — the model
  loses to buy-and-hold and shows no skill after deflation. SHAP spread thin (top
  feature ~7%), no leakage smell. Routed all future work to *improving inputs*, not
  to strategy.
- **Phase 6 (macro):** Macro features dominate SHAP (~66–69%) yet carry no OOS skill —
  a clean demonstration that high in-sample reliance + null OOS is exactly what the
  validation framework exists to expose, and argues *against* leakage.
- **Phase 10 (TA):** Proved a *dead lever by construction* — TA is a transformation of
  the same price stream and cannot inject new information. PBO swinging in opposite
  directions on the two symbols was the tell that the 7 TA cols are near-collinear
  price transforms that destabilise selection without adding signal.
- **Phase 14 (microstructure):** The first lever carrying *genuinely different*
  information (order flow), tested at adequate effective N (~1,000/symbol). Fold-local
  class weighting worked (predictions did not collapse to the ~78–80% timeout class).
  The null narrows the remaining search to deeper order flow (MBP-10, paid) or new
  information entirely (news/sentiment).
- **Phase 19 (sentiment):** The second genuinely-new-information lever (public news text,
  not a price transform), pre-registered before any modeling and run as a clean opt-in A/B
  (the only delta is the `s2` block). SHAP shows the treatment arm leaned **heavily** on
  sentiment (`sent_1d_topic_rates_mean_score` #1 on both symbols), so the null is *not*
  "the model ignored it." ES was a 5/6 near-miss: adding sentiment lifted gross DSR (0.84)
  and the CPCV median (+0.010), but **PBO blew out 0.476→0.734** — the search found an
  in-sample-stronger config whose selection is overfit, exactly what PBO exists to catch.
  Coverage (Phase 18) was real and high; coverage was never the question — out-of-sample
  edge was, and there is none.
- **Phase 20 (meta-labeling):** The first attempt to change the *framing* rather than the
  inputs — a fixed causal primary (`sign(ofi_top)`) with a binary gate deciding whether to
  act, the canonical remedy for a low-precision/imbalanced regime. The **meta-skill gate
  passed on both symbols** (the gate genuinely lifted the acted-on hit-rate above the
  always-act base rate: ES 0.122→0.134, NQ 0.103→0.130), so the precision filter *worked* —
  but the lift was far too small to survive deflation (gross DSR ES 0.06 / NQ 0.25, all CPCV
  path-medians negative; NQ PBO 0.72). Because meta-labeling is the textbook fix for exactly
  this regime, its failure is strong evidence the binding constraint is **sample size / edge
  existence at this horizon, not model framing** — which sharpens the remaining forks below.

## Current state and what is allowed next

- **Current state: no promoted model.**
- **No strategy / economic backtest is authorized.**
- **No live trading is authorized** (and the system is hard-locked to paper).
- **Sentiment is the fifth honest null (Phase 19).** The zero-spend news/sentiment lever was
  scaffolded (15), live-shape-validated (16), given a PIT feature layer (17), measured for
  historical coverage (18 — the pre-registered coverage gates passed, sent_1d has_any 98.3%),
  then run as the **pre-registered Phase 19 A/B edge verdict** (`mm1` OFI vs `mm2` = `mm1` +
  `s2`). **Both symbols returned no significant edge.** Coverage was real and high; the edge
  the gates test was not. The lever is **not re-litigated** (no sentiment re-tuning). See
  `docs/PHASE19_AB.md` + `docs/PHASE19_PREREGISTRATION.md`.
- **Meta-labeling is the sixth honest null (Phase 20).** A fixed causal primary
  (`sign(ofi_top)`) + a binary gate over OFI + `s2` deciding whether to act — the canonical
  remedy for this regime. The gate added real precision (meta-skill passed on both) but
  nowhere near enough for a deflated edge; both symbols failed the gross gates. The lever is
  **not re-litigated** (no τ tuning, no primary-rule swapping). See `docs/PHASE20_META.md` +
  `docs/PHASE20_PREREGISTRATION.md`.
- **Remaining forks (operator decision, none auto-authorized):** (a) MBP-10 multi-level
  order-flow depth — **paid, billing-gated, BLOCKED** until billing is deliberately unfrozen
  (`docs/BILLING_SAFETY.md`, `src/options_system/common/databento_guard.py`); or (b) a
  deliberate horizon/regime redesign of the labels/target. **Phase 20 sharpened the read:**
  since even the textbook meta-labeling fix could not extract an edge, the bottleneck is most
  likely sample size / edge existence at this horizon — so the forks that change *what data
  exists* (more depth via MBP-10, or a different horizon/market) are favoured over further
  modeling of the current set. Six levers in, **no strategy / economic backtest / risk /
  execution / live trading is authorized** until a lever clears the unchanged bar.

Per-phase method docs: `docs/MODEL.md` (5), `docs/MACRO.md` (6), `docs/TA_FEATURES.md`
(10), `docs/MICRO_MODEL.md` (14), `docs/PHASE19_AB.md` (19), `docs/PHASE20_META.md` (20).
