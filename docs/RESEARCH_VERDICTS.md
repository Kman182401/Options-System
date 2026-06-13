# Research Verdicts — Options-System

**Status as of Phase 19 (2026-06-13): no promoted model. No strategy, backtest, or
live trading is authorized.** Five feature/model levers have been tested through the
*same* fixed validation framework (purged K-fold + CPCV + PBO + trial-deflated DSR,
skill measured *over and above* a perma-long benchmark). All five returned honest
nulls. An honest null was an explicit success criterion; a fabricated edge was the
one unacceptable outcome.

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
- **Remaining forks (operator decision, none auto-authorized):** (a) MBP-10 multi-level
  order-flow depth — **paid, billing-gated, BLOCKED** until billing is deliberately unfrozen
  (`docs/BILLING_SAFETY.md`, `src/options_system/common/databento_guard.py`); or (b) a
  deliberate horizon/regime redesign of the labels/target. Five levers in, **no strategy /
  economic backtest / risk / execution / live trading is authorized** until a lever clears
  the unchanged bar.

Per-phase method docs: `docs/MODEL.md` (5), `docs/MACRO.md` (6), `docs/TA_FEATURES.md`
(10), `docs/MICRO_MODEL.md` (14), `docs/PHASE19_AB.md` (19).
