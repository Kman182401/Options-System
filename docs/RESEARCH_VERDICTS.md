# Research Verdicts — Options-System

**Status as of Phase 15 (2026-06-09): no promoted model. No strategy, backtest, or
live trading is authorized.** Four feature/model levers have been tested through the
*same* fixed validation framework (purged K-fold + CPCV + PBO + trial-deflated DSR,
skill measured *over and above* a perma-long benchmark). All four returned honest
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

## Current state and what is allowed next

- **Current state: no promoted model.**
- **No strategy / economic backtest is authorized.**
- **No live trading is authorized** (and the system is hard-locked to paper).
- **Next allowed lever: zero-spend news/sentiment *feasibility* only** — design,
  scaffold, and point-in-time correctness on fixtures (Phase 15). No model verdict,
  no strategy, no broad ingestion. See `docs/SENTIMENT.md`.
- **Paid-data escalation (e.g. MBP-10 multi-level OFI) is BLOCKED** unless the operator
  explicitly unfreezes billing and authorizes a new capped run. See
  `docs/BILLING_SAFETY.md` and `src/options_system/common/databento_guard.py`.

Per-phase method docs: `docs/MODEL.md` (5), `docs/MACRO.md` (6), `docs/TA_FEATURES.md`
(10), `docs/MICRO_MODEL.md` (14).
