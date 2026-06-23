# PHASE24_FREEDATA.md — Free-data incremental-value verdict (Phase 24)

## What it is
Phase 23 confirmed a 1-day realized-volatility forecast skill using only the existing features (the
Phase-22 free-data blocks were held OFF — "confirm first"). Phase 24 is the deferred question: do the
**free** Phase-22 blocks add incremental accuracy *on top of* that confirmed model? Frozen contract:
`docs/PHASE24_PREREGISTRATION.md` / `config/phase24_freedata.yaml` (`freedata_version=fd1`). The
modeling core is inherited verbatim from the Phase-23 contract, so the **baseline arm is the confirmed
Phase-23 model**. Forecast-skill verdict only — no spend, no trading.

## Design
Per free-data block, per symbol, an incremental A/B at **h = 1**:
- **Baseline B0** = the confirmed Phase-23 feature set (`with_marketdata=False`, `with_gkg=False`).
- **Augmented A_B** = B0 + the block (`x1` market-data, or `s3` GKG news-tone).
- Same fixed LightGBM, same walk-forward / regime / DM machinery; the only delta is the block, and
  both arms are scored on the **same OOS rows and folds** (the row gate is unchanged).

Gates (per block, per symbol, all required): **G1** incremental accuracy (augmented QLIKE <
baseline, one-sided DM significant), **G2** regime robustness (augmented ≤ baseline in both calm and
turbulent), **G3** temporal stability (augmented beats baseline in ≥ 13/18 folds). A block below the
**80 % OOS coverage** precondition is **deferred**, not run as a null.

## Result — measured 2026-06-23 (OOS n = 1,139 per symbol)

The baseline QLIKE reproduces the confirmed Phase-23 model byte-for-byte (MES 0.246401, MNQ
0.224083) — the comparison is faithful.

### `x1` — VIX/VXN + cross-asset (FRED): NO incremental value (both symbols)
| Symbol | baseline QLIKE | augmented QLIKE | Δ (lower=better) | DM p (aug better) | folds aug>base | SHAP share | G1 | G2 | G3 | verdict |
|--------|---------------:|----------------:|-----------------:|-------------------|----------------|-----------:|----|----|----|---------|
| **MES** | 0.246401 | 0.259877 | **−0.0135** (worse) | 0.997 | 8/18 | 0.266 | ✗ | ✗ | ✗ | **no value** |
| **MNQ** | 0.224083 | 0.233533 | **−0.0094** (worse) | 0.996 | 10/18 | 0.274 | ✗ | ✗ | ✗ | **no value** |

Adding the market-data block made the forecast **worse**, not better (the one-sided DM for
"augmented more accurate" has p ≈ 1, i.e. it is significantly *less* accurate). Coverage was full
(1.0).

**Regime detail — the block hurts exactly where it matters:**
| Symbol | calm QLIKE (base / aug) | turbulent QLIKE (base / aug) |
|--------|-------------------------|------------------------------|
| **MES** | 0.2390 / **0.2353** ✓ | 0.2549 / **0.2880** ✗ |
| **MNQ** | 0.2115 / **0.2073** ✓ | 0.2390 / **0.2646** ✗ |

The block helps *marginally* in calm markets but **degrades the forecast in turbulent markets** — the
same failure mode that killed Phase-21 at h = 5, and the reason G2 fails. SHAP shows the model
**used** the block heavily (~27 % of importance), so this is not "the model ignored it" — the VIX /
cross-asset daily features simply add noise (and turbulent-regime harm) to a model whose edge already
comes ~76 % from intraday realized-vol structure.

### `s3` — GKG news tone: DEFERRED (coverage 0.197 < 0.80)
The GKG bulk backfill currently reaches ~2022-11 (chronological fill from 2019), covering ~20 % of the
2022→2026 OOS window. Coverage is measured by **per-day partition-set membership** (robust to gaps),
so an incomplete backfill cannot falsely pass the gate. The arm is **deferred** and re-runs unchanged
once the backfill lifts coverage ≥ 80 %. (No s3 model was trained.)

## Decision (per the frozen rule)
- **`x1` (market-data): no incremental value — both symbols.** Not folded in; the confirmed Phase-23
  baseline stands unchanged.
- **`s3` (GKG news): deferred on coverage.** Pending the backfill.

## Honest interpretation
- An honest null for the market-data block — the program's modal outcome, and consistent with the
  Phase-21/23 finding that the 1-day RV edge is driven by *intraday* realized-vol features. Daily VIX
  and cross-asset levels, added to a fixed (no-retune) model, do not improve day-ahead RV accuracy and
  hurt in turbulence. The lever is **not re-litigated** (no feature-subset fishing, no re-tuning).
- The confirmed Phase-23 h = 1 skill is unchanged and remains the program's one positive result.
- This is a *forecast-skill* study. No strategy / backtest / risk / execution / live trading is
  authorized; the system stays hard-locked to paper.

## Run it
```sh
uv run python -m options_system.volatility.run_freedata --symbols MES MNQ
```
Per-symbol summaries → `data/volatility/runs_fd/<symbol>.json`; combined verdict →
`data/volatility/runs_fd/verdict.json` (gitignored). MLflow experiment `volatility-freedata-fd1`.
Tests: `tests/test_phase24_freedata.py`.
