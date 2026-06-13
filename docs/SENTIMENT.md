# Sentiment / News Layer (Phase 15 — scaffold only)

## Why sentiment is the next candidate

Four levers have now produced honest nulls through the same fixed validation framework
— price (Phase 5), macro (Phase 6), TA (Phase 10), and microstructure/order-flow
(Phase 14). See `docs/RESEARCH_VERDICTS.md`. Three of those (price, macro, TA) are
transformations or correlates of the same price stream; microstructure carries new
information but did not clear the gates. The remaining genuinely-new-information lever
that costs no money is **public news/text sentiment**: free/open metadata (GDELT),
no-auth filing data (SEC EDGAR), and a **local** FinBERT scorer.

## Why this phase is scaffold only

This phase builds the *infrastructure and safety rails* and proves *point-in-time
correctness on fixtures*. It deliberately does **not**:

- train a model or run any edge verdict,
- build a strategy, backtest, risk, or execution path,
- run any paid data pull, or any broad/bulk network ingestion,
- touch IBKR or live trading.

**No sentiment model verdict has been run yet. This phase does not authorize
strategy, backtest, or live trading.**

## Source policy (fail-closed)

Enforced in code by `options_system.common.external_data_policy` (the authoritative
registry; unknown sources fail closed). Declared, cross-checked, in `config/sentiment.yaml`.

| Source | Policy | Notes |
|--------|--------|-------|
| **GDELT** | `free_no_auth` | DOC 2.0 ArtList — free/open global news metadata + tone, no key. |
| **SEC EDGAR** | `free_no_auth` | `data.sec.gov` — no API key; requires a descriptive `User-Agent`. Future context for index/earnings/megacap-tech themes, not a trading signal yet. |
| **FinBERT (local)** | `local_only` | `ProsusAI/finbert` via local `transformers`/`torch`. Never networks; never calls a hosted inference API. |
| **Finnhub** | `paid_blocked` | Credentialed/paid — blocked. |
| **Databento** | `paid_blocked` | Paid per-byte — blocked (keeps its own stricter env gate). |

**Network is OFF by default.** A real fetch happens only when a `free_no_auth` source
is paired with an explicit `--allow-network`. Everything else is offline/fixture-only.

## Point-in-time rules

Every raw event carries three timestamps, with `published_at <= observed_at <=
ingested_at` enforced at construction:

- `published_at` — when the source says it was published.
- `observed_at` — the earliest moment our system could first have known it. For
  backfills this must be set **conservatively** from source metadata (GDELT `seendate`,
  SEC acceptance datetime) and **never earlier than the source supports**.
- `ingested_at` — when we stored it.

Feature generation may use an event only when `observed_at <= t` for the label/event
time `t` (`schema.filter_point_in_time`). Duplicates are removed by `content_hash`
(a stable hash of the identifying fields) or a stable `source_id`.

## Storage schema

Two Parquet datasets under `data/` (gitignored), kept separate so re-scoring never
rewrites raw text:

- `data/sentiment_raw/source=<src>/…` — `RawNewsEvent` rows. Idempotent on `content_hash`.
- `data/sentiment_scores/…` — `ScoredNewsEvent` rows. Idempotent on `(content_hash, model_name)`.

Scored output: `positive_score`, `negative_score`, `neutral_score`,
`sentiment_score = positive − negative`, `model_name`, `model_version_or_hash`,
`scored_at`.

## Fixture-only commands (no network, no spend)

```sh
# Parse a GDELT fixture offline, score with the deterministic FakeScorer, dry-run:
uv run python -m options_system.sentiment.build \
    --source gdelt --fixture tests/fixtures/sentiment/gdelt_fed.json --score --dry-run

# Parse a SEC EDGAR submissions fixture offline:
uv run python -m options_system.sentiment.build \
    --source sec_edgar --fixture tests/fixtures/sentiment/sec_submissions.json --dry-run

# Show the bounded GDELT request a real fetch WOULD make — without making it:
uv run python -m options_system.sentiment.build --source gdelt --topic fed

# Tests (no network):
uv run pytest tests/test_sentiment_policy.py tests/test_sentiment_*.py -q
```

## Scoring

`scoring.py` provides two scorers. `FakeScorer` is a deterministic lexicon stand-in —
the **default test path**, needing no weights and no network. `FinbertScorer` is the
optional real model: it loads weights with `local_files_only=True`, and if they are
absent it **fails with instructions rather than downloading** anything silently.

## Observability

`observability/sentiment_health.py` is a pure summary over the local lake frames:
rows by source/topic, `published_at`/`observed_at` ranges, duplicate rate, missing-
timestamp count, the scored sentiment distribution, per-source policy status, and a
`network_used` flag.

## Phase 16 — bounded free/no-auth live-shape smoke (2026-06-09)

A live-shape **validation** phase only: prove the adapters still match real source
responses, with the smallest possible bounded fetch. **No model, no features, no
label-join, no strategy/backtest, no live trading. No paid source touched. No model
verdict was run.**

A `--smoke` mode was added to the build CLI, fail-closed and hard-capped:
`--allow-network` required · free_no_auth source only · GDELT ≤ 5 records / SEC ≤ 2 ·
window ≤ 2 days · prints `network_used=true/false`. Bounds are enforced by the pure
`enforce_smoke_bounds` (unit-tested); paid/unknown/local-only network sources are
refused before any call.

**GDELT smoke — run.**
- Command:
  `uv run python -m options_system.sentiment.build --smoke --source gdelt --topic fed --max-records 5 --start 2026-06-08 --end 2026-06-09 --allow-network --score`
- Egress reached GDELT. A request with the adapter's exact URL + `User-Agent` returned
  **HTTP 200** once (request shape valid and accepted by GDELT).
- Repeated attempts then returned **HTTP 429** — GDELT's documented per-IP rate limit
  (response body: *"Please limit requests to one every 5 seconds"*) on the shared VPN
  egress IP. The adapter handled it **cleanly**: exit 1, `network_used=true`, no crash,
  **no partial/garbage write** to the lake. Records persisted via the live fetch: **0**
  (rate-limited, an environmental block, not an adapter fault).
- **Live response-shape validated** against a realistic live-shaped fixture
  (`tests/fixtures/sentiment/gdelt_live_shape.json` — real ArtList fields incl.
  `url_mobile`, `socialimage`, `domain`, `sourcecountry`, `seendate`): the parser
  **ignores the extra live fields**, the PIT schema is intact
  (`published_at == observed_at == seendate <= ingested_at`), **0 degraded, 0
  duplicate**. **Schema match: yes** — no normalization fix needed.
- End-to-end pipeline + scoring exercised on the live-shaped payload via the CLI:
  parse 2 → **FakeScorer** (`fake-lexicon-v1`) → write 2 raw + 2 scored → **idempotent
  rerun 0 + 0**. Health summary: 2 rows, sentiment mean −0.25, `network_used=false`,
  `gdelt: free_no_auth`. **Real FinBERT was NOT invoked; no model weights downloaded.**
- Adapter hygiene fix during this phase: GDELT/SEC requests now send
  `Accept: application/json` + `Accept-Encoding: identity` (standard JSON-client
  headers; urllib sends none by default).

**SEC EDGAR smoke — skipped (correctly).** Reason: a compliant `User-Agent` is not
configured (`config/sentiment.yaml` `fetch_limits.sec_user_agent` is the placeholder
`"...set-in-env"`). The smoke path detects the placeholder and **skips before any
network call** (`network_used=false`, exit 0). To enable later, set a real UA
(name + contact email) and pass `--cik <CIK>`.

**This phase does not authorize strategy, backtest, or live trading, and ran no model
verdict.**

## Phase 17 — point-in-time feature aggregation + label-join (2026-06-09)

A **fixture-first, offline** feature-engineering scaffold. It answers one question:
*given raw/scored sentiment events on disk, can we turn them into causal, versioned
sentiment features and attach them to labels with `observed_at <= label t0`, without
leakage?* It builds the aggregation + join + coverage tooling and proves point-in-time
correctness on fixtures. **No network, no scoring, no model training, no signal verdict,
no strategy/backtest, no broad ingestion, no IBKR/execution.**

### Versioning — two separate axes

- `sentiment_feature_version` (**s1**) — the raw/scored **event schema** version, stamped
  on every `RawNewsEvent`/`ScoredNewsEvent`. **Unchanged** by this phase.
- `aggregation.feature_version` (**s2**) — the **aggregate feature layer** version. s2 is
  the *first aggregate feature version*, built on s1 events. Emitted feature/coverage
  frames stamp `sentiment_feature_version = s2`. These are different layers (events vs
  aggregates), so the version was not bumped silently — s1 events keep their s1 stamp.

### The aggregate features (`config/sentiment.yaml` → `aggregation`)

- **Windows** (trailing, UTC): `15m`, `60m`, `240m`, `1d`.
- **Groups**: `all_sources_all_topics` (the full field set), `by_source`, `by_topic`
  (the reduced `breakdown_fields` over curated, **sanitized**, config-vetted source/topic
  lists — never arbitrary raw strings).
- **Global fields** (9): `event_count`, `degraded_count`, `mean_sentiment_score`,
  `sum_sentiment_score`, `mean_positive_score`, `mean_negative_score`,
  `mean_neutral_score`, `max_abs_sentiment_score`, `latest_observed_age_minutes`; plus a
  per-window `has_any` Int8 missing flag.
- **Breakdown fields** (per source / per topic): `event_count`, `mean_sentiment_score`.
- Conservative first pass: **80 columns total**, all prefixed `sent_` with stable names
  (e.g. `sent_15m_count`, `sent_1d_source_gdelt_mean_score`, `sent_60m_topic_fed_count`).
  `sentiment_feature_names(cfg)` is deterministic (same config → same ordered names).

### `observed_at` vs `published_at` vs `ingested_at` — why `observed_at` is the PIT key

Aggregation keys on **`observed_at`** only. It is the leakage-safe clock — the earliest
moment our system could have known the item. `published_at` can be *earlier* than we
could have known it (using it would feed information from before we had it);
`ingested_at` can be *later* (an implementation artefact of when we stored it). The window
is **half-open `(t - window, t]`**: an event exactly at `t` is knowable and included; an
event exactly at `t - window` has just aged out and is excluded. Deterministic, testable.

### Missing-data behavior

Empty window ⇒ count fields `0`, score aggregates **null** (not 0, so a model can tell
"no events" from a real zero), `has_any = 0`. Rows are **never dropped** for missing
sentiment. Events are deduped to one row per `content_hash` (latest `scored_at` wins), so
a headline scored by several models is counted once. Degraded events (unscoreable raw
items) are counted in `degraded_count` only and excluded from score aggregates.

### Label-join design (`sentiment/join.py`)

`attach_to_micro_labels` / `attach_to_daily_labels` attach the aggregate features onto the
micro (`data/micro_labels/`) and daily (`data/labels/`) label tables on **`t0`** (the
event/decision time). They **never** read `t1`, returns, or the label outcome; outcome
columns flow through as passengers. Every label row is preserved. Each returns
`(attached_frame, coverage_metadata)`.

### Coverage report (`python -m options_system.sentiment.coverage`)

Read-only, offline. Attaches features to labels and summarizes coverage — label rows,
sentiment rows, `rows_with_any_sentiment`, coverage rate, coverage by window /
source / topic, `events_used`, duplicate count, degraded count, null-feature count,
feature-column stability, and the `observed_at` / label-time ranges. Never writes the data
lake. With no sentiment/labels on disk it prints a clean **0% coverage** report and
exits 0.

```sh
# Coverage on fixtures (offline; no network, no spend):
uv run python -m options_system.sentiment.coverage \
    --label-type micro \
    --fixture tests/fixtures/sentiment/scored_events_pit.json \
    --label-fixture tests/fixtures/sentiment/micro_labels_for_join.json --no-write

# Coverage against the local lake (default; offline):
uv run python -m options_system.sentiment.coverage --label-type micro --symbols ES NQ
```

This phase used **fixture/local data only**. **No model verdict was run. No
strategy/backtest/live trading is authorized.** Actual historical sentiment coverage still
needs to be measured (a bounded, free/no-auth GDELT plan) before any model training.

## Phase 18 — Bounded GDELT historical backfill + coverage verdict (2026-06-13)

This phase executed the bounded GDELT backfill designed in `docs/DECISIONS.md` (Phase 18),
scored the result with the **local** FinBERT, and evaluated the **pre-registered coverage
gates**. It is **coverage measurement only**: no model was trained, no edge verdict was run,
no paid data was touched (`OPTIONS_DATABENTO_SPEND_OK` stayed unset throughout), and the
network reached **only** GDELT (free/no-auth).

### The backfill run

The detached resumable chain (systemd user unit, breadth-first / supported-region-first)
recorded 15 runs and stopped cleanly on the 240-minute wall-clock cap with the checkpoint
intact (`outcome=capped_max_wall_clock_minutes`, exit 3 — a clean cap, not a crash).
Authoritative manifest totals:

- **1,172 day×topic slices** attempted (1,100 ok, 72 failed `rate_limited` after the
  5-attempt cap — the run continues past failures by design).
- **839 supported / 333 unsupported-archive** slices (both attempted; supported first).
- **185,533 records returned → 145,654 written** after dedup on `content_hash` (≈40k
  cross-topic / cross-bisection duplicates collapsed to one row each).
- **396 slices truncated** (hit GDELT's 250-record, no-pagination cap at the 1-hour
  bisection floor) — disclosed; truncation costs *depth*, not `has_any` coverage.
- **5,623 requests, 4,537 HTTP 429s** — GDELT's ~1-req/5s per-IP limit throttled the shared
  egress heavily; exponential backoff + retries (over the AirVPN egress route) absorbed it.
  The cap was reached on breadth, not exhaustion: the full plan did not complete, but the
  **supported region — where the gates are evaluated — was covered**.

Lake after the run: `data/sentiment_raw` holds **145,656 raw events** (145,654 backfill + 2
Phase-16 smoke), spanning `observed_at` 2026-01-26 → 2026-06-10. Topic mix (largest first):
inflation 24,748 · earnings 23,017 · fed 20,724 · rates 17,423 · ai_capex 16,219 · recession
11,867 · risk_off 10,671 · semiconductors 10,594 · megacap_tech 10,393.

### FinBERT scoring

`python -m options_system.sentiment.score_backfill` scored the 140,764 unscored rows with
the **local** `ProsusAI/finbert` (snapshot revision `4556d130…`, `local_files_only=True`,
CUDA) in 54.9 s (2,564 rows/s), idempotent on `(content_hash, model_name)`. The lake now
holds **145,629 scored rows** (29 raw events were degraded/unscoreable and excluded from
score aggregates). Scored distribution: mean −0.048, std 0.560 — a healthy, slightly
net-negative spread typical of macro-news headlines. `network_used=false`.

### Pre-registered gates

The gates below were **fixed at backfill launch (2026-06-10), before any data arrived**, to
decide *feasibility only* — whether enough point-in-time sentiment exists to justify a Phase
19 model verdict. They were set deliberately **low/conservative** (GDELT was expected to be
sparse and rate-limited). They are evaluated **only over the supported archive region**
(label `t0` ≥ the 92-day cutoff `2026-03-10`), pooled across ES + NQ micro labels.

> Provenance note: these thresholds were referenced from `docs/DECISIONS.md` ("gates
> pre-registered … see `docs/SENTIMENT.md`") but not transcribed into this file at the time.
> They are recorded verbatim here, unchanged, alongside the result. The verdict clears them
> by a wide margin under every interpretation, so the transcription gap does not affect the
> outcome.

| Gate | Definition (supported region) | Threshold |
|------|-------------------------------|-----------|
| **G1** | fraction of label rows with **any** prior sentiment in the `1d` window (`sent_1d` has_any) | ≥ 60% |
| **G2** | fraction of label rows with any prior sentiment in the `240m` window | ≥ 35% |
| **G3** | **pooled** count of label rows with `sent_1d` has_any | ≥ 1,400 |

### Coverage result

Coverage from `python -m options_system.sentiment.coverage --label-type micro
--archive-cutoff 2026-03-10` (offline, read-only; JSON saved under
`data/sentiment_backfill/coverage_*.json`):

| Region (pooled ES+NQ) | label rows | sent_1d has_any | rate | sent_240m rate | sent_60m | sent_15m |
|-----------------------|-----------:|----------------:|-----:|---------------:|---------:|---------:|
| **supported** (`t0` ≥ 2026-03-10) | 2,168 | 2,131 | **98.3%** | 98.3% | 98.3% | 86.3% |
| unsupported_archive (`t0` < cutoff) | 1,042 | 925 | 88.8% | 85.9% | 85.7% | 70.3% |

Per symbol (supported region): **ES** 1,132 labels → 1,111 has_any (98.1%); **NQ** 1,036 →
1,020 (98.5%). The monotone `15m < 60m ≈ 240m < 1d` shape is the expected correctness
signature (a wider lookback can only add events), not a leakage smell — the join uses
`observed_at ≤ t0` on a half-open window, the same PIT rule proven by Phase 17's leakage
teeth (re-confirmed here by a 200-sample brute-force recompute of the `(t0−1d, t0]` window
against `sent_1d_count`: 0 mismatches).

Two honest notes on reading these numbers: **(1)** in the supported region the `240m` and
`1d` coverage are *identical* (both 2,131 / 98.3%) — not a transcription error but a density
effect: almost every label with any prior-4h macro news also has prior-24h news. The windows
diverge where data is sparser (unsupported region: 895 vs 925; and the `15m` window, 86.3%,
is where they genuinely separate). **(2) G3 is a pooled gate by pre-registration** — per
symbol the `sent_1d` has_any counts (ES 1,111 / NQ 1,020) are each individually *below*
1,400. The ≥1,400 threshold was fixed against the **pooled** ES+NQ total (2,131), and only
that pooled figure is claimed to clear it; the per-symbol figures are reported for
transparency, not as independent G3 passes.

### Verdict — ALL GATES PASS

| Gate | Threshold | Actual (supported) | |
|------|-----------|--------------------|---|
| G1 — sent_1d has_any rate | ≥ 60% | **98.3%** | ✅ |
| G2 — sent_240m rate | ≥ 35% | **98.3%** | ✅ |
| G3 — pooled sent_1d has_any rows | ≥ 1,400 | **2,131** | ✅ |

**Phase 19 (sentiment micro-model A/B verdict) is authorized.** Enough free, point-in-time
sentiment coverage exists to train a model and run it through the unchanged edge bar.

**Disclosed caveats (forward into Phase 19):**

- **Coverage ≠ edge.** Passing these gates authorizes *building/running* the Phase 19 model
  verdict; it does **not** predict an edge. Phase 19 must clear the same fixed bar (dir. acc
  > 0.52, PBO < 0.5, excess DSR > 0.5, positive mean excess-over-long, positive CPCV median)
  — see `docs/RESEARCH_VERDICTS.md`. An honest null remains a likely and acceptable outcome.
- **Low has_any variance.** At ~98% `1d`/`240m` coverage the *presence* flags are nearly
  constant in the supported region, so discriminative signal (if any) will come from the
  **score aggregates** (mean / max-abs sentiment) and the **shorter windows** (15m at 86.3%),
  not from `has_any`. A modelling consideration, not a coverage failure.
- **Truncation / topic-undercount.** 396 truncated slices, and first-write-wins
  `content_hash` attribution means by-topic breakdowns undercount multi-topic articles
  (disclosed in the Phase 18 design). `has_any` coverage is unaffected.
- **Unsupported-region bonus.** GDELT returned usable data well before its officially
  supported 92-day window (88.8% `1d` coverage in the unsupported region). Reported for
  transparency but **not** part of the gate — the verdict rests only on the supported region,
  as pre-registered.
- **Capped run.** The chain stopped on the wall-clock cap, not plan exhaustion; the supported
  region was covered but the full historical plan was not. Two separately-scoped follow-ups
  remain available if ever needed: forward live collection on a non-throttled egress, and the
  GDELT 15-minute raw-file route. Neither is required for Phase 19.

**This phase does not authorize any strategy, backtest, risk, execution, or live trading.**

## Next (not yet implemented here)

With feasibility established, the next step is **Phase 19 — the sentiment micro-model A/B
verdict**: train the `mm1`-style micro model with vs. without the `s2` sentiment features
through the *same* purged-K-fold + CPCV + PBO + deflated-DSR framework, and report an honest
edge verdict. Still **no strategy / backtest / risk / execution / live trading** until a
lever clears the bar.
