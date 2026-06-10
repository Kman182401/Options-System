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

## Next (not yet authorized to implement here)

The GDELT live shape matches the fixtures (no adapter normalization needed). Next is a
**fixture-first point-in-time sentiment feature aggregation + label-join design**
(still offline / scaffold). If a future live run on a non-rate-limited egress surfaces
field differences, normalize the adapter first. **Do not** recommend model training
until actual historical sentiment coverage is measured.
