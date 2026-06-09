# sentiment/

**Zero-spend news/sentiment scaffold (Phase 15).** This is the next candidate
"genuinely new information" lever after four honest no-edge results — price,
macro, TA, microstructure (see `docs/RESEARCH_VERDICTS.md`). **This is scaffold
only: no model verdict, no strategy, no broad ingestion, no paid calls.** Real
historical text ingestion is a later, deliberately-authorized phase.

## What's here

| Module | Role |
|--------|------|
| `config.py` | Typed loader for `config/sentiment.yaml` (`SentimentConfig`). |
| `schema.py` | Point-in-time `RawNewsEvent` / `ScoredNewsEvent`, content-hash dedup, PIT filter. |
| `sources.py` | Registry of source adapters + their safety policy. |
| `gdelt.py` | GDELT DOC 2.0 adapter — fixture parse now; bounded real fetch is gated. |
| `sec_edgar.py` | SEC EDGAR submissions adapter — fixture parse now; gated fetch. |
| `scoring.py` | `FakeScorer` (deterministic, default) + optional local `FinbertScorer`. |
| `lake.py` | Idempotent Parquet store: `sentiment_raw` + `sentiment_scores`. |
| `build.py` | CLI. **Offline/fixture by default; network is opt-in and gated.** |

## Safety model (fail-closed)

Source access is governed by `options_system.common.external_data_policy`, the
authoritative registry. Sources are classified `free_no_auth` (GDELT, SEC EDGAR),
`local_only` (local FinBERT), `paid_blocked` (Finnhub, Databento), or
`unknown_blocked` (everything else — **fails closed**). A network fetch happens
**only** when the source is `free_no_auth` **and** `--allow-network` is explicitly
passed. The default does nothing over the network. Databento keeps its own,
stricter env-gated guard (`databento_guard.py`); nothing here can spend money.

## Point-in-time correctness

Every raw event carries three timestamps with `published_at <= observed_at <=
ingested_at` enforced. Feature generation may only use events with
`observed_at <= t` (the label/event time). Backfills must set `observed_at`
conservatively from source metadata (GDELT `seendate`, SEC acceptance time) —
never earlier than the source supports.

## Fixture-only examples (no network, no spend)

```sh
# Parse a GDELT fixture offline, score with the deterministic FakeScorer, dry-run:
uv run python -m options_system.sentiment.build \
    --source gdelt --fixture tests/fixtures/sentiment/gdelt_fed.json --score --dry-run

# Parse a SEC EDGAR submissions fixture offline:
uv run python -m options_system.sentiment.build \
    --source sec_edgar --fixture tests/fixtures/sentiment/sec_submissions.json --dry-run
```

Tests: `uv run pytest tests/test_sentiment*.py tests/test_external_data_policy.py -q`.

**No sentiment model verdict has been run yet. This phase does not authorize
strategy, backtest, or live trading.** See `docs/SENTIMENT.md`.
