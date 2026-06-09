# Billing & External-Data Safety

This document is the operator-facing record of the Databento billing incident and the
guards that now prevent a repeat, plus the generic policy for every other external
source. **Default posture: no paid data, no network. Spending requires a deliberate,
per-run opt-in.**

## What happened (2026-06-09 Databento billing incident)

Databento bills **per byte** against whatever payment method the API key's *account*
has on file. The code selects an account purely by which API key resolves
(`pass databento/api_key_2` for the microstructure ingest;
`OPTIONS_DATABENTO_API_KEY` for the daily loader) — it **cannot** tell from the API
whether a download draws on free trial credits or charges a real card. The only guard
at the time was a **dollar cap**, which bounds *how much* is spent but knows nothing
about the *funding source*.

The original free-credit key (`databento/api_key`, ~$125 trial) was depleted and
removed on 2026-06-08. `api_key_2` became active and bills a real card. So the Phase 12
($163.82) and Phase 13 ($40.10) pulls charged the card while every dollar-cap guard
still showed green. The operator was (rightly) upset and **froze the card**.

## What the guard does

`src/options_system/common/databento_guard.py` adds a **fail-closed** authorization
gate. Every real (billable) download path — microstructure `run_ingest` and the daily
loader `_download_and_store` — calls `assert_spend_authorized()` **before a single
byte is fetched**. It refuses unless the operator has explicitly attested, per process,
via the environment.

- **Default behaviour: BLOCKED.** No env var set → no paid download is possible.
- **Free dry-run cost estimation is unaffected.** `metadata.get_cost` /
  `get_billable_size` are free and are **not** gated — you can always estimate.
- The dollar cap (`databento_budget_usd_cap`) still applies *on top of* the gate.

## What the guard does NOT know

The guard **cannot verify free credits vs. a real card.** That is an account property
the Databento API does not expose to the code. The guard only enforces that a human
deliberately opted in for this run; the human is responsible for confirming the
account is actually safe to spend on.

## Required operator state before any paid download

At least one of, and ideally all of:

1. **Card frozen** / removed (the true funding-source guarantee — currently in place), or
2. a **hard spend limit** on the Databento dashboard at/below the remaining free credit, or
3. **no card on file**.

The frozen card + the fail-closed env gate are layered: account-level + code-level.

## Required env var for a paid download

```sh
OPTIONS_DATABENTO_SPEND_OK=1 uv run python -m options_system.microstructure.ingest ... --confirm
```

- **Default: blocked** (variable unset).
- Truthy values accepted: `1`, `true`, `yes`, `on`.
- Set it **only** for the single run that should spend, **only** after confirming the
  account is safe, and **only** with the operator's explicit per-run approval. Never
  export it globally.

## Dry-run estimates are always allowed

Cost estimates (`metadata.get_cost`) make no billable call and are never gated. Use them
freely to size a future, explicitly-authorized pull.

## Generic policy for non-Databento sources

`src/options_system/common/external_data_policy.py` generalises the idea to every other
source. Sources are classified, with **unknown sources failing closed**:

| Policy | Meaning | Network |
|--------|---------|---------|
| `free_no_auth` | Free/open, no key/account/card (GDELT, SEC EDGAR) | Only with an explicit `--allow-network` opt-in |
| `local_only` | Runs entirely on this machine (local FinBERT) | Never |
| `paid_blocked` | Costs money / needs a card or subscription (Finnhub, Databento) | Blocked |
| `unknown_blocked` | Anything not vetted | **Blocked (fail closed)** |

The new sentiment scaffold (Phase 15) honours this: it is fixture/offline by default
and performs **no** network call unless a `free_no_auth` source is paired with an
explicit `--allow-network`. Databento keeps its own, stricter env-gated guard on top.

## Standing rule

**All future paid-data phases require a fresh, explicit operator approval.** Do not set
`OPTIONS_DATABENTO_SPEND_OK`, do not unfreeze the card, and do not run any paid pull
without it. Most learning-loop work (training, validation, this sentiment scaffold)
needs **no** new ingest at all.
