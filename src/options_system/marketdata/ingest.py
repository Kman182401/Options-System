"""Daily market-state ingestion CLI (System B) — key-gated AND policy-gated.

    uv run python -m options_system.marketdata.ingest --allow-network [--series VIXCLS ...]

Pulls each configured FRED daily series and writes it idempotently to the
``market_daily`` lake. Two independent gates must both pass for any network call:

* **Policy** — ``fred`` is ``FREE_AUTH`` in :mod:`options_system.common.external_data_policy`;
  a real fetch requires an explicit ``--allow-network`` (default is offline / no-op).
* **Key** — ``OPTIONS_FRED_API_KEY`` must be set (free key; bridge it from ``pass``). With
  it unset the command no-ops with a clear message, like the macro layer.

Per-series failures (a bad id, a transient FRED error) are logged and skipped — one bad
series never aborts the rest.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import polars as pl

from config.settings import Settings

from ..common.external_data_policy import ExternalAccessNotAuthorized, assert_network_allowed
from ..common.logging import get_logger
from .config import MarketDataConfig
from .fred_daily import fetch_daily_series
from .lake import MarketDailyLake, build_rows

logger = get_logger(__name__)

SOURCE = "fred"


def ingest(
    cfg: MarketDataConfig | None = None,
    settings: Settings | None = None,
    *,
    allow_network: bool,
    series_ids: list[str] | None = None,
) -> dict[str, int]:
    """Ingest configured daily series. Returns rows written per series id (+ ``_written``).

    Fail-closed: refuses unless the ``fred`` policy permits AND ``allow_network`` is True
    AND the FRED key is set.
    """
    cfg = cfg or MarketDataConfig.load()
    settings = settings or Settings()
    assert_network_allowed(SOURCE, allow_network=allow_network)  # raises if not permitted
    if settings.fred_api_key is None:
        logger.warning(
            "OPTIONS_FRED_API_KEY is not set — market-data ingestion is a no-op. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html."
        )
        return {}
    api_key = settings.fred_api_key.get_secret_value()

    wanted = set(series_ids) if series_ids else {s.id for s in cfg.series}
    lake = MarketDailyLake(dataset=cfg.storage.dataset)

    ingested_at = datetime.now(UTC)
    out: dict[str, int] = {}
    total = 0
    for spec in cfg.series:
        if spec.id not in wanted:
            continue
        try:
            obs = fetch_daily_series(spec.id, api_key, observation_start=cfg.observation_start)
        except RuntimeError as exc:
            logger.warning(f"[{spec.id}] fetch failed; skipping: {exc}")
            continue
        if obs.is_empty():
            logger.warning(f"[{spec.id}] FRED returned no observations; skipping")
            out[spec.id] = 0
            continue
        written = lake.write(build_rows(spec.id, obs, ingested_at=ingested_at))
        out[spec.id] = written
        total += written
        logger.info(f"[{spec.id}] {written} new daily rows ({spec.label})")
    out["_written"] = total
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="marketdata.ingest", description=__doc__)
    p.add_argument("--series", nargs="+", default=None, help="subset of configured FRED ids")
    p.add_argument("--allow-network", action="store_true", help="REQUIRED for any real fetch")
    p.add_argument("--show", action="store_true", help="print a sample after ingest")
    args = p.parse_args(argv)

    cfg = MarketDataConfig.load()
    print(
        f"marketdata version={cfg.marketdata_feature_version} series={[s.id for s in cfg.series]}"
    )
    try:
        result = ingest(cfg, allow_network=args.allow_network, series_ids=args.series)
    except ExternalAccessNotAuthorized as exc:
        print(f"BLOCKED: {exc}")
        return 2
    if not result:
        print("  (no FRED key — nothing ingested)")
        return 0
    print(f"  written={result.get('_written', 0)}")
    for k, v in sorted(result.items()):
        if not k.startswith("_"):
            print(f"    {k}: {v}")
    if args.show:
        df = MarketDailyLake(dataset=cfg.storage.dataset).read()
        print(df.group_by("series_id").agg(rows=pl.len(), last=pl.col("obs_date").max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
