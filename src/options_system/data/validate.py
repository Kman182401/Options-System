"""Data validation over the lake — report problems, never repair them.

Checks (all read-only):

* **monotonic** — each contract's ``ts_event`` is strictly increasing (no two
  bars share a timestamp).
* **duplicates** — rows sharing the natural key ``(contract_id, ts_event)``.
* **ohlc** — ``high >= max(open, close, low)``, ``low <= min(open, close)``, and
  no negative prices/volume.
* **ingest** — ``ts_ingest >= ts_event`` (a row stamped as received before it
  happened means a clock problem; never let it pass silently).
* **gaps** — within RTH, consecutive bars more than ~2 intervals apart (a hole in
  data we should have). ETH gaps are not flagged (overnight trade is thin and
  legitimately sparse).

This module only *reports*. It never forward-fills, drops, or synthesizes data —
gaps stay gaps. Calendar-precise (holiday-aware) gap detection via
``exchange-calendars`` is deferred until we need it (see docs/DECISIONS.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import polars as pl

_EXPECTED_SECONDS = {"1m": 60, "5s": 5}


@dataclass
class Finding:
    check: str
    severity: str  # "error" | "warning"
    detail: str
    contract_id: str | None = None
    count: int = 1


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if there are no error-severity findings."""
        return not any(f.severity == "error" for f in self.findings)

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    def checks_failed(self) -> set[str]:
        return {f.check for f in self.findings}

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.n_errors,
            "warnings": self.n_warnings,
            "checks_failed": sorted(self.checks_failed()),
        }


def check_duplicates(df: pl.DataFrame) -> list[Finding]:
    dupes = df.group_by(["contract_id", "ts_event"]).len().filter(pl.col("len") > 1)
    if dupes.is_empty():
        return []
    return [Finding("duplicates", "error", f"{dupes.height} duplicated (contract_id, ts_event)")]


def check_monotonic(df: pl.DataFrame) -> list[Finding]:
    out: list[Finding] = []
    for (cid,), g in df.group_by("contract_id"):
        if g["ts_event"].n_unique() != g.height:
            out.append(
                Finding("monotonic", "error", "ts_event not strictly increasing", contract_id=cid)
            )
    return out


def check_ohlc(df: pl.DataFrame) -> list[Finding]:
    bad = df.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
        | (pl.col("open") < 0)
        | (pl.col("high") < 0)
        | (pl.col("low") < 0)
        | (pl.col("close") < 0)
        | (pl.col("volume") < 0)
    )
    if bad.is_empty():
        return []
    return [Finding("ohlc", "error", f"{bad.height} rows violate OHLC sanity", count=bad.height)]


def check_ingest(df: pl.DataFrame) -> list[Finding]:
    bad = df.filter(pl.col("ts_ingest") < pl.col("ts_event"))
    if bad.is_empty():
        return []
    return [
        Finding("ingest", "error", f"{bad.height} rows have ts_ingest < ts_event", count=bad.height)
    ]


def check_gaps(df: pl.DataFrame, freq: str) -> list[Finding]:
    exp = _EXPECTED_SECONDS.get(freq)
    if exp is None or df.is_empty():
        return []
    out: list[Finding] = []
    for (cid,), g in df.group_by("contract_id"):
        g = g.sort("ts_event").with_columns(
            pl.col("ts_event").diff().dt.total_seconds().alias("_gap")
        )
        gaps = g.filter((pl.col("_gap") > 2 * exp) & (pl.col("session") == "RTH"))
        if not gaps.is_empty():
            out.append(
                Finding(
                    "gaps",
                    "warning",
                    f"{gaps.height} RTH gaps > {2 * exp}s",
                    contract_id=cid,
                    count=gaps.height,
                )
            )
    return out


def validate_bars(df: pl.DataFrame, freq: str = "1m") -> ValidationReport:
    """Run every bar check and return a structured report (no mutation)."""
    if df.is_empty():
        return ValidationReport()
    findings: list[Finding] = []
    findings += check_duplicates(df)
    findings += check_monotonic(df)
    findings += check_ohlc(df)
    findings += check_ingest(df)
    findings += check_gaps(df, freq)
    return ValidationReport(findings)


def validate_quotes(df: pl.DataFrame) -> ValidationReport:
    """Lighter checks for L1 quotes: dupes, ingest order, crossed book."""
    if df.is_empty():
        return ValidationReport()
    findings: list[Finding] = []
    findings += check_duplicates(df)
    findings += check_ingest(df)
    crossed = df.filter(
        pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("bid") > pl.col("ask"))
    )
    if not crossed.is_empty():
        findings.append(
            Finding(
                "crossed_book",
                "warning",
                f"{crossed.height} quotes with bid > ask",
                count=crossed.height,
            )
        )
    return ValidationReport(findings)


# --- Coverage + CLI driver (report-only) -----------------------------------

_FREQ_DATASET = {"1m": "bars_1m", "5s": "bars_5s"}


def coverage(df: pl.DataFrame) -> list[dict]:
    """Per-contract coverage: row count + first/last bar ``ts_event``."""
    if df.is_empty():
        return []
    return (
        df.group_by("contract_id")
        .agg(
            rows=pl.len(),
            first_bar=pl.col("ts_event").min(),
            last_bar=pl.col("ts_event").max(),
        )
        .sort("first_bar")
        .to_dicts()
    )


def main(argv: list[str] | None = None) -> int:
    """Run validation across the lake for one or more symbols. Reports only."""
    import argparse

    from config.settings import Settings

    from .lake import Lake

    p = argparse.ArgumentParser(prog="validate", description="Validate lake bars; report only.")
    p.add_argument("--symbols", nargs="+", default=None, help="default: settings.record_symbols")
    p.add_argument("--freq", choices=sorted(_FREQ_DATASET), default="1m")
    p.add_argument(
        "--outrights-only",
        action="store_true",
        help="exclude calendar-spread instruments (contract_id containing '-')",
    )
    args = p.parse_args(argv)

    symbols = args.symbols or Settings().record_symbols
    dataset = _FREQ_DATASET[args.freq]
    lake = Lake()
    overall_ok = True
    for sym in symbols:
        # cast: collect() (no background) yields a DataFrame; polars' overload
        # stub widens the return to a union that ty can't narrow.
        df = cast("pl.DataFrame", lake.scan(dataset, sym).collect())
        if not df.is_empty():
            n_spread = df.filter(pl.col("contract_id").str.contains("-")).height
            if args.outrights_only:
                df = df.filter(~pl.col("contract_id").str.contains("-"))
            tag = " (outrights only)" if args.outrights_only else ""
            disp = "excluded" if args.outrights_only else "included"
            print(
                f"\n=== {sym} ({args.freq}){tag} — {df.height:,} rows "
                f"[{n_spread:,} spread rows {disp}] ==="
            )
        else:
            print(f"\n=== {sym} ({args.freq}) — 0 rows ===")
        if df.is_empty():
            print("  (no data on disk)")
            continue
        for c in coverage(df):
            print(
                f"  {c['contract_id']:>10}  rows={c['rows']:>10,}  "
                f"{c['first_bar']}  ->  {c['last_bar']}"
            )
        rep = validate_bars(df, args.freq)
        print(f"  validation: ok={rep.ok}  errors={rep.n_errors}  warnings={rep.n_warnings}")
        for f in rep.findings:
            where = f" (contract={f.contract_id})" if f.contract_id else ""
            print(f"    [{f.severity}] {f.check}: {f.detail}{where}")
        overall_ok = overall_ok and rep.ok
    print(f"\nOVERALL OK (no error-severity findings): {overall_ok}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
