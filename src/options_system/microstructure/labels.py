"""Short-horizon triple-barrier labels on the microstructure dollar bars.

This is the intraday sibling of :mod:`options_system.labeling` (the daily
triple-barrier labels). Same López de Prado methodology — symmetric CUSUM event
sampling, ±σ horizontal barriers, a vertical time barrier, ``t1`` resolution and
average-uniqueness sample weights — re-scaled to a **30-minute** horizon and run
on the m1 dollar bars (``microstructure.ingest.read_micro_bars``) instead of the
1-minute price bars. Outputs are stamped ``micro_label_version`` and stored in
their own lake (``data/micro_labels/``), isolated from the daily ``data/labels/``.

What is REUSED unchanged from the daily layer (imported, not reimplemented):

* :func:`options_system.labeling.events.cusum_events` — the symmetric CUSUM
  filter for event sampling.
* :func:`options_system.labeling.weights.sample_weights` — concurrency-based
  average uniqueness + normalized weights (and hence effective N).

What is NEW here, because the dollar bars are NOT time-uniform and a short
intraday horizon has a session-close leakage trap the daily layer never faces:

1. **σ scaled to a wall-clock horizon.** σ_bar (causal EWM std of per-bar mid log
   returns) is scaled to 30 min via ``sqrt(vertical_seconds / EWMA(duration_s))``
   — i.e. by the causally-estimated number of bars in 30 minutes — rather than by
   a fixed bar count. Dollar bars carry ~equal variance per bar, so 30-min
   variance ≈ (bars per 30 min) · (per-bar variance).
2. **Wall-clock vertical barrier.** The vertical barrier is the first bar whose
   close ``ts_event ≥ t0 + 30 min``, not a fixed bar count.
3. **Session-close guards (the key short-horizon leakage trap).** Everything is
   processed **per RTH session block** ``(contract_id, ET-date)``, so a label can
   physically never read a bar from the next session. On top of that: (a) events
   whose ``t0 + 30 min`` would fall after the session close are excluded; (b) if
   no bar reaches the 30-min mark before the session ends, the label is hard-capped
   and resolved at the last in-session bar (``barrier_touched = "close"``).

Leakage discipline (proven in tests/test_micro_labeling_leakage.py): a label looks
FORWARD to its barrier touch (that is what a label is), but it never reads a price
beyond its own ``t1``, and never beyond its session close. σ at ``t0`` uses only
bars with ``ts_event ≤ t0``. The construction is deterministic.
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, date, datetime
from datetime import time as dtime
from glob import glob as _glob
from pathlib import Path
from typing import cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from config.settings import Settings

from ..common.logging import get_logger
from ..data.store import DuckStore
from ..labeling.events import cusum_events
from ..labeling.weights import sample_weights
from .config import MicrostructureConfig, SessionCfg
from .ingest import read_micro_bars
from .label_config import MicroLabelConfig

logger = get_logger(__name__)

# Columns this module needs on the input micro-bar frame.
REQUIRED_INPUT = ("ts_event", "mid_close", "duration_s", "contract_id", "bar_complete")

_DATASET = "micro_labels"
_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)

# Generator output (one row per resolved event), before symbol/ts_ingest stamps.
_OUTPUT_SCHEMA = {
    "t0": pl.Datetime("us", "UTC"),
    "t1": pl.Datetime("us", "UTC"),
    "contract_id": pl.Utf8,
    "session_date": pl.Date,
    "label": pl.Int8,
    "barrier_touched": pl.Utf8,  # upper | lower | vertical | close
    "ret_t1": pl.Float64,
    "sigma": pl.Float64,
    "n_bars": pl.Int32,
    "resolved_at_close": pl.Boolean,
    "uniqueness_weight": pl.Float64,  # average uniqueness (LdP), in (0, 1]
    "sample_weight": pl.Float64,  # normalized to mean 1.0
    "micro_label_version": pl.Utf8,
}

# Persisted column order: keys first, then outcome, weights, stamps.
_PERSIST_COLUMNS = (
    "t0",
    "t1",
    "symbol",
    "contract_id",
    "session_date",
    "label",
    "barrier_touched",
    "ret_t1",
    "sigma",
    "n_bars",
    "resolved_at_close",
    "uniqueness_weight",
    "sample_weight",
    "micro_label_version",
    "ts_ingest",
)


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_OUTPUT_SCHEMA)


def _attach_block_sigma(block: pl.DataFrame, cfg: MicroLabelConfig) -> pl.DataFrame:
    """Attach ``_r`` (per-bar mid log return) and ``_sigma`` (σ scaled to 30 min).

    Causal: σ_bar is an EWM std of per-bar mid log returns (``adjust=False`` →
    trailing recursion, null during the first ``min_samples`` bars), and the
    bar-rate used to scale it to the horizon is an EWMA of ``duration_s`` over the
    same trailing window. Every value at bar t uses only bars ≤ t.
    """
    v = cfg.volatility
    vertical_seconds = cfg.barriers.vertical_minutes * 60.0
    logmid = pl.col("mid_close").log()
    r = logmid - logmid.shift(1)
    sigma_bar = r.ewm_std(span=v.ewm_span, adjust=False, bias=False, min_samples=v.min_samples)
    dur_ewma = pl.col("duration_s").ewm_mean(span=v.dur_ewm_span, adjust=False, min_samples=1)
    # σ_H = σ_bar · sqrt(seconds_in_horizon / seconds_per_bar) = σ_bar · sqrt(bars_in_horizon)
    sigma = sigma_bar * (vertical_seconds / dur_ewma).sqrt()
    return block.with_columns(_r=r, _sigma=sigma)


def _label_block(
    ts: np.ndarray,
    logmid: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    *,
    contract_id: str,
    session_date: date,
    session_close: np.datetime64,
    cfg: MicroLabelConfig,
) -> list[dict]:
    """Resolve triple-barrier labels for one RTH session block.

    ``ts`` / ``logmid`` / ``sigma`` / ``r`` are the block's bars (sorted ascending);
    ``session_close`` is the 16:00-ET close as a naive-UTC ``datetime64[us]``. All
    forward scanning is confined to this block, so a label can never read the next
    session. Mirrors the daily ``label_events`` first-touch logic, with a wall-clock
    vertical barrier and the session-close hard cap.
    """
    n = ts.shape[0]
    b = cfg.barriers
    vertical_delta = np.timedelta64(int(round(b.vertical_minutes * 60.0 * 1e6)), "us")

    rets = np.nan_to_num(r, nan=0.0)
    thresh = sigma * cfg.events.cusum_mult  # NaN where σ is warmup → CUSUM resets, no event
    event_idx = cusum_events(rets, thresh)

    rows: list[dict] = []
    for p in event_idx.tolist():
        sig = sigma[p]
        if not (math.isfinite(sig) and sig > 0.0):
            continue  # warmup guard (events should already exclude these)

        t0 = ts[p]
        vertical_ts = t0 + vertical_delta
        # GUARD 1 — exclude events whose 30-min window would run past the close,
        # so every retained label gets its full intraday horizon within one session.
        if vertical_ts > session_close:
            continue

        # Vertical barrier = first bar at/after t0 + 30 min, searched within the block.
        rel = int(np.searchsorted(ts[p + 1 :], vertical_ts, side="left"))
        if rel < n - (p + 1):
            vbar = p + 1 + rel
            reached_vertical = True
        else:
            # GUARD 2 — no bar reached the 30-min mark before the session ended;
            # hard-cap at the last in-session bar (the close).
            vbar = n - 1
            reached_vertical = False

        if vbar <= p:
            continue  # no forward bar to evaluate

        cr = logmid[p + 1 : vbar + 1] - logmid[p]
        up = b.pt_mult * sig
        dn = -b.sl_mult * sig
        up_mask = cr >= up
        dn_mask = cr <= dn
        first_up = int(up_mask.argmax()) if up_mask.any() else -1
        first_dn = int(dn_mask.argmax()) if dn_mask.any() else -1

        if first_up >= 0 and (first_dn < 0 or first_up <= first_dn):
            k, label, barrier, at_close = first_up, 1, "upper", False
        elif first_dn >= 0:
            k, label, barrier, at_close = first_dn, -1, "lower", False
        else:
            k = cr.shape[0] - 1
            if reached_vertical:
                barrier, at_close = "vertical", False
            else:
                barrier, at_close = "close", True
            label = int(np.sign(cr[k])) if b.vertical_label_sign else 0

        pos = p + 1 + k
        rows.append(
            {
                "t0": t0,
                "t1": ts[pos],
                "contract_id": contract_id,
                "session_date": session_date,
                "label": label,
                "barrier_touched": barrier,
                "ret_t1": float(cr[k]),
                "sigma": float(sig),
                "n_bars": pos - p,
                "resolved_at_close": at_close,
            }
        )
    return rows


def generate_micro_labels(
    df: pl.DataFrame,
    cfg: MicroLabelConfig,
    session: SessionCfg,
    *,
    diag: dict | None = None,
) -> pl.DataFrame:
    """End-to-end short-horizon labels for one symbol's micro bars.

    ``df`` is one symbol's dollar bars (any order). Returns the full
    ``_OUTPUT_SCHEMA`` (one row per resolved event) with uniqueness + sample
    weights attached. If ``diag`` is given it is filled with sampling counts
    (``events_sampled``, ``dropped_final30_or_no_forward``, ``retained``) for QA.
    """
    if df.is_empty() or df.width == 0:
        return _empty()
    missing = [c for c in REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(f"generate_micro_labels: input missing columns {missing}")
    if df.select(pl.col("contract_id").str.contains("-").any()).item():
        raise ValueError("generate_micro_labels: spread contract_id (containing '-') in input")

    df = df.filter(pl.col("bar_complete")).sort("ts_event")
    if df.is_empty():
        return _empty()
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone(session.tz).dt.date().alias("_session_date")
    )

    et = ZoneInfo(session.tz)
    close_h, close_m = divmod(session.rth_close_min, 60)

    events_sampled = 0
    rows: list[dict] = []
    for (cid, sdate), block in df.group_by(["contract_id", "_session_date"], maintain_order=True):
        block = _attach_block_sigma(block, cfg)
        ts = block["ts_event"].to_numpy()
        logmid = np.log(block["mid_close"].to_numpy())
        sigma = block["_sigma"].to_numpy()
        r = block["_r"].to_numpy()

        # CUSUM count (before exclusion/scan drops) for honest QA reporting.
        thresh = sigma * cfg.events.cusum_mult
        events_sampled += int(cusum_events(np.nan_to_num(r, nan=0.0), thresh).size)

        close_dt = (
            datetime.combine(cast("date", sdate), dtime(close_h, close_m), et)
            .astimezone(UTC)
            .replace(tzinfo=None)
        )
        rows.extend(
            _label_block(
                ts,
                logmid,
                sigma,
                r,
                contract_id=cast("str", cid),
                session_date=cast("date", sdate),
                session_close=np.datetime64(close_dt, "us"),
                cfg=cfg,
            )
        )

    if diag is not None:
        diag["events_sampled"] = events_sampled
        diag["retained"] = len(rows)
        diag["dropped_final30_or_no_forward"] = events_sampled - len(rows)

    if not rows:
        return _empty()

    t0_arr = np.array([row["t0"] for row in rows], dtype="datetime64[us]")
    t1_arr = np.array([row["t1"] for row in rows], dtype="datetime64[us]")
    out = pl.DataFrame(
        {
            "t0": pl.Series(t0_arr).dt.replace_time_zone("UTC"),
            "t1": pl.Series(t1_arr).dt.replace_time_zone("UTC"),
            "contract_id": [row["contract_id"] for row in rows],
            "session_date": [row["session_date"] for row in rows],
            "label": [row["label"] for row in rows],
            "barrier_touched": [row["barrier_touched"] for row in rows],
            "ret_t1": [row["ret_t1"] for row in rows],
            "sigma": [row["sigma"] for row in rows],
            "n_bars": [row["n_bars"] for row in rows],
            "resolved_at_close": [row["resolved_at_close"] for row in rows],
        }
    )

    # Uniqueness over the symbol's full bar timeline (labels never overlap across
    # session blocks, so a global concurrency equals the per-block one).
    bar_ts = df["ts_event"].to_numpy()
    starts = np.searchsorted(bar_ts, out["t0"].to_numpy(), side="left")
    ends = np.searchsorted(bar_ts, out["t1"].to_numpy(), side="left")
    w = sample_weights(
        starts,
        ends,
        returns=out["ret_t1"].to_numpy(),
        scheme=cfg.weights.scheme,
        time_decay=cfg.weights.time_decay,
    )
    out = out.with_columns(
        pl.Series("uniqueness_weight", w["avg_uniqueness"]),
        pl.Series("sample_weight", w["weight"]),
        pl.col("label").cast(pl.Int8),
        pl.col("n_bars").cast(pl.Int32),
        pl.lit(cfg.micro_label_version).alias("micro_label_version"),
    )
    return out.select(list(_OUTPUT_SCHEMA)).sort(["t0", "contract_id"])


# --- storage (versioned, idempotent, leak-aware) --------------------------- #


def _root() -> Path:
    return Settings().data_dir / _DATASET


def partition_glob(symbol: str | None = None) -> str:
    sym = "*" if symbol is None else f"symbol={symbol}"
    return str(_root() / sym / "date=*" / "*.parquet")


def _existing_keys(part_dir: Path) -> set:
    if not part_dir.exists() or not any(part_dir.glob("*.parquet")):
        return set()
    keys = cast("pl.DataFrame", pl.scan_parquet(part_dir / "*.parquet").select("t0").collect())
    return set(keys["t0"])


def _write_symbol(frame: pl.DataFrame, symbol: str) -> int:
    """Append a symbol's label frame; idempotent on ``t0`` per date partition."""
    if frame.is_empty():
        return 0
    frame = frame.with_columns(pl.col("t0").dt.date().alias("_date"))
    written = 0
    for (day,), group in frame.group_by(["_date"], maintain_order=True):
        part_dir = _root() / f"symbol={symbol}" / f"date={day}"
        seen = _existing_keys(part_dir)
        new = group.filter(~pl.col("t0").is_in(list(seen))) if seen else group
        if new.is_empty():
            continue
        part_dir.mkdir(parents=True, exist_ok=True)
        new.drop("_date").write_parquet(
            part_dir / f"part-{uuid4().hex}.parquet", compression="zstd"
        )
        written += new.height
    return written


def read_micro_labels(
    symbol: str, start: datetime, end: datetime, store: DuckStore | None = None
) -> pl.DataFrame:
    """Label rows for ``symbol`` with ``t0`` in ``[start, end]`` (UTC), latest-ingest wins."""
    own = store is None
    store = store or DuckStore()
    try:
        if not _glob(partition_glob(symbol)):
            return pl.DataFrame()
        glob_str = partition_glob(symbol)
        return store.con.execute(
            f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (PARTITION BY t0 ORDER BY ts_ingest DESC) AS rn
                FROM read_parquet('{glob_str}', hive_partitioning=false)
                WHERE t0 >= ? AND t0 <= ?
            ) WHERE rn = 1
            ORDER BY t0
            """,
            [start, end],
        ).pl()
    finally:
        if own:
            store.close()


# --- QA report ------------------------------------------------------------- #


def label_qa(labels: pl.DataFrame, *, events_sampled: int | None = None) -> dict:
    """Per-symbol QA stats for a generated label frame (honest, no tuning)."""
    n = labels.height
    qa: dict = {"events_sampled": events_sampled, "retained": n}
    if n == 0:
        return qa
    hold_s = (labels["t1"] - labels["t0"]).dt.total_microseconds().to_numpy().astype(
        np.float64
    ) / 1e6
    label_arr = labels["label"].to_numpy()
    uniq = labels["uniqueness_weight"].to_numpy()
    n_days = int(labels["session_date"].n_unique())
    effective_n = float(uniq.sum())
    qa.update(
        {
            "n_session_days": n_days,
            "label_balance": {
                "pos": float(np.mean(label_arr == 1)),
                "neg": float(np.mean(label_arr == -1)),
                "zero": float(np.mean(label_arr == 0)),
            },
            "barrier_touched": {
                k: int(v) for k, v in labels["barrier_touched"].value_counts().iter_rows()
            },
            "hold_minutes": {
                "median": float(np.median(hold_s) / 60.0),
                "iqr_lo": float(np.percentile(hold_s, 25) / 60.0),
                "iqr_hi": float(np.percentile(hold_s, 75) / 60.0),
            },
            "frac_resolved_at_close": float(np.mean(labels["resolved_at_close"].to_numpy())),
            "avg_uniqueness": float(uniq.mean()),
            "effective_n": effective_n,
            "effective_n_per_day": effective_n / n_days if n_days else 0.0,
        }
    )
    return qa


def log_label_stats(cfg: MicroLabelConfig, qa_by_symbol: dict[str, dict]) -> str | None:
    """Log label-config + QA stats to the local MLflow file store (best-effort, no model)."""
    try:
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
    except Exception as exc:  # noqa: BLE001 - tracking optional
        logger.warning(f"mlflow unavailable ({exc}); skipping micro-label tracking")
        return None
    mlflow.set_tracking_uri((Settings().data_dir / "mlruns").as_uri())
    mlflow.set_experiment("micro-labels")
    with mlflow.start_run(run_name=f"micro-labels-{cfg.micro_label_version}") as run:
        mlflow.log_params(
            {
                "micro_label_version": cfg.micro_label_version,
                "pt_mult": cfg.barriers.pt_mult,
                "sl_mult": cfg.barriers.sl_mult,
                "vertical_minutes": cfg.barriers.vertical_minutes,
                "cusum_mult": cfg.events.cusum_mult,
                "ewm_span": cfg.volatility.ewm_span,
                "min_samples": cfg.volatility.min_samples,
                "dur_ewm_span": cfg.volatility.dur_ewm_span,
                "weights_scheme": cfg.weights.scheme,
                "symbols": ",".join(qa_by_symbol),
            }
        )
        for sym, qa in qa_by_symbol.items():
            if not qa.get("retained"):
                continue
            mlflow.log_metrics(
                {
                    f"{sym}_events_sampled": float(qa.get("events_sampled") or 0),
                    f"{sym}_retained": float(qa["retained"]),
                    f"{sym}_frac_pos": qa["label_balance"]["pos"],
                    f"{sym}_frac_neg": qa["label_balance"]["neg"],
                    f"{sym}_frac_zero": qa["label_balance"]["zero"],
                    f"{sym}_frac_resolved_at_close": qa["frac_resolved_at_close"],
                    f"{sym}_avg_uniqueness": qa["avg_uniqueness"],
                    f"{sym}_effective_n": qa["effective_n"],
                    f"{sym}_effective_n_per_day": qa["effective_n_per_day"],
                    f"{sym}_hold_minutes_median": qa["hold_minutes"]["median"],
                }
            )
        mlflow.log_dict({"config": cfg.to_dict(), "qa": qa_by_symbol}, "micro_labels_qa.json")
        return run.info.run_id


# --- build (compute + QA + store) ------------------------------------------ #


def build_micro_labels(
    symbols: list[str],
    *,
    write_start: datetime | None = None,
    write_end: datetime | None = None,
    cfg: MicroLabelConfig | None = None,
    mcfg: MicrostructureConfig | None = None,
    store: DuckStore | None = None,
    log_mlflow: bool = True,
) -> dict[str, dict]:
    """Compute short-horizon labels over all on-disk micro bars and write the lake.

    No Databento calls — reads only the ``micro_bars`` already on disk via
    :func:`read_micro_bars`. ``write_start/write_end`` bound only the rows
    persisted (labels are computed over everything available). Returns
    ``{symbol: {"rows_written", "qa"}}``.
    """
    cfg = cfg or MicroLabelConfig.load()
    mcfg = mcfg or MicrostructureConfig.load()
    own = store is None
    store = store or DuckStore()
    try:
        ts_ingest = datetime.now(UTC)
        result: dict[str, dict] = {}
        qa_by_symbol: dict[str, dict] = {}
        for sym in symbols:
            bars = read_micro_bars(sym, _WIDE_START, _WIDE_END, store=store)
            if bars.is_empty() or bars.width == 0:
                result[sym] = {"rows_written": 0, "qa": {"events_sampled": 0, "retained": 0}}
                qa_by_symbol[sym] = result[sym]["qa"]
                continue
            diag: dict = {}
            labels = generate_micro_labels(bars, cfg, mcfg.session, diag=diag)
            qa = label_qa(labels, events_sampled=diag.get("events_sampled"))
            qa_by_symbol[sym] = qa
            labels = labels.with_columns(
                pl.lit(sym).alias("symbol"),
                pl.lit(ts_ingest).cast(pl.Datetime("us", "UTC")).alias("ts_ingest"),
            )
            if write_start is not None:
                labels = labels.filter(pl.col("t0") >= write_start)
            if write_end is not None:
                labels = labels.filter(pl.col("t0") <= write_end)
            written = (
                _write_symbol(labels.select(_PERSIST_COLUMNS), sym) if not labels.is_empty() else 0
            )
            result[sym] = {"rows_written": written, "qa": qa}
        if log_mlflow:
            run_id = log_label_stats(cfg, qa_by_symbol)
            if run_id:
                logger.info(f"micro-label QA logged to MLflow run {run_id}")
        return result
    finally:
        if own:
            store.close()


def _print_report(cfg: MicroLabelConfig, result: dict[str, dict]) -> None:
    print(f"micro_label_version={cfg.micro_label_version} cusum_mult={cfg.events.cusum_mult}")
    for sym, r in result.items():
        qa = r["qa"]
        if not qa.get("retained"):
            print(f"  {sym}: 0 labels (events_sampled={qa.get('events_sampled')})")
            continue
        bal = qa["label_balance"]
        print(
            f"  {sym}: +{r['rows_written']:,} rows | events_sampled={qa['events_sampled']} "
            f"retained={qa['retained']} | balance +/-/0="
            f"{bal['pos']:.2f}/{bal['neg']:.2f}/{bal['zero']:.2f}"
        )
        print(
            f"      hold(min) median={qa['hold_minutes']['median']:.1f} "
            f"IQR[{qa['hold_minutes']['iqr_lo']:.1f},{qa['hold_minutes']['iqr_hi']:.1f}] | "
            f"resolved_at_close={qa['frac_resolved_at_close']:.3f} | "
            f"avg_uniqueness={qa['avg_uniqueness']:.3f}"
        )
        print(
            f"      effective_N={qa['effective_n']:.1f} over {qa['n_session_days']} RTH days "
            f"-> {qa['effective_n_per_day']:.1f}/day | barriers={qa['barrier_touched']}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="microstructure.labels", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: microstructure instruments")
    p.add_argument("--start", help="write-window start YYYY-MM-DD (UTC); compute is always full")
    p.add_argument("--end", help="write-window end YYYY-MM-DD (UTC)")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = p.parse_args(argv)

    cfg = MicroLabelConfig.load()
    mcfg = MicrostructureConfig.load()
    symbols = args.symbols or mcfg.symbols()

    def _parse(d: str | None) -> datetime | None:
        return datetime.fromisoformat(d).replace(tzinfo=UTC) if d else None

    result = build_micro_labels(
        symbols,
        write_start=_parse(args.start),
        write_end=_parse(args.end),
        cfg=cfg,
        mcfg=mcfg,
        log_mlflow=not args.no_mlflow,
    )
    _print_report(cfg, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
