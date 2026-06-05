"""Triple-barrier label generator (López de Prado, AFML ch. 3).

For each event ``t0`` we place three barriers and the label is decided by the
**first one touched**:

* upper (profit-take) at ``+pt_mult · σ_{t0}`` cumulative log-return → label ``+1``
* lower (stop-loss)   at ``−sl_mult · σ_{t0}`` cumulative log-return → label ``−1``
* vertical (time) at ``t0 + max_hold_bars`` → label ``0`` (or ``sign(ret)`` if
  ``vertical_label_sign``)

**The instrument is real, the path is in return space.** We walk the
back-adjusted *continuous* close in cumulative-log-return space
(``ln(close_τ) − ln(close_{t0})``). Within a contract that equals the raw
front-month return; across a roll the ratio-adjustment makes the seam return
~0, which is exactly the realistic "rolled position" (the roll is not a P&L
event — costs are modelled later). Because we only ever use return
*differences*, the labels are degree-0 in the price scale → invariant to ratio
back-adjustment (proven in ``tests/test_labeling_triple_barrier.py``).

**Honest right-censoring.** An event whose vertical window runs past the end of
the available data *and* never touches a price barrier is **dropped**, not
resolved early — we genuinely do not know its outcome without future bars.
Resolving it at the last bar would be a shorter, biased hold.

**Roll handling** (config ``roll.handling``):

* ``adjust`` (default) — walk straight through the seam on the continuous path.
* ``close`` — cap the path at the roll bar and resolve there (barrier ``roll``).

Either way a window that spans a roll is flagged ``roll_crossed=True``.

Every label records ``t1`` (resolution time) — non-negotiable: the validation
framework needs it to purge/embargo overlapping samples.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl

from .config import LabelConfig
from .events import compute_sigma, sample_events

# Columns the generator needs on the input continuous frame.
REQUIRED_INPUT = ("ts_event", "close", "contract_id", "session")

# Output schema (keys first, then outcome, then carried metadata + meta hook).
_OUTPUT_SCHEMA = {
    "t0": pl.Datetime("us", "UTC"),
    "t1": pl.Datetime("us", "UTC"),
    "ret": pl.Float64,
    "label": pl.Int8,
    "barrier": pl.Utf8,
    "sigma": pl.Float64,
    "n_bars": pl.Int32,
    "contract_id": pl.Utf8,
    "roll_crossed": pl.Boolean,
    "session": pl.Utf8,
    "degraded": pl.Boolean,
    "side": pl.Int8,  # meta-labeling hook: primary signal side (null until used)
    "meta_label": pl.Int8,  # meta-labeling hook: act/size flag (null until used)
    "label_version": pl.Utf8,
}


def _roll_bar_positions(ts: np.ndarray, rolls: pl.DataFrame | None) -> np.ndarray:
    """Bar index of the first bar at/after each roll ``ts_event`` (sorted, unique)."""
    if rolls is None or rolls.is_empty():
        return np.asarray([], dtype=np.int64)
    roll_ts = rolls.sort("ts_event")["ts_event"].to_numpy()
    pos = np.searchsorted(ts, roll_ts, side="left").astype(np.int64)
    pos = pos[(pos > 0) & (pos < ts.shape[0])]
    return np.unique(pos)


def label_events(
    df: pl.DataFrame,
    event_idx: np.ndarray,
    cfg: LabelConfig,
    *,
    rolls: pl.DataFrame | None = None,
    degraded_dates: frozenset[date] = frozenset(),
) -> pl.DataFrame:
    """Resolve triple-barrier labels for the events at ``event_idx`` over ``df``.

    ``df`` is one symbol's continuous bars (sorted ascending) carrying ``sigma``
    (run :func:`~options_system.labeling.events.compute_sigma` first). Returns one
    row per *resolved* event with the full label schema. Right-censored events
    (vertical window past data, no barrier hit) are dropped.
    """
    missing = [c for c in (*REQUIRED_INPUT, "sigma") if c not in df.columns]
    if missing:
        raise ValueError(f"label_events: input missing columns {missing}")
    n = df.height
    if n == 0 or event_idx.size == 0:
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    b = cfg.barriers
    handling = cfg.roll.handling
    ts = df["ts_event"].to_numpy()
    logc = np.log(df["close"].to_numpy())
    sigma = df["sigma"].to_numpy()
    contract = df["contract_id"].to_list()
    session = df["session"].to_list()

    bar_dates = df["ts_event"].dt.date().to_list()
    is_deg = np.fromiter((d in degraded_dates for d in bar_dates), dtype=bool, count=n)
    prefix_deg = np.concatenate(([0], np.cumsum(is_deg.astype(np.int64))))

    roll_pos = _roll_bar_positions(ts, rolls)

    p_idx: list[int] = []
    pos_idx: list[int] = []
    rets: list[float] = []
    labels: list[int] = []
    barriers: list[str] = []
    sigmas: list[float] = []
    nbars: list[int] = []
    contracts: list[str] = []
    rolls_crossed: list[bool] = []
    sessions: list[str] = []
    degradeds: list[bool] = []

    for p in event_idx.tolist():
        sig = sigma[p]
        if not (math.isfinite(sig) and sig > 0.0):
            continue  # warmup guard (events should already exclude these)

        vbar = p + b.max_hold_bars  # ideal vertical (may exceed data)
        scan_end = min(vbar, n - 1)
        # roll crossing within the evaluable window (p, scan_end]
        crossed = bool(roll_pos.size and np.any((roll_pos > p) & (roll_pos <= scan_end)))
        capped_by_roll = False
        if handling == "close" and crossed:
            first_roll = int(roll_pos[(roll_pos > p) & (roll_pos <= scan_end)][0])
            scan_end = min(scan_end, first_roll)
            capped_by_roll = True

        if scan_end <= p:
            continue  # no forward bar to evaluate

        cr = logc[p + 1 : scan_end + 1] - logc[p]
        up = b.pt_mult * sig
        dn = -b.sl_mult * sig
        up_mask = cr >= up
        dn_mask = cr <= dn
        first_up = int(up_mask.argmax()) if up_mask.any() else -1
        first_dn = int(dn_mask.argmax()) if dn_mask.any() else -1

        if first_up >= 0 and (first_dn < 0 or first_up <= first_dn):
            k, label, barrier = first_up, 1, "up"
        elif first_dn >= 0:
            k, label, barrier = first_dn, -1, "dn"
        else:
            # no price barrier touched within the scanned window
            if capped_by_roll:
                k, barrier = cr.shape[0] - 1, "roll"
            elif vbar <= n - 1:
                k, barrier = cr.shape[0] - 1, "time"
            else:
                continue  # right-censored: window runs past data, outcome unknown
            label = int(np.sign(cr[k])) if b.vertical_label_sign else 0

        pos = p + 1 + k
        p_idx.append(p)
        pos_idx.append(pos)
        rets.append(float(cr[k]))
        labels.append(label)
        barriers.append(barrier)
        sigmas.append(float(sig))
        nbars.append(pos - p)
        contracts.append(contract[p])
        rolls_crossed.append(crossed)
        sessions.append(session[p])
        degradeds.append(bool(prefix_deg[pos + 1] - prefix_deg[p] > 0))

    if not p_idx:
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    ts_col = df["ts_event"]  # gather keeps the Datetime(us, UTC) dtype + tz
    out = pl.DataFrame(
        {
            "t0": ts_col.gather(p_idx),
            "t1": ts_col.gather(pos_idx),
            "ret": rets,
            "label": labels,
            "barrier": barriers,
            "sigma": sigmas,
            "n_bars": nbars,
            "contract_id": contracts,
            "roll_crossed": rolls_crossed,
            "session": sessions,
            "degraded": degradeds,
        }
    )
    out = out.with_columns(
        pl.col("label").cast(pl.Int8),
        pl.col("n_bars").cast(pl.Int32),
        pl.lit(None, dtype=pl.Int8).alias("side"),
        pl.lit(None, dtype=pl.Int8).alias("meta_label"),
        pl.lit(cfg.label_version).alias("label_version"),
    )
    return out.select(list(_OUTPUT_SCHEMA)).sort("t0")


def generate_labels(
    df: pl.DataFrame,
    cfg: LabelConfig,
    *,
    rolls: pl.DataFrame | None = None,
    degraded_dates: frozenset[date] = frozenset(),
) -> pl.DataFrame:
    """End-to-end: σ → event sampling → triple-barrier labels for one symbol.

    ``df`` is one symbol's continuous (outright) bars. Convenience wrapper used by
    the builder and tests. Returns the full label schema (see ``_OUTPUT_SCHEMA``).
    """
    if df.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    if df.select(pl.col("contract_id").str.contains("-").any()).item():
        raise ValueError("generate_labels: spread contract_id (containing '-') in input")
    df = df.sort("ts_event").pipe(compute_sigma, cfg)
    idx = sample_events(df, cfg)
    return label_events(df, idx, cfg, rolls=rolls, degraded_dates=degraded_dates)
