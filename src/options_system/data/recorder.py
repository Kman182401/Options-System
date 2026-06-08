"""Live forward-recorder: stream MES/MNQ L1 + bars from IBKR into the Parquet lake.

This is the whole point of starting now: every minute it runs builds free,
point-in-time-correct, survivorship-free history we own forever.

What it records (paper Gateway, **read-only** API — never places orders):

* **5-second real-time bars** (``reqRealTimeBars``) stored as ``bars_5s`` and
  aggregated into ``bars_1m``.
* **L1 top-of-book** quotes (``reqMktData``) stored as ``quotes_l1``.

Deliberately NOT recorded: L2 / market depth (needs the paid CME depth sub). A
clearly-marked extension point is left in :meth:`Recorder._subscribe_symbol`.

Robustness: reconnects on disconnect, survives the IBKR daily-restart window,
tags every row RTH/ETH, logs throughput + last-bar age, flushes on an interval,
and exits non-zero on a fatal error. Run it:

    uv run python -m options_system.data.recorder

or as the ``options-recorder`` systemd user unit (see scripts/systemd/).

NOTE: the IBKR-connected paths are unverified until the first paper login; the
pure aggregation/session logic below is unit-tested.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import polars as pl

from config.settings import Settings
from options_system.common.logging import get_logger

from .continuous import Contract, pick_front_month
from .lake import SCHEMA_VERSION, Lake

if TYPE_CHECKING:
    from ib_async import IB

log = get_logger("data.recorder")

_ET = ZoneInfo("America/New_York")
# CME equity-index regular trading hours, in Eastern time.
_RTH_OPEN = (9, 30)
_RTH_CLOSE = (16, 0)


def session_for(ts: datetime) -> str:
    """Tag a UTC timestamp as ``RTH`` (regular hours) or ``ETH`` (extended)."""
    et = ts.astimezone(_ET)
    if et.weekday() >= 5:  # Sat/Sun
        return "ETH"
    minutes = et.hour * 60 + et.minute
    open_m = _RTH_OPEN[0] * 60 + _RTH_OPEN[1]
    close_m = _RTH_CLOSE[0] * 60 + _RTH_CLOSE[1]
    return "RTH" if open_m <= minutes < close_m else "ETH"


class MinuteAggregator:
    """Aggregate 5-second bars into 1-minute bars, per contract.

    A completed minute is emitted when the first bar of the *next* minute
    arrives (or via :meth:`flush` on shutdown). Pure and unit-tested — no IBKR.
    """

    def __init__(self) -> None:
        self._cur: dict[str, dict] = {}

    def add(
        self,
        contract_id: str,
        ts_event: datetime,
        o: float,
        h: float,
        low: float,
        c: float,
        volume: float,
        wap: float,
        n_trades: int,
    ) -> dict | None:
        minute = ts_event.replace(second=0, microsecond=0)
        cur = self._cur.get(contract_id)
        emitted: dict | None = None
        if cur is None or cur["minute"] != minute:
            if cur is not None:
                emitted = self._finalize(contract_id, cur)
            self._cur[contract_id] = {
                "minute": minute,
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": volume,
                "vw": wap * volume,  # for volume-weighted wap
                "n_trades": n_trades,
            }
        else:
            cur["high"] = max(cur["high"], h)
            cur["low"] = min(cur["low"], low)
            cur["close"] = c
            cur["volume"] += volume
            cur["vw"] += wap * volume
            cur["n_trades"] += n_trades
        return emitted

    @staticmethod
    def _finalize(contract_id: str, cur: dict) -> dict:
        vol = cur["volume"]
        return {
            "ts_event": cur["minute"],
            "contract_id": contract_id,
            "open": cur["open"],
            "high": cur["high"],
            "low": cur["low"],
            "close": cur["close"],
            "volume": vol,
            "wap": (cur["vw"] / vol) if vol else cur["close"],
            "n_trades": cur["n_trades"],
        }

    def flush(self) -> list[dict]:
        """Finalize and return all in-progress minutes (call on shutdown)."""
        out = [self._finalize(cid, cur) for cid, cur in self._cur.items()]
        self._cur.clear()
        return out


class Recorder:
    """Streams MES/MNQ L1 + bars into the lake. Supervises its own connection."""

    def __init__(self, settings: Settings | None = None, lake: Lake | None = None) -> None:
        self.settings = settings or Settings()
        self.lake = lake or Lake()
        self._agg = MinuteAggregator()
        # buffered rows awaiting flush
        self._buf: dict[str, list[dict]] = defaultdict(list)
        # contract_id -> symbol, and the ib contract object
        self._contract_symbol: dict[str, str] = {}
        self._ib_contracts: dict[str, object] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self._last_bar_ts: dict[str, datetime] = {}
        self._stop = False
        self.ib: IB | None = None  # set in run()

    # -- row builders (pure) ------------------------------------------------
    def _bar_row(self, symbol: str, contract_id: str, con_id: int | None, b: dict) -> dict:
        ts = b["ts_event"]
        return {
            **b,
            "ts_ingest": datetime.now(UTC),
            "symbol": symbol,
            "con_id": con_id,
            "session": session_for(ts),
            "source": "ibkr",
            "schema_version": SCHEMA_VERSION,
        }

    def _buffer_bar5s(self, symbol, contract_id, con_id, ts, o, h, low, c, vol, wap, count) -> None:
        self._buf["bars_5s"].append(
            self._bar_row(
                symbol,
                contract_id,
                con_id,
                {
                    "ts_event": ts,
                    "contract_id": contract_id,
                    "open": o,
                    "high": h,
                    "low": low,
                    "close": c,
                    "volume": vol,
                    "wap": wap,
                    "n_trades": count,
                },
            )
        )
        minute = self._agg.add(contract_id, ts, o, h, low, c, vol, wap, count)
        if minute is not None:
            self._buf["bars_1m"].append(self._bar_row(symbol, contract_id, con_id, minute))
        self._counts[contract_id] += 1
        self._last_bar_ts[contract_id] = ts

    def _buffer_quote(self, symbol, contract_id, con_id, ts, bid, ask, bsz, asz, last, lsz) -> None:
        self._buf["quotes_l1"].append(
            {
                "ts_event": ts,
                "ts_ingest": datetime.now(UTC),
                "symbol": symbol,
                "contract_id": contract_id,
                "con_id": con_id,
                "bid": bid,
                "ask": ask,
                "bid_size": bsz,
                "ask_size": asz,
                "last": last,
                "last_size": lsz,
                "session": session_for(ts),
                "source": "ibkr",
                "schema_version": SCHEMA_VERSION,
            }
        )

    # -- flush --------------------------------------------------------------
    def flush(self) -> int:
        written = 0
        for dataset, rows in list(self._buf.items()):
            if not rows:
                continue
            written += self.lake.write(dataset, pl.DataFrame(rows))
            self._buf[dataset].clear()
        return written

    def _log_health(self) -> None:
        now = datetime.now(UTC)
        for cid, last in self._last_bar_ts.items():
            age = (now - last).total_seconds()
            log.info(f"{cid}: {self._counts[cid]} bars, last-bar-age {age:.0f}s")

    # ===================== IBKR-connected paths (untested live) ============
    async def _resolve_front(self, symbol: str):
        from ib_async import Future

        assert self.ib is not None  # connected in _main() before any resolve
        details = await self.ib.reqContractDetailsAsync(
            Future(symbol=symbol, exchange="CME", currency="USD")
        )
        if not details:
            raise RuntimeError(f"no contract details for {symbol} (market data permissions?)")
        candidates: list[Contract] = []
        ib_by_id: dict[str, object] = {}
        for cd in details:
            c = cd.contract
            assert c is not None  # IBKR ContractDetails always carry a contract
            expiry = datetime.strptime(c.lastTradeDateOrContractMonth[:8], "%Y%m%d").date()
            cid = c.localSymbol or f"{symbol}{c.lastTradeDateOrContractMonth}"
            candidates.append(Contract(cid, expiry, c.conId))
            ib_by_id[cid] = c
        front = pick_front_month(
            candidates, datetime.now(UTC).date(), self.settings.roll_calendar_days
        )
        return ib_by_id[front.contract_id], front.contract_id, front.con_id

    async def _subscribe_symbol(self, symbol: str) -> None:
        assert self.ib is not None  # connected in _main() before any subscribe
        ib_contract, contract_id, con_id = await self._resolve_front(symbol)
        self._contract_symbol[contract_id] = symbol
        self._ib_contracts[contract_id] = ib_contract
        log.info(f"{symbol}: front month {contract_id} (conId={con_id})")

        # 5-second real-time bars -> bars_5s + aggregated bars_1m
        bars = self.ib.reqRealTimeBars(ib_contract, 5, "TRADES", useRTH=False)
        bars.updateEvent += lambda bars, has_new: self._on_bar(
            symbol, contract_id, con_id, bars, has_new
        )

        # L1 top-of-book
        self.ib.reqMarketDataType(3)  # delayed is fine for Phase 0/1
        ticker = self.ib.reqMktData(ib_contract, "", False, False)
        ticker.updateEvent += lambda t: self._on_ticker(symbol, contract_id, con_id, t)

        # --- L2 / market-depth extension point (DEFERRED — needs paid CME depth) ---
        # self.ib.reqMktDepth(ib_contract, numRows=10)  # do NOT enable yet

    def _on_bar(self, symbol, contract_id, con_id, bars, has_new_bar) -> None:
        if not has_new_bar or not bars:
            return
        b = bars[-1]
        ts = b.time.astimezone(UTC) if b.time.tzinfo else b.time.replace(tzinfo=UTC)
        self._buffer_bar5s(
            symbol,
            contract_id,
            con_id,
            ts,
            float(b.open_),
            float(b.high),
            float(b.low),
            float(b.close),
            float(b.volume),
            float(b.wap),
            int(b.count),
        )

    def _on_ticker(self, symbol, contract_id, con_id, t) -> None:
        ts = (t.time or datetime.now(UTC)).astimezone(UTC)

        def _num(x):
            return float(x) if x is not None and x == x else None  # filters NaN

        self._buffer_quote(
            symbol,
            contract_id,
            con_id,
            ts,
            _num(t.bid),
            _num(t.ask),
            _num(t.bidSize),
            _num(t.askSize),
            _num(t.last),
            _num(t.lastSize),
        )

    async def _connect(self) -> None:
        assert self.ib is not None  # _main() constructs self.ib before _connect
        await self.ib.connectAsync(
            self.settings.ibkr_host,
            self.settings.ibkr_port,
            clientId=self.settings.recorder_client_id,
            timeout=15,
            readonly=True,
        )
        log.info(f"connected to {self.settings.ibkr_host}:{self.settings.ibkr_port}")
        for symbol in self.settings.record_symbols:
            await self._subscribe_symbol(symbol)

    async def _main(self) -> None:
        from ib_async import IB

        self.ib = IB()
        await self._connect()
        flush_s = self.settings.recorder_flush_seconds
        while not self._stop:
            await asyncio.sleep(flush_s)
            if not self.ib.isConnected():
                log.warning("disconnected; reconnecting (daily restart window?)...")
                try:
                    await self._connect()
                except Exception as exc:  # noqa: BLE001
                    log.error(f"reconnect failed: {exc}; retrying next cycle")
                    continue
            n = self.flush()
            if n:
                log.debug(f"flushed {n} rows")
            self._log_health()
        # final drain
        for minute in self._agg.flush():
            sym = self._contract_symbol.get(minute["contract_id"], minute["contract_id"][:3])
            self._buf["bars_1m"].append(self._bar_row(sym, minute["contract_id"], None, minute))
        self.flush()
        if self.ib.isConnected():
            self.ib.disconnect()

    def run(self) -> int:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):  # pragma: no cover
                loop.add_signal_handler(sig, self._request_stop)
        try:
            loop.run_until_complete(self._main())
            return 0
        except Exception:
            log.exception("recorder fatal error")
            return 1
        finally:
            loop.close()

    def _request_stop(self) -> None:
        log.info("stop requested; draining buffers...")
        self._stop = True


def main() -> int:
    settings = Settings()
    log.info(f"recorder starting: symbols={settings.record_symbols} mode={settings.mode}")
    return Recorder(settings).run()


if __name__ == "__main__":
    raise SystemExit(main())
