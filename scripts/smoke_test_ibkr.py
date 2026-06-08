"""IBKR **paper** connectivity smoke test.

Connects to the IB Gateway/TWS **paper** account (host/port/clientId from
`config.settings`), prints the account summary, resolves the front-month MES
contract, and pulls one recent historical bar — then disconnects cleanly.

This places NO orders (the API session is opened read-only). If it can't
connect, it prints exactly what to fix and exits non-zero. That is the correct,
expected behavior when IB Gateway isn't running — see `docs/SETUP.md`.

Run:  uv run python scripts/smoke_test_ibkr.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Make the repo root importable so `config` resolves under `python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ib_async import IB, Future  # noqa: E402

from config.settings import Settings  # noqa: E402

# Account-summary tags worth printing for a quick health check.
_SUMMARY_TAGS = {
    "NetLiquidation",
    "AvailableFunds",
    "BuyingPower",
    "TotalCashValue",
    "AccountType",
}


def _print_setup_help(settings: Settings, error: Exception) -> None:
    print("\n" + "=" * 72)
    print("Could NOT connect to the IBKR paper gateway.")
    print(
        f"  tried: {settings.ibkr_host}:{settings.ibkr_port} (clientId={settings.ibkr_client_id})"
    )
    print(f"  error: {type(error).__name__}: {error}")
    print("-" * 72)
    print("Checklist (see docs/SETUP.md for full steps):")
    print("  1. IB Gateway (or TWS) is running and logged into the PAPER account.")
    print("  2. API is enabled:  Configure > Settings > API > Enable ActiveX and Socket Clients.")
    print("  3. Socket port matches IBKR_PORT (IB Gateway paper = 4002, TWS paper = 7497).")
    print("  4. 127.0.0.1 is in the API 'Trusted IPs' list.")
    print("=" * 72)


def main() -> int:
    settings = Settings()  # mode is hard-locked to 'paper'
    ib = IB()

    try:
        # readonly=True: this smoke test can never place/modify orders.
        ib.connect(
            settings.ibkr_host,
            settings.ibkr_port,
            clientId=settings.ibkr_client_id,
            timeout=10,
            readonly=True,
        )
    except Exception as exc:  # noqa: BLE001 - we want a friendly message for any failure
        _print_setup_help(settings, exc)
        return 1

    try:
        print(
            "Connected to IBKR paper gateway "
            f"({settings.ibkr_host}:{settings.ibkr_port}, clientId={settings.ibkr_client_id})."
        )

        accounts = ib.managedAccounts()
        print(f"\nManaged accounts: {accounts}")
        if accounts and accounts[0].startswith("DU"):
            print("  -> account id starts with 'DU' = PAPER account. Good.")

        # --- Account summary ---
        print("\nAccount summary:")
        summary = ib.accountSummary()
        if not summary:
            print("  (no summary rows returned)")
        for row in summary:
            if row.tag in _SUMMARY_TAGS:
                print(f"  {row.tag:16} {row.value} {row.currency}")

        # --- Resolve front-month MES ---
        print("\nResolving front-month MES (CME)...")
        details = ib.reqContractDetails(Future(symbol="MES", exchange="CME", currency="USD"))
        if not details:
            print("  No contract details returned for MES. Check market-data/permissions.")
            return 1

        this_month = datetime.now().strftime("%Y%m")
        contracts = sorted(
            (cd.contract for cd in details if cd.contract is not None),
            key=lambda c: c.lastTradeDateOrContractMonth,
        )
        front = next(
            (c for c in contracts if c.lastTradeDateOrContractMonth[:6] >= this_month),
            contracts[0],
        )
        print(
            f"  Front month: {front.localSymbol or front.symbol} "
            f"expiry={front.lastTradeDateOrContractMonth} conId={front.conId}"
        )

        # --- One recent historical bar (delayed data is fine for Phase 0) ---
        ib.reqMarketDataType(3)  # 3 = delayed; harmless for historical
        bars = ib.reqHistoricalData(
            front,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="1 hour",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
        )
        if bars:
            b = bars[-1]
            print("\nMost recent 1-hour bar:")
            print(f"  {b.date}  O={b.open} H={b.high} L={b.low} C={b.close} V={b.volume}")
        else:
            print(
                "\nNo historical bars returned (likely market-data permissions on the "
                "paper account). Connection + contract resolution still succeeded."
            )

        print("\nIBKR smoke test OK.")
        return 0
    finally:
        ib.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
