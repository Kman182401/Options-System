# execution/

**Sends risk-approved orders to the broker and tracks their lifecycle.** This is
the live wiring: a `nautilus_trader` live node connected through **`ib_async`**
to **Interactive Brokers** — pointed at the **paper** account/Gateway only,
until the human explicitly approves real-money trading. It submits orders,
handles fills/partial-fills/rejections/cancels, reconciles positions, and keeps
the engine's view of broker state honest. It contains no decision-making and no
risk logic — by the time an order arrives here it has already been decided by
`strategy` and approved (and sized, and stop-protected) by `risk`. Connection
setup, the API socket port, and client-id come from typed config; see
`docs/SETUP.md` for the IB Gateway steps.
