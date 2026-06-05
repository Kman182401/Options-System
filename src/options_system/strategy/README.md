# strategy/

**The trading logic itself, expressed as a `nautilus_trader` Strategy.** This is
where model inference becomes intent: it consumes features + the champion
model's signal (and later sentiment), and decides *whether*, *which way*, and
*how much* to trade — then proposes orders. Because it subclasses
`nautilus_trader`'s `Strategy`, the **exact same code runs in backtest and in
live**, which is what prevents "worked in backtest, broke live" drift. The
strategy proposes; it never has the final say — every decision is passed to the
`risk` module, which can veto or resize it. **The actual strategy will be
researched and selected by Claude in a later phase** via hypothesis-driven work
and rigorous walk-forward validation; this folder is intentionally empty of
logic until then.
