# risk/

**The safety layer that sits between every decision and every order — it is
sacrosanct.** Nothing reaches the broker without passing through here. Its
responsibilities: position **sizing** (defined-risk per trade), hard **caps**
(max position, max concurrent risk), a **daily-loss kill-switch** that flattens
and halts trading when a loss limit is hit, and — critically — ensuring every
live position rests a **hard stop-loss order at IBKR itself**, so that a crashed
process, a dead machine, or a dropped internet connection still cannot leave a
position unprotected. The Risk Manager can **veto** or **resize** any proposed
trade and can **flatten** everything on command. It is deliberately simple,
conservative, and exhaustively tested. When in doubt, it says no.
