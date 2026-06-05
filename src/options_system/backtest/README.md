# backtest/

**Evaluates strategies and models honestly, before any money is at risk.** This
module wraps `nautilus_trader`'s backtest engine so the *same* strategy code
used live can be replayed over historical data, and adds the validation
discipline on top: **walk-forward** analysis (train on a window, test on the
next, roll forward) instead of a single in-sample fit, with **realistic costs
and slippage** modeled so results aren't fantasy. The output feeds the
champion–challenger gate in `models/`. The guiding rule of this module is the
project's anti-overfitting stance: a result that only looks good in-sample is
treated as no result at all. Pure evaluation — it places no live orders.
