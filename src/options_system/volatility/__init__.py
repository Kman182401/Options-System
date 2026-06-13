"""Phase-21 volatility-forecast skill model — a SEPARATE concern from the direction models.

After six honest nulls on intraday *direction*, this package redirects the proven leak-safe
framework at a genuinely forecastable target — **realized volatility** — at a daily / multi-day
horizon. It builds a noise-reduced 5-minute realized-variance estimator on `bars_1m`, the
standard HAR-RV benchmark (Corsi 2009), and a single fixed regularized LightGBM regressor, then
asks one pre-registered question: does the ML model forecast h-day-ahead RV more accurately than
HAR-RV (by QLIKE, with a significant Diebold-Mariano improvement robust across regimes), per
symbol? It is a **forecast-skill verdict only** — forecast skill is not tradeable money, and
nothing here trades. The frozen contract is `docs/PHASE21_PREREGISTRATION.md`.
"""
