"""Phase-14 microstructure signal model — a SEPARATE model from the daily one.

This package trains and honestly validates a short-horizon **3-class** LightGBM
signal model on the MBP-1 microstructure dataset (m1 order-flow features on dollar
bars, ml1 triple-barrier labels). It reuses the Phase-4 leak-safe validation
machinery (purged K-fold, CPCV, PBO, PSR/DSR) and adds what this regime needs:
**fold-local class weighting** for the ~78-80% timeout class and a **gross
signal-return** proxy. It is a SIGNAL VERDICT only — not a strategy, not an
economic backtest, and nothing here trades.
"""
