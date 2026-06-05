"""Options-System: an autonomous, local-first futures-trading system.

Phase 1 trades CME micro futures (MES, MNQ) intraday; Phase 2 (later) adds
options vertical spreads. See CLAUDE.md for the full architecture and the
"two brains" split (deterministic live engine vs. offline learning loop).

This is the top-level package. Each subpackage owns one concern and is
documented by its own README.md. No trading logic lives here.
"""

__version__ = "0.0.0"
