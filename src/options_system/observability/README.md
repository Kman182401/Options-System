# observability/

**How the human sees and is alerted about the system.** Two surfaces. (1) A
**Streamlit** dashboard for at-a-glance local monitoring — open positions, P&L,
recent decisions, model/risk state, data freshness, system health. (2)
**Telegram** alerts (`python-telegram-bot`) for things that need attention when
the human isn't watching the screen: fills, risk vetoes, the daily-loss
kill-switch firing, connection loss, errors. This module only *reads* system
state and *reports* it; it never makes or influences trading decisions. It
exists to serve the prime directive — the human must be able to understand what
the system is doing at all times.
