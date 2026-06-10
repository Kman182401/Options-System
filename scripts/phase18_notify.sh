#!/usr/bin/env bash
# Phase 18 completion notifier — fires a Telegram message when the detached GDELT
# backfill systemd unit (gdelt-backfill-chain) goes inactive. Polls the unit; on
# exit, summarizes the manifest and sends one clearly-labeled [Options-System]
# message via the security bot. Self-contained; not part of the app.
set -u

UNIT="gdelt-backfill-chain.service"
REPO="/home/karson/Options-System"
MANIFEST="$REPO/data/sentiment_backfill/manifest.json"
LOG="/tmp/backfill_chain.log"

send() {
  local text="$1"
  local tok chat
  tok="$(pass show braxen/telegram-security-bot-token 2>/dev/null)" || return 1
  chat="$(pass show braxen/telegram-security-chat-id 2>/dev/null)" || return 1
  TOK="$tok" CHAT="$chat" TEXT="$text" python3 - <<'PY'
import os, json, urllib.request, urllib.parse
data = urllib.parse.urlencode({"chat_id": os.environ["CHAT"], "text": os.environ["TEXT"]}).encode()
try:
    r = urllib.request.urlopen(
        f"https://api.telegram.org/bot{os.environ['TOK']}/sendMessage", data=data, timeout=20
    )
    print("sent ok=%s" % json.load(r).get("ok"))
except Exception as exc:  # noqa: BLE001
    print("send FAILED: %s" % exc)
PY
}

# Wait for the unit to leave 'active' (poll every 60s).
while systemctl --user is-active --quiet "$UNIT"; do
  sleep 60
done

SUMMARY="$(python3 - "$MANIFEST" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1]))
except Exception as exc:  # noqa: BLE001
    print(f"manifest unreadable: {exc}")
    raise SystemExit
s = m["slices"]
ok = sum(1 for e in s.values() if e.get("failed") is None)
fail = sum(1 for e in s.values() if e.get("failed") is not None)
trunc = sum(1 for e in s.values() if e.get("truncated"))
recs = sum(int(e.get("records_written") or 0) for e in s.values())
runs = m.get("runs", [])
outcome = runs[-1].get("outcome") if runs else "?"
print(
    f"slices_attempted={len(s)} ok={ok} failed_now={fail} truncated={trunc} "
    f"records_written={recs} runs={len(runs)} last_outcome={outcome}"
)
PY
)"

LASTLOG="$(tail -n 3 "$LOG" 2>/dev/null | tr '\n' ' ')"

send "$(printf '\U0001F7E2 [Options-System] Phase 18 GDELT backfill chain FINISHED.\n%s\nlog tail: %s\n\nNext: FinBERT scoring + coverage gate verdict (G1/G2/G3). Reopen the Claude chat and say "continue Phase 18" to run it, or it runs automatically if the chat is still open.' "$SUMMARY" "$LASTLOG")"
