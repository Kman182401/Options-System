#!/usr/bin/env fish
# Launch IB Gateway (PAPER) via IBC with auto-login.
#
# Credentials come from .env (OPTIONS_IBKR_USERNAME / OPTIONS_IBKR_PASSWORD) and
# are rendered into a tmpfs IBC config (mode 600) that is never persisted to disk
# or committed. IBC's own gatewaystart.sh hardcodes its variables, so we run a
# patched COPY (in tmpfs) and leave the installed IBC untouched.
#
# Stop with: scripts/stop_gateway.fish
#
# NOTE: unverified until your first successful paper login. If IBC can't find the
# Gateway install, adjust TWS_PATH / TWS_MAJOR_VRSN below (Gateway is at
# ~/ibgateway, version 10.45). Manual launch always works: ~/ibgateway/ibgateway

set -l here (status dirname)
set -l repo (path resolve $here/..)
cd $repo; or exit 1

set -l runtime $XDG_RUNTIME_DIR
test -n "$runtime"; or set runtime /tmp
set -l cfg "$runtime/options-ibc-config.ini"
set -l launcher "$runtime/options-gatewaystart.sh"

# Render IBC config (creds from .env) to tmpfs; aborts cleanly if creds unset.
uv run python scripts/render_ibc_config.py "$cfg"; or exit 1

mkdir -p "$repo/logs/ibc"

# Patch a private copy of IBC's launcher with our paths/version/config.
sed -e "s|^TWS_MAJOR_VRSN=.*|TWS_MAJOR_VRSN=1045|" \
    -e "s|^IBC_INI=.*|IBC_INI=$cfg|" \
    -e "s|^TRADING_MODE=.*|TRADING_MODE=paper|" \
    -e "s|^IBC_PATH=.*|IBC_PATH=$HOME/ibc|" \
    -e "s|^TWS_PATH=.*|TWS_PATH=$HOME/ibgateway|" \
    -e "s|^LOG_PATH=.*|LOG_PATH=$repo/logs/ibc|" \
    "$HOME/ibc/gatewaystart.sh" >"$launcher"
chmod +x "$launcher"

echo "Launching IB Gateway via IBC (paper). Logs: $repo/logs/ibc"
exec bash "$launcher"
