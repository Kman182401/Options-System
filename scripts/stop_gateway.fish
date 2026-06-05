#!/usr/bin/env fish
# Stop the IBC-launched IB Gateway and remove the tmpfs credential config.
#
# If you run the Gateway under systemd instead, prefer:
#   systemctl --user stop options-gateway.service

set -l runtime $XDG_RUNTIME_DIR
test -n "$runtime"; or set runtime /tmp

# IBC ships a stop helper that talks to its command server.
if test -x "$HOME/ibc/stop.sh"
    bash "$HOME/ibc/stop.sh" 2>/dev/null
end

# Fallback: terminate the Gateway / IBC java processes.
pkill -f "ibgateway" 2>/dev/null
pkill -f "ibcalpha|IBC.jar" 2>/dev/null

# Scrub the rendered credential config from tmpfs.
rm -f "$runtime/options-ibc-config.ini" "$runtime/options-gatewaystart.sh"
echo "IB Gateway stopped; tmpfs credential config removed."
