"""Render a complete IBC config.ini to a tmpfs path, credentials from .env.

Called by ``scripts/start_gateway.fish``. Takes IBC's installed base config
(``~/ibc/config.ini``) and overrides only the keys this project needs (paper
mode, API port, read-only API, trusted localhost, auto-login credentials),
writing the result to the path you pass — with ``0600`` perms.

The rendered file contains the IBKR password in plaintext, so it must live on
tmpfs (e.g. ``$XDG_RUNTIME_DIR``) and is never committed. This script writes only
to the output path argument. If credentials aren't set in ``.env`` it exits
non-zero with a clear message (log in to IB Gateway by hand instead).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import Settings  # noqa: E402

IBC_BASE_CONFIG = Path.home() / "ibc" / "config.ini"


def render(out_path: Path, base: Path = IBC_BASE_CONFIG) -> int:
    settings = Settings()
    if not settings.ibkr_username or not settings.ibkr_password:
        print(
            "OPTIONS_IBKR_USERNAME / OPTIONS_IBKR_PASSWORD are not set in .env.\n"
            "Either set them for IBC auto-login, or log in to IB Gateway by hand\n"
            "(see docs/SETUP.md section 3).",
            file=sys.stderr,
        )
        return 1
    if not base.is_file():
        print(f"IBC base config not found at {base}. Is IBC installed?", file=sys.stderr)
        return 1

    overrides = {
        "IbLoginId": settings.ibkr_username,
        "IbPassword": settings.ibkr_password.get_secret_value(),
        "TradingMode": settings.mode,  # hard-locked to 'paper'
        "OverrideTwsApiPort": str(settings.ibkr_port),
        "ReadOnlyApi": "yes",  # the data recorder must never place orders
        "TrustedTwsApiClientIPs": "127.0.0.1",
        "AcceptIncomingConnectionAction": "accept",
        "ExistingSessionDetectedAction": "primary",
        "AcceptNonBrokerageAccountWarning": "yes",
        "DismissPasswordExpiryWarning": "yes",
        "IbAutoClosedown": "no",
    }

    seen: set[str] = set()
    out_lines: list[str] = []
    for line in base.read_text().splitlines():
        stripped = line.lstrip()
        key = (
            line.split("=", 1)[0].strip()
            if ("=" in line and not stripped.startswith("#"))
            else None
        )
        if key in overrides:
            out_lines.append(f"{key}={overrides[key]}")
            seen.add(key)
        else:
            out_lines.append(line)
    for key, val in overrides.items():
        if key not in seen:
            out_lines.append(f"{key}={val}")

    # Open with 0600 BEFORE writing — the file holds the password in plaintext.
    fd = os.open(str(out_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    print(f"Rendered IBC config -> {out_path} (mode 600, paper, port {settings.ibkr_port}).")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: render_ibc_config.py <output_path>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(render(Path(sys.argv[1])))
