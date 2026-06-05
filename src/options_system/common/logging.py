"""Logging for the whole system — a thin wrapper around ``loguru``.

Every module gets its logger via ``get_logger(__name__)``. The first call
configures two sinks (idempotently): a colorized console sink and a rotating
file sink under the configured ``logs/`` directory. ``diagnose`` is disabled on
the file sink so exception logs never serialize local variables (which could
contain secrets) to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[name]}</cyan> | "
    "<level>{message}</level>"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[name]} | {message}"


def _configure(log_dir: Path, console_level: str) -> None:
    """Set up loguru sinks exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Default the bound `name` so the format never raises if a caller forgets to bind.
    logger.configure(extra={"name": "options_system"})
    logger.remove()  # drop loguru's default stderr handler

    logger.add(sys.stderr, level=console_level, format=_CONSOLE_FORMAT, enqueue=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "options_system.log",
        level="DEBUG",
        format=_FILE_FORMAT,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,  # safe under threads / asyncio (nautilus is async)
        backtrace=True,
        diagnose=False,  # never write variable values (may hold secrets) to disk
    )
    _CONFIGURED = True


def get_logger(
    name: str = "options_system",
    *,
    log_dir: str | Path | None = None,
    console_level: str = "INFO",
):
    """Return a logger bound to ``name``.

    On the first call, configures the console + rotating-file sinks. ``log_dir``
    defaults to ``Settings().logs_dir`` (imported lazily to avoid a config import
    at module load time).
    """
    if log_dir is None:
        from config.settings import Settings

        log_dir = Settings().logs_dir
    _configure(Path(log_dir), console_level)
    return logger.bind(name=name)
