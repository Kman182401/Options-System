"""Phase 0 sanity tests — NOT trading tests.

These only confirm the skeleton is wired together: the package imports and the
typed config loads in paper mode. Real behavior is tested in later phases.
"""

from __future__ import annotations


def test_package_imports() -> None:
    import options_system

    assert options_system.__version__ is not None


def test_settings_loads_in_paper_mode() -> None:
    from config.settings import Settings

    settings = Settings()
    assert settings.mode == "paper"


def test_live_mode_is_rejected() -> None:
    """Safety: configuration must never be able to select live trading."""
    import pytest

    from config.settings import Settings

    with pytest.raises(ValueError):
        Settings(mode="live")
