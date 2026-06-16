"""External-data safety policy: fail-closed classification + network gating."""

from __future__ import annotations

import pytest

from options_system.common.external_data_policy import (
    ExternalAccessNotAuthorized,
    SourcePolicy,
    assert_network_allowed,
    assert_source_usable,
    classify,
    requires_network,
)


def test_unknown_source_blocked():
    assert classify("some_random_api") is SourcePolicy.UNKNOWN_BLOCKED
    with pytest.raises(ExternalAccessNotAuthorized):
        assert_network_allowed("some_random_api", allow_network=True)
    with pytest.raises(ExternalAccessNotAuthorized):
        assert_source_usable("some_random_api")


@pytest.mark.parametrize("src", ["databento", "finnhub"])
def test_paid_source_blocked(src):
    assert classify(src) is SourcePolicy.PAID_BLOCKED
    with pytest.raises(ExternalAccessNotAuthorized):
        assert_network_allowed(src, allow_network=True)
    with pytest.raises(ExternalAccessNotAuthorized):
        assert_source_usable(src)


def test_free_source_needs_explicit_network_allow():
    assert classify("gdelt") is SourcePolicy.FREE_NO_AUTH
    # Blocked by default (no explicit opt-in).
    with pytest.raises(ExternalAccessNotAuthorized):
        assert_network_allowed("gdelt", allow_network=False)
    # Allowed ONLY with the explicit opt-in.
    assert_network_allowed("gdelt", allow_network=True)  # does not raise
    assert requires_network("gdelt") is True
    # Usable for offline scaffolding work.
    assert assert_source_usable("gdelt") is SourcePolicy.FREE_NO_AUTH


def test_local_only_never_needs_network():
    assert classify("finbert_local") is SourcePolicy.LOCAL_ONLY
    assert requires_network("finbert_local") is False
    # A network request for a local-only source is a programming error -> refused.
    with pytest.raises(ExternalAccessNotAuthorized):
        assert_network_allowed("finbert_local", allow_network=True)
    # But it is a usable (local) source.
    assert assert_source_usable("finbert_local") is SourcePolicy.LOCAL_ONLY


def test_gkg_bulk_is_free_no_auth():
    assert classify("gdelt_gkg") is SourcePolicy.FREE_NO_AUTH
    with pytest.raises(ExternalAccessNotAuthorized):
        assert_network_allowed("gdelt_gkg", allow_network=False)
    assert_network_allowed("gdelt_gkg", allow_network=True)  # explicit opt-in only
    assert requires_network("gdelt_gkg") is True
    assert assert_source_usable("gdelt_gkg") is SourcePolicy.FREE_NO_AUTH


def test_fred_is_free_auth_and_gated_like_free():
    assert classify("fred") is SourcePolicy.FREE_AUTH
    # FREE_AUTH is network-eligible and still requires the explicit opt-in.
    assert requires_network("fred") is True
    with pytest.raises(ExternalAccessNotAuthorized):
        assert_network_allowed("fred", allow_network=False)
    assert_network_allowed("fred", allow_network=True)  # key check is the caller's job
    # Usable for offline work (a free, vetted source).
    assert assert_source_usable("fred") is SourcePolicy.FREE_AUTH


def test_case_insensitive():
    assert classify("  GDELT ") is SourcePolicy.FREE_NO_AUTH
    assert classify("Databento") is SourcePolicy.PAID_BLOCKED
    assert classify(" Fred ") is SourcePolicy.FREE_AUTH
