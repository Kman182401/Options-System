"""GKG adapter (parse + filter + tone mapping) and config tests — all offline."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from options_system.sentiment import gkg
from options_system.sentiment.gkg_config import GkgConfig

INGEST = datetime(2026, 1, 1, tzinfo=UTC)


def _row(**over: str) -> str:
    """Build one 27-field GKG TSV row with sensible defaults; override by index name."""
    f = [""] * 27
    f[0] = "20240102000000-1"
    f[1] = "20240102000000"
    f[2] = "1"
    f[3] = "reuters.com"
    f[4] = "https://reuters.com/a"
    f[7] = "ECON_STOCKMARKET;EPU_ECONOMY"
    f[15] = "2.5,5.0,2.5,7.5,20,5,150"
    f[26] = "<PAGE_TITLE>Markets rally</PAGE_TITLE>"
    idx = {"id": 0, "date": 1, "source": 3, "url": 4, "themes": 7, "tone": 15, "extras": 26}
    for k, v in over.items():
        f[idx[k]] = v
    return "\t".join(f)


def _parse(text: str):
    return gkg.parse_gkg_file(
        text,
        theme_prefixes=("ECON_", "EPU_"),
        query_topic="gkg_markets",
        event_version="g1",
        tone_model_name="gdelt_v2tone",
        ingested_at=INGEST,
    )


# --- low-level helpers -------------------------------------------------------- #


def test_parse_datetime_utc():
    assert gkg.parse_gkg_datetime("20240102031500") == datetime(2024, 1, 2, 3, 15, tzinfo=UTC)
    with pytest.raises(ValueError):
        gkg.parse_gkg_datetime("not-a-date")


def test_parse_tone_and_malformed():
    assert gkg.parse_tone("2.5,5.0,2.5,7.5") == (2.5, 5.0, 2.5)
    assert gkg.parse_tone("") is None
    assert gkg.parse_tone("a,b,c") is None
    assert gkg.parse_tone("1.0,2.0") is None  # too few components


def test_tone_to_score_math_and_clamp():
    s = gkg.tone_to_score(2.5, 5.0, 2.5, model_name="gdelt_v2tone", scored_at=INGEST)
    assert s.sentiment_score == pytest.approx(0.025)
    assert s.positive_score == pytest.approx(0.05)
    assert s.negative_score == pytest.approx(0.025)
    assert s.neutral_score == pytest.approx(0.925)
    # Out-of-range components are clamped so the score schema can never be violated.
    s2 = gkg.tone_to_score(250.0, 130.0, -5.0, model_name="m", scored_at=INGEST)
    assert s2.sentiment_score == 1.0 and s2.positive_score == 1.0 and s2.negative_score == 0.0


def test_extract_title_and_unescape():
    assert gkg.extract_title("<PAGE_TITLE>A &amp; B</PAGE_TITLE>") == "A & B"
    assert gkg.extract_title("no title here") == ""


def test_theme_match_prefix_ci():
    assert gkg.matches_themes("SPORTS;ECON_STOCKMARKET", ("econ_",)) is True
    assert gkg.matches_themes("SPORTS;ARTS", ("ECON_", "EPU_")) is False
    assert gkg.matches_themes("", ("ECON_",)) is False


# --- file parsing ------------------------------------------------------------- #


def test_kept_event_is_pit_correct_and_linked():
    res = _parse(_row())
    assert (res.n_rows, res.n_kept, res.n_malformed) == (1, 1, 0)
    ev, sc = res.raw[0], res.scored[0]
    assert ev.source == "gdelt_gkg" and ev.title == "Markets rally"
    # PIT: published == observed == file DATE, <= ingested.
    assert ev.published_at == ev.observed_at == datetime(2024, 1, 2, tzinfo=UTC)
    assert ev.observed_at <= ev.ingested_at
    # raw and scored share the content hash (so the score attaches to the right item).
    assert ev.content_hash == sc.content_hash
    assert sc.score.model_name == "gdelt_v2tone"


def test_theme_filter_drops_nonmatching():
    res = _parse(_row(themes="SPORTS;ARTS"))
    assert (res.n_rows, res.n_kept, res.n_malformed) == (1, 0, 0)


def test_empty_tone_dropped_not_malformed():
    res = _parse(_row(tone=""))
    assert (res.n_rows, res.n_kept, res.n_malformed) == (1, 0, 0)


def test_short_row_is_malformed():
    res = _parse("a\tb\tc")  # far fewer than the tone field index
    assert (res.n_rows, res.n_kept, res.n_malformed) == (1, 0, 1)


def test_bad_date_is_malformed():
    res = _parse(_row(date="garbage"))
    assert (res.n_rows, res.n_kept, res.n_malformed) == (1, 0, 1)


def test_blank_lines_ignored_and_title_fallback():
    # Missing page title -> fall back to the source domain so the item still has a label.
    res = _parse("\n" + _row(extras="") + "\n\n")
    assert res.n_rows == 1 and res.n_kept == 1
    assert res.raw[0].title == "reuters.com"


def test_url_missing_uses_record_id_as_source_id():
    res = _parse(_row(url="", id="REC-42"))
    assert res.n_kept == 1
    assert res.raw[0].source_id == "REC-42" and res.raw[0].source_url is None


# --- config ------------------------------------------------------------------- #


def test_config_loads_and_validates():
    cfg = GkgConfig.load()
    assert cfg.SOURCE == "gdelt_gkg"
    assert cfg.theme_prefixes  # non-empty
    assert cfg.storage.raw_dataset not in {"sentiment_raw", "sentiment_scores"}
    assert cfg.window.end >= cfg.window.start


def test_config_rejects_finbert_lake_collision():
    cfg = GkgConfig.load()
    data = cfg.model_dump(mode="json")
    data["storage"]["raw_dataset"] = "sentiment_raw"
    with pytest.raises(ValueError, match="distinct from the FinBERT"):
        GkgConfig.model_validate(data)


def test_config_rejects_policy_disagreement():
    cfg = GkgConfig.load()
    data = cfg.model_dump(mode="json")
    data["source_policy"] = {"gdelt_gkg": "paid_blocked"}
    with pytest.raises(ValueError, match="disagrees with the authoritative"):
        GkgConfig.model_validate(data)
