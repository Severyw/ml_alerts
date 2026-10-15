import re

from src import config


def test_pinned_commit_is_a_full_sha_not_an_abbreviation_or_branch():
    assert re.fullmatch(r"[0-9a-f]{40}", config.PINNED_COMMIT), (
        "PINNED_COMMIT must be a full 40-char git SHA, so 'the pinned data "
        "version' is unambiguous -- an abbreviated hash or a branch name "
        "like 'main' would silently track future upstream changes."
    )


def test_fetch_data_script_checks_out_the_same_pinned_commit():
    """Guards against config.PINNED_COMMIT and scripts/fetch_data.sh drifting
    apart (two places that both need to agree on which snapshot we used)."""
    script = (config.REPO_ROOT / "scripts" / "fetch_data.sh").read_text()
    assert config.PINNED_COMMIT in script


def test_excluded_oblasts_contains_luhansk():
    assert "Luhanska oblast" in config.EXCLUDED_OBLASTS


def test_modelled_oblasts_are_24_unique_and_exclude_luhansk():
    assert len(config.MODELLED_OBLASTS) == 24
    assert len(set(config.MODELLED_OBLASTS)) == 24
    assert not set(config.MODELLED_OBLASTS) & set(config.EXCLUDED_OBLASTS)
    assert "Kyiv City" in config.MODELLED_OBLASTS
