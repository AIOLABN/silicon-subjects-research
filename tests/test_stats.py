import math

import pytest

from silicon_subjects import stats


def test_wilson_matches_reference() -> None:

    ci = stats.wilson(81, 263)
    assert ci.lo == pytest.approx(0.2553, abs=1e-3)
    assert ci.hi == pytest.approx(0.3662, abs=1e-3)


def test_newcombe_diff_reference() -> None:

    ci = stats.newcombe_diff(56, 70, 48, 80)
    assert ci.estimate == pytest.approx(0.2)
    assert ci.lo == pytest.approx(0.0524, abs=1e-3)
    assert ci.hi == pytest.approx(0.3339, abs=1e-3)


def test_tost_equivalence_and_non_equivalence() -> None:
    assert stats.tost(0.0, 0.02, 0.10).equivalent()
    assert not stats.tost(0.09, 0.05, 0.10).equivalent()


def test_holm_is_monotone_and_bounded() -> None:
    adj = stats.holm([0.01, 0.04, 0.03, math.nan])
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)
    assert math.isnan(adj[3])


def test_icc_extremes() -> None:
    assert stats.icc_binary_two_level([[1, 1], [0, 0], [1, 1], [0, 0]]) == pytest.approx(1.0)
    assert stats.icc_binary_two_level([[1, 0], [0, 1], [1, 0], [0, 1]]) < 0


def test_cohens_h_sign() -> None:
    assert stats.cohens_h(0.7, 0.4) > 0
    assert stats.cohens_h(0.5, 0.5) == 0


def test_tost_power_reference() -> None:

    assert stats.tost_power(0.0524, 0.10) == pytest.approx(0.21, abs=0.01)
