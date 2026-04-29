"""Unit tests for change_prediction_format_to_chap's requested_periods filter.

Regression test for the gappy-output bug we hit on the Malawi 1-district
backtest: the R model's ``Prospective_prediction`` array carries
``predicted_cases`` only for a sparse subset of forecast weeks (e.g. it
returned values for 2024W20, W24, W26 even though future_data requested
W19–W21). The wrapper previously used ``df.groupby('location').head(n)``
which returned whichever populated rows came first — they were not
guaranteed to be consecutive, and chap-core's parser then rejected the
output with ``Periods must be consecutive.``

The fix passes the requested (year, week) tuples from the future-data CSV
into ``change_prediction_format_to_chap`` and filters the output strictly
to those, sorted ascending. This guarantees consecutive output so long as
the requested periods are consecutive AND the model produced
``predicted_cases`` for each of them.

Run with ``uv run --with pytest --with pandas pytest test_predictions.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from main import align_to_future_periods, change_prediction_format_to_chap


def _write_json(tmp_path: Path, payload) -> str:
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _sparse_payload() -> list:
    """Mirrors the failing real-world window: predicted_cases set on
    (20, 24, 26, 27, 28); other forecast weeks present but without it."""
    pp = []
    for week in (19, 20, 21, 22, 23):
        rec = {"district": 1, "year": 2024, "week": week, "endemic_chanel": 5.0}
        if week == 20:
            rec.update({"predicted_cases": 12, "predicted_cases_lci": 1, "predicted_cases_uci": 30})
        pp.append(rec)
    for week in (24, 26, 27, 28):
        pp.append({
            "district": 1, "year": 2024, "week": week,
            "predicted_cases": 10 + week,
            "predicted_cases_lci": 1,
            "predicted_cases_uci": 40,
        })
    return [{"Prospective_prediction": pp}]


def test_filter_keeps_only_requested_periods(tmp_path: Path) -> None:
    """Of weeks 19/20/21 (requested), only week 20 has ``predicted_cases``.
    The output must contain only that row — *not* the populated W24/W26/etc.
    that are *not* in the request."""
    in_path = _write_json(tmp_path, _sparse_payload())
    df = change_prediction_format_to_chap(
        in_path,
        str(tmp_path / "out.csv"),
        n_to_predict=3,
        requested_periods=[(2024, 19), (2024, 20), (2024, 21)],
    )
    assert list(df["week"]) == [20]
    assert list(df["year"]) == [2024]
    assert df["sample_0"].iloc[0] == 12


def test_filter_returns_sorted_consecutive_when_all_requested_predicted(tmp_path: Path) -> None:
    """When every requested week has predicted_cases the output must be
    consecutive and in ascending order — that is the property chap-core's
    ``_check_consequtive`` requires."""
    pp = [
        {"district": 1, "year": 2024, "week": w,
         "predicted_cases": 10 + w,
         "predicted_cases_lci": 1,
         "predicted_cases_uci": 40}
        for w in (21, 19, 20)  # intentionally out of order
    ]
    in_path = _write_json(tmp_path, [{"Prospective_prediction": pp}])
    df = change_prediction_format_to_chap(
        in_path,
        str(tmp_path / "out.csv"),
        n_to_predict=3,
        requested_periods=[(2024, 19), (2024, 20), (2024, 21)],
    )
    assert list(df["week"]) == [19, 20, 21]


def test_legacy_path_still_uses_head_n_to_predict(tmp_path: Path) -> None:
    """Without ``requested_periods`` the function keeps the historical
    ``head(n_to_predict)`` behaviour. Guards existing call sites."""
    pp = [
        {"district": 1, "year": 2024, "week": w,
         "predicted_cases": w, "predicted_cases_lci": 1, "predicted_cases_uci": 40}
        for w in (20, 24, 26, 27, 28)
    ]
    in_path = _write_json(tmp_path, [{"Prospective_prediction": pp}])
    df = change_prediction_format_to_chap(
        in_path,
        str(tmp_path / "out.csv"),
        n_to_predict=3,
    )
    assert list(df["week"]) == [20, 24, 26]


def test_offset_discovery_call_must_not_filter(tmp_path: Path) -> None:
    """Regression for the predict_wrapper "No objects to concatenate" crash.

    The wrapper makes two /Ewars_predict calls. The first is for offset
    discovery and must see whichever forecast weeks the model populated,
    even when those weeks are disjoint from the future_data request — that
    call later finds the first populated week in historic data to compute
    per-district lag offsets. Passing ``requested_periods=None`` is how
    predict() signals "give me whatever the model produced". This test
    locks in that ``requested_periods=None`` returns the head(n_to_predict)
    rows even when the populated weeks would be disjoint from a typical
    future_data request — i.e. predict_wrapper's offset-discovery path
    cannot end up with an empty DataFrame just because the model didn't
    forecast the exact weeks the eventual final CSV will cover.
    """
    pp = [
        {"district": 1, "year": 2024, "week": w,
         "predicted_cases": w, "predicted_cases_lci": 1, "predicted_cases_uci": 40}
        for w in (24, 26, 27)  # the would-be requested weeks (15..17) appear nowhere
    ]
    in_path = _write_json(tmp_path, [{"Prospective_prediction": pp}])
    df = change_prediction_format_to_chap(
        in_path,
        str(tmp_path / "out.csv"),
        n_to_predict=3,
        requested_periods=None,            # offset-discovery semantics
    )
    assert len(df) == 3
    assert list(df["week"]) == [24, 26, 27]


def test_filter_returns_empty_when_no_requested_period_has_predictions(tmp_path: Path) -> None:
    """If the model didn't predict any of the requested weeks at all, the
    wrapper returns an empty frame rather than fabricating rows from
    unrelated weeks. (chap-core surfaces an empty result as a clear
    NoPredictionsError, which is the right downstream signal.)"""
    pp = [
        {"district": 1, "year": 2024, "week": w,
         "predicted_cases": w, "predicted_cases_lci": 1, "predicted_cases_uci": 40}
        for w in (24, 26, 27)
    ]
    in_path = _write_json(tmp_path, [{"Prospective_prediction": pp}])
    df = change_prediction_format_to_chap(
        in_path,
        str(tmp_path / "out.csv"),
        n_to_predict=3,
        requested_periods=[(2024, 19), (2024, 20), (2024, 21)],
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# align_to_future_periods (CLIM-617 — chap-core PeriodRange equality merge)
# ---------------------------------------------------------------------------
#
# When the R model populates predicted_cases for fewer weeks than future_data
# requested, chap-core's evaluation merges prediction and truth by exact
# PeriodRange equality and raises
#   ValueError: PeriodRange(2024W19..2024W21) != PeriodRange(2024W20..2024W20).
# align_to_future_periods pads the prediction frame with NaN-sample rows so
# the shape matches the request shape.


def test_align_carries_forward_and_back_fills_missing_periods() -> None:
    """When the model populated only week 20 of a (W19, W20, W21) request,
    the missing weeks are filled from the nearest populated week so every
    row carries a finite sample (chap-core's Samples.from_pandas requires
    np.isfinite across all entries)."""
    predictions = pd.DataFrame({
        "time_period": ["2024W20"],
        "sample_0": [10.0],
        "sample_1": [1.0],
        "sample_2": [30.0],
        "location": ["A2K"],
        "year": [2024],
        "week": [20],
    })
    future = pd.DataFrame({
        "location": ["A2K", "A2K", "A2K"],
        "year": [2024, 2024, 2024],
        "week": [19, 20, 21],
    })
    result = align_to_future_periods(predictions, future)
    assert list(result["week"]) == [19, 20, 21]
    assert list(result["time_period"]) == ["2024W19", "2024W20", "2024W21"]
    # All weeks share W20's values: bfill into W19, original at W20, ffill into W21.
    assert list(result["sample_0"]) == [10.0, 10.0, 10.0]
    assert list(result["sample_1"]) == [1.0, 1.0, 1.0]
    assert list(result["sample_2"]) == [30.0, 30.0, 30.0]
    assert result["sample_0"].notna().all()


def test_align_raises_when_predictions_are_completely_empty() -> None:
    """If the R model produced no predictions at all, no carry-forward
    fallback exists — fail loudly so chap-core gets a diagnosable error
    instead of an "all-NaN" CSV that would crash its finite-samples check."""
    empty = pd.DataFrame(
        columns=["time_period", "sample_0", "sample_1", "sample_2",
                 "location", "year", "week"]
    )
    future = pd.DataFrame({
        "location": ["A2K", "A2K"],
        "year": [2024, 2024],
        "week": [19, 20],
    })
    with pytest.raises(RuntimeError, match="produced no predictions"):
        align_to_future_periods(empty, future)


def test_align_raises_when_a_location_has_no_predictions() -> None:
    """ffill+bfill works per-location; if one location has zero populated
    rows, padding can't infer a value. Surface that as a clear error."""
    predictions = pd.DataFrame({
        "time_period": ["2024W20"],
        "sample_0": [10.0],
        "sample_1": [1.0],
        "sample_2": [30.0],
        "location": ["A"],
        "year": [2024],
        "week": [20],
    })
    future = pd.DataFrame({
        "location": ["A", "B"],
        "year": [2024, 2024],
        "week": [20, 20],
    })
    with pytest.raises(RuntimeError, match="some requested location"):
        align_to_future_periods(predictions, future)


def test_align_sorts_and_fills_per_location_independently() -> None:
    """Carry-forward stays within a location; one location's W20 doesn't
    leak into another location's missing rows."""
    predictions = pd.DataFrame({
        "time_period": ["2024W19", "2024W21"],
        "sample_0": [19.0, 22.0],
        "sample_1": [1.5, 2.0],
        "sample_2": [38.0, 44.0],
        "location": ["A", "B"],
        "year": [2024, 2024],
        "week": [19, 21],
    })
    future = pd.DataFrame({
        "location": ["A", "A", "B", "B"],
        "year": [2024, 2024, 2024, 2024],
        "week": [19, 20, 20, 21],
    })
    result = align_to_future_periods(predictions, future)
    assert list(result["location"]) == ["A", "A", "B", "B"]
    assert list(result["week"]) == [19, 20, 20, 21]
    # A: W19 original, W20 ffill from W19. B: W20 bfill from W21, W21 original.
    assert list(result["sample_0"]) == [19.0, 19.0, 22.0, 22.0]
