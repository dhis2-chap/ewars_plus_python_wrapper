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

from main import change_prediction_format_to_chap


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
