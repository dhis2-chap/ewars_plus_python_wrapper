"""Unit tests for main.py error-handling helpers.

Guards against regressions in:

* ``change_prediction_format_to_chap`` — must reject non-list / dict-with-error
  responses with a clear ``RuntimeError`` instead of the original
  ``AttributeError: 'str' object has no attribute 'get'`` (CLIM-614).
* ``run_command`` — must raise on non-zero curl exit codes (CLIM-614 / CLIM-618),
  including the ``curl: (28)`` timeout case introduced by ``--max-time``.
* ``_check_api_response`` — must surface JSON-encoded server errors as
  ``RuntimeError`` and tolerate empty / non-JSON bodies (CLIM-614).

These tests run with plain ``pytest`` from the repo root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from main import (
    _check_api_response,
    change_prediction_format_to_chap,
    run_command,
)


# ---------------------------------------------------------------------------
# change_prediction_format_to_chap
# ---------------------------------------------------------------------------


def _write_json(tmp_path: Path, payload) -> str:
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_change_prediction_format_to_chap_happy_path(tmp_path: Path) -> None:
    payload = [
        {
            "Prospective_prediction": [
                {
                    "district": 1,
                    "year": 2024,
                    "week": 19,
                    "predicted_cases": 11,
                    "predicted_cases_lci": 1,
                    "predicted_cases_uci": 37,
                },
                {
                    "district": 1,
                    "year": 2024,
                    "week": 20,
                    "predicted_cases": 12,
                    "predicted_cases_lci": 2,
                    "predicted_cases_uci": 40,
                },
                # No predicted_cases — must be skipped.
                {"district": 1, "year": 2024, "week": 13},
            ]
        }
    ]
    in_path = _write_json(tmp_path, payload)
    df = change_prediction_format_to_chap(in_path, str(tmp_path / "out.csv"), n_to_predict=5)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["time_period"]) == ["2024W19", "2024W20"]
    assert list(df["sample_0"]) == [11, 12]
    assert list(df["sample_1"]) == [1, 2]
    assert list(df["sample_2"]) == [37, 40]
    assert (df["location"] == 1).all()


def test_change_prediction_format_to_chap_dict_with_error_raises(tmp_path: Path) -> None:
    in_path = _write_json(tmp_path, {"error": "training did not finish"})
    with pytest.raises(RuntimeError, match="training did not finish"):
        change_prediction_format_to_chap(in_path, str(tmp_path / "out.csv"), n_to_predict=5)


def test_change_prediction_format_to_chap_dict_without_error_raises(tmp_path: Path) -> None:
    """Pre-CLIM-614 this would have crashed with AttributeError when iterating
    a dict yielded its string keys; now it surfaces the shape mismatch."""
    in_path = _write_json(tmp_path, {"unexpected": "shape"})
    with pytest.raises(RuntimeError, match="Expected a list"):
        change_prediction_format_to_chap(in_path, str(tmp_path / "out.csv"), n_to_predict=5)


def test_change_prediction_format_to_chap_bare_string_raises(tmp_path: Path) -> None:
    """JSON-decoded bare string is what an HTML/text error body looks like."""
    in_path = _write_json(tmp_path, "Internal Server Error")
    with pytest.raises(RuntimeError, match="Expected a list"):
        change_prediction_format_to_chap(in_path, str(tmp_path / "out.csv"), n_to_predict=5)


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------


def test_run_command_returns_stdout_on_success() -> None:
    output = run_command("printf hello")
    assert output == b"hello"


def test_run_command_raises_on_nonzero_exit() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        run_command("sh -c 'echo boom 1>&2; exit 1'")
    msg = str(exc_info.value)
    assert "exit code 1" in msg
    assert "boom" in msg  # stderr should be surfaced


def test_run_command_raises_on_curl_timeout_exit_28() -> None:
    """curl --max-time exceeded exits 28; the wrapper must propagate it."""
    with pytest.raises(RuntimeError, match="exit code 28"):
        run_command("sh -c 'exit 28'")


# ---------------------------------------------------------------------------
# _check_api_response
# ---------------------------------------------------------------------------


def test_check_api_response_empty_is_noop() -> None:
    _check_api_response(b"", "/Ewars_run")
    _check_api_response("", "/Ewars_run")


def test_check_api_response_non_json_is_noop() -> None:
    """Successful endpoints can return CSV/HTML/binary; only structured JSON
    error bodies should raise."""
    _check_api_response(b"this is not json", "/Ewars_run")


def test_check_api_response_list_is_noop() -> None:
    _check_api_response(json.dumps([{"district": 1}]).encode(), "/retrieve_predicted_cases")


def test_check_api_response_dict_without_error_is_noop() -> None:
    _check_api_response(json.dumps({"status": "ok"}).encode(), "/Ewars_run")


def test_check_api_response_dict_with_error_raises() -> None:
    body = json.dumps({"error": "500 - Internal server error"}).encode()
    with pytest.raises(RuntimeError) as exc_info:
        _check_api_response(body, "/Ewars_run")
    msg = str(exc_info.value)
    assert "/Ewars_run" in msg
    assert "500" in msg
