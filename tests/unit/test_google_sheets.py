"""
tests/unit/test_google_sheets.py
Unit tests for nexus/integrations/google/sheets.py — Faz 53.

TEST PLAN
---------
read_range:
  1. Happy path — returns normalised 2-D grid.
  2. Empty response (no "values" key) — returns [].

write_range:
  3. Happy path — PUT called, read-back matches → True.
  4. Formula cell in target range → PolicyBlockedError raised, no PUT.
  5. check_formulas=False with formula cell — no block, write proceeds.
  6. Write succeeds but read-back mismatches → returns False.
  7. Write succeeds, read-back has fewer rows than written → returns False.

get_sheet_names:
  8. Returns list of sheet titles in order.
  9. Spreadsheet with no sheets → [].

check_concurrent_edit:
  10. modifiedTime unchanged → False, sleep called once.
  11. modifiedTime changed → True.
  12. modifiedTime missing in both responses → False.

quota / retry (_handle_quota):
  13. First call 429, second call succeeds → retried, result returned.
  14. All 6 calls return 429 → CloudQuotaExceededError raised.
  15. Non-quota error raised immediately (no retry).

_normalise_grid:
  16. Jagged rows padded to widest row.
  17. Empty input returns [].
"""
from __future__ import annotations

from typing import Any

import pytest

from nexus.core.errors import CloudQuotaExceededError, PolicyBlockedError
from nexus.integrations.google.sheets import GoogleSheetsClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKEN = "ya29.test_token"
_SPREADSHEET_ID = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"


def _make_client(
    *,
    get_responses: list[dict[str, Any]] | None = None,
    put_responses: list[dict[str, Any]] | None = None,
    sleep_calls: list[float] | None = None,
) -> GoogleSheetsClient:
    """
    Build a GoogleSheetsClient backed by queue-style mock callables.

    *get_responses* is consumed in order for each GET.
    *put_responses* is consumed in order for each PUT.
    Passing an Exception instance in a response list causes that call to raise.
    """
    _gets = list(get_responses or [])
    _puts = list(put_responses or [])
    _sleeps = sleep_calls if sleep_calls is not None else []

    def _fake_get(url: str, headers: dict[str, str]) -> dict[str, Any]:
        resp = _gets.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    def _fake_put(
        url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, Any]:
        resp = _puts.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    return GoogleSheetsClient(
        get_token_fn=lambda: _TOKEN,
        _get_fn=_fake_get,
        _put_fn=_fake_put,
        _sleep_fn=lambda s: _sleeps.append(s),
    )


def _values_response(rows: list[list[str]]) -> dict[str, Any]:
    return {"range": "Sheet1!A1:Z100", "majorDimension": "ROWS", "values": rows}


# ---------------------------------------------------------------------------
# read_range
# ---------------------------------------------------------------------------


class TestReadRange:
    def test_happy_path_normalised(self):
        """read_range returns a normalised 2-D grid (jagged rows padded)."""
        rows = [["A", "B", "C"], ["D", "E"], ["F"]]
        client = _make_client(get_responses=[_values_response(rows)])
        result = client.read_range(_SPREADSHEET_ID, "Sheet1!A1:C3")

        assert result == [
            ["A", "B", "C"],
            ["D", "E", ""],
            ["F", "", ""],
        ]

    def test_empty_response(self):
        """No 'values' key in response → empty list."""
        client = _make_client(get_responses=[{}])
        result = client.read_range(_SPREADSHEET_ID, "Sheet1!A1:C3")
        assert result == []


# ---------------------------------------------------------------------------
# write_range
# ---------------------------------------------------------------------------


class TestWriteRange:
    def test_happy_path_returns_true(self):
        """Successful write with matching read-back → True."""
        values = [["hello", "world"], ["foo", "bar"]]
        # GET for formula check, PUT for write, GET for verification
        client = _make_client(
            get_responses=[
                _values_response([["old1", "old2"], ["old3", "old4"]]),  # formula check
                _values_response(values),  # verification read-back
            ],
            put_responses=[{"updatedCells": 4}],
        )
        result = client.write_range(_SPREADSHEET_ID, "Sheet1!A1:B2", values)
        assert result is True

    def test_formula_cell_raises_policy_blocked(self):
        """Formula cell in target → PolicyBlockedError, no PUT issued."""
        values = [["new_val"]]
        put_calls: list[Any] = []

        def _fake_put(url: str, headers: dict, body: dict) -> dict:
            put_calls.append(body)
            return {}

        client = GoogleSheetsClient(
            get_token_fn=lambda: _TOKEN,
            _get_fn=lambda url, h: _values_response([["=SUM(A1:A10)"]]),
            _put_fn=_fake_put,
            _sleep_fn=lambda s: None,
        )

        with pytest.raises(PolicyBlockedError) as exc_info:
            client.write_range(_SPREADSHEET_ID, "Sheet1!A1", values)

        assert exc_info.value.rule == "sheets_formula_protection"
        assert put_calls == []  # PUT must NOT be called

    def test_check_formulas_false_skips_block(self):
        """check_formulas=False — formula cell is overwritten without error."""
        values = [["new_val"]]
        client = _make_client(
            get_responses=[
                _values_response(values),  # verification read-back only
            ],
            put_responses=[{"updatedCells": 1}],
        )
        result = client.write_range(
            _SPREADSHEET_ID, "Sheet1!A1", values, check_formulas=False
        )
        assert result is True

    def test_verification_mismatch_returns_false(self):
        """Read-back value differs from written value → False."""
        written = [["expected"]]
        read_back = [["something_else"]]
        client = _make_client(
            get_responses=[
                _values_response([["plain"]]),  # formula check (no formula)
                _values_response(read_back),    # verification
            ],
            put_responses=[{"updatedCells": 1}],
        )
        result = client.write_range(_SPREADSHEET_ID, "Sheet1!A1", written)
        assert result is False

    def test_verification_missing_row_returns_false(self):
        """Read-back returns fewer rows than written → False."""
        written = [["r1c1"], ["r2c1"]]
        client = _make_client(
            get_responses=[
                _values_response([["x"], ["y"]]),  # formula check
                _values_response([["r1c1"]]),       # only 1 row back
            ],
            put_responses=[{"updatedCells": 2}],
        )
        result = client.write_range(_SPREADSHEET_ID, "Sheet1!A1:A2", written)
        assert result is False


# ---------------------------------------------------------------------------
# get_sheet_names
# ---------------------------------------------------------------------------


class TestGetSheetNames:
    def test_returns_titles_in_order(self):
        response = {
            "sheets": [
                {"properties": {"title": "Sheet1"}},
                {"properties": {"title": "Data"}},
                {"properties": {"title": "Summary"}},
            ]
        }
        client = _make_client(get_responses=[response])
        names = client.get_sheet_names(_SPREADSHEET_ID)
        assert names == ["Sheet1", "Data", "Summary"]

    def test_no_sheets_returns_empty(self):
        client = _make_client(get_responses=[{"sheets": []}])
        assert client.get_sheet_names(_SPREADSHEET_ID) == []


# ---------------------------------------------------------------------------
# check_concurrent_edit
# ---------------------------------------------------------------------------


class TestCheckConcurrentEdit:
    def test_no_change_returns_false(self):
        """Same modifiedTime on both polls → False."""
        ts = "2026-04-09T10:00:00.000Z"
        meta = {"properties": {"modifiedTime": ts}}
        sleeps: list[float] = []
        client = _make_client(
            get_responses=[meta, meta],
            sleep_calls=sleeps,
        )
        assert client.check_concurrent_edit(_SPREADSHEET_ID) is False
        assert sleeps == [1.0]

    def test_changed_returns_true(self):
        """Different modifiedTime → True."""
        meta1 = {"properties": {"modifiedTime": "2026-04-09T10:00:00.000Z"}}
        meta2 = {"properties": {"modifiedTime": "2026-04-09T10:00:05.000Z"}}
        client = _make_client(get_responses=[meta1, meta2])
        assert client.check_concurrent_edit(_SPREADSHEET_ID) is True

    def test_missing_time_returns_false(self):
        """No modifiedTime in either response → False."""
        client = _make_client(get_responses=[{}, {}])
        assert client.check_concurrent_edit(_SPREADSHEET_ID) is False


# ---------------------------------------------------------------------------
# Quota / retry
# ---------------------------------------------------------------------------


class TestQuotaRetry:
    def test_single_429_then_success(self):
        """One 429 followed by a successful response → retried, data returned."""
        sleeps: list[float] = []
        rows = [["ok"]]
        calls = [0]

        def _get(url: str, headers: dict) -> dict:
            calls[0] += 1
            if calls[0] == 1:
                raise OSError("HTTP Error 429: Too Many Requests")
            return _values_response(rows)

        client = GoogleSheetsClient(
            get_token_fn=lambda: _TOKEN,
            _get_fn=_get,
            _put_fn=lambda *_: {},
            _sleep_fn=lambda s: sleeps.append(s),
        )
        result = client.read_range(_SPREADSHEET_ID, "Sheet1!A1")
        assert result == [["ok"]]
        assert len(sleeps) == 1
        assert sleeps[0] == 1.0  # first back-off = 1 s

    def test_all_retries_exhausted_raises(self):
        """Six consecutive 429 responses → CloudQuotaExceededError."""
        def _get(url: str, headers: dict) -> dict:
            raise OSError("HTTP Error 429: quota exceeded")

        client = GoogleSheetsClient(
            get_token_fn=lambda: _TOKEN,
            _get_fn=_get,
            _put_fn=lambda *_: {},
            _sleep_fn=lambda s: None,
        )
        with pytest.raises(CloudQuotaExceededError):
            client.read_range(_SPREADSHEET_ID, "Sheet1!A1")

    def test_non_quota_error_not_retried(self):
        """Non-429 errors propagate immediately without retry."""
        calls = [0]

        def _get(url: str, headers: dict) -> dict:
            calls[0] += 1
            raise ValueError("unexpected error")

        client = GoogleSheetsClient(
            get_token_fn=lambda: _TOKEN,
            _get_fn=_get,
            _put_fn=lambda *_: {},
            _sleep_fn=lambda s: None,
        )
        with pytest.raises(ValueError):
            client.read_range(_SPREADSHEET_ID, "Sheet1!A1")
        assert calls[0] == 1  # called exactly once


# ---------------------------------------------------------------------------
# _normalise_grid (unit)
# ---------------------------------------------------------------------------


class TestNormaliseGrid:
    def test_jagged_rows_padded(self):
        rows = [["a", "b", "c"], ["d"], ["e", "f"]]
        result = GoogleSheetsClient._normalise_grid(rows)
        assert result == [["a", "b", "c"], ["d", "", ""], ["e", "f", ""]]

    def test_empty_input(self):
        assert GoogleSheetsClient._normalise_grid([]) == []
