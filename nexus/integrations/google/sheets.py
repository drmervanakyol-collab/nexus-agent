"""
nexus/integrations/google/sheets.py
Google Sheets API v4 client for Nexus Agent.

GoogleSheetsClient
------------------
Thin, fully-injectable wrapper around the Sheets REST API.  Every network
call is provided via injectable callables so the class is testable without
a live Google account.

read_range(spreadsheet_id, range_) -> list[list]
  GET spreadsheets/{id}/values/{range}.
  Returns a 2-D list of cell values (strings).  Empty rows/cells are
  represented as empty strings.

write_range(spreadsheet_id, range_, values, *, check_formulas=True) -> bool
  1. If check_formulas=True, fetch the existing range and block writes
     to any cell that starts with "=" (formula protection).
  2. PUT spreadsheets/{id}/values/{range}?valueInputOption=USER_ENTERED.
  3. Read the range back and compare with the written values
     (SOURCE-level verification).  Returns True on match, False on mismatch.
  Raises PolicyBlockedError when a formula cell would be overwritten.

get_sheet_names(spreadsheet_id) -> list[str]
  GET spreadsheets/{id}?fields=sheets.properties.title.

check_concurrent_edit(spreadsheet_id) -> bool
  Fetch spreadsheet metadata twice (1 s apart) and compare the
  ``updatedTime`` field.  Returns True when an external edit is detected.

_handle_quota(exc):
  Called internally on HTTP 429.  Applies exponential back-off up to
  _MAX_RETRIES (5) attempts.  Raises CloudQuotaExceededError when retries
  are exhausted.

Injectable callables
--------------------
_get_fn  : (url, headers) -> dict
    Perform a GET request and return the parsed JSON body.
    Default: urllib.request + json.loads.
_put_fn  : (url, headers, body) -> dict
    Perform a PUT request with a JSON body and return the parsed response.
    Default: urllib.request + json.loads.
_sleep_fn: (seconds: float) -> None
    Pause between retry attempts.  Default: time.sleep.
_now_fn  : () -> float
    Current Unix timestamp.  Default: time.time.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from nexus.core.errors import CloudQuotaExceededError, PolicyBlockedError
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SHEETS_BASE = "https://sheets.googleapis.com/v4/spreadsheets"
_MAX_RETRIES = 5
_BACKOFF_BASE_S = 1.0  # first retry waits 1 s, then 2, 4, 8, 16


# ---------------------------------------------------------------------------
# Default injectable implementations
# ---------------------------------------------------------------------------


def _default_get(url: str, headers: dict[str, str]) -> dict[str, Any]:
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _default_put(
    url: str, headers: dict[str, str], body: dict[str, Any]
) -> dict[str, Any]:
    import urllib.request  # noqa: PLC0415

    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# GoogleSheetsClient
# ---------------------------------------------------------------------------


class GoogleSheetsClient:
    """
    Google Sheets API v4 client with formula protection and write verification.

    Parameters
    ----------
    get_token_fn:
        Callable that returns a valid OAuth 2.0 access token string.
        Typically ``GoogleAuthManager.get_valid_token``.
    _get_fn:
        ``(url, headers) -> dict``.  HTTP GET implementation.
    _put_fn:
        ``(url, headers, body) -> dict``.  HTTP PUT implementation.
    _sleep_fn:
        ``(seconds: float) -> None``.  Sleep implementation for back-off.
    _now_fn:
        ``() -> float``.  Current Unix timestamp.
    """

    def __init__(
        self,
        get_token_fn: Callable[[], str],
        *,
        _get_fn: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
        _put_fn: (
            Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]] | None
        ) = None,
        _sleep_fn: Callable[[float], None] | None = None,
        _now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._get_token = get_token_fn
        self._get = _get_fn or _default_get
        self._put = _put_fn or _default_put
        self._sleep = _sleep_fn or time.sleep
        self._now = _now_fn or time.time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_range(
        self, spreadsheet_id: str, range_: str
    ) -> list[list[str]]:
        """
        Read a range from a Google Sheet.

        Parameters
        ----------
        spreadsheet_id:
            The ID of the spreadsheet (from its URL).
        range_:
            A1 notation range, e.g. ``"Sheet1!A1:C10"``.

        Returns
        -------
        2-D list of cell values (strings).  Missing trailing cells in a
        row are normalised to empty strings up to the widest row.
        """
        url = f"{_SHEETS_BASE}/{spreadsheet_id}/values/{quote(range_, safe='!:')}"
        response = self._request_with_retry("GET", url)
        raw_rows: list[list[str]] = response.get("values", [])
        return self._normalise_grid(raw_rows)

    def write_range(
        self,
        spreadsheet_id: str,
        range_: str,
        values: list[list[Any]],
        *,
        check_formulas: bool = True,
    ) -> bool:
        """
        Write *values* into *range_* with optional formula protection.

        Steps
        -----
        1. If *check_formulas* is True, read the existing range and raise
           ``PolicyBlockedError`` if any target cell contains a formula.
        2. PUT the new values.
        3. Read back and verify each written cell matches the supplied value
           (SOURCE-level verification).

        Parameters
        ----------
        spreadsheet_id:
            Spreadsheet ID.
        range_:
            A1 notation range.
        values:
            2-D list of values to write.
        check_formulas:
            When True (default), block writes to formula cells.

        Returns
        -------
        True when the written values verify correctly, False on mismatch.

        Raises
        ------
        PolicyBlockedError
            When *check_formulas=True* and a formula is detected in the
            target range.
        """
        if check_formulas:
            self._assert_no_formulas(spreadsheet_id, range_, values)

        url = (
            f"{_SHEETS_BASE}/{spreadsheet_id}/values/{quote(range_, safe='!:')}"
            "?valueInputOption=USER_ENTERED"
        )
        body: dict[str, Any] = {
            "range": range_,
            "majorDimension": "ROWS",
            "values": [[str(v) for v in row] for row in values],
        }
        self._request_with_retry("PUT", url, body=body)
        _log.debug("write_range_put_ok", spreadsheet_id=spreadsheet_id, range=range_)

        # SOURCE-level verification
        return self._verify_write(spreadsheet_id, range_, values)

    def get_sheet_names(self, spreadsheet_id: str) -> list[str]:
        """
        Return the tab names of *spreadsheet_id* in order.

        Parameters
        ----------
        spreadsheet_id:
            Spreadsheet ID.

        Returns
        -------
        List of sheet title strings.
        """
        url = f"{_SHEETS_BASE}/{spreadsheet_id}?fields=sheets.properties.title"
        response = self._request_with_retry("GET", url)
        sheets: list[dict[str, Any]] = response.get("sheets", [])
        return [s["properties"]["title"] for s in sheets]

    def check_concurrent_edit(self, spreadsheet_id: str) -> bool:
        """
        Detect whether an external actor modified *spreadsheet_id* recently.

        Fetches spreadsheet metadata twice, 1 second apart, and compares
        the ``updatedTime`` field.

        Returns
        -------
        True when an external edit is detected between the two polls,
        False otherwise.
        """
        url = f"{_SHEETS_BASE}/{spreadsheet_id}?fields=properties.modifiedTime"
        first = self._request_with_retry("GET", url)
        t1 = first.get("properties", {}).get("modifiedTime", "")
        self._sleep(1.0)
        second = self._request_with_retry("GET", url)
        t2 = second.get("properties", {}).get("modifiedTime", "")
        changed = t1 != t2 and bool(t2)
        if changed:
            _log.warning(
                "concurrent_edit_detected",
                spreadsheet_id=spreadsheet_id,
                before=t1,
                after=t2,
            )
        return changed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Build the Authorization + Content-Type headers."""
        token = self._get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a GET or PUT request with exponential back-off on HTTP 429.

        Raises
        ------
        CloudQuotaExceededError
            After *_MAX_RETRIES* consecutive 429 responses.
        """
        for attempt in range(_MAX_RETRIES + 1):
            headers = self._auth_headers()
            try:
                if method == "GET":
                    return self._get(url, headers)
                return self._put(url, headers, body or {})
            except Exception as exc:
                if self._is_quota_error(exc) and attempt < _MAX_RETRIES:
                    wait = _BACKOFF_BASE_S * (2**attempt)
                    _log.warning(
                        "quota_backoff",
                        attempt=attempt + 1,
                        wait_s=wait,
                        url=url,
                    )
                    self._sleep(wait)
                    continue
                if self._is_quota_error(exc):
                    raise CloudQuotaExceededError(
                        "Google Sheets: quota exceeded after max retries.",
                        context={"url": url, "retries": _MAX_RETRIES},
                    ) from exc
                raise
        # unreachable — satisfies type checker
        raise CloudQuotaExceededError("quota_exceeded")  # pragma: no cover

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        """Return True when *exc* represents an HTTP 429 response."""
        msg = str(exc).lower()
        return "429" in msg or "quota" in msg or "rate" in msg

    def _assert_no_formulas(
        self,
        spreadsheet_id: str,
        range_: str,
        values: list[list[Any]],
    ) -> None:
        """
        Read the target range and raise PolicyBlockedError if any cell that
        would be overwritten currently contains a formula (starts with '=').
        """
        existing = self.read_range(spreadsheet_id, range_)
        for row_idx, row in enumerate(values):
            for col_idx in range(len(row)):
                try:
                    cell_val = existing[row_idx][col_idx]
                except IndexError:
                    continue
                if isinstance(cell_val, str) and cell_val.startswith("="):
                    raise PolicyBlockedError(
                        f"Google Sheets: formula protection — cell at "
                        f"row {row_idx}, col {col_idx} contains a formula "
                        f"({cell_val!r}).  Set check_formulas=False to override.",
                        rule="sheets_formula_protection",
                        severity="block",
                        context={
                            "spreadsheet_id": spreadsheet_id,
                            "range": range_,
                            "row": row_idx,
                            "col": col_idx,
                            "formula": cell_val,
                        },
                    )

    def _verify_write(
        self,
        spreadsheet_id: str,
        range_: str,
        expected: list[list[Any]],
    ) -> bool:
        """
        SOURCE-level verification: read back the written range and compare
        each cell against *expected*.

        Returns True when every written cell matches, False otherwise.
        """
        actual = self.read_range(spreadsheet_id, range_)
        for row_idx, row in enumerate(expected):
            for col_idx, exp_val in enumerate(row):
                try:
                    act_val = actual[row_idx][col_idx]
                except IndexError:
                    _log.warning(
                        "verify_write_missing_cell",
                        row=row_idx,
                        col=col_idx,
                        expected=str(exp_val),
                    )
                    return False
                if str(exp_val) != str(act_val):
                    _log.warning(
                        "verify_write_mismatch",
                        row=row_idx,
                        col=col_idx,
                        expected=str(exp_val),
                        actual=str(act_val),
                    )
                    return False
        _log.debug("verify_write_ok", range=range_)
        return True

    @staticmethod
    def _normalise_grid(rows: list[list[str]]) -> list[list[str]]:
        """Pad all rows to the same width with empty strings."""
        if not rows:
            return rows
        width = max(len(r) for r in rows)
        return [r + [""] * (width - len(r)) for r in rows]
