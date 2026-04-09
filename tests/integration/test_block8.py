"""
tests/integration/test_block8.py
Blok 8 Integration Tests — Faz 55

Google OAuth + Sheets pipeline end-to-end, exercised entirely with
lightweight stubs.  No real HTTP calls, browser, or OS credential store
are touched.

TEST 1 — Auth + Sheets pipeline
  GoogleAuthManager.get_valid_token() returns a token →
  GoogleSheetsClient.read_range() uses that token in the Authorization
  header and returns the expected 2-D grid.

TEST 2 — Formula protection
  Target cell contains "=SUM(A1:A10)" →
  GoogleSheetsClient.write_range() raises PolicyBlockedError.
  No PUT is issued.

TEST 3 — Quota backoff
  GET fails with HTTP 429 three times, then succeeds →
  read_range() returns data on the 4th attempt.
  Exactly 3 back-off sleeps recorded (1 s, 2 s, 4 s).

TEST 4 — Token refresh on expiry
  Stored token expires in 60 s (< 300 s buffer) →
  GoogleAuthManager.get_valid_token() calls _refresh_token_fn and
  returns the new access token.

TEST 5 — Write SOURCE_LEVEL verification
  write_range() writes values → reads back the same values →
  returns True (verification passed).
  Confirm read-back GET was actually issued after the PUT.
"""
from __future__ import annotations

import pytest

from nexus.core.errors import InvalidAPIKeyError, PolicyBlockedError
from nexus.integrations.google.auth import GoogleAuthManager, TokenData
from nexus.integrations.google.sheets import GoogleSheetsClient

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_CLIENT_ID = "test-client.apps.googleusercontent.com"
_CLIENT_SECRET = "test-secret"
_ACCESS_TOKEN = "ya29.access_v1"
_NEW_ACCESS_TOKEN = "ya29.access_v2"
_REFRESH_TOKEN = "1//refresh_v1"
_SPREADSHEET_ID = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"
_RANGE = "Sheet1!A1:C2"

# Fixed "now" well before expiry in normal cases
_NOW = 1_700_000_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_token(
    access: str = _ACCESS_TOKEN,
    refresh: str | None = _REFRESH_TOKEN,
    expires_in: float = 3600.0,
) -> TokenData:
    return TokenData(
        access_token=access,
        refresh_token=refresh,
        expires_at=_NOW + expires_in,
    )


def _cred_store(token: TokenData | None = None) -> dict[str, str]:
    store: dict[str, str] = {}
    if token is not None:
        store["nexus-google-oauth:token"] = token.to_json()
    return store


def _make_auth(
    store: dict[str, str],
    *,
    now: float = _NOW,
    refresh_result: TokenData | None = None,
) -> GoogleAuthManager:
    """Auth manager backed by an in-memory credential store."""
    return GoogleAuthManager(
        _CLIENT_ID,
        _CLIENT_SECRET,
        _open_browser_fn=lambda url: None,
        _wait_for_code_fn=lambda _port: "auth-code",
        _exchange_code_fn=lambda _code: _fresh_token(),
        _refresh_token_fn=lambda _rt: refresh_result,
        _revoke_fn=lambda _token: None,
        _save_cred_fn=lambda svc, key, val: store.update({f"{svc}:{key}": val}),
        _load_cred_fn=lambda svc, key: store.get(f"{svc}:{key}"),
        _delete_cred_fn=lambda svc, key: store.pop(f"{svc}:{key}", None),
        _now_fn=lambda: now,
    )


def _values_response(rows: list[list[str]]) -> dict:
    return {"range": _RANGE, "majorDimension": "ROWS", "values": rows}


# ---------------------------------------------------------------------------
# TEST 1 — Auth + Sheets pipeline
# ---------------------------------------------------------------------------


class TestAuthSheetsPipeline:
    """
    get_valid_token() → token forwarded to Sheets client →
    read_range() returns expected grid.
    """

    def test_token_used_in_authorization_header(self):
        store = _cred_store(_fresh_token())
        auth = _make_auth(store)

        captured_headers: list[dict] = []
        data = [["Name", "Score", "Grade"], ["Alice", "95", "A"]]

        def _get(url: str, headers: dict) -> dict:
            captured_headers.append(dict(headers))
            return _values_response(data)

        sheets = GoogleSheetsClient(
            get_token_fn=auth.get_valid_token,
            _get_fn=_get,
        )
        result = sheets.read_range(_SPREADSHEET_ID, _RANGE)

        assert result == data
        assert len(captured_headers) == 1
        assert captured_headers[0]["Authorization"] == f"Bearer {_ACCESS_TOKEN}"


# ---------------------------------------------------------------------------
# TEST 2 — Formula protection
# ---------------------------------------------------------------------------


class TestFormulaProtection:
    """
    Existing cell has a formula → write_range raises PolicyBlockedError.
    PUT must never be called.
    """

    def test_formula_cell_blocks_write(self):
        store = _cred_store(_fresh_token())
        auth = _make_auth(store)

        existing = [["=SUM(B2:B10)", "header"]]
        put_calls: list[dict] = []

        def _get(url: str, headers: dict) -> dict:
            return _values_response(existing)

        def _put(url: str, headers: dict, body: dict) -> dict:
            put_calls.append(body)
            return {}

        sheets = GoogleSheetsClient(
            get_token_fn=auth.get_valid_token,
            _get_fn=_get,
            _put_fn=_put,
        )

        with pytest.raises(PolicyBlockedError) as exc_info:
            sheets.write_range(
                _SPREADSHEET_ID, _RANGE, [["new_value", "other"]]
            )

        assert exc_info.value.rule == "sheets_formula_protection"
        assert put_calls == [], "PUT must not be issued when formula protection fires"


# ---------------------------------------------------------------------------
# TEST 3 — Quota backoff
# ---------------------------------------------------------------------------


class TestQuotaBackoff:
    """
    First 3 GET calls return HTTP 429 → 4th succeeds.
    Exactly 3 sleeps: 1 s, 2 s, 4 s (exponential back-off).
    """

    def test_three_429_then_success(self):
        store = _cred_store(_fresh_token())
        auth = _make_auth(store)

        sleeps: list[float] = []
        calls = [0]
        data = [["ok"]]

        def _get(url: str, headers: dict) -> dict:
            calls[0] += 1
            if calls[0] <= 3:
                raise OSError("HTTP Error 429: Too Many Requests")
            return _values_response(data)

        sheets = GoogleSheetsClient(
            get_token_fn=auth.get_valid_token,
            _get_fn=_get,
            _sleep_fn=lambda s: sleeps.append(s),
        )
        result = sheets.read_range(_SPREADSHEET_ID, _RANGE)

        assert result == [["ok"]]
        assert calls[0] == 4
        assert sleeps == [1.0, 2.0, 4.0], f"Expected [1, 2, 4], got {sleeps}"


# ---------------------------------------------------------------------------
# TEST 4 — Token refresh on expiry
# ---------------------------------------------------------------------------


class TestTokenRefreshOnExpiry:
    """
    Stored token expires in 60 s (inside 300 s buffer) →
    get_valid_token() silently refreshes and returns the new token.
    """

    def test_expiring_token_triggers_refresh(self):
        expiring = _fresh_token(expires_in=60.0)  # 1 min left
        refreshed = _fresh_token(
            access=_NEW_ACCESS_TOKEN, expires_in=3600.0
        )
        store = _cred_store(expiring)

        auth = _make_auth(store, now=_NOW, refresh_result=refreshed)
        token = auth.get_valid_token()

        assert token == _NEW_ACCESS_TOKEN
        # Persisted token must also be the refreshed one
        saved = TokenData.from_json(store["nexus-google-oauth:token"])
        assert saved.access_token == _NEW_ACCESS_TOKEN

    def test_no_token_raises_invalid_api_key(self):
        auth = _make_auth(_cred_store())  # empty store

        with pytest.raises(InvalidAPIKeyError):
            auth.get_valid_token()


# ---------------------------------------------------------------------------
# TEST 5 — Write SOURCE_LEVEL verification
# ---------------------------------------------------------------------------


class TestWriteSourceVerification:
    """
    write_range() → PUT → read-back GET issued → values match → True.
    Confirms the full write+verify round-trip.
    """

    def test_verification_read_issued_after_write(self):
        store = _cred_store(_fresh_token())
        auth = _make_auth(store)

        values = [["Alice", "95"], ["Bob", "87"]]
        get_calls: list[str] = []
        put_calls: list[dict] = []

        call_order: list[str] = []

        def _get(url: str, headers: dict) -> dict:
            get_calls.append(url)
            call_order.append("GET")
            # First GET = formula check (existing plain values), second = verify
            return _values_response(values)

        def _put(url: str, headers: dict, body: dict) -> dict:
            put_calls.append(body)
            call_order.append("PUT")
            return {"updatedCells": 4}

        sheets = GoogleSheetsClient(
            get_token_fn=auth.get_valid_token,
            _get_fn=_get,
            _put_fn=_put,
        )
        result = sheets.write_range(_SPREADSHEET_ID, _RANGE, values)

        assert result is True
        # Order must be: GET (formula check) → PUT → GET (verification)
        assert call_order == ["GET", "PUT", "GET"], (
            f"Expected [GET, PUT, GET], got {call_order}"
        )
        assert len(put_calls) == 1
        assert len(get_calls) == 2

    def test_verification_mismatch_returns_false(self):
        store = _cred_store(_fresh_token())
        auth = _make_auth(store)

        written = [["expected_value"]]
        read_back = [["different_value"]]
        get_responses = [
            _values_response([["plain"]]),   # formula check — no formula
            _values_response(read_back),     # verification — mismatch
        ]

        def _get(url: str, headers: dict) -> dict:
            return get_responses.pop(0)

        sheets = GoogleSheetsClient(
            get_token_fn=auth.get_valid_token,
            _get_fn=_get,
            _put_fn=lambda *_: {"updatedCells": 1},
        )
        result = sheets.write_range(_SPREADSHEET_ID, _RANGE, written)

        assert result is False
