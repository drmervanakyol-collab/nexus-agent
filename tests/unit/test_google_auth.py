"""
tests/unit/test_google_auth.py
Unit tests for nexus/integrations/google/auth.py — Faz 52.

TEST PLAN
---------
GoogleAuthManager.authorize:
  1. Full happy-path flow — code received, exchange succeeds → True,
     token saved to credential manager.
  2. wait_for_code returns None (server timeout) → False, no save.
  3. exchange_code returns None (Google error) → False, no save.

GoogleAuthManager.get_valid_token:
  4. Valid token, not expiring → returned as-is, no refresh.
  5. Token expiring within 5 min → refresh called, new token returned.
  6. Token expiring, refresh returns None → InvalidAPIKeyError raised.
  7. No credentials stored → InvalidAPIKeyError raised.
  8. Token has no refresh_token and is expiring → InvalidAPIKeyError raised.

GoogleAuthManager.is_authorized:
  9. Non-expired token stored → True.
  10. Expired token stored → False.
  11. No token stored → False.

GoogleAuthManager.revoke:
  12. Token stored → revoke_fn called, credential deleted.
  13. No token stored → revoke_fn NOT called, no error.
"""
from __future__ import annotations

import pytest

from nexus.core.errors import InvalidAPIKeyError
from nexus.integrations.google.auth import GoogleAuthManager, TokenData

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CLIENT_ID = "test-client-id.apps.googleusercontent.com"
_CLIENT_SECRET = "test-client-secret"
_ACCESS_TOKEN = "ya29.access_token_v1"
_REFRESH_TOKEN = "1//refresh_token_v1"
_NEW_ACCESS_TOKEN = "ya29.access_token_v2"

# A fixed "now" for deterministic expiry tests
_NOW = 1_700_000_000.0


def _token(
    *,
    access_token: str = _ACCESS_TOKEN,
    refresh_token: str | None = _REFRESH_TOKEN,
    expires_at: float = _NOW + 3600.0,  # 1h from now — fresh
) -> TokenData:
    return TokenData(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
    )


def _cred_store(initial: TokenData | None = None) -> dict[str, str]:
    """In-memory credential store dict; pre-populated when *initial* given."""
    store: dict[str, str] = {}
    if initial is not None:
        store["nexus-google-oauth:token"] = initial.to_json()
    return store


def _make_manager(
    *,
    store: dict[str, str] | None = None,
    open_browser_calls: list[str] | None = None,
    wait_code: str | None = "auth-code-abc",
    exchange_result: TokenData | None = None,
    refresh_result: TokenData | None = None,
    revoke_calls: list[str] | None = None,
    now: float = _NOW,
) -> GoogleAuthManager:
    _store = store if store is not None else {}
    _open = open_browser_calls if open_browser_calls is not None else []
    _revokes = revoke_calls if revoke_calls is not None else []

    return GoogleAuthManager(
        _CLIENT_ID,
        _CLIENT_SECRET,
        _open_browser_fn=lambda url: _open.append(url),
        _wait_for_code_fn=lambda _port: wait_code,
        _exchange_code_fn=lambda _code: exchange_result,
        _refresh_token_fn=lambda _rt: refresh_result,
        _revoke_fn=lambda token: _revokes.append(token),
        _save_cred_fn=lambda svc, key, val: _store.update({f"{svc}:{key}": val}),
        _load_cred_fn=lambda svc, key: _store.get(f"{svc}:{key}"),
        _delete_cred_fn=lambda svc, key: _store.pop(f"{svc}:{key}", None),
        _now_fn=lambda: now,
    )


# ---------------------------------------------------------------------------
# authorize() tests
# ---------------------------------------------------------------------------


class TestAuthorize:
    @pytest.mark.asyncio
    async def test_happy_path_saves_token(self):
        """Full flow: code received → exchange succeeds → token saved."""
        store: dict[str, str] = {}
        fresh_token = _token()

        manager = _make_manager(store=store, exchange_result=fresh_token)
        result = await manager.authorize(
            ["https://www.googleapis.com/auth/gmail.readonly"]
        )

        assert result is True
        # Token must be persisted in the credential store
        saved_raw = store.get("nexus-google-oauth:token")
        assert saved_raw is not None
        saved = TokenData.from_json(saved_raw)
        assert saved.access_token == fresh_token.access_token
        assert saved.refresh_token == fresh_token.refresh_token

    @pytest.mark.asyncio
    async def test_browser_opened_with_scopes(self):
        """Browser URL contains the requested scopes."""
        open_calls: list[str] = []
        manager = _make_manager(
            open_browser_calls=open_calls,
            exchange_result=_token(),
        )
        await manager.authorize(["scope.a", "scope.b"])

        assert len(open_calls) == 1
        assert "scope.a" in open_calls[0]
        assert "scope.b" in open_calls[0]

    @pytest.mark.asyncio
    async def test_no_code_returns_false(self):
        """wait_for_code returns None → authorize returns False."""
        store: dict[str, str] = {}
        manager = _make_manager(store=store, wait_code=None)
        result = await manager.authorize(["scope.a"])

        assert result is False
        assert store == {}  # nothing saved

    @pytest.mark.asyncio
    async def test_exchange_fails_returns_false(self):
        """exchange_code returns None → authorize returns False."""
        store: dict[str, str] = {}
        manager = _make_manager(store=store, exchange_result=None)
        result = await manager.authorize(["scope.a"])

        assert result is False
        assert store == {}


# ---------------------------------------------------------------------------
# get_valid_token() tests
# ---------------------------------------------------------------------------


class TestGetValidToken:
    def test_fresh_token_returned_as_is(self):
        """Token not near expiry → returned without refresh."""
        fresh = _token(expires_at=_NOW + 3600.0)
        store = _cred_store(fresh)
        refresh_calls: list[str] = []

        manager = GoogleAuthManager(
            _CLIENT_ID,
            _CLIENT_SECRET,
            _refresh_token_fn=lambda rt: refresh_calls.append(rt) or None,  # type: ignore[func-returns-value]
            _load_cred_fn=lambda svc, key: store.get(f"{svc}:{key}"),
            _save_cred_fn=lambda svc, key, val: None,
            _delete_cred_fn=lambda svc, key: None,
            _now_fn=lambda: _NOW,
        )
        token = manager.get_valid_token()

        assert token == _ACCESS_TOKEN
        assert refresh_calls == []

    def test_expiring_token_refreshed(self):
        """Token expires in 4 min (< 5 min buffer) → refresh called."""
        expiring = _token(expires_at=_NOW + 240.0)  # 4 min left
        refreshed = _token(
            access_token=_NEW_ACCESS_TOKEN,
            expires_at=_NOW + 3600.0,
        )
        store = _cred_store(expiring)

        manager = _make_manager(
            store=store,
            refresh_result=refreshed,
            now=_NOW,
        )
        manager._load_cred = lambda svc, key: store.get(f"{svc}:{key}")

        token = manager.get_valid_token()

        assert token == _NEW_ACCESS_TOKEN
        # Refreshed token must be persisted
        saved = TokenData.from_json(store["nexus-google-oauth:token"])
        assert saved.access_token == _NEW_ACCESS_TOKEN

    def test_refresh_fail_raises_invalid_api_key(self):
        """Refresh returns None → InvalidAPIKeyError raised."""
        expiring = _token(expires_at=_NOW + 60.0)  # about to expire
        store = _cred_store(expiring)

        manager = _make_manager(
            store=store,
            refresh_result=None,  # refresh fails
            now=_NOW,
        )
        manager._load_cred = lambda svc, key: store.get(f"{svc}:{key}")

        with pytest.raises(InvalidAPIKeyError):
            manager.get_valid_token()

    def test_no_credentials_raises_invalid_api_key(self):
        """No stored credentials → InvalidAPIKeyError raised."""
        manager = _make_manager(store={}, now=_NOW)

        with pytest.raises(InvalidAPIKeyError):
            manager.get_valid_token()

    def test_no_refresh_token_when_expiring_raises(self):
        """Token expiring + no refresh_token → InvalidAPIKeyError."""
        expiring = _token(
            refresh_token=None,  # no refresh token
            expires_at=_NOW + 60.0,
        )
        store = _cred_store(expiring)

        manager = _make_manager(store=store, now=_NOW)
        manager._load_cred = lambda svc, key: store.get(f"{svc}:{key}")

        with pytest.raises(InvalidAPIKeyError):
            manager.get_valid_token()


# ---------------------------------------------------------------------------
# is_authorized() tests
# ---------------------------------------------------------------------------


class TestIsAuthorized:
    def test_fresh_token_authorized(self):
        """Non-expired token → True."""
        store = _cred_store(_token(expires_at=_NOW + 3600.0))
        manager = _make_manager(store=store, now=_NOW)
        manager._load_cred = lambda svc, key: store.get(f"{svc}:{key}")

        assert manager.is_authorized() is True

    def test_expired_token_not_authorized(self):
        """Expired token → False."""
        store = _cred_store(_token(expires_at=_NOW - 1.0))
        manager = _make_manager(store=store, now=_NOW)
        manager._load_cred = lambda svc, key: store.get(f"{svc}:{key}")

        assert manager.is_authorized() is False

    def test_no_token_not_authorized(self):
        """Empty store → False."""
        manager = _make_manager(store={}, now=_NOW)

        assert manager.is_authorized() is False


# ---------------------------------------------------------------------------
# revoke() tests
# ---------------------------------------------------------------------------


class TestRevoke:
    def test_revoke_calls_fn_and_deletes_cred(self):
        """Token stored → _revoke_fn called with access token, cred deleted."""
        stored = _token()
        store = _cred_store(stored)
        revoke_calls: list[str] = []

        manager = _make_manager(
            store=store,
            revoke_calls=revoke_calls,
            now=_NOW,
        )
        manager._load_cred = lambda svc, key: store.get(f"{svc}:{key}")

        manager.revoke()

        assert _ACCESS_TOKEN in revoke_calls
        assert "nexus-google-oauth:token" not in store

    def test_revoke_no_token_no_error(self):
        """No stored token → revoke_fn NOT called, no exception."""
        revoke_calls: list[str] = []
        manager = _make_manager(store={}, revoke_calls=revoke_calls, now=_NOW)

        manager.revoke()  # must not raise

        assert revoke_calls == []


# ---------------------------------------------------------------------------
# TokenData serialisation round-trip
# ---------------------------------------------------------------------------


class TestTokenDataSerialization:
    def test_round_trip(self):
        original = TokenData(
            access_token="tok123",
            refresh_token="ref456",
            expires_at=1_700_001_000.0,
        )
        restored = TokenData.from_json(original.to_json())

        assert restored.access_token == original.access_token
        assert restored.refresh_token == original.refresh_token
        assert restored.expires_at == original.expires_at

    def test_none_refresh_token_survives_json(self):
        original = TokenData(
            access_token="tok",
            refresh_token=None,
            expires_at=1_700_001_000.0,
        )
        restored = TokenData.from_json(original.to_json())

        assert restored.refresh_token is None
