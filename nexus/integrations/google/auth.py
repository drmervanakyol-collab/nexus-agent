"""
nexus/integrations/google/auth.py
Google OAuth 2.0 manager for Nexus Agent.

GoogleAuthManager
-----------------
Manages a Google OAuth 2.0 authorization code flow with PKCE-ready
redirect URI on localhost.  All external I/O is injectable so the class
is fully testable without a real browser or Google endpoint.

authorize(scopes) -> bool
  1. Build the Google consent URL from *scopes*.
  2. Open the user's default browser via ``_open_browser_fn``.
  3. Start a local HTTP server on ``redirect_port`` (default 8080) via
     ``_wait_for_code_fn``; block until Google redirects with the auth
     code, or until ``server_timeout_s`` elapses.
  4. Exchange the auth code for an access + refresh token pair via
     ``_exchange_code_fn``.
  5. Persist the TokenData JSON to the OS credential store via
     ``_save_cred_fn``.
  Returns True on success, False on any failure.

get_valid_token() -> str
  1. Load stored TokenData from the credential store.
  2. Raise InvalidAPIKeyError when no credentials are stored.
  3. If the access token expires in less than ``_REFRESH_BUFFER_S`` (300s),
     attempt a silent refresh via ``_refresh_token_fn``.
  4. Raise InvalidAPIKeyError when the refresh call fails or returns None.
  5. Persist refreshed TokenData and return the access token string.

is_authorized() -> bool
  Return True when a non-expired access token is stored.

revoke() -> None
  Call ``_revoke_fn`` (best-effort) then delete the stored credential.

Injectable callables
--------------------
_open_browser_fn     : (url: str) -> None
    Open the Google consent URL.  Default: ``webbrowser.open``.
_wait_for_code_fn    : (port: int) -> str | None
    Start a local HTTP redirect server, block until the auth code
    arrives, return it (or None on timeout/error).
    Default: threaded ``http.server.HTTPServer`` on the given port.
_exchange_code_fn    : (code: str) -> TokenData | None
    POST to Google's token endpoint to exchange the code.
    Default: ``aiohttp`` POST to ``_GOOGLE_TOKEN_URL``.
_refresh_token_fn    : (refresh_token: str) -> TokenData | None
    POST to Google's token endpoint with grant_type=refresh_token.
    Default: ``aiohttp`` POST to ``_GOOGLE_TOKEN_URL``.
_revoke_fn           : (access_token: str) -> None
    POST to Google's revoke endpoint.  Default: best-effort aiohttp.
_save_cred_fn        : (service: str, key: str, value: str) -> None
    Persist a string secret.  Default: Windows Credential Manager.
_load_cred_fn        : (service: str, key: str) -> str | None
    Load a string secret.  Default: Windows Credential Manager.
_delete_cred_fn      : (service: str, key: str) -> None
    Delete a stored secret.  Default: Windows Credential Manager.
_now_fn              : () -> float
    Current time as a Unix timestamp.  Default: ``time.time``.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode

from nexus.core.errors import InvalidAPIKeyError
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Google OAuth 2.0 endpoints
# ---------------------------------------------------------------------------

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

# Seconds before expiry at which a proactive refresh is triggered
_REFRESH_BUFFER_S: int = 300  # 5 minutes

# Credential store keys
_CRED_SERVICE = "nexus-google-oauth"
_CRED_KEY = "token"


# ---------------------------------------------------------------------------
# TokenData
# ---------------------------------------------------------------------------


@dataclass
class TokenData:
    """
    Persisted OAuth 2.0 token bundle.

    Attributes
    ----------
    access_token:
        Short-lived Google API bearer token.
    refresh_token:
        Long-lived token used to obtain new access tokens.  May be None
        when the server omits it (subsequent authorisations).
    expires_at:
        Unix timestamp (float) at which the access token expires.
    """

    access_token: str
    refresh_token: str | None
    expires_at: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> TokenData:
        d = json.loads(s)
        return cls(**d)


# ---------------------------------------------------------------------------
# Default injectable implementations (all lazy-import to stay testable)
# ---------------------------------------------------------------------------


def _default_open_browser(url: str) -> None:
    import webbrowser  # noqa: PLC0415

    webbrowser.open(url)


def _default_wait_for_code(port: int) -> str | None:
    """
    Spin up a one-shot HTTP server on *port* and wait for Google's
    redirect to ``/callback?code=<auth_code>``.  Times out after 120s.
    """
    import queue  # noqa: PLC0415
    import threading  # noqa: PLC0415
    from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: PLC0415
    from urllib.parse import parse_qs, urlparse  # noqa: PLC0415

    code_q: queue.Queue[str | None] = queue.Queue(maxsize=1)

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            code = params.get("code", [None])[0]
            code_q.put(code)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Authorization complete. You may close this tab.")

        def log_message(self, *_: Any) -> None:
            pass  # suppress server log output

    server = HTTPServer(("localhost", port), _Handler)
    server.timeout = 120.0

    def _serve() -> None:
        server.handle_request()
        server.server_close()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    t.join(timeout=125)
    try:
        return code_q.get_nowait()
    except Exception:
        return None


def _default_save_cred(service: str, key: str, value: str) -> None:
    try:
        import win32cred  # noqa: PLC0415

        win32cred.CredWrite(
            {
                "Type": win32cred.CRED_TYPE_GENERIC,
                "TargetName": f"{service}/{key}",
                "UserName": key,
                "CredentialBlob": value.encode("utf-8"),
                "Persist": win32cred.CRED_PERSIST_LOCAL_MACHINE,
            },
            0,
        )
    except Exception as exc:
        _log.warning("cred_save_failed", service=service, key=key, error=str(exc))


def _default_load_cred(service: str, key: str) -> str | None:
    try:
        import win32cred  # noqa: PLC0415

        cred = win32cred.CredRead(
            f"{service}/{key}", win32cred.CRED_TYPE_GENERIC, 0
        )
        blob: bytes = cred["CredentialBlob"]
        return blob.decode("utf-8")
    except Exception:
        return None


def _default_delete_cred(service: str, key: str) -> None:
    try:
        import win32cred  # noqa: PLC0415

        win32cred.CredDelete(
            f"{service}/{key}", win32cred.CRED_TYPE_GENERIC, 0
        )
    except Exception:
        pass


def _default_revoke(access_token: str) -> None:
    try:
        import urllib.request  # noqa: PLC0415

        data = urlencode({"token": access_token}).encode()
        req = urllib.request.Request(_GOOGLE_REVOKE_URL, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as exc:
        _log.debug("revoke_failed", error=str(exc))


def _default_exchange_code(
    code: str,
    *,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> TokenData | None:
    try:
        import urllib.request  # noqa: PLC0415

        payload = urlencode(
            {
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            }
        ).encode()
        req = urllib.request.Request(
            _GOOGLE_TOKEN_URL, data=payload, method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        expires_in = body.get("expires_in", 3600)
        return TokenData(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token"),
            expires_at=time.time() + expires_in,
        )
    except Exception as exc:
        _log.warning("exchange_code_failed", error=str(exc))
        return None


def _default_refresh_token(
    refresh_token: str,
    *,
    client_id: str,
    client_secret: str,
) -> TokenData | None:
    try:
        import urllib.request  # noqa: PLC0415

        payload = urlencode(
            {
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
            }
        ).encode()
        req = urllib.request.Request(
            _GOOGLE_TOKEN_URL, data=payload, method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        expires_in = body.get("expires_in", 3600)
        return TokenData(
            access_token=body["access_token"],
            refresh_token=refresh_token,  # Google doesn't re-issue refresh tokens
            expires_at=time.time() + expires_in,
        )
    except Exception as exc:
        _log.warning("refresh_token_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# GoogleAuthManager
# ---------------------------------------------------------------------------


class GoogleAuthManager:
    """
    Google OAuth 2.0 manager — browser consent flow with token persistence.

    Parameters
    ----------
    client_id:
        Google API client ID (from Cloud Console credentials).
    client_secret:
        Google API client secret.
    redirect_port:
        Port for the local redirect HTTP server.  Default: 8080.
    _open_browser_fn:
        ``(url: str) -> None``.  Opens the consent URL.
    _wait_for_code_fn:
        ``(port: int) -> str | None``.  Returns the auth code.
    _exchange_code_fn:
        ``(code: str) -> TokenData | None``.  Code → token exchange.
    _refresh_token_fn:
        ``(refresh_token: str) -> TokenData | None``.  Silent refresh.
    _revoke_fn:
        ``(access_token: str) -> None``.  Best-effort token revocation.
    _save_cred_fn:
        ``(service: str, key: str, value: str) -> None``.
    _load_cred_fn:
        ``(service: str, key: str) -> str | None``.
    _delete_cred_fn:
        ``(service: str, key: str) -> None``.
    _now_fn:
        ``() -> float``.  Current Unix timestamp.  Default: ``time.time``.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        redirect_port: int = 8080,
        _open_browser_fn: Callable[[str], None] | None = None,
        _wait_for_code_fn: Callable[[int], str | None] | None = None,
        _exchange_code_fn: Callable[[str], TokenData | None] | None = None,
        _refresh_token_fn: Callable[[str], TokenData | None] | None = None,
        _revoke_fn: Callable[[str], None] | None = None,
        _save_cred_fn: Callable[[str, str, str], None] | None = None,
        _load_cred_fn: Callable[[str, str], str | None] | None = None,
        _delete_cred_fn: Callable[[str, str], None] | None = None,
        _now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_port = redirect_port
        self._redirect_uri = f"http://localhost:{redirect_port}/callback"

        self._open_browser: Callable[[str], None] = (
            _open_browser_fn or _default_open_browser
        )
        self._wait_for_code: Callable[[int], str | None] = (
            _wait_for_code_fn or _default_wait_for_code
        )
        self._exchange_code: Callable[[str], TokenData | None] = (
            _exchange_code_fn
            or (
                lambda code: _default_exchange_code(
                    code,
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                    redirect_uri=self._redirect_uri,
                )
            )
        )
        self._refresh_token: Callable[[str], TokenData | None] = (
            _refresh_token_fn
            or (
                lambda rt: _default_refresh_token(
                    rt,
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                )
            )
        )
        self._revoke: Callable[[str], None] = _revoke_fn or _default_revoke
        self._save_cred: Callable[[str, str, str], None] = (
            _save_cred_fn or _default_save_cred
        )
        self._load_cred: Callable[[str, str], str | None] = (
            _load_cred_fn or _default_load_cred
        )
        self._delete_cred: Callable[[str, str], None] = (
            _delete_cred_fn or _default_delete_cred
        )
        self._now: Callable[[], float] = _now_fn or time.time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def authorize(self, scopes: list[str]) -> bool:
        """
        Run the browser-based OAuth 2.0 consent flow.

        Opens the user's browser to the Google consent page, waits for
        the redirect callback, exchanges the code for tokens, and
        persists them in the OS credential store.

        Parameters
        ----------
        scopes:
            List of Google API scope URLs.

        Returns
        -------
        True on success, False on any failure (no exception raised).
        """
        auth_url = self._build_auth_url(scopes)
        _log.debug("authorize_opening_browser", url=auth_url)
        self._open_browser(auth_url)

        code = self._wait_for_code(self._redirect_port)
        if not code:
            _log.warning("authorize_no_code_received")
            return False

        token_data = self._exchange_code(code)
        if token_data is None:
            _log.warning("authorize_exchange_failed")
            return False

        self._persist(token_data)
        _log.debug(
            "authorize_ok",
            expires_at=token_data.expires_at,
        )
        return True

    def get_valid_token(self) -> str:
        """
        Return a valid access token, refreshing silently when needed.

        Raises
        ------
        InvalidAPIKeyError
            When no credentials are stored, or when a required token
            refresh fails.
        """
        token_data = self._load_token()
        if token_data is None:
            raise InvalidAPIKeyError(
                "Google OAuth: no credentials stored — call authorize() first.",
                context={"client_id": self._client_id},
            )

        now = self._now()
        needs_refresh = now + _REFRESH_BUFFER_S >= token_data.expires_at

        if needs_refresh:
            if not token_data.refresh_token:
                raise InvalidAPIKeyError(
                    "Google OAuth: access token expiring and no refresh token stored.",
                    context={"expires_at": token_data.expires_at},
                )
            _log.debug(
                "get_valid_token_refreshing",
                seconds_remaining=token_data.expires_at - now,
            )
            refreshed = self._refresh_token(token_data.refresh_token)
            if refreshed is None:
                raise InvalidAPIKeyError(
                    "Google OAuth: token refresh failed.",
                    context={"client_id": self._client_id},
                )
            self._persist(refreshed)
            token_data = refreshed
            _log.debug("get_valid_token_refreshed", expires_at=token_data.expires_at)

        return token_data.access_token

    def is_authorized(self) -> bool:
        """
        Return True when a non-expired access token is stored.

        Does not attempt a refresh; use ``get_valid_token()`` to get a
        guaranteed-valid token.
        """
        token_data = self._load_token()
        if token_data is None:
            return False
        return self._now() < token_data.expires_at

    def revoke(self) -> None:
        """
        Revoke the stored access token and delete all credentials.

        Best-effort: revocation failure is logged but does not raise.
        The credential store entry is always deleted.
        """
        token_data = self._load_token()
        if token_data is not None:
            try:
                self._revoke(token_data.access_token)
            except Exception as exc:
                _log.debug("revoke_best_effort_failed", error=str(exc))
        self._delete_cred(_CRED_SERVICE, _CRED_KEY)
        _log.debug("revoke_complete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_auth_url(self, scopes: list[str]) -> str:
        """Build the Google OAuth 2.0 consent page URL."""
        params = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",
            "prompt": "consent",
        }
        return f"{_GOOGLE_AUTH_URL}?{urlencode(params)}"

    def _persist(self, token_data: TokenData) -> None:
        """Serialise and save *token_data* to the credential store."""
        self._save_cred(_CRED_SERVICE, _CRED_KEY, token_data.to_json())

    def _load_token(self) -> TokenData | None:
        """Load and deserialise TokenData from the credential store."""
        raw = self._load_cred(_CRED_SERVICE, _CRED_KEY)
        if raw is None:
            return None
        try:
            return TokenData.from_json(raw)
        except Exception as exc:
            _log.warning("load_token_corrupt", error=str(exc))
            return None
