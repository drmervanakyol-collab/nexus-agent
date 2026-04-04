"""
nexus/cloud/credentials.py
Cloud API Credential Manager — ADR-009: Windows Credential Manager storage.

Architecture
------------
Credentials are stored as GENERIC credentials in the Windows Credential
Manager (backed by DPAPI encryption tied to the current user account).
The target name format is ``nexus/{provider}`` which namespaces all
Nexus credentials and makes them queryable via ``CredEnumerate``.

SecretStr (pydantic) wraps every retrieved key to prevent accidental
logging or printing of the plain-text value.  The intermediate plain-text
Python ``str`` produced during blob decoding is explicitly deleted and
followed by ``gc.collect()`` to reduce its lifetime in heap memory.

Public API
----------
  save_key(provider, key)       Store or update the key for *provider*.
  load_key(provider)            Return SecretStr | None.
  delete_key(provider)          Remove the entry; no-op when absent.
  has_key(provider)             bool — True when a key exists.
  list_configured()             Sorted list of configured provider names.
  validate_key(provider, key)   Format check; returns (bool, reason).

RAM safety
----------
  load_key() decodes the blob to a plain ``str``, immediately wraps it
  in SecretStr, then deletes the temporary string and calls gc.collect().
  Callers should likewise ``del`` their SecretStr when done and call
  ``gc.collect()`` to clear the value from heap.

  SecretStr.__str__ / __repr__ never expose the secret value — they
  emit ``'**********'``.  Use ``.get_secret_value()`` only at the point
  of consumption (e.g. building an HTTP header).

Windows constants (used without importing win32cred at module level)
--------------------------------------------------------------------
  CRED_TYPE_GENERIC          = 1
  CRED_PERSIST_LOCAL_MACHINE = 2
  ERROR_NOT_FOUND            = 1168
"""
from __future__ import annotations

import gc
from typing import Any

from pydantic import SecretStr

from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Windows constants (avoid importing win32cred at module level so the module
# loads cleanly on all platforms and in test environments without pywin32).
# ---------------------------------------------------------------------------

_CRED_TYPE_GENERIC: int = 1
_CRED_PERSIST_LOCAL_MACHINE: int = 2
_ERROR_NOT_FOUND: int = 1168   # Windows ERROR_NOT_FOUND

# Target name prefix for all Nexus credentials
_PREFIX: str = "nexus/"
_USERNAME: str = "nexus-agent"

# Minimum key length (chars) accepted by validate_key
_MIN_KEY_LEN: int = 8


# ---------------------------------------------------------------------------
# Backend shim (module-level functions for easy mocking in tests)
# ---------------------------------------------------------------------------


def _cred_write(cred: dict[str, Any], flags: int) -> None:
    """Write a credential entry via Windows Credential Manager."""
    import win32cred  # noqa: PLC0415

    win32cred.CredWrite(cred, flags)


def _cred_read(target: str, cred_type: int, flags: int) -> dict[str, Any]:
    """Read a credential entry; raises on not-found."""
    import win32cred  # noqa: PLC0415

    return dict(win32cred.CredRead(target, cred_type, flags))


def _cred_delete(target: str, cred_type: int, flags: int) -> None:
    """Delete a credential entry; raises on not-found."""
    import win32cred  # noqa: PLC0415

    win32cred.CredDelete(target, cred_type, flags)


def _cred_enumerate(filter_str: str, flags: int) -> list[dict[str, Any]]:
    """Enumerate credentials matching *filter_str*; raises on no matches."""
    import win32cred  # noqa: PLC0415

    return [dict(c) for c in win32cred.CredEnumerate(filter_str, flags)]


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------


def _is_not_found(exc: Exception) -> bool:
    """Return True when *exc* represents Windows ERROR_NOT_FOUND (1168)."""
    return getattr(exc, "winerror", None) == _ERROR_NOT_FOUND


# ---------------------------------------------------------------------------
# Provider format rules
# ---------------------------------------------------------------------------

_PROVIDER_RULES: dict[str, tuple[str, str]] = {
    # provider_lower → (required_prefix, description)
    "anthropic": ("sk-ant-", "Anthropic keys must start with 'sk-ant-'"),
    "openai":    ("sk-",     "OpenAI keys must start with 'sk-'"),
}


# ---------------------------------------------------------------------------
# CredentialManager
# ---------------------------------------------------------------------------


class CredentialManager:
    """
    Secure credential storage via Windows Credential Manager (DPAPI).

    All credentials are namespaced under the target-name prefix
    ``nexus/{provider}`` so they are isolated from other applications.

    Notes
    -----
    This class is thread-safe for independent providers; concurrent
    reads/writes to the *same* provider are serialised by Windows.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_key(self, provider: str, key: str) -> None:
        """
        Store or update the API key for *provider*.

        Parameters
        ----------
        provider:
            Short identifier for the cloud provider (e.g. "anthropic",
            "openai").  Must be non-empty and contain no ``/`` characters.
        key:
            Plain-text API key.  Must be non-empty.

        Raises
        ------
        ValueError
            When *provider* or *key* is invalid.
        OSError
            On Windows Credential Manager write failure.
        """
        _check_provider(provider)
        if not key:
            raise ValueError("key must not be empty")

        target = _PREFIX + provider
        cred: dict[str, Any] = {
            "Type": _CRED_TYPE_GENERIC,
            "TargetName": target,
            "CredentialBlob": key,
            "Persist": _CRED_PERSIST_LOCAL_MACHINE,
            "UserName": _USERNAME,
        }
        _cred_write(cred, 0)
        _log.debug("credential_saved", provider=provider)

    def load_key(self, provider: str) -> SecretStr | None:
        """
        Return the stored key for *provider* wrapped in SecretStr, or
        None when no credential is found.

        The plain-text intermediate string is deleted and gc is run
        before this method returns.
        """
        _check_provider(provider)
        target = _PREFIX + provider
        try:
            result = _cred_read(target, _CRED_TYPE_GENERIC, 0)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise

        blob: bytes = result["CredentialBlob"]
        key_str: str = blob.decode("utf-16-le")
        secret = SecretStr(key_str)

        # Wipe the plain-text string from local scope immediately.
        del key_str
        gc.collect()

        _log.debug("credential_loaded", provider=provider)
        return secret

    def delete_key(self, provider: str) -> None:
        """
        Remove the stored key for *provider*.

        No-op when the provider has no stored credential.

        Raises
        ------
        ValueError
            When *provider* is invalid.
        OSError
            On unexpected Windows Credential Manager errors.
        """
        _check_provider(provider)
        target = _PREFIX + provider
        try:
            _cred_delete(target, _CRED_TYPE_GENERIC, 0)
        except Exception as exc:
            if _is_not_found(exc):
                _log.debug("credential_delete_noop", provider=provider)
                return
            raise
        _log.debug("credential_deleted", provider=provider)

    def has_key(self, provider: str) -> bool:
        """Return True when a key is stored for *provider*."""
        return self.load_key(provider) is not None

    def list_configured(self) -> list[str]:
        """
        Return a sorted list of configured provider names.

        Only credentials under the ``nexus/`` prefix are considered.
        """
        try:
            entries = _cred_enumerate(_PREFIX + "*", 0)
        except Exception as exc:
            if _is_not_found(exc):
                return []
            raise
        providers = sorted(
            e["TargetName"][len(_PREFIX):]
            for e in entries
            if e.get("TargetName", "").startswith(_PREFIX)
        )
        return providers

    def validate_key(self, provider: str, key: str) -> tuple[bool, str]:
        """
        Validate *key* format for *provider*.

        Performs client-side format checks only (no live API call).

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` when the key passes all checks.
            ``(False, reason)`` when a check fails.
        """
        _check_provider(provider)

        if not key or not key.strip():
            return False, "Key must not be empty"

        if len(key) < _MIN_KEY_LEN:
            return False, f"Key is too short (minimum {_MIN_KEY_LEN} characters)"

        rule = _PROVIDER_RULES.get(provider.lower())
        if rule is not None:
            prefix, message = rule
            if not key.startswith(prefix):
                return False, message

        return True, ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_provider(provider: str) -> None:
    """Raise ValueError when *provider* is not a valid identifier."""
    if not provider:
        raise ValueError("provider must not be empty")
    if "/" in provider:
        raise ValueError(f"provider must not contain '/': {provider!r}")
