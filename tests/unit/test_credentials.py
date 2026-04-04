"""
tests/unit/test_credentials.py
Unit tests for nexus/cloud/credentials.py — FAZ 32.

Windows Credential Manager calls are intercepted via module-level patches;
no real pywin32 / DPAPI calls are made.

Sections:
  1.  Mock store fixture
  2.  save_key + load_key roundtrip
  3.  delete_key
  4.  has_key
  5.  list_configured
  6.  validate_key
  7.  RAM safety (SecretStr masking + gc)
  8.  Error propagation
  9.  Provider validation
  10. Edge cases
"""
from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import SecretStr

import nexus.cloud.credentials as cred_module
from nexus.cloud.credentials import CredentialManager

# ---------------------------------------------------------------------------
# In-memory Credential Store stub
# ---------------------------------------------------------------------------


class _NotFoundError(Exception):
    """Simulates pywintypes.error with winerror=1168 (ERROR_NOT_FOUND)."""

    winerror: int = 1168


class _MemStore:
    """
    Dict-backed stand-in for Windows Credential Manager.

    Mirrors the win32cred calling conventions:
      - CredWrite: stores blob as the raw str (CredentialManager passes str).
      - CredRead:  returns {'CredentialBlob': bytes (utf-16-le encoded)}.
      - CredDelete: removes entry; raises _NotFoundError when absent.
      - CredEnumerate: returns matching entries or raises _NotFoundError.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def write(self, cred: dict[str, Any], flags: int) -> None:
        blob = cred["CredentialBlob"]
        self.store[cred["TargetName"]] = str(blob)

    def read(self, target: str, cred_type: int, flags: int) -> dict[str, Any]:
        if target not in self.store:
            raise _NotFoundError(f"Not found: {target}")
        raw: str = self.store[target]
        return {"CredentialBlob": raw.encode("utf-16-le"), "TargetName": target}

    def delete(self, target: str, cred_type: int, flags: int) -> None:
        if target not in self.store:
            raise _NotFoundError(f"Not found: {target}")
        del self.store[target]

    def enumerate(self, filter_str: str, flags: int) -> list[dict[str, Any]]:
        prefix = filter_str.rstrip("*")
        matches = [
            {"TargetName": k} for k in self.store if k.startswith(prefix)
        ]
        if not matches:
            raise _NotFoundError(f"No matches for: {filter_str}")
        return matches


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> _MemStore:
    return _MemStore()


@pytest.fixture()
def manager(store: _MemStore) -> CredentialManager:
    """CredentialManager wired to the in-memory store."""
    with _patched(store):
        yield CredentialManager()


@contextmanager
def _patched(store: _MemStore):
    """Context manager that patches all four backend shims."""
    with (
        patch.object(cred_module, "_cred_write", store.write),
        patch.object(cred_module, "_cred_read", store.read),
        patch.object(cred_module, "_cred_delete", store.delete),
        patch.object(cred_module, "_cred_enumerate", store.enumerate),
    ):
        yield


# ---------------------------------------------------------------------------
# 1. Mock store health-check
# ---------------------------------------------------------------------------


class TestMockStore:
    def test_write_read_roundtrip(self, store: _MemStore) -> None:
        store.write(
            {"TargetName": "nexus/test", "CredentialBlob": "secret123"},
            0,
        )
        result = store.read("nexus/test", 1, 0)
        recovered = result["CredentialBlob"].decode("utf-16-le")
        assert recovered == "secret123"

    def test_delete_removes_entry(self, store: _MemStore) -> None:
        store.write({"TargetName": "nexus/x", "CredentialBlob": "k"}, 0)
        store.delete("nexus/x", 1, 0)
        with pytest.raises(_NotFoundError):
            store.read("nexus/x", 1, 0)

    def test_not_found_error_has_winerror_1168(self) -> None:
        err = _NotFoundError("test")
        assert err.winerror == 1168


# ---------------------------------------------------------------------------
# 2. save_key + load_key roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_load_returns_secret_str(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-test-key-value")
        secret = manager.load_key("openai")
        assert isinstance(secret, SecretStr)

    def test_roundtrip_value_correct(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-roundtrip-test")
        secret = manager.load_key("openai")
        assert secret is not None
        assert secret.get_secret_value() == "sk-roundtrip-test"

    def test_multiple_providers_independent(
        self, manager: CredentialManager
    ) -> None:
        manager.save_key("openai", "sk-openai-key")
        manager.save_key("anthropic", "sk-ant-anthropic-key")
        oa = manager.load_key("openai")
        ant = manager.load_key("anthropic")
        assert oa is not None and oa.get_secret_value() == "sk-openai-key"
        assert ant is not None and ant.get_secret_value() == "sk-ant-anthropic-key"

    def test_overwrite_updates_value(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-original")
        manager.save_key("openai", "sk-updated")
        secret = manager.load_key("openai")
        assert secret is not None
        assert secret.get_secret_value() == "sk-updated"

    def test_load_missing_returns_none(self, manager: CredentialManager) -> None:
        assert manager.load_key("nonexistent") is None

    def test_unicode_key_roundtrip(self, manager: CredentialManager) -> None:
        """Keys with non-ASCII characters survive utf-16-le encode/decode."""
        unicode_key = "sk-tëst-ünïcödé-kéy"
        manager.save_key("custom", unicode_key)
        secret = manager.load_key("custom")
        assert secret is not None
        assert secret.get_secret_value() == unicode_key

    def test_long_key_roundtrip(self, manager: CredentialManager) -> None:
        long_key = "sk-ant-" + "a" * 200
        manager.save_key("anthropic", long_key)
        secret = manager.load_key("anthropic")
        assert secret is not None
        assert secret.get_secret_value() == long_key

    def test_target_name_uses_prefix(self, store: _MemStore) -> None:
        """Credential is stored under 'nexus/{provider}' key."""
        with _patched(store):
            mgr = CredentialManager()
            mgr.save_key("openai", "sk-x")
        assert "nexus/openai" in store.store


# ---------------------------------------------------------------------------
# 3. delete_key
# ---------------------------------------------------------------------------


class TestDeleteKey:
    def test_delete_existing_key(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-to-delete")
        manager.delete_key("openai")
        assert manager.load_key("openai") is None

    def test_delete_nonexistent_is_noop(self, manager: CredentialManager) -> None:
        # Must not raise
        manager.delete_key("nonexistent")

    def test_delete_twice_is_noop(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-x")
        manager.delete_key("openai")
        manager.delete_key("openai")  # second call must not raise

    def test_delete_one_leaves_others(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-openai")
        manager.save_key("anthropic", "sk-ant-x")
        manager.delete_key("openai")
        assert manager.load_key("openai") is None
        assert manager.load_key("anthropic") is not None


# ---------------------------------------------------------------------------
# 4. has_key
# ---------------------------------------------------------------------------


class TestHasKey:
    def test_has_key_true_when_saved(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-test")
        assert manager.has_key("openai") is True

    def test_has_key_false_when_absent(self, manager: CredentialManager) -> None:
        assert manager.has_key("nonexistent") is False

    def test_has_key_false_after_delete(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-test")
        manager.delete_key("openai")
        assert manager.has_key("openai") is False

    def test_has_key_true_after_overwrite(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-v1")
        manager.save_key("openai", "sk-v2")
        assert manager.has_key("openai") is True


# ---------------------------------------------------------------------------
# 5. list_configured
# ---------------------------------------------------------------------------


class TestListConfigured:
    def test_empty_when_nothing_saved(self, manager: CredentialManager) -> None:
        assert manager.list_configured() == []

    def test_lists_saved_providers(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-a")
        manager.save_key("anthropic", "sk-ant-b")
        providers = manager.list_configured()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_sorted_alphabetically(self, manager: CredentialManager) -> None:
        for p in ["zzz", "aaa", "mmm"]:
            manager.save_key(p, f"key-{p}")
        assert manager.list_configured() == ["aaa", "mmm", "zzz"]

    def test_deleted_provider_not_listed(
        self, manager: CredentialManager
    ) -> None:
        manager.save_key("openai", "sk-a")
        manager.save_key("anthropic", "sk-ant-b")
        manager.delete_key("openai")
        assert "openai" not in manager.list_configured()
        assert "anthropic" in manager.list_configured()

    def test_single_provider_listed(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-x")
        assert manager.list_configured() == ["openai"]

    def test_list_configured_strips_prefix(
        self, store: _MemStore
    ) -> None:
        """Returned names must NOT include the 'nexus/' prefix."""
        with _patched(store):
            mgr = CredentialManager()
            mgr.save_key("myprovider", "key123")
            providers = mgr.list_configured()
        assert all("nexus/" not in p for p in providers)
        assert "myprovider" in providers


# ---------------------------------------------------------------------------
# 6. validate_key
# ---------------------------------------------------------------------------


class TestValidateKey:
    def test_empty_key_invalid(self, manager: CredentialManager) -> None:
        ok, reason = manager.validate_key("openai", "")
        assert ok is False
        assert reason

    def test_whitespace_only_invalid(self, manager: CredentialManager) -> None:
        ok, reason = manager.validate_key("openai", "   ")
        assert ok is False
        assert reason

    def test_too_short_invalid(self, manager: CredentialManager) -> None:
        ok, reason = manager.validate_key("openai", "sk-")
        assert ok is False
        assert reason

    def test_anthropic_valid_prefix(self, manager: CredentialManager) -> None:
        ok, reason = manager.validate_key("anthropic", "sk-ant-valid-key-12345")
        assert ok is True
        assert reason == ""

    def test_anthropic_wrong_prefix(self, manager: CredentialManager) -> None:
        ok, reason = manager.validate_key("anthropic", "sk-wrong-prefix-key12345")
        assert ok is False
        assert "sk-ant-" in reason

    def test_openai_valid_prefix(self, manager: CredentialManager) -> None:
        ok, reason = manager.validate_key("openai", "sk-valid-openai-key-123456")
        assert ok is True
        assert reason == ""

    def test_openai_wrong_prefix(self, manager: CredentialManager) -> None:
        ok, reason = manager.validate_key("openai", "invalid-openai-key-12345")
        assert ok is False
        assert "sk-" in reason

    def test_unknown_provider_accepts_any_nonempty_key(
        self, manager: CredentialManager
    ) -> None:
        ok, reason = manager.validate_key("custom_provider", "any-key-value-here")
        assert ok is True
        assert reason == ""

    def test_validate_returns_tuple(self, manager: CredentialManager) -> None:
        result = manager.validate_key("openai", "sk-valid-key-12345678")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


# ---------------------------------------------------------------------------
# 7. RAM safety
# ---------------------------------------------------------------------------


class TestRAMSafety:
    def test_load_key_returns_secret_str_not_plain_str(
        self, manager: CredentialManager
    ) -> None:
        manager.save_key("openai", "sk-secret-value")
        result = manager.load_key("openai")
        assert isinstance(result, SecretStr)
        assert not isinstance(result, str)

    def test_str_does_not_expose_key(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-secret-ram-test")
        secret = manager.load_key("openai")
        assert secret is not None
        assert "sk-secret-ram-test" not in str(secret)

    def test_repr_does_not_expose_key(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-repr-test-key-value")
        secret = manager.load_key("openai")
        assert secret is not None
        assert "sk-repr-test-key-value" not in repr(secret)

    def test_secret_str_is_masked(self, manager: CredentialManager) -> None:
        manager.save_key("openai", "sk-mask-me")
        secret = manager.load_key("openai")
        assert secret is not None
        # pydantic SecretStr always shows '**********' in str/repr
        assert "**********" in str(secret) or "**********" in repr(secret)

    def test_get_secret_value_returns_correct_value(
        self, manager: CredentialManager
    ) -> None:
        manager.save_key("openai", "sk-correct-value")
        secret = manager.load_key("openai")
        assert secret is not None
        assert secret.get_secret_value() == "sk-correct-value"

    def test_del_and_gc_collect_run_without_error(
        self, manager: CredentialManager
    ) -> None:
        """del + gc.collect() must not raise."""
        manager.save_key("openai", "sk-gc-test-key")
        secret = manager.load_key("openai")
        del secret
        gc.collect()

    def test_no_extra_gc_references_after_del(
        self, manager: CredentialManager
    ) -> None:
        """
        After del + gc.collect(), the key string must not have additional
        copies floating in the gc-tracked object graph.

        We use a unique sentinel so it can be distinguished from any other
        string in the process.  The sentinel variable itself is still
        reachable; we check that no *additional* instances beyond the one
        local variable exist in gc-tracked objects.
        """
        import uuid

        sentinel = "SK-NEXUS-GC-" + uuid.uuid4().hex.upper()
        manager.save_key("gcprovider", sentinel)

        secret = manager.load_key("gcprovider")
        assert secret is not None

        del secret
        gc.collect()
        gc.collect()  # two passes for cyclic references

        # Count all string objects that ARE the sentinel value
        matches = [
            obj
            for obj in gc.get_objects()
            if isinstance(obj, str) and obj == sentinel
        ]
        # Only the local variable `sentinel` should remain
        assert len(matches) <= 1, (
            f"Found {len(matches)} copies of sentinel in gc objects "
            f"after del+gc (expected ≤1)"
        )


# ---------------------------------------------------------------------------
# 8. Error propagation
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    def test_unexpected_write_error_propagates(self, store: _MemStore) -> None:
        class _BadStore(_MemStore):
            def write(self, cred: dict, flags: int) -> None:
                err = OSError("Disk full")
                err.winerror = 112  # ERROR_DISK_FULL
                raise err

        with _patched(_BadStore()):
            mgr = CredentialManager()
            with pytest.raises(OSError, match="Disk full"):
                mgr.save_key("openai", "sk-test")

    def test_unexpected_read_error_propagates(self, store: _MemStore) -> None:
        class _BadStore(_MemStore):
            def read(self, target: str, cred_type: int, flags: int) -> dict:
                err = PermissionError("Access denied")
                err.winerror = 5  # ERROR_ACCESS_DENIED
                raise err

        with _patched(_BadStore()):
            mgr = CredentialManager()
            with pytest.raises(PermissionError, match="Access denied"):
                mgr.load_key("openai")

    def test_unexpected_delete_error_propagates(self, store: _MemStore) -> None:
        class _BadStore(_MemStore):
            def delete(self, target: str, cred_type: int, flags: int) -> None:
                err = PermissionError("Access denied")
                err.winerror = 5
                raise err

        with _patched(_BadStore()):
            mgr = CredentialManager()
            with pytest.raises(PermissionError):
                mgr.delete_key("openai")

    def test_not_found_on_load_returns_none(
        self, manager: CredentialManager
    ) -> None:
        """ERROR_NOT_FOUND (winerror=1168) must return None, not raise."""
        result = manager.load_key("absent_provider")
        assert result is None

    def test_not_found_on_delete_is_noop(
        self, manager: CredentialManager
    ) -> None:
        """ERROR_NOT_FOUND on delete must be silently ignored."""
        manager.delete_key("absent_provider")  # must not raise

    def test_not_found_on_enumerate_returns_empty(
        self, manager: CredentialManager
    ) -> None:
        """ERROR_NOT_FOUND from CredEnumerate → empty list."""
        result = manager.list_configured()
        assert result == []


# ---------------------------------------------------------------------------
# 9. Provider validation
# ---------------------------------------------------------------------------


class TestProviderValidation:
    def test_empty_provider_raises(self, manager: CredentialManager) -> None:
        with pytest.raises(ValueError, match="empty"):
            manager.save_key("", "sk-key")

    def test_slash_in_provider_raises(self, manager: CredentialManager) -> None:
        with pytest.raises(ValueError, match="'/'"):
            manager.save_key("open/ai", "sk-key")

    def test_empty_provider_load_raises(self, manager: CredentialManager) -> None:
        with pytest.raises(ValueError):
            manager.load_key("")

    def test_empty_provider_delete_raises(
        self, manager: CredentialManager
    ) -> None:
        with pytest.raises(ValueError):
            manager.delete_key("")

    def test_empty_provider_has_key_raises(
        self, manager: CredentialManager
    ) -> None:
        with pytest.raises(ValueError):
            manager.has_key("")

    def test_empty_key_save_raises(self, manager: CredentialManager) -> None:
        with pytest.raises(ValueError, match="empty"):
            manager.save_key("openai", "")

    def test_valid_provider_names_accepted(
        self, manager: CredentialManager
    ) -> None:
        for name in ["openai", "anthropic", "azure_openai", "my-provider-v2"]:
            manager.save_key(name, "sk-valid-key-1234")
            assert manager.has_key(name) is True


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_case_sensitive_providers(self, manager: CredentialManager) -> None:
        """'OpenAI' and 'openai' are stored as different entries."""
        manager.save_key("OpenAI", "sk-upper")
        manager.save_key("openai", "sk-lower")
        assert manager.load_key("OpenAI").get_secret_value() == "sk-upper"  # type: ignore[union-attr]
        assert manager.load_key("openai").get_secret_value() == "sk-lower"  # type: ignore[union-attr]

    def test_key_with_special_characters(
        self, manager: CredentialManager
    ) -> None:
        special = "sk-ant-key=with+special/chars&encoded%20spaces"
        manager.save_key("anthropic", special)
        secret = manager.load_key("anthropic")
        assert secret is not None
        assert secret.get_secret_value() == special

    def test_save_multiple_then_list_all(
        self, manager: CredentialManager
    ) -> None:
        providers = ["aaa", "bbb", "ccc", "ddd"]
        for p in providers:
            manager.save_key(p, f"key-{p}")
        listed = manager.list_configured()
        assert listed == sorted(providers)

    def test_validate_then_save_then_load(
        self, manager: CredentialManager
    ) -> None:
        """validate_key → save_key → load_key full workflow."""
        key = "sk-ant-valid-key-for-workflow-test"
        ok, _ = manager.validate_key("anthropic", key)
        assert ok is True
        manager.save_key("anthropic", key)
        secret = manager.load_key("anthropic")
        assert secret is not None
        assert secret.get_secret_value() == key

    def test_credential_manager_uses_windows_cred_manager(
        self, store: _MemStore
    ) -> None:
        """
        Verify the production code path goes through the backend shims
        (not direct module import) — ensures mock isolation works.
        """
        write_calls: list[dict] = []

        class _TrackingStore(_MemStore):
            def write(self, cred: dict, flags: int) -> None:
                write_calls.append(dict(cred))
                super().write(cred, flags)

        with _patched(_TrackingStore()):
            mgr = CredentialManager()
            mgr.save_key("openai", "sk-tracked")

        assert len(write_calls) == 1
        assert write_calls[0]["TargetName"] == "nexus/openai"
        assert write_calls[0]["CredentialBlob"] == "sk-tracked"
