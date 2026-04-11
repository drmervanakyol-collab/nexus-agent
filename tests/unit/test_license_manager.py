"""
Unit tests for nexus/release/license_manager.py

Strategy
--------
- WMI subprocess calls are mocked so tests run on any OS (Linux CI).
- Trial state is written to a tmp_path file via NEXUS_TRIAL_PATH env var.
- License key generation uses the same dev secret as validation.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

import nexus.release.license_manager as lm_mod
from nexus.release.license_manager import (
    TRIAL_DAILY_TASK_LIMIT,
    TRIAL_DAYS,
    LicenseManager,
    LicenseType,
    _b64url_decode,
    _b64url_encode,
)

# Capture the real (unpatched) generate_machine_id before any test-level patching.
_real_generate_machine_id = LicenseManager.generate_machine_id


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FAKE_MID = "abcd1234abcd1234"


@pytest.fixture(autouse=True)
def fixed_machine_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin machine_id so tests don't depend on host hardware."""
    monkeypatch.setattr(LicenseManager, "generate_machine_id", lambda self: FAKE_MID)


@pytest.fixture()
def trial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect trial state to a temp file."""
    p = tmp_path / "trial.json"
    monkeypatch.setenv("NEXUS_TRIAL_PATH", str(p))
    return p


@pytest.fixture()
def lm() -> LicenseManager:
    return LicenseManager()


def make_key(
    machine_id: str = FAKE_MID,
    license_type: LicenseType = LicenseType.FULL,
    expires: date | None = None,
) -> str:
    return LicenseManager.generate_key(machine_id, license_type, expires)


def _today() -> date:
    return lm_mod._today()


# ---------------------------------------------------------------------------
# Machine ID
# ---------------------------------------------------------------------------


class TestMachineId:
    def test_returns_16_hex_chars(self, lm: LicenseManager) -> None:
        mid = lm.generate_machine_id()
        assert len(mid) == 16
        assert all(c in "0123456789abcdef" for c in mid)

    def test_consistent_across_calls(self, lm: LicenseManager) -> None:
        assert lm.generate_machine_id() == lm.generate_machine_id()

    def test_wmi_cpu_disk_used_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lm2 = LicenseManager()
        monkeypatch.setattr(LicenseManager, "generate_machine_id", LicenseManager.generate_machine_id)
        with (
            patch.object(lm_mod, "_wmi_cpu_serial", return_value="CPUSERIAL123"),
            patch.object(lm_mod, "_wmi_disk_serial", return_value="DISKSERIAL456"),
        ):
            mid = lm2.generate_machine_id()
        assert len(mid) == 16

    def test_fallback_used_when_wmi_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lm2 = LicenseManager()
        monkeypatch.setattr(LicenseManager, "generate_machine_id", LicenseManager.generate_machine_id)
        with (
            patch.object(lm_mod, "_wmi_cpu_serial", return_value=""),
            patch.object(lm_mod, "_wmi_disk_serial", return_value=""),
            patch.object(lm_mod, "_fallback_machine_components", return_value="node|x64|intel"),
        ):
            mid = lm2.generate_machine_id()
        assert len(mid) == 16

    def test_cached_after_first_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Restore the real implementation (overrides autouse patch) so we can
        # verify that _machine_id is returned directly on subsequent calls.
        monkeypatch.setattr(LicenseManager, "generate_machine_id", _real_generate_machine_id)
        with (
            patch.object(lm_mod, "_wmi_cpu_serial", return_value="CPU"),
            patch.object(lm_mod, "_wmi_disk_serial", return_value="DISK"),
        ):
            lm2 = LicenseManager()
            first = lm2.generate_machine_id()
            # Pre-set cache to a different value
            lm2._machine_id = "cached0000000000"
            # Must return cached value, not recompute
            assert lm2.generate_machine_id() == "cached0000000000"
            assert first != "cached0000000000"


# ---------------------------------------------------------------------------
# generate_key / b64 helpers
# ---------------------------------------------------------------------------


class TestKeyGeneration:
    def test_key_has_two_parts(self) -> None:
        key = make_key()
        parts = key.split(".")
        assert len(parts) == 2

    def test_payload_decodable(self) -> None:
        key = make_key()
        payload_b64 = key.rsplit(".", 1)[0]
        payload = json.loads(_b64url_decode(payload_b64))
        assert payload["mid"] == FAKE_MID
        assert payload["typ"] == "full"

    def test_perpetual_license_has_empty_exp(self) -> None:
        key = make_key(expires=None)
        payload = json.loads(_b64url_decode(key.rsplit(".", 1)[0]))
        assert payload["exp"] == ""

    def test_expiry_included_in_payload(self) -> None:
        exp = date(2030, 1, 1)
        key = make_key(expires=exp)
        payload = json.loads(_b64url_decode(key.rsplit(".", 1)[0]))
        assert payload["exp"] == "2030-01-01"

    def test_trial_type_in_payload(self) -> None:
        key = make_key(license_type=LicenseType.TRIAL)
        payload = json.loads(_b64url_decode(key.rsplit(".", 1)[0]))
        assert payload["typ"] == "trial"


# ---------------------------------------------------------------------------
# validate_license — success paths
# ---------------------------------------------------------------------------


class TestValidateLicenseSuccess:
    def test_valid_full_perpetual(self, lm: LicenseManager) -> None:
        key = make_key()
        result = lm.validate_license(key)
        assert result.valid is True
        assert result.license_type == LicenseType.FULL
        assert result.expires is None
        assert result.machine_id == FAKE_MID

    def test_valid_full_future_expiry(self, lm: LicenseManager) -> None:
        future = _today() + timedelta(days=365)
        key = make_key(expires=future)
        result = lm.validate_license(key)
        assert result.valid is True
        assert result.expires == future

    def test_valid_trial_key(self, lm: LicenseManager) -> None:
        future = _today() + timedelta(days=7)
        key = make_key(license_type=LicenseType.TRIAL, expires=future)
        result = lm.validate_license(key)
        assert result.valid is True
        assert result.license_type == LicenseType.TRIAL


# ---------------------------------------------------------------------------
# validate_license — failure paths
# ---------------------------------------------------------------------------


class TestValidateLicenseFailures:
    def test_empty_key_invalid(self, lm: LicenseManager) -> None:
        result = lm.validate_license("")
        assert result.valid is False

    def test_no_dot_invalid(self, lm: LicenseManager) -> None:
        result = lm.validate_license("nodot")
        assert result.valid is False

    def test_tampered_payload_fails_hmac(self, lm: LicenseManager) -> None:
        key = make_key()
        payload_b64, sig_b64 = key.rsplit(".", 1)
        tampered = _b64url_encode(b"tampered")
        result = lm.validate_license(f"{tampered}.{sig_b64}")
        assert result.valid is False

    def test_tampered_sig_fails_hmac(self, lm: LicenseManager) -> None:
        key = make_key()
        payload_b64, _ = key.rsplit(".", 1)
        result = lm.validate_license(f"{payload_b64}.badsig")
        assert result.valid is False

    def test_wrong_machine_id_rejected(self, lm: LicenseManager) -> None:
        key = make_key(machine_id="0000000000000000")
        result = lm.validate_license(key)
        assert result.valid is False
        assert "makine" in result.message

    def test_expired_key_rejected(self, lm: LicenseManager) -> None:
        past = _today() - timedelta(days=1)
        key = make_key(expires=past)
        result = lm.validate_license(key)
        assert result.valid is False
        assert "doldu" in result.message

    def test_today_is_last_valid_day(self, lm: LicenseManager) -> None:
        today = _today()
        key = make_key(expires=today)
        result = lm.validate_license(key)
        assert result.valid is True

    def test_garbage_key_does_not_raise(self, lm: LicenseManager) -> None:
        result = lm.validate_license("!!!.###")
        assert result.valid is False


# ---------------------------------------------------------------------------
# Trial mode
# ---------------------------------------------------------------------------


class TestTrialStatus:
    def test_initial_state(self, lm: LicenseManager, trial_file: Path) -> None:
        status = lm.trial_status()
        assert status["days_used"] == 1
        assert status["expired"] is False
        assert status["today_tasks"] == 0
        assert status["daily_limit_reached"] is False

    def test_trial_file_created(self, lm: LicenseManager, trial_file: Path) -> None:
        lm.trial_status()
        assert trial_file.is_file()
        state = json.loads(trial_file.read_text())
        assert "started_on" in state

    def test_consistent_machine_id(self, lm: LicenseManager, trial_file: Path) -> None:
        mid1 = lm.generate_machine_id()
        mid2 = lm.generate_machine_id()
        assert mid1 == mid2

    def test_expires_after_14_days(
        self, lm: LicenseManager, trial_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started = _today() - timedelta(days=TRIAL_DAYS)
        trial_file.write_text(
            json.dumps({"started_on": started.isoformat(), "today_tasks": 0, "last_task_date": _today().isoformat()})
        )
        status = lm.trial_status()
        assert status["expired"] is True

    def test_not_expired_on_day_14(
        self, lm: LicenseManager, trial_file: Path
    ) -> None:
        started = _today() - timedelta(days=TRIAL_DAYS - 1)
        trial_file.write_text(
            json.dumps({"started_on": started.isoformat(), "today_tasks": 0, "last_task_date": _today().isoformat()})
        )
        status = lm.trial_status()
        assert status["expired"] is False

    def test_daily_counter_increments(self, lm: LicenseManager, trial_file: Path) -> None:
        for _i in range(3):
            allowed = lm.increment_trial_task()
            assert allowed is True
        status = lm.trial_status()
        assert status["today_tasks"] == 3

    def test_daily_limit_blocks_at_10(self, lm: LicenseManager, trial_file: Path) -> None:
        for _ in range(TRIAL_DAILY_TASK_LIMIT):
            lm.increment_trial_task()
        allowed = lm.increment_trial_task()
        assert allowed is False
        status = lm.trial_status()
        assert status["daily_limit_reached"] is True

    def test_daily_counter_resets_next_day(
        self, lm: LicenseManager, trial_file: Path
    ) -> None:
        yesterday = (_today() - timedelta(days=1)).isoformat()
        trial_file.write_text(
            json.dumps({
                "started_on": (_today() - timedelta(days=2)).isoformat(),
                "today_tasks": 9,
                "last_task_date": yesterday,
            })
        )
        status = lm.trial_status()
        assert status["today_tasks"] == 0
        assert status["daily_limit_reached"] is False


# ---------------------------------------------------------------------------
# Trial messages
# ---------------------------------------------------------------------------


class TestTrialMessages:
    def test_active_trial_message(self, lm: LicenseManager, trial_file: Path) -> None:
        msg = lm.trial_message()
        assert "Trial modu" in msg
        assert "gün" in msg

    def test_expired_message(
        self, lm: LicenseManager, trial_file: Path
    ) -> None:
        started = _today() - timedelta(days=TRIAL_DAYS)
        trial_file.write_text(
            json.dumps({"started_on": started.isoformat(), "today_tasks": 0, "last_task_date": _today().isoformat()})
        )
        msg = lm.trial_message()
        assert "Trial sona erdi" in msg

    def test_daily_limit_message(self, lm: LicenseManager, trial_file: Path) -> None:
        for _ in range(TRIAL_DAILY_TASK_LIMIT):
            lm.increment_trial_task()
        msg = lm.trial_message()
        assert "limit" in msg.lower()


# ---------------------------------------------------------------------------
# is_licensed
# ---------------------------------------------------------------------------


class TestIsLicensed:
    def test_valid_full_key_via_env(
        self, lm: LicenseManager, monkeypatch: pytest.MonkeyPatch, trial_file: Path
    ) -> None:
        key = make_key()
        monkeypatch.setenv("NEXUS_LICENSE_KEY", key)
        assert lm.is_licensed() is True

    def test_expired_key_falls_through_to_trial(
        self, lm: LicenseManager, monkeypatch: pytest.MonkeyPatch, trial_file: Path
    ) -> None:
        past = _today() - timedelta(days=1)
        key = make_key(expires=past)
        monkeypatch.setenv("NEXUS_LICENSE_KEY", key)
        # Trial still valid (day 1)
        assert lm.is_licensed() is True

    def test_trial_expired_returns_false(
        self, lm: LicenseManager, monkeypatch: pytest.MonkeyPatch, trial_file: Path
    ) -> None:
        monkeypatch.delenv("NEXUS_LICENSE_KEY", raising=False)
        started = _today() - timedelta(days=TRIAL_DAYS)
        trial_file.write_text(
            json.dumps({"started_on": started.isoformat(), "today_tasks": 0, "last_task_date": _today().isoformat()})
        )
        assert lm.is_licensed() is False

    def test_daily_limit_reached_returns_false(
        self, lm: LicenseManager, monkeypatch: pytest.MonkeyPatch, trial_file: Path
    ) -> None:
        monkeypatch.delenv("NEXUS_LICENSE_KEY", raising=False)
        trial_file.write_text(
            json.dumps({
                "started_on": _today().isoformat(),
                "today_tasks": TRIAL_DAILY_TASK_LIMIT,
                "last_task_date": _today().isoformat(),
            })
        )
        assert lm.is_licensed() is False

    def test_no_key_fresh_trial_is_licensed(
        self, lm: LicenseManager, monkeypatch: pytest.MonkeyPatch, trial_file: Path
    ) -> None:
        monkeypatch.delenv("NEXUS_LICENSE_KEY", raising=False)
        assert lm.is_licensed() is True

    def test_full_license_bypasses_trial_counter(
        self, lm: LicenseManager, monkeypatch: pytest.MonkeyPatch, trial_file: Path
    ) -> None:
        key = make_key()
        monkeypatch.setenv("NEXUS_LICENSE_KEY", key)
        # Even with daily limit reached, full license passes
        trial_file.write_text(
            json.dumps({
                "started_on": _today().isoformat(),
                "today_tasks": TRIAL_DAILY_TASK_LIMIT,
                "last_task_date": _today().isoformat(),
            })
        )
        assert lm.is_licensed() is True
