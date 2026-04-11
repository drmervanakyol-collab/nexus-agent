"""
nexus/release/license_manager.py
Offline license validation and trial-mode enforcement for Nexus Agent.

License key format
------------------
    <b64url(payload_json)>.<b64url(hmac_sha256)>

Payload JSON keys:
    mid  — machine_id (SHA-256 hex of CPU + disk serial, 16 chars prefix)
    exp  — ISO-8601 expiry date string ("YYYY-MM-DD")  or "" for perpetual
    typ  — "full" | "trial"

HMAC secret
-----------
Resolved at runtime via env var NEXUS_LICENSE_SECRET.  Falls back to a
development-only constant that is replaced with the real secret at build
time (the placeholder is intentionally obvious so it cannot be confused
with a real secret).

Trial mode
----------
- 14-day window from first run (stored in ~/.nexus/trial.json).
- Max 10 tasks per calendar day (counter in same file).
- Once expired: is_licensed() returns False; callers receive the message
  "Trial sona erdi."

Environment variables
---------------------
NEXUS_LICENSE_KEY    — license key string (overrides key file lookup)
NEXUS_LICENSE_SECRET — HMAC secret (production: injected by build pipeline)
NEXUS_TRIAL_PATH     — override path to trial state file (useful in tests)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIAL_DAYS = 14
TRIAL_DAILY_TASK_LIMIT = 10

_DEV_SECRET = "nexus-dev-secret-REPLACE-AT-BUILD-TIME"  # noqa: S105

_DEFAULT_TRIAL_PATH = Path.home() / ".nexus" / "trial.json"
_DEFAULT_KEY_PATH = Path.home() / ".nexus" / "license.key"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class LicenseType(StrEnum):
    FULL = "full"
    TRIAL = "trial"
    NONE = "none"


@dataclass
class LicenseResult:
    valid: bool
    license_type: LicenseType
    expires: date | None
    message: str
    machine_id: str = field(default="")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _secret() -> bytes:
    raw = os.environ.get("NEXUS_LICENSE_SECRET", _DEV_SECRET)
    return raw.encode()


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _sign_payload(payload_json: str) -> str:
    sig = hmac.new(_secret(), payload_json.encode(), hashlib.sha256).digest()
    return _b64url_encode(sig)


def _wmi_cpu_serial() -> str:
    """Return CPU serial via WMI; empty string on non-Windows or WMI error."""
    try:
        import subprocess  # noqa: PLC0415
        result = subprocess.run(
            ["wmic", "cpu", "get", "ProcessorId", "/value"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "ProcessorId=" in line:
                return line.split("=", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _wmi_disk_serial() -> str:
    """Return first physical disk serial via WMI; empty string on error."""
    try:
        import subprocess  # noqa: PLC0415
        result = subprocess.run(
            ["wmic", "diskdrive", "get", "SerialNumber", "/value"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "SerialNumber=" in line:
                serial = line.split("=", 1)[1].strip()
                if serial:
                    return serial
    except Exception:  # noqa: BLE001
        pass
    return ""


def _fallback_machine_components() -> str:
    """Best-effort stable identifier when WMI is unavailable."""
    node = platform.node()
    machine = platform.machine()
    processor = platform.processor()
    return f"{node}|{machine}|{processor}"


def _today() -> date:
    return datetime.now().date()


def _trial_path() -> Path:
    env = os.environ.get("NEXUS_TRIAL_PATH", "")
    return Path(env) if env else _DEFAULT_TRIAL_PATH


def _load_trial_state() -> dict[str, Any]:
    path = _trial_path()
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_trial_state(state: dict[str, Any]) -> None:
    path = _trial_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not save trial state: %s", exc)


# ---------------------------------------------------------------------------
# LicenseManager
# ---------------------------------------------------------------------------


class LicenseManager:
    """
    Offline license manager for Nexus Agent.

    Typical usage::

        lm = LicenseManager()
        if not lm.is_licensed():
            print(lm.trial_message())
            sys.exit(1)
    """

    def __init__(self) -> None:
        self._machine_id: str | None = None

    # ------------------------------------------------------------------
    # Machine identity
    # ------------------------------------------------------------------

    def generate_machine_id(self) -> str:
        """
        Return a stable 16-character hex identifier for this machine.

        Built from SHA-256 of (CPU ProcessorId + disk SerialNumber) via WMI.
        Falls back to platform node/machine/processor strings when WMI is
        unavailable (Linux CI, test environments).
        """
        if self._machine_id is not None:
            return self._machine_id

        cpu = _wmi_cpu_serial()
        disk = _wmi_disk_serial()

        raw = f"{cpu}|{disk}" if cpu or disk else _fallback_machine_components()

        digest = hashlib.sha256(raw.encode()).hexdigest()
        self._machine_id = digest[:16]
        return self._machine_id

    # ------------------------------------------------------------------
    # Key generation (used by license issuance tool, not distributed)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_key(
        machine_id: str,
        license_type: LicenseType = LicenseType.FULL,
        expires: date | None = None,
    ) -> str:
        """
        Generate a signed license key for *machine_id*.

        This is the issuer-side method; it requires access to the HMAC secret.
        """
        payload = {
            "mid": machine_id,
            "exp": expires.isoformat() if expires else "",
            "typ": license_type.value,
        }
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        payload_b64 = _b64url_encode(payload_json.encode())
        sig_b64 = _sign_payload(payload_json)
        return f"{payload_b64}.{sig_b64}"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_license(self, key: str) -> LicenseResult:
        """
        Validate *key* offline (HMAC verification + machine_id + expiry).

        Parameters
        ----------
        key:
            License key string in ``<payload_b64>.<sig_b64>`` format.

        Returns
        -------
        LicenseResult
            ``.valid`` is True only when signature, machine_id, and expiry
            all pass.
        """
        mid = self.generate_machine_id()

        if not key or "." not in key:
            return LicenseResult(
                valid=False,
                license_type=LicenseType.NONE,
                expires=None,
                message="Geçersiz lisans anahtarı formatı.",
                machine_id=mid,
            )

        payload_b64, sig_b64 = key.rsplit(".", 1)

        # --- Decode payload ---
        try:
            payload_json = _b64url_decode(payload_b64).decode()
            payload = json.loads(payload_json)
        except Exception:  # noqa: BLE001
            return LicenseResult(
                valid=False,
                license_type=LicenseType.NONE,
                expires=None,
                message="Lisans anahtarı çözümlenemedi.",
                machine_id=mid,
            )

        # --- HMAC verification ---
        expected_sig = _sign_payload(payload_json)
        try:
            sig_bytes = _b64url_decode(sig_b64)
            expected_bytes = _b64url_decode(expected_sig)
        except Exception:  # noqa: BLE001
            return LicenseResult(
                valid=False,
                license_type=LicenseType.NONE,
                expires=None,
                message="İmza doğrulanamadı.",
                machine_id=mid,
            )

        if not hmac.compare_digest(sig_bytes, expected_bytes):
            return LicenseResult(
                valid=False,
                license_type=LicenseType.NONE,
                expires=None,
                message="Lisans imzası geçersiz.",
                machine_id=mid,
            )

        # --- Machine ID check ---
        key_mid = payload.get("mid", "")
        if key_mid != mid:
            return LicenseResult(
                valid=False,
                license_type=LicenseType.NONE,
                expires=None,
                message=f"Bu lisans bu makine için değil (beklenen: {mid}).",
                machine_id=mid,
            )

        # --- Expiry check ---
        exp_str = payload.get("exp", "")
        expires: date | None = None
        if exp_str:
            try:
                expires = date.fromisoformat(exp_str)
            except ValueError:
                return LicenseResult(
                    valid=False,
                    license_type=LicenseType.NONE,
                    expires=None,
                    message="Geçersiz son kullanma tarihi.",
                    machine_id=mid,
                )
            if _today() > expires:
                return LicenseResult(
                    valid=False,
                    license_type=LicenseType.NONE,
                    expires=expires,
                    message=f"Lisans süresi doldu ({expires.isoformat()}).",
                    machine_id=mid,
                )

        typ_str = payload.get("typ", LicenseType.FULL.value)
        try:
            lic_type = LicenseType(typ_str)
        except ValueError:
            lic_type = LicenseType.FULL

        return LicenseResult(
            valid=True,
            license_type=lic_type,
            expires=expires,
            message="Lisans geçerli.",
            machine_id=mid,
        )

    # ------------------------------------------------------------------
    # Trial mode
    # ------------------------------------------------------------------

    def trial_status(self) -> dict[str, Any]:
        """
        Return current trial state as a dict:
            started_on  : ISO date str or ""
            days_used   : int
            expired     : bool
            today_tasks : int
            daily_limit_reached : bool
        """
        state = _load_trial_state()
        today_str = _today().isoformat()

        started_on_str: str = state.get("started_on", "")
        if not started_on_str:
            # First run — initialise trial (preserve today_tasks if already set)
            started_on_str = today_str
            state["started_on"] = started_on_str
            if "today_tasks" not in state:
                state["today_tasks"] = 0
            if "last_task_date" not in state:
                state["last_task_date"] = today_str
            _save_trial_state(state)

        started_on = date.fromisoformat(started_on_str)
        days_used = (_today() - started_on).days + 1
        expired = days_used > TRIAL_DAYS

        # Reset daily counter if new day
        last_task_date = state.get("last_task_date", today_str)
        if last_task_date != today_str:
            state["today_tasks"] = 0
            state["last_task_date"] = today_str
            _save_trial_state(state)

        today_tasks: int = state.get("today_tasks", 0)

        return {
            "started_on": started_on_str,
            "days_used": days_used,
            "expired": expired,
            "today_tasks": today_tasks,
            "daily_limit_reached": today_tasks >= TRIAL_DAILY_TASK_LIMIT,
        }

    def increment_trial_task(self) -> bool:
        """
        Increment today's task counter for trial mode.

        Returns True if the task is allowed (under daily limit), False if the
        daily limit has been reached.  Does nothing and returns True when a
        full license is active.
        """
        if self._full_license_active():
            return True

        state = _load_trial_state()
        today_str = _today().isoformat()

        if state.get("last_task_date") != today_str:
            state["today_tasks"] = 0
            state["last_task_date"] = today_str

        count: int = state.get("today_tasks", 0)
        if count >= TRIAL_DAILY_TASK_LIMIT:
            return False

        state["today_tasks"] = count + 1
        _save_trial_state(state)
        return True

    def trial_message(self) -> str:
        """Human-readable trial status message (Turkish)."""
        status = self.trial_status()
        if status["expired"]:
            return "Trial sona erdi. Lütfen lisans satın alın."
        remaining = TRIAL_DAYS - status["days_used"] + 1
        if status["daily_limit_reached"]:
            return (
                f"Günlük görev limitine ulaşıldı "
                f"({TRIAL_DAILY_TASK_LIMIT} görev/gün). "
                f"Trial: {remaining} gün kaldı."
            )
        return (
            f"Trial modu: {status['days_used']}/{TRIAL_DAYS} gün, "
            f"bugün {status['today_tasks']}/{TRIAL_DAILY_TASK_LIMIT} görev."
        )

    # ------------------------------------------------------------------
    # is_licensed
    # ------------------------------------------------------------------

    def _full_license_active(self) -> bool:
        """Internal: True when a valid full/paid license key is present."""
        key = self._resolve_key()
        if not key:
            return False
        result = self.validate_license(key)
        return result.valid and result.license_type == LicenseType.FULL

    def _resolve_key(self) -> str:
        """Return license key from env var or key file, empty string if none."""
        env_key = os.environ.get("NEXUS_LICENSE_KEY", "")
        if env_key:
            return env_key.strip()
        key_file = _DEFAULT_KEY_PATH
        if key_file.is_file():
            try:
                return key_file.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        return ""

    def is_licensed(self) -> bool:
        """
        Return True when the agent may run without restriction.

        A full license always satisfies this.  A trial license satisfies this
        while within the 14-day window *and* under the daily task limit.
        """
        if self._full_license_active():
            return True

        key = self._resolve_key()
        if key:
            result = self.validate_license(key)
            if result.valid and result.license_type == LicenseType.TRIAL:
                status = self.trial_status()
                return not status["expired"] and not status["daily_limit_reached"]
            # Invalid key — fall through to bare trial
            if not result.valid:
                logger.warning("License key invalid: %s", result.message)

        # Bare trial (no key)
        status = self.trial_status()
        return not status["expired"] and not status["daily_limit_reached"]
