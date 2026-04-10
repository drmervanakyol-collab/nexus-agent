"""
tests/unit/test_onboarding.py
Unit tests for nexus/ui/onboarding.py — Faz 56.

TEST PLAN
---------
is_first_run:
  1. No consents stored → True.
  2. Only 'privacy' stored → True (both required).
  3. Both 'privacy' + 'terms' stored → False.

run() — full happy-path (first run):
  4. All steps complete → True, both consents saved, browser launched.

run() — returning user:
  5. Both consents already stored → True immediately (no prompts).

run() — cancellation paths:
  6. Welcome step cancelled ('q') → False, no consents.
  7. Health step: overall="fail" → False.
  8. Health step: overall="warn", user declines → False.
  9. Privacy declined → False, privacy consent NOT saved.
  10. Terms declined → False, terms consent NOT saved.
  11. API key step: invalid provider choice → False.
  12. API key step: blank key → False.

API key validation with retry:
  13. First key invalid, second valid → flow continues.

Browser setup:
  14. User declines browser setup → _launch_browser_fn NOT called.
  15. User accepts, launch succeeds → print confirms.
  16. User accepts, launch fails → warning printed, flow continues.

Connectivity:
  17. test_api returns (True, 0.001) → cost printed.
  18. test_api returns (False, 0.0) → failure message, flow still completes.

Consent persistence:
  19. After full run, both 'privacy' and 'terms' are in consent store.
"""
from __future__ import annotations

from nexus.infra.health import CheckResult, HealthReport
from nexus.ui.onboarding import OnboardingFlow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report(overall: str) -> HealthReport:
    """Build a minimal HealthReport with a single check at *overall* level."""
    check = CheckResult(
        name="CHECK_PYTHON_VERSION",
        status=overall,  # type: ignore[arg-type]
        message="test",
        fix_hint="fix it" if overall != "ok" else "",
    )
    report = HealthReport(checks=[check])
    return report


def _flow(
    *,
    prompts: list[str],
    consent_store: dict[str, bool] | None = None,
    health_overall: str = "ok",
    validate_results: list[tuple[bool, str]] | None = None,
    launch_result: bool = True,
    test_api_result: tuple[bool, float] = (True, 0.001),
    printed: list[str] | None = None,
) -> OnboardingFlow:
    """
    Build an OnboardingFlow backed by queue-style stubs.

    *prompts* is consumed in order for each prompt call.
    *consent_store* maps scope → granted (default empty dict).
    *validate_results* is consumed in order for each validate_key call.
    """
    _store: dict[str, bool] = consent_store if consent_store is not None else {}
    _prompts = list(prompts)
    _validate = list(validate_results or [(True, "ok")])
    _printed = printed if printed is not None else []

    def _print(text: str) -> None:
        _printed.append(text)

    def _prompt(label: str) -> str:
        if not _prompts:
            return ""
        return _prompts.pop(0)

    def _has_consent(scope: str) -> bool:
        return _store.get(scope, False)

    def _save_consent(scope: str) -> None:
        _store[scope] = True

    def _health() -> HealthReport:
        return _make_report(health_overall)

    def _validate_key(provider: str, key: str) -> tuple[bool, str]:
        if not _validate:
            return True, "ok"
        return _validate.pop(0)

    def _launch() -> bool:
        return launch_result

    def _test_api(provider: str, key: str) -> tuple[bool, float]:
        return test_api_result

    return OnboardingFlow(
        _print_fn=_print,
        _prompt_fn=_prompt,
        _has_consent_fn=_has_consent,
        _save_consent_fn=_save_consent,
        _health_check_fn=_health,
        _validate_key_fn=_validate_key,
        _launch_browser_fn=_launch,
        _test_api_fn=_test_api,
    )


# ---------------------------------------------------------------------------
# is_first_run
# ---------------------------------------------------------------------------


class TestIsFirstRun:
    def test_no_consents_is_first_run(self):
        flow = _flow(prompts=[])
        assert flow.is_first_run() is True

    def test_only_privacy_is_first_run(self):
        store = {"privacy": True}
        flow = _flow(prompts=[], consent_store=store)
        assert flow.is_first_run() is True

    def test_both_consents_not_first_run(self):
        store = {"privacy": True, "terms": True}
        flow = _flow(prompts=[], consent_store=store)
        assert flow.is_first_run() is False


# ---------------------------------------------------------------------------
# run() — full happy-path
# ---------------------------------------------------------------------------


class TestRunHappyPath:
    def test_full_flow_returns_true(self):
        """All steps succeed → True, both consents persisted."""
        store: dict[str, bool] = {}
        prompts = [
            "",    # welcome: press Enter
            "e",   # privacy: accept
            "e",   # terms: accept
            "1",   # provider: anthropic
            "sk-ant-valid_key_here",  # api key
            "e",   # browser setup: yes
        ]
        flow = _flow(prompts=prompts, consent_store=store)
        result = flow.run()

        assert result is True
        assert store.get("privacy") is True
        assert store.get("terms") is True

    def test_returning_user_skips_prompts(self):
        """Both consents already stored → run() returns True with no prompts."""
        store = {"privacy": True, "terms": True}
        prompt_calls: list[str] = []
        flow = OnboardingFlow(
            _print_fn=lambda _: None,
            _prompt_fn=lambda label: prompt_calls.append(label) or "",  # type: ignore[func-returns-value]
            _has_consent_fn=lambda scope: store.get(scope, False),
            _save_consent_fn=lambda scope: None,
        )
        result = flow.run()

        assert result is True
        assert prompt_calls == [], "No prompts should be issued for returning user"


# ---------------------------------------------------------------------------
# run() — cancellation paths
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_welcome_quit_returns_false(self):
        flow = _flow(prompts=["q"])
        assert flow.run() is False

    def test_health_fail_returns_false(self):
        flow = _flow(
            prompts=[""],   # welcome Enter
            health_overall="fail",
        )
        assert flow.run() is False

    def test_health_warn_user_declines_returns_false(self):
        flow = _flow(
            prompts=["", "h"],   # welcome Enter, health warn → decline
            health_overall="warn",
        )
        assert flow.run() is False

    def test_privacy_declined_returns_false(self):
        store: dict[str, bool] = {}
        flow = _flow(
            prompts=["", "h"],   # welcome Enter, privacy decline
            consent_store=store,
            health_overall="ok",
        )
        assert flow.run() is False
        assert "privacy" not in store

    def test_terms_declined_returns_false(self):
        store: dict[str, bool] = {}
        flow = _flow(
            prompts=["", "e", "h"],   # welcome, privacy accept, terms decline
            consent_store=store,
        )
        assert flow.run() is False
        assert store.get("privacy") is True
        assert "terms" not in store

    def test_invalid_provider_choice_returns_false(self):
        flow = _flow(
            prompts=["", "e", "e", "9"],   # welcome, privacy, terms, bad choice
        )
        assert flow.run() is False

    def test_blank_key_returns_false(self):
        flow = _flow(
            prompts=["", "e", "e", "1", ""],  # welcome, privacy, terms, provider, empty key
        )
        assert flow.run() is False


# ---------------------------------------------------------------------------
# API key retry
# ---------------------------------------------------------------------------


class TestAPIKeyRetry:
    def test_first_invalid_second_valid_continues(self):
        """First key fails validation; second passes → flow completes."""
        store: dict[str, bool] = {}
        prompts = [
            "",              # welcome
            "e",             # privacy
            "e",             # terms
            "1",             # provider: anthropic
            "bad_key",       # first key → invalid
            "sk-ant-good",   # second key → valid
            "h",             # browser: no
        ]
        flow = _flow(
            prompts=prompts,
            consent_store=store,
            validate_results=[(False, "key too short"), (True, "ok")],
        )
        result = flow.run()
        assert result is True


# ---------------------------------------------------------------------------
# Browser setup step
# ---------------------------------------------------------------------------


class TestBrowserSetup:
    def test_declined_browser_not_launched(self):
        launch_calls: list[bool] = []

        def _launch() -> bool:
            launch_calls.append(True)
            return True

        prompts = ["", "e", "e", "1", "sk-ant-valid_key", "h"]
        flow = OnboardingFlow(
            _print_fn=lambda _: None,
            _prompt_fn=lambda _: prompts.pop(0) if prompts else "",
            _has_consent_fn=lambda _: False,
            _save_consent_fn=lambda _: None,
            _health_check_fn=lambda: _make_report("ok"),
            _validate_key_fn=lambda p, k: (True, "ok"),
            _launch_browser_fn=_launch,
            _test_api_fn=lambda p, k: (True, 0.001),
        )
        result = flow.run()

        assert result is True
        assert launch_calls == [], "Browser must not be launched when user declines"

    def test_accepted_launch_success_prints_confirmation(self):
        printed: list[str] = []
        prompts = ["", "e", "e", "1", "sk-ant-valid_key", "e"]
        flow = OnboardingFlow(
            _print_fn=lambda t: printed.append(t),
            _prompt_fn=lambda _: prompts.pop(0) if prompts else "",
            _has_consent_fn=lambda _: False,
            _save_consent_fn=lambda _: None,
            _health_check_fn=lambda: _make_report("ok"),
            _validate_key_fn=lambda p, k: (True, "ok"),
            _launch_browser_fn=lambda: True,
            _test_api_fn=lambda p, k: (True, 0.001),
        )
        flow.run()
        assert any("debug modunda başlatıldı" in line for line in printed)

    def test_accepted_launch_failure_prints_warning(self):
        printed: list[str] = []
        prompts = ["", "e", "e", "1", "sk-ant-valid_key", "e"]
        flow = OnboardingFlow(
            _print_fn=lambda t: printed.append(t),
            _prompt_fn=lambda _: prompts.pop(0) if prompts else "",
            _has_consent_fn=lambda _: False,
            _save_consent_fn=lambda _: None,
            _health_check_fn=lambda: _make_report("ok"),
            _validate_key_fn=lambda p, k: (True, "ok"),
            _launch_browser_fn=lambda: False,
            _test_api_fn=lambda p, k: (True, 0.001),
        )
        flow.run()
        assert any("başlatılamadı" in line for line in printed)


# ---------------------------------------------------------------------------
# Connectivity step
# ---------------------------------------------------------------------------


class TestConnectivity:
    def _run_to_connectivity(
        self, test_api_result: tuple[bool, float]
    ) -> list[str]:
        printed: list[str] = []
        prompts = ["", "e", "e", "1", "sk-ant-valid_key", "h"]
        OnboardingFlow(
            _print_fn=lambda t: printed.append(t),
            _prompt_fn=lambda _: prompts.pop(0) if prompts else "",
            _has_consent_fn=lambda _: False,
            _save_consent_fn=lambda _: None,
            _health_check_fn=lambda: _make_report("ok"),
            _validate_key_fn=lambda p, k: (True, "ok"),
            _launch_browser_fn=lambda: False,
            _test_api_fn=lambda p, k: test_api_result,
        ).run()
        return printed

    def test_success_prints_cost(self):
        printed = self._run_to_connectivity((True, 0.001))
        assert any("$0.0010" in line for line in printed)

    def test_failure_prints_warning(self):
        printed = self._run_to_connectivity((False, 0.0))
        assert any("başarısız" in line for line in printed)


# ---------------------------------------------------------------------------
# Consent persistence
# ---------------------------------------------------------------------------


class TestConsentPersistence:
    def test_both_consents_saved_after_full_run(self):
        store: dict[str, bool] = {}
        prompts = ["", "e", "e", "2", "sk-valid_openai_key", "h"]
        flow = _flow(
            prompts=prompts,
            consent_store=store,
            validate_results=[(True, "ok")],
        )
        result = flow.run()

        assert result is True
        assert store.get("privacy") is True, "privacy consent must be saved"
        assert store.get("terms") is True, "terms consent must be saved"
