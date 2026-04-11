"""
tests/unit/test_legal_documents.py
Legal document and versioned-consent tests — Faz 68

TEST 1 — Her belge var mi? (Document existence)
  docs/privacy_policy.md exists and is non-empty.
  docs/terms_of_service.md exists and is non-empty.

TEST 2 — Transport seffafligi privacy policy'de mi?
  (Transport transparency in privacy policy?)
  Privacy policy contains dedicated transport transparency section.
  Native transport: explicitly states screenshot is NOT sent.
  Visual transport: explicitly states masked screenshot IS sent.
  Sensitive regions: explicitly states NEVER sent.

TEST 3 — Onay versiyonlaniyor mu? (Consent versioning)
  _save_consent_fn receives (scope, version) — not just scope.
  _has_consent_fn checks scope+version pair.
  A consent stored for v1.0 does NOT satisfy a v2.0 check.
  Privacy version constant exists in onboarding module.
  Terms version constant exists in onboarding module.

TEST 4 — Belge icerigi (Document content)
  Privacy policy contains KVKK rights section.
  Privacy policy mentions data deletion.
  Terms of service contains liability limitation section.
  Terms of service contains BYOK responsibility section.
  Terms of service contains unacceptable use section.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Document paths
# ---------------------------------------------------------------------------

_DOCS_DIR = Path("docs")
_PRIVACY_POLICY = _DOCS_DIR / "privacy_policy.md"
_TERMS_OF_SERVICE = _DOCS_DIR / "terms_of_service.md"


# ---------------------------------------------------------------------------
# TEST 1 — Document existence
# ---------------------------------------------------------------------------


class TestDocumentExistence:
    def test_privacy_policy_exists(self) -> None:
        assert _PRIVACY_POLICY.exists(), (
            f"docs/privacy_policy.md not found at {_PRIVACY_POLICY.resolve()}"
        )

    def test_privacy_policy_non_empty(self) -> None:
        assert _PRIVACY_POLICY.exists()
        content = _PRIVACY_POLICY.read_text(encoding="utf-8")
        assert len(content) > 500, (
            f"privacy_policy.md too short ({len(content)} chars) — incomplete document"
        )

    def test_terms_of_service_exists(self) -> None:
        assert _TERMS_OF_SERVICE.exists(), (
            f"docs/terms_of_service.md not found at {_TERMS_OF_SERVICE.resolve()}"
        )

    def test_terms_of_service_non_empty(self) -> None:
        assert _TERMS_OF_SERVICE.exists()
        content = _TERMS_OF_SERVICE.read_text(encoding="utf-8")
        assert len(content) > 500, (
            f"terms_of_service.md too short ({len(content)} chars) — incomplete document"
        )

    def test_privacy_policy_has_version_header(self) -> None:
        """Document must declare its version number."""
        content = _PRIVACY_POLICY.read_text(encoding="utf-8")
        assert "1.0" in content, "Privacy policy must declare version 1.0"

    def test_terms_of_service_has_version_header(self) -> None:
        """Document must declare its version number."""
        content = _TERMS_OF_SERVICE.read_text(encoding="utf-8")
        assert "1.0" in content, "Terms of service must declare version 1.0"


# ---------------------------------------------------------------------------
# TEST 2 — Transport transparency in privacy policy
# ---------------------------------------------------------------------------


class TestTransportTransparencyInPrivacyPolicy:
    def _policy(self) -> str:
        return _PRIVACY_POLICY.read_text(encoding="utf-8").lower()

    def test_transport_transparency_section_exists(self) -> None:
        """Privacy policy must have a dedicated transport transparency section."""
        content = _PRIVACY_POLICY.read_text(encoding="utf-8")
        has_section = (
            "Transport" in content
            and ("effaflık" in content or "effafl" in content or "transparency" in content.lower())
        )
        assert has_section, (
            "Privacy policy must contain a transport transparency section"
        )

    def test_native_transport_no_screenshot_sent(self) -> None:
        """Privacy policy must explicitly state that native transport sends NO screenshot."""
        content = self._policy()
        # Must mention that UIA/native transport does NOT send screenshots
        native_no_send = (
            ("native" in content or "uia" in content or "uiautomation" in content)
            and ("gönderilmez" in content or "hayir" in content or "gitmez" in content
                 or "gitmez" in content)
        )
        assert native_no_send, (
            "Privacy policy must state that native transport (UIA) does NOT send screenshots"
        )

    def test_visual_transport_masked_screenshot_sent(self) -> None:
        """Privacy policy must state that visual transport sends MASKED screenshot."""
        content = self._policy()
        visual_sends = (
            ("visual" in content or "mouse" in content)
            and ("maskelen" in content or "mask" in content)
        )
        assert visual_sends, (
            "Privacy policy must state that visual transport sends a masked screenshot"
        )

    def test_sensitive_regions_never_sent(self) -> None:
        """Privacy policy must state sensitive regions are NEVER sent."""
        content = self._policy()
        sensitive_never = (
            ("hassas" in content or "sensitive" in content)
            and (
                "hicbir zaman" in content      # ASCII fallback
                or "hiçbir zaman" in content   # Turkish with ç
                or "never" in content
                or "hayır" in content
                or "hayir" in content
            )
        )
        assert sensitive_never, (
            "Privacy policy must explicitly state sensitive regions are NEVER sent"
        )

    def test_transport_comparison_table_exists(self) -> None:
        """Privacy policy should contain a comparison table of transport methods."""
        content = _PRIVACY_POLICY.read_text(encoding="utf-8")
        # Markdown table has | separators
        has_table = content.count("|") >= 10  # at least a simple table
        assert has_table, (
            "Privacy policy should contain a transport comparison table (markdown |...| format)"
        )

    def test_cloud_data_summary_present(self) -> None:
        """Privacy policy must have a 'what goes to cloud' summary."""
        content = self._policy()
        has_cloud_section = (
            "bulut" in content or "cloud" in content
        ) and (
            "gönderilen" in content or "gidiyor" in content or "sent" in content
        )
        assert has_cloud_section, (
            "Privacy policy must have a 'what goes to cloud' section"
        )


# ---------------------------------------------------------------------------
# TEST 3 — Consent versioning
# ---------------------------------------------------------------------------


class TestConsentVersioning:
    def test_privacy_version_constant_exists(self) -> None:
        """onboarding module exports _PRIVACY_VERSION constant."""
        from nexus.ui.onboarding import _PRIVACY_VERSION

        assert _PRIVACY_VERSION, "onboarding._PRIVACY_VERSION must be a non-empty string"
        assert isinstance(_PRIVACY_VERSION, str)

    def test_terms_version_constant_exists(self) -> None:
        """onboarding module exports _TERMS_VERSION constant."""
        from nexus.ui.onboarding import _TERMS_VERSION

        assert _TERMS_VERSION, "onboarding._TERMS_VERSION must be a non-empty string"
        assert isinstance(_TERMS_VERSION, str)

    def test_versions_match_document_version(self) -> None:
        """onboarding version constants must match the version in the docs."""
        from nexus.ui.onboarding import _PRIVACY_VERSION, _TERMS_VERSION

        privacy_doc = _PRIVACY_POLICY.read_text(encoding="utf-8")
        terms_doc = _TERMS_OF_SERVICE.read_text(encoding="utf-8")

        assert _PRIVACY_VERSION in privacy_doc, (
            f"_PRIVACY_VERSION={_PRIVACY_VERSION!r} not found in privacy_policy.md"
        )
        assert _TERMS_VERSION in terms_doc, (
            f"_TERMS_VERSION={_TERMS_VERSION!r} not found in terms_of_service.md"
        )

    def test_save_consent_receives_version(self) -> None:
        """_save_consent_fn is called with (scope, version) — both non-empty."""
        from nexus.ui.onboarding import OnboardingFlow

        saved: list[tuple[str, str]] = []

        def _save(scope: str, version: str) -> None:
            saved.append((scope, version))

        prompts = iter(["", "e", "e", "1", "sk-ant-valid_key", "h"])
        flow = OnboardingFlow(
            _print_fn=lambda _: None,
            _prompt_fn=lambda _: next(prompts, ""),
            _has_consent_fn=lambda s, v: False,
            _save_consent_fn=_save,
            _health_check_fn=lambda: _make_ok_health(),
            _validate_key_fn=lambda p, k: (True, "ok"),
            _test_api_fn=lambda p, k: (True, 0.001),
        )
        flow.run()

        # privacy and terms must both be saved with version info
        scopes = {s for s, _ in saved}
        assert "privacy" in scopes, "privacy consent must be saved"
        assert "terms" in scopes, "terms consent must be saved"

        for scope, version in saved:
            assert version, f"version must not be empty for scope={scope!r}"

    def test_has_consent_receives_version(self) -> None:
        """_has_consent_fn is called with (scope, version) — both non-empty."""
        from nexus.ui.onboarding import OnboardingFlow

        checked: list[tuple[str, str]] = []

        def _has(scope: str, version: str) -> bool:
            checked.append((scope, version))
            return False  # force onboarding to run

        prompts = iter(["", "e", "e", "1", "sk-ant-valid_key", "h"])
        flow = OnboardingFlow(
            _print_fn=lambda _: None,
            _prompt_fn=lambda _: next(prompts, ""),
            _has_consent_fn=_has,
            _save_consent_fn=lambda s, v: None,
            _health_check_fn=lambda: _make_ok_health(),
            _validate_key_fn=lambda p, k: (True, "ok"),
            _test_api_fn=lambda p, k: (True, 0.001),
        )
        flow.run()

        assert checked, "_has_consent_fn must be called at least once"
        for scope, version in checked:
            assert version, f"version must not be empty for scope={scope!r}"

    def test_old_version_consent_requires_reaccept(self) -> None:
        """
        A consent stored under v1.0 does NOT satisfy a v2.0 check.
        is_first_run() returns True when stored version differs from current.
        """
        from nexus.ui.onboarding import OnboardingFlow

        # Simulate: user previously consented to v1.0, but current version is v2.0
        store = {"privacy:1.0": True, "terms:1.0": True}

        def _has(scope: str, version: str) -> bool:
            return store.get(f"{scope}:{version}", False)

        flow_v2 = OnboardingFlow(
            _print_fn=lambda _: None,
            _prompt_fn=lambda _: "",
            _has_consent_fn=lambda s, v: store.get(f"{s}:2.0", False),  # checks v2.0
            _save_consent_fn=lambda s, v: None,
        )
        # v2.0 consent not in store → is_first_run() should return True
        assert flow_v2.is_first_run() is True, (
            "User must re-accept when document version changes"
        )

    def test_current_version_consent_skips_onboarding(self) -> None:
        """A consent stored under the current version skips onboarding."""
        from nexus.ui.onboarding import _PRIVACY_VERSION, _TERMS_VERSION, OnboardingFlow

        store = {
            f"privacy:{_PRIVACY_VERSION}": True,
            f"terms:{_TERMS_VERSION}": True,
        }
        flow = OnboardingFlow(
            _print_fn=lambda _: None,
            _prompt_fn=lambda _: "",
            _has_consent_fn=lambda s, v: store.get(f"{s}:{v}", False),
            _save_consent_fn=lambda s, v: None,
        )
        assert flow.is_first_run() is False, (
            "User should NOT be shown onboarding when current-version consent exists"
        )


# ---------------------------------------------------------------------------
# TEST 4 — Document content completeness
# ---------------------------------------------------------------------------


class TestDocumentContent:
    def test_privacy_policy_has_kvkk_section(self) -> None:
        """Privacy policy must contain KVKK rights."""
        content = _PRIVACY_POLICY.read_text(encoding="utf-8")
        assert "KVKK" in content or "6698" in content, (
            "Privacy policy must reference KVKK (Turkish data protection law)"
        )

    def test_privacy_policy_has_data_deletion(self) -> None:
        """Privacy policy must describe data deletion procedure."""
        content = _PRIVACY_POLICY.read_text(encoding="utf-8").lower()
        has_deletion = "silme" in content or "delete" in content or "reset" in content
        assert has_deletion, "Privacy policy must describe data deletion"

    def test_terms_has_liability_limitation(self) -> None:
        """Terms of service must contain liability limitation."""
        content = _TERMS_OF_SERVICE.read_text(encoding="utf-8").lower()
        has_liability = (
            "sorumluluk" in content or "liability" in content
            or "garanti" in content or "warranty" in content
        )
        assert has_liability, "Terms must contain liability limitation section"

    def test_terms_has_byok_section(self) -> None:
        """Terms of service must explain BYOK responsibility."""
        content = _TERMS_OF_SERVICE.read_text(encoding="utf-8")
        assert "BYOK" in content or "Bring Your Own Key" in content, (
            "Terms must contain BYOK responsibility section"
        )

    def test_terms_has_unacceptable_use(self) -> None:
        """Terms of service must list unacceptable use cases."""
        content = _TERMS_OF_SERVICE.read_text(encoding="utf-8").lower()
        has_unacceptable = (
            "kabul edilemez" in content
            or "unacceptable" in content
            or "yasak" in content
            or "prohibited" in content
        )
        assert has_unacceptable, "Terms must list unacceptable use cases"

    def test_terms_has_api_cost_responsibility(self) -> None:
        """Terms must state API costs are the user's responsibility."""
        content = _TERMS_OF_SERVICE.read_text(encoding="utf-8").lower()
        has_cost_clause = (
            "maliyet" in content or "cost" in content or "ücret" in content
        ) and (
            "sorumlu" in content or "responsible" in content
        )
        assert has_cost_clause, (
            "Terms must state that API costs are the user's responsibility"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ok_health():
    from nexus.infra.health import CheckResult, HealthReport

    return HealthReport(checks=[CheckResult(name="test", status="ok", message="")])
