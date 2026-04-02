"""Structural integrity tests for the nexus-agent project skeleton."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent

EXPECTED_PACKAGES = [
    "nexus",
    "nexus/core",
    "nexus/infra",
    "nexus/source",
    "nexus/source/uia",
    "nexus/source/dom",
    "nexus/source/file",
    "nexus/source/transport",
    "nexus/capture",
    "nexus/perception",
    "nexus/perception/arbitration",
    "nexus/decision",
    "nexus/action",
    "nexus/action/handlers",
    "nexus/verification",
    "nexus/memory",
    "nexus/cloud",
    "nexus/integrations",
    "nexus/integrations/google",
    "nexus/skills",
    "nexus/skills/browser",
    "nexus/skills/spreadsheet",
    "nexus/skills/pdf",
    "nexus/skills/desktop",
    "nexus/ui",
    "nexus/release",
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/golden",
    "tests/adversarial",
    "tests/benchmarks",
]

EXPECTED_DIRS = [
    "docs/adr",
    "configs",
    "scripts",
]

EXPECTED_FILES = [
    "pyproject.toml",
    "ruff.toml",
    "mypy.ini",
    "pytest.ini",
    "Makefile",
]


class TestDirectoryStructure:
    def test_package_dirs_exist(self) -> None:
        for pkg in EXPECTED_PACKAGES:
            path = ROOT / pkg
            assert path.is_dir(), f"Missing directory: {pkg}"

    def test_non_package_dirs_exist(self) -> None:
        for d in EXPECTED_DIRS:
            path = ROOT / d
            assert path.is_dir(), f"Missing directory: {d}"

    def test_init_files_present(self) -> None:
        for pkg in EXPECTED_PACKAGES:
            init = ROOT / pkg / "__init__.py"
            assert init.is_file(), f"Missing __init__.py in: {pkg}"

    def test_config_files_present(self) -> None:
        for fname in EXPECTED_FILES:
            path = ROOT / fname
            assert path.is_file(), f"Missing file: {fname}"


class TestPyprojectToml:
    def _load(self) -> dict:  # type: ignore[type-arg]
        with open(ROOT / "pyproject.toml", "rb") as f:
            return tomllib.load(f)

    def test_parseable(self) -> None:
        data = self._load()
        assert isinstance(data, dict)

    def test_project_section(self) -> None:
        data = self._load()
        assert "project" in data
        project = data["project"]
        assert project["name"] == "nexus-agent"
        assert "requires-python" in project

    def test_python_constraint(self) -> None:
        data = self._load()
        constraint = data["project"]["requires-python"]
        assert "3.11" in constraint

    def test_production_dependencies(self) -> None:
        data = self._load()
        deps = data["project"]["dependencies"]
        dep_names = {d.split(">=")[0].split("[")[0].lower() for d in deps}
        required = {
            "pydantic", "pydantic-settings", "structlog", "aiosqlite",
            "dxcam", "pytesseract", "pillow", "pywin32", "openai",
            "anthropic", "langdetect", "numpy", "opencv-python",
            "aiohttp", "websockets",
        }
        for req in required:
            assert req in dep_names, f"Missing production dependency: {req}"

    def test_dev_dependencies(self) -> None:
        data = self._load()
        dev_deps = data["project"]["optional-dependencies"]["dev"]
        dev_names = {d.split(">=")[0].split("[")[0].lower() for d in dev_deps}
        required = {
            "pytest", "pytest-asyncio", "pytest-cov", "pytest-mock",
            "pytest-timeout", "ruff", "mypy", "hypothesis",
            "factory-boy", "bandit", "safety",
        }
        for req in required:
            assert req in dev_names, f"Missing dev dependency: {req}"


class TestDocumentation:
    ADR_DIR = ROOT / "docs" / "adr"
    EXPECTED_ADRS = [
        "ADR-001", "ADR-002", "ADR-003", "ADR-004", "ADR-005",
        "ADR-006", "ADR-007", "ADR-008", "ADR-009", "ADR-010", "ADR-011",
    ]
    GLOSSARY_TERMS = [
        "source_layer", "transport_resolver", "uia_adapter", "dom_adapter",
        "file_adapter", "visual_fallback", "capture", "frame",
        "stabilization_gate", "dirty_region", "locator", "reader",
        "matcher", "temporal_expert", "spatial_graph", "perception_result",
        "arbitration", "confidence_score", "ambiguity_score", "decision",
        "action_spec", "transport_layer", "macroaction", "preflight",
        "verification", "visual_verification", "semantic_verification",
        "source_verification", "verification_policy", "fingerprint",
        "correction_memory", "hitl", "suspend", "resume", "byok",
        "cost_ledger", "budget_cap", "golden_scenario", "adversarial_test",
    ]

    def test_adr_count(self) -> None:
        adr_files = list(self.ADR_DIR.glob("ADR-*.md"))
        found = {re.match(r"(ADR-\d+)", f.name).group(1) for f in adr_files}  # type: ignore[union-attr]
        for expected in self.EXPECTED_ADRS:
            assert expected in found, f"Missing ADR file for: {expected}"
        assert len(found) >= 11, f"Expected at least 11 ADRs, found {len(found)}"

    def test_glossary_exists(self) -> None:
        assert (ROOT / "docs" / "glossary.md").is_file()

    def test_glossary_term_count(self) -> None:
        content = (ROOT / "docs" / "glossary.md").read_text(encoding="utf-8")
        missing = [term for term in self.GLOSSARY_TERMS if term not in content]
        assert not missing, f"Missing glossary terms: {missing}"
        assert len(self.GLOSSARY_TERMS) >= 35

    def test_v1_scope_exists(self) -> None:
        assert (ROOT / "docs" / "v1_scope.md").is_file()

    def test_v1_scope_sections(self) -> None:
        content = (ROOT / "docs" / "v1_scope.md").read_text(encoding="utf-8")
        assert "V1-core" in content, "v1_scope.md missing V1-core section"
        assert "V1-nice" in content, "v1_scope.md missing V1-nice section"
        assert "V2" in content, "v1_scope.md missing V2 section"
