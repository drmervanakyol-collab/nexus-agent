"""
tests/unit/test_spreadsheet_skills.py
Unit tests for nexus/skills/spreadsheet/ — Faz 48.

TEST PLAN
---------
Safety guard:
  1.  verify_correct_cell — correct column  → True
  2.  verify_correct_cell — wrong column    → False
  3.  verify_correct_cell — no cell ref     → False
  4.  check_formula_protection — formula cell  → True
  5.  check_formula_protection — plain value   → False
  6.  row_identity_lock — matching identifier  → True
  7.  row_identity_lock — wrong identifier     → False
  8.  row_identity_lock — no current cell      → False
  9.  check_calculation_mode — returns injected value
  10. bypass_autocorrect — fraction  → prefixed
  11. bypass_autocorrect — date-like → prefixed
  12. bypass_autocorrect — plain text → unchanged

Navigation:
  13. go_to_cell — UIA Name Box found   → set_value + Enter, no keyboard
  14. go_to_cell — UIA not found        → keyboard fallback (F5 + type + Enter)
  15. get_current_cell — returns Name Box value
  16. get_current_cell — no Name Box    → empty string
  17. get_sheet_names — injected fn     → list returned

Extraction:
  18. extract_table with header → list[dict] keyed by header row
  19. extract_table no header   → col_0, col_1 keys
  20. extract_table empty graph → []
  21. extract_table header-only → []
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from nexus.core.settings import NexusSettings
from nexus.core.types import ElementId, Rect
from nexus.perception.arbitration.arbitrator import ArbitrationResult
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.matcher.matcher import Affordance, SemanticLabel
from nexus.perception.orchestrator import PerceptionResult
from nexus.perception.spatial_graph import SpatialGraph
from nexus.perception.temporal.temporal_expert import ScreenState, StateType
from nexus.skills.spreadsheet.extraction import SpreadsheetExtractor
from nexus.skills.spreadsheet.navigation import SpreadsheetNavigator
from nexus.skills.spreadsheet.safety import SpreadsheetSafetyGuard
from nexus.source.resolver import SourceResult
from nexus.source.uia.adapter import UIAAdapter, UIAElement

# ---------------------------------------------------------------------------
# Helpers — UIAAdapter stubs
# ---------------------------------------------------------------------------


def _make_settings() -> NexusSettings:
    return NexusSettings.model_validate({})


def _uia_stub(*, available: bool = True) -> UIAAdapter:
    """Return a UIAAdapter that uses a mock COM factory."""
    mock_auto = MagicMock()
    stub = UIAAdapter(_make_settings(), _automation_factory=lambda: mock_auto)
    if not available:
        stub._automation_factory = lambda: (_ for _ in ()).throw(
            RuntimeError("UIA unavailable")
        )
    return stub


def _make_uia_element(
    *,
    name: str = "B5",
    value: str = "",
    automation_id: str = "Box",
) -> UIAElement:
    return UIAElement(
        automation_id=automation_id,
        name=name,
        control_type=50000,
        bounding_rect=Rect(0, 0, 80, 20),
        is_enabled=True,
        is_visible=True,
        value=value or None,
        _raw=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Helpers — PerceptionResult stubs
# ---------------------------------------------------------------------------


def _make_ui_element(
    eid: ElementId | None = None,
    *,
    rect: Rect,
) -> UIElement:
    return UIElement(
        id=eid or ElementId(str(uuid.uuid4())),
        element_type=ElementType.LABEL,
        bounding_box=rect,
        confidence=0.9,
        is_visible=True,
        is_occluded=False,
        occlusion_ratio=0.0,
        z_order_estimate=1,
    )


def _make_semantic(element_id: ElementId) -> SemanticLabel:
    return SemanticLabel(
        element_id=element_id,
        primary_label="label",
        secondary_labels=[],
        confidence=0.9,
        affordance=Affordance.READ_ONLY,
        is_destructive=False,
    )


def _make_perception(
    cells: list[tuple[str, Rect]],
) -> PerceptionResult:
    """Build a PerceptionResult whose graph has one node per (text, rect) pair."""
    elements = []
    labels = []
    texts: dict[ElementId, str] = {}
    for text, rect in cells:
        el = _make_ui_element(rect=rect)
        sem = _make_semantic(el.id)
        elements.append(el)
        labels.append(sem)
        texts[el.id] = text

    stable = ScreenState(
        state_type=StateType.STABLE,
        confidence=1.0,
        blocks_perception=False,
        reason="stable",
        retry_after_ms=0,
    )
    arb = ArbitrationResult(
        resolved_elements=(),
        resolved_labels=(),
        conflicts_detected=0,
        conflicts_resolved=0,
        temporal_blocked=False,
        overall_confidence=1.0,
    )
    return PerceptionResult(
        spatial_graph=SpatialGraph(elements, labels, texts),
        screen_state=stable,
        arbitration=arb,
        source_result=SourceResult(
            source_type="visual",
            data={},
            confidence=0.8,
            latency_ms=0.0,
        ),
        perception_ms=0.0,
        frame_sequence=1,
        timestamp="2026-04-08T00:00:00Z",
    )


def _empty_perception() -> PerceptionResult:
    return _make_perception([])


# ---------------------------------------------------------------------------
# 1-12: SpreadsheetSafetyGuard
# ---------------------------------------------------------------------------


class TestVerifyCorrectCell:
    def test_correct_column_returns_true(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(
            uia, _get_current_cell_fn=lambda: "B5"
        )
        assert guard.verify_correct_cell("B") is True

    def test_wrong_column_returns_false(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(
            uia, _get_current_cell_fn=lambda: "C10"
        )
        assert guard.verify_correct_cell("B") is False

    def test_no_cell_returns_false(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(
            uia, _get_current_cell_fn=lambda: None
        )
        assert guard.verify_correct_cell("A") is False

    def test_case_insensitive(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(
            uia, _get_current_cell_fn=lambda: "aa10"
        )
        assert guard.verify_correct_cell("AA") is True

    def test_multi_letter_column(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(
            uia, _get_current_cell_fn=lambda: "AB3"
        )
        assert guard.verify_correct_cell("AB") is True


class TestCheckFormulaProtection:
    def test_formula_cell_returns_true(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(uia)
        assert guard.check_formula_protection("=SUM(A1:A5)") is True

    def test_plain_value_returns_false(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(uia)
        assert guard.check_formula_protection("Hello World") is False

    def test_empty_string_returns_false(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(uia)
        assert guard.check_formula_protection("") is False

    def test_number_returns_false(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(uia)
        assert guard.check_formula_protection("42") is False


class TestRowIdentityLock:
    def test_matching_identifier_returns_true(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(
            uia,
            _get_current_cell_fn=lambda: "B7",
            _get_cell_value_fn=lambda ref: "INV-001" if ref == "A7" else None,
        )
        assert guard.row_identity_lock("INV-001") is True

    def test_wrong_identifier_returns_false(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(
            uia,
            _get_current_cell_fn=lambda: "B7",
            _get_cell_value_fn=lambda ref: "INV-999",
        )
        assert guard.row_identity_lock("INV-001") is False

    def test_no_current_cell_returns_false(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(
            uia, _get_current_cell_fn=lambda: None
        )
        assert guard.row_identity_lock("anything") is False


class TestCheckCalculationMode:
    def test_returns_injected_value(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(
            uia, _get_calc_mode_fn=lambda: "manual"
        )
        assert guard.check_calculation_mode() == "manual"

    def test_default_returns_automatic(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(uia)
        assert guard.check_calculation_mode() == "automatic"


class TestBypassAutocorrect:
    def test_fraction_gets_prefix(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(uia)
        assert guard.bypass_autocorrect("1/2") == "'1/2"

    def test_date_like_gets_prefix(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(uia)
        assert guard.bypass_autocorrect("3-5") == "'3-5"

    def test_scientific_gets_prefix(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(uia)
        assert guard.bypass_autocorrect("1e5") == "'1e5"

    def test_leading_zero_gets_prefix(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(uia)
        assert guard.bypass_autocorrect("007") == "'007"

    def test_plain_text_unchanged(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(uia)
        assert guard.bypass_autocorrect("Hello") == "Hello"

    def test_normal_number_unchanged(self):
        uia = _uia_stub()
        guard = SpreadsheetSafetyGuard(uia)
        assert guard.bypass_autocorrect("123") == "123"


# ---------------------------------------------------------------------------
# 13-17: SpreadsheetNavigator
# ---------------------------------------------------------------------------


class TestGoToCell:
    @pytest.mark.asyncio
    async def test_uia_first_no_keyboard_fallback(self):
        """When Name Box is found and set_value succeeds, no keyboard is used."""
        name_box = _make_uia_element(name="A1", value="A1")
        special_key_calls: list[str] = []

        async def spy_special_key(key: str) -> bool:
            special_key_calls.append(key)
            return True

        type_calls: list[str] = []

        async def spy_type(text: str) -> bool:
            type_calls.append(text)
            return True

        uia = MagicMock(spec=UIAAdapter)
        uia.set_value.return_value = True

        nav = SpreadsheetNavigator(
            uia,
            _find_name_box_fn=lambda: name_box,
            _special_key_fn=spy_special_key,
            _type_fn=spy_type,
        )

        result = await nav.go_to_cell("B5")

        assert result is True
        uia.set_value.assert_called_once_with(name_box, "B5")
        assert special_key_calls == ["enter"]
        assert type_calls == []   # keyboard type NOT used

    @pytest.mark.asyncio
    async def test_keyboard_fallback_when_uia_fails(self):
        """When Name Box not found, F5 + type + Enter keyboard path is used."""
        special_key_calls: list[str] = []

        async def spy_special_key(key: str) -> bool:
            special_key_calls.append(key)
            return True

        type_calls: list[str] = []

        async def spy_type(text: str) -> bool:
            type_calls.append(text)
            return True

        uia = MagicMock(spec=UIAAdapter)

        nav = SpreadsheetNavigator(
            uia,
            _find_name_box_fn=lambda: None,   # UIA fails
            _special_key_fn=spy_special_key,
            _type_fn=spy_type,
        )

        result = await nav.go_to_cell("C10")

        assert result is True
        assert "f5" in special_key_calls
        assert "C10" in type_calls
        assert "enter" in special_key_calls

    @pytest.mark.asyncio
    async def test_uia_set_value_fails_uses_keyboard(self):
        """When set_value returns False, falls through to keyboard."""
        name_box = _make_uia_element()
        special_key_calls: list[str] = []

        async def spy_special_key(key: str) -> bool:
            special_key_calls.append(key)
            return True

        async def spy_type(text: str) -> bool:
            return True

        uia = MagicMock(spec=UIAAdapter)
        uia.set_value.return_value = False   # set_value fails

        nav = SpreadsheetNavigator(
            uia,
            _find_name_box_fn=lambda: name_box,
            _special_key_fn=spy_special_key,
            _type_fn=spy_type,
        )

        await nav.go_to_cell("D1")

        assert "f5" in special_key_calls


class TestGetCurrentCell:
    def test_returns_name_box_value(self):
        name_box = _make_uia_element(value="B5")
        uia = MagicMock(spec=UIAAdapter)
        uia.get_value.return_value = "B5"

        nav = SpreadsheetNavigator(uia, _find_name_box_fn=lambda: name_box)

        assert nav.get_current_cell() == "B5"

    def test_no_name_box_returns_empty(self):
        uia = MagicMock(spec=UIAAdapter)

        nav = SpreadsheetNavigator(uia, _find_name_box_fn=lambda: None)

        assert nav.get_current_cell() == ""


class TestGetSheetNames:
    def test_returns_injected_names(self):
        uia = MagicMock(spec=UIAAdapter)
        names = ["Sheet1", "Sheet2", "Summary"]

        nav = SpreadsheetNavigator(uia, _get_sheet_names_fn=lambda: names)

        assert nav.get_sheet_names() == names

    def test_returns_empty_when_no_tabs(self):
        uia = MagicMock(spec=UIAAdapter)

        nav = SpreadsheetNavigator(uia, _get_sheet_names_fn=lambda: [])

        assert nav.get_sheet_names() == []


# ---------------------------------------------------------------------------
# 18-21: SpreadsheetExtractor
# ---------------------------------------------------------------------------

# Cell layout helpers — construct grid positions
# Row 0 (y≈10):  Name(0,0), Age(100,0), City(200,0)
# Row 1 (y≈60):  Alice(0,50), 30(100,50), London(200,50)
# Row 2 (y≈110): Bob(0,100), 25(100,100), Paris(200,100)

_ROW_H = 30
_COL_W = 100


def _cell_rect(row: int, col: int) -> Rect:
    return Rect(col * _COL_W, row * 50, _COL_W - 5, _ROW_H)


class TestSpreadsheetExtractor:
    def test_extract_table_with_header(self):
        """First row → headers, subsequent rows → data dicts."""
        cells = [
            ("Name",   _cell_rect(0, 0)),
            ("Age",    _cell_rect(0, 1)),
            ("City",   _cell_rect(0, 2)),
            ("Alice",  _cell_rect(1, 0)),
            ("30",     _cell_rect(1, 1)),
            ("London", _cell_rect(1, 2)),
            ("Bob",    _cell_rect(2, 0)),
            ("25",     _cell_rect(2, 1)),
            ("Paris",  _cell_rect(2, 2)),
        ]
        perception = _make_perception(cells)
        extractor = SpreadsheetExtractor()
        result = extractor.extract_table(perception, has_header=True)

        assert len(result) == 2
        assert result[0] == {"Name": "Alice", "Age": "30", "City": "London"}
        assert result[1] == {"Name": "Bob", "Age": "25", "City": "Paris"}

    def test_extract_table_no_header(self):
        """Without header, all rows are data with col_N keys."""
        cells = [
            ("Alice", _cell_rect(0, 0)),
            ("30",    _cell_rect(0, 1)),
            ("Bob",   _cell_rect(1, 0)),
            ("25",    _cell_rect(1, 1)),
        ]
        perception = _make_perception(cells)
        extractor = SpreadsheetExtractor()
        result = extractor.extract_table(perception, has_header=False)

        assert len(result) == 2
        assert result[0] == {"col_0": "Alice", "col_1": "30"}
        assert result[1] == {"col_0": "Bob", "col_1": "25"}

    def test_extract_table_empty_graph(self):
        """Empty graph → empty list."""
        perception = _empty_perception()
        extractor = SpreadsheetExtractor()
        result = extractor.extract_table(perception)
        assert result == []

    def test_extract_table_header_only(self):
        """Only one row with has_header=True → no data rows → empty list."""
        cells = [
            ("Name", _cell_rect(0, 0)),
            ("Age",  _cell_rect(0, 1)),
        ]
        perception = _make_perception(cells)
        extractor = SpreadsheetExtractor()
        result = extractor.extract_table(perception, has_header=True)
        assert result == []

    def test_extract_table_short_rows_padded(self):
        """Rows shorter than the header are padded with empty strings."""
        cells = [
            ("Name",  _cell_rect(0, 0)),
            ("Age",   _cell_rect(0, 1)),
            ("City",  _cell_rect(0, 2)),
            ("Alice", _cell_rect(1, 0)),
            # Age and City missing for Alice's row
        ]
        perception = _make_perception(cells)
        extractor = SpreadsheetExtractor()
        result = extractor.extract_table(perception, has_header=True)

        assert len(result) == 1
        assert result[0]["Name"] == "Alice"
        assert result[0]["Age"] == ""
        assert result[0]["City"] == ""
