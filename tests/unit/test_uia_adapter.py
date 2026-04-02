"""
tests/unit/test_uia_adapter.py
Unit tests for nexus/source/uia/adapter.py using mock COM objects.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from nexus.core.settings import NexusSettings  # noqa: E402
from nexus.core.types import Rect
from nexus.source.uia.adapter import (
    UIAAdapter,
    UIAElement,
    _UIA_ExpandCollapsePatternId,
    _UIA_InvokePatternId,
    _UIA_SelectionItemPatternId,
    _UIA_ValuePatternId,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(timeout_ms: int = 5000) -> NexusSettings:
    return NexusSettings.model_validate(
        {"source": {"uia_timeout_ms": timeout_ms}}
    )


def _make_element(*, raw: Any = None, name: str = "btn") -> UIAElement:
    """Return a minimal UIAElement, optionally with a raw COM object."""
    return UIAElement(
        automation_id="id1",
        name=name,
        control_type=50000,
        bounding_rect=Rect(0, 0, 100, 30),
        is_enabled=True,
        is_visible=True,
        value=None,
        _raw=raw,
    )


def _make_raw_element(
    *,
    auto_id: str = "aid",
    name: str = "elem",
    control_type: int = 50000,
    is_enabled: bool = True,
    is_offscreen: bool = False,
    value: str | None = None,
    rect: Any = None,
    supports_invoke: bool = False,
    supports_value: bool = False,
    supports_selection: bool = False,
    supports_expand: bool = False,
) -> MagicMock:
    """Return a mock COM element that UIAAdapter._raw_to_element can consume."""
    prop_map = {
        30011: auto_id,
        30005: name,
        30003: control_type,
        30010: is_enabled,
        30022: is_offscreen,
        30045: value,
        30001: rect,
        30020: supports_invoke,
        30043: supports_value,
        30030: supports_selection,
        30018: supports_expand,
    }
    raw = MagicMock()
    raw.GetCurrentPropertyValue.side_effect = lambda pid: prop_map.get(pid)
    return raw


def _make_mock_automation() -> MagicMock:
    """Return a mock IUIAutomation COM object."""
    auto = MagicMock()
    auto.CreateTrueCondition.return_value = MagicMock()
    auto.CreatePropertyCondition.return_value = MagicMock()
    return auto


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_returns_true_when_factory_succeeds(self):
        settings = _make_settings()
        factory = MagicMock(return_value=MagicMock())
        adapter = UIAAdapter(settings, _automation_factory=factory)
        assert adapter.is_available() is True

    def test_returns_false_when_factory_raises(self):
        settings = _make_settings()
        factory = MagicMock(side_effect=OSError("COM unavailable"))
        adapter = UIAAdapter(settings, _automation_factory=factory)
        assert adapter.is_available() is False

    def test_factory_called_once_on_repeated_calls(self):
        settings = _make_settings()
        factory = MagicMock(return_value=MagicMock())
        adapter = UIAAdapter(settings, _automation_factory=factory)
        adapter.is_available()
        adapter.is_available()
        factory.assert_called_once()


# ---------------------------------------------------------------------------
# Action methods — _raw is None → False immediately
# ---------------------------------------------------------------------------


class TestActionMethodsNoRaw:
    def _adapter(self) -> UIAAdapter:
        settings = _make_settings()
        return UIAAdapter(settings, _automation_factory=MagicMock())

    def test_invoke_no_raw(self):
        elem = _make_element(raw=None)
        assert self._adapter().invoke(elem) is False

    def test_set_value_no_raw(self):
        elem = _make_element(raw=None)
        assert self._adapter().set_value(elem, "x") is False

    def test_select_no_raw(self):
        elem = _make_element(raw=None)
        assert self._adapter().select(elem) is False

    def test_expand_no_raw(self):
        elem = _make_element(raw=None)
        assert self._adapter().expand(elem) is False

    def test_get_value_no_raw(self):
        elem = _make_element(raw=None)
        assert self._adapter().get_value(elem) is None


# ---------------------------------------------------------------------------
# invoke()
# ---------------------------------------------------------------------------


class TestInvoke:
    def _adapter_with_raw(self):
        settings = _make_settings()
        auto = _make_mock_automation()
        adapter = UIAAdapter(settings, _automation_factory=lambda: auto)
        raw = MagicMock()
        pattern = MagicMock()
        raw.GetCurrentPattern.return_value = pattern
        elem = _make_element(raw=raw)
        return adapter, elem, raw, pattern

    def test_invoke_calls_get_current_pattern_with_correct_id(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        adapter.invoke(elem)
        raw.GetCurrentPattern.assert_called_once_with(_UIA_InvokePatternId)

    def test_invoke_calls_invoke_on_pattern(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        adapter.invoke(elem)
        pattern.Invoke.assert_called_once()

    def test_invoke_returns_true_on_success(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        assert adapter.invoke(elem) is True

    def test_invoke_returns_false_on_com_exception(self):
        settings = _make_settings()
        adapter = UIAAdapter(settings, _automation_factory=MagicMock())
        raw = MagicMock()
        raw.GetCurrentPattern.side_effect = Exception("COM error")
        elem = _make_element(raw=raw)
        assert adapter.invoke(elem) is False

    def test_invoke_returns_false_when_pattern_invoke_raises(self):
        settings = _make_settings()
        adapter = UIAAdapter(settings, _automation_factory=MagicMock())
        raw = MagicMock()
        pattern = MagicMock()
        pattern.Invoke.side_effect = Exception("Invoke failed")
        raw.GetCurrentPattern.return_value = pattern
        elem = _make_element(raw=raw)
        assert adapter.invoke(elem) is False


# ---------------------------------------------------------------------------
# set_value()
# ---------------------------------------------------------------------------


class TestSetValue:
    def _adapter_with_raw(self):
        settings = _make_settings()
        adapter = UIAAdapter(settings, _automation_factory=MagicMock())
        raw = MagicMock()
        pattern = MagicMock()
        raw.GetCurrentPattern.return_value = pattern
        elem = _make_element(raw=raw)
        return adapter, elem, raw, pattern

    def test_set_value_calls_get_current_pattern_with_correct_id(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        adapter.set_value(elem, "hello")
        raw.GetCurrentPattern.assert_called_once_with(_UIA_ValuePatternId)

    def test_set_value_calls_set_value_on_pattern(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        adapter.set_value(elem, "hello")
        pattern.SetValue.assert_called_once_with("hello")

    def test_set_value_returns_true_on_success(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        assert adapter.set_value(elem, "x") is True

    def test_set_value_returns_false_on_com_exception(self):
        settings = _make_settings()
        adapter = UIAAdapter(settings, _automation_factory=MagicMock())
        raw = MagicMock()
        raw.GetCurrentPattern.side_effect = Exception("COM error")
        elem = _make_element(raw=raw)
        assert adapter.set_value(elem, "x") is False

    def test_set_value_passes_value_string_correctly(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        adapter.set_value(elem, "test_value_123")
        pattern.SetValue.assert_called_once_with("test_value_123")


# ---------------------------------------------------------------------------
# select()
# ---------------------------------------------------------------------------


class TestSelect:
    def _adapter_with_raw(self):
        settings = _make_settings()
        adapter = UIAAdapter(settings, _automation_factory=MagicMock())
        raw = MagicMock()
        pattern = MagicMock()
        raw.GetCurrentPattern.return_value = pattern
        elem = _make_element(raw=raw)
        return adapter, elem, raw, pattern

    def test_select_calls_get_current_pattern_with_correct_id(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        adapter.select(elem)
        raw.GetCurrentPattern.assert_called_once_with(_UIA_SelectionItemPatternId)

    def test_select_calls_select_on_pattern(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        adapter.select(elem)
        pattern.Select.assert_called_once()

    def test_select_returns_true_on_success(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        assert adapter.select(elem) is True

    def test_select_returns_false_on_com_exception(self):
        settings = _make_settings()
        adapter = UIAAdapter(settings, _automation_factory=MagicMock())
        raw = MagicMock()
        raw.GetCurrentPattern.side_effect = Exception("COM error")
        elem = _make_element(raw=raw)
        assert adapter.select(elem) is False


# ---------------------------------------------------------------------------
# expand()
# ---------------------------------------------------------------------------


class TestExpand:
    def _adapter_with_raw(self):
        settings = _make_settings()
        adapter = UIAAdapter(settings, _automation_factory=MagicMock())
        raw = MagicMock()
        pattern = MagicMock()
        raw.GetCurrentPattern.return_value = pattern
        elem = _make_element(raw=raw)
        return adapter, elem, raw, pattern

    def test_expand_calls_get_current_pattern_with_correct_id(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        adapter.expand(elem)
        raw.GetCurrentPattern.assert_called_once_with(_UIA_ExpandCollapsePatternId)

    def test_expand_calls_expand_on_pattern(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        adapter.expand(elem)
        pattern.Expand.assert_called_once()

    def test_expand_returns_true_on_success(self):
        adapter, elem, raw, pattern = self._adapter_with_raw()
        assert adapter.expand(elem) is True

    def test_expand_returns_false_on_com_exception(self):
        settings = _make_settings()
        adapter = UIAAdapter(settings, _automation_factory=MagicMock())
        raw = MagicMock()
        raw.GetCurrentPattern.side_effect = Exception("COM error")
        elem = _make_element(raw=raw)
        assert adapter.expand(elem) is False


# ---------------------------------------------------------------------------
# get_value()
# ---------------------------------------------------------------------------


class TestGetValue:
    def test_get_value_returns_string_on_success(self):
        settings = _make_settings()
        adapter = UIAAdapter(settings, _automation_factory=MagicMock())
        raw = MagicMock()
        pattern = MagicMock()
        pattern.CurrentValue = "some_text"
        raw.GetCurrentPattern.return_value = pattern
        elem = _make_element(raw=raw)
        result = adapter.get_value(elem)
        assert result == "some_text"

    def test_get_value_returns_none_on_exception(self):
        settings = _make_settings()
        adapter = UIAAdapter(settings, _automation_factory=MagicMock())
        raw = MagicMock()
        raw.GetCurrentPattern.side_effect = Exception("no pattern")
        elem = _make_element(raw=raw)
        assert adapter.get_value(elem) is None

    def test_get_value_uses_value_pattern_id(self):
        settings = _make_settings()
        adapter = UIAAdapter(settings, _automation_factory=MagicMock())
        raw = MagicMock()
        pattern = MagicMock()
        pattern.CurrentValue = "v"
        raw.GetCurrentPattern.return_value = pattern
        elem = _make_element(raw=raw)
        adapter.get_value(elem)
        raw.GetCurrentPattern.assert_called_once_with(_UIA_ValuePatternId)


# ---------------------------------------------------------------------------
# get_elements() — timeout
# ---------------------------------------------------------------------------


class TestGetElementsTimeout:
    def test_returns_none_when_factory_fails(self):
        settings = _make_settings(timeout_ms=1000)
        factory = MagicMock(side_effect=OSError("no COM"))
        adapter = UIAAdapter(settings, _automation_factory=factory)
        result = adapter.get_elements(12345)
        assert result is None

    def test_returns_none_on_timeout(self):
        import time

        settings = _make_settings(timeout_ms=50)

        def slow_factory():
            return _make_mock_automation()

        auto = _make_mock_automation()

        def slow_find_all(*_args):
            time.sleep(1.0)
            return MagicMock(Length=0)

        auto.ElementFromHandle.return_value.FindAll = slow_find_all
        adapter = UIAAdapter(settings, _automation_factory=lambda: auto)
        result = adapter.get_elements(12345)
        assert result is None

    def test_returns_list_on_success(self):
        settings = _make_settings(timeout_ms=2000)
        auto = _make_mock_automation()

        raw_elem = _make_raw_element(name="btn1")
        collection = MagicMock()
        collection.Length = 1
        collection.GetElement.return_value = raw_elem
        root = MagicMock()
        root.FindAll.return_value = collection
        auto.ElementFromHandle.return_value = root

        adapter = UIAAdapter(settings, _automation_factory=lambda: auto)
        result = adapter.get_elements(99999)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].name == "btn1"

    def test_respects_timeout_ms_override(self):
        settings = _make_settings(timeout_ms=5000)
        auto = _make_mock_automation()
        collection = MagicMock()
        collection.Length = 0
        root = MagicMock()
        root.FindAll.return_value = collection
        auto.ElementFromHandle.return_value = root

        adapter = UIAAdapter(settings, _automation_factory=lambda: auto)
        result = adapter.get_elements(12345, timeout_ms=2000)
        assert result == []


# ---------------------------------------------------------------------------
# find_by_name / find_by_automation_id
# ---------------------------------------------------------------------------


class TestFindMethods:
    def test_find_by_name_returns_element(self):
        settings = _make_settings()
        auto = _make_mock_automation()

        raw_elem = _make_raw_element(name="MyButton")
        auto.GetRootElement.return_value.FindFirst.return_value = raw_elem

        adapter = UIAAdapter(settings, _automation_factory=lambda: auto)
        result = adapter.find_by_name("MyButton")
        assert result is not None
        assert result.name == "MyButton"

    def test_find_by_name_returns_none_when_not_found(self):
        settings = _make_settings()
        auto = _make_mock_automation()
        auto.GetRootElement.return_value.FindFirst.return_value = None

        adapter = UIAAdapter(settings, _automation_factory=lambda: auto)
        result = adapter.find_by_name("ghost")
        assert result is None

    def test_find_by_name_returns_none_on_exception(self):
        settings = _make_settings()
        auto = _make_mock_automation()
        auto.GetRootElement.side_effect = Exception("COM crash")

        adapter = UIAAdapter(settings, _automation_factory=lambda: auto)
        result = adapter.find_by_name("anything")
        assert result is None

    def test_find_by_automation_id_returns_element(self):
        settings = _make_settings()
        auto = _make_mock_automation()

        raw_elem = _make_raw_element(auto_id="btn_ok", name="OK")
        auto.GetRootElement.return_value.FindFirst.return_value = raw_elem

        adapter = UIAAdapter(settings, _automation_factory=lambda: auto)
        result = adapter.find_by_automation_id("btn_ok")
        assert result is not None
        assert result.automation_id == "btn_ok"

    def test_find_by_automation_id_returns_none_when_not_found(self):
        settings = _make_settings()
        auto = _make_mock_automation()
        auto.GetRootElement.return_value.FindFirst.return_value = None

        adapter = UIAAdapter(settings, _automation_factory=lambda: auto)
        result = adapter.find_by_automation_id("nonexistent")
        assert result is None

    def test_find_by_automation_id_returns_none_on_exception(self):
        settings = _make_settings()
        auto = _make_mock_automation()
        auto.GetRootElement.side_effect = Exception("COM crash")

        adapter = UIAAdapter(settings, _automation_factory=lambda: auto)
        result = adapter.find_by_automation_id("anything")
        assert result is None


# ---------------------------------------------------------------------------
# _parse_rect
# ---------------------------------------------------------------------------


class TestParseRect:
    def test_parse_rect_from_struct_with_left_top_right_bottom(self):
        rect_struct = MagicMock()
        rect_struct.left = 10
        rect_struct.top = 20
        rect_struct.right = 110
        rect_struct.bottom = 70
        result = UIAAdapter._parse_rect(rect_struct)
        assert result == Rect(10, 20, 100, 50)

    def test_parse_rect_from_sequence(self):
        result = UIAAdapter._parse_rect([5, 10, 200, 80])
        assert result == Rect(5, 10, 200, 80)

    def test_parse_rect_returns_none_for_none(self):
        assert UIAAdapter._parse_rect(None) is None

    def test_parse_rect_clamps_negative_width(self):
        rect_struct = MagicMock()
        rect_struct.left = 100
        rect_struct.top = 50
        rect_struct.right = 80  # right < left → width should be 0
        rect_struct.bottom = 100
        result = UIAAdapter._parse_rect(rect_struct)
        assert result is not None
        assert result.width == 0

    def test_parse_rect_returns_none_on_bad_data(self):
        result = UIAAdapter._parse_rect("not_a_rect")
        assert result is None


# ---------------------------------------------------------------------------
# _raw_to_element — field mapping
# ---------------------------------------------------------------------------


class TestRawToElement:
    def _adapter(self) -> UIAAdapter:
        settings = _make_settings()
        return UIAAdapter(settings, _automation_factory=MagicMock())

    def test_visible_when_not_offscreen(self):
        adapter = self._adapter()
        raw = _make_raw_element(is_offscreen=False)
        elem = adapter._raw_to_element(raw)
        assert elem is not None
        assert elem.is_visible is True

    def test_not_visible_when_offscreen(self):
        adapter = self._adapter()
        raw = _make_raw_element(is_offscreen=True)
        elem = adapter._raw_to_element(raw)
        assert elem is not None
        assert elem.is_visible is False

    def test_supports_flags_mapped_correctly(self):
        adapter = self._adapter()
        raw = _make_raw_element(
            supports_invoke=True,
            supports_value=False,
            supports_selection=True,
            supports_expand=False,
        )
        elem = adapter._raw_to_element(raw)
        assert elem is not None
        assert elem.supports_invoke is True
        assert elem.supports_value is False
        assert elem.supports_selection is True
        assert elem.supports_expand_collapse is False

    def test_value_none_when_empty_string(self):
        adapter = self._adapter()
        raw = _make_raw_element(value="")
        elem = adapter._raw_to_element(raw)
        assert elem is not None
        assert elem.value is None

    def test_value_set_when_non_empty(self):
        adapter = self._adapter()
        raw = _make_raw_element(value="hello")
        elem = adapter._raw_to_element(raw)
        assert elem is not None
        assert elem.value == "hello"

    def test_returns_none_on_exception(self):
        adapter = self._adapter()
        raw = MagicMock()
        raw.GetCurrentPropertyValue.side_effect = Exception("crash")
        elem = adapter._raw_to_element(raw)
        assert elem is None

    def test_raw_reference_stored_on_element(self):
        adapter = self._adapter()
        raw = _make_raw_element()
        elem = adapter._raw_to_element(raw)
        assert elem is not None
        assert elem._raw is raw
