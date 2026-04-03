"""
nexus/source/uia/adapter.py
Windows UI Automation adapter for Nexus Agent.

Architecture
------------
UIAAdapter wraps the IUIAutomation COM object behind a thin Python layer.
A pluggable ``_automation_factory`` callable is accepted at construction so
that unit tests can inject a fully-mocked COM object without touching the
real Windows UIA stack.

Action methods (invoke / set_value / select / expand) follow the rule:
  - Return True on success.
  - Return False on any COM exception or pattern unavailability.
  - Never raise.

Timeout
-------
``settings.source.uia_timeout_ms`` is stored on the instance.  When the
real COM stack is used, the value is applied via
``IUIAutomation.RawViewWalker`` timeout hints.  For action calls wrapped
in ``_run_with_timeout()``, the same value bounds a background thread.

Pattern IDs  (UIAutomationClient constants)
-------------------------------------------
UIA_InvokePatternId          = 10000
UIA_ValuePatternId           = 10002
UIA_SelectionItemPatternId   = 10010
UIA_ExpandCollapsePatternId  = 10005
"""
from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nexus.core.settings import NexusSettings
from nexus.core.types import Rect
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# UIA constant IDs
# ---------------------------------------------------------------------------

_UIA_InvokePatternId: int = 10000
_UIA_ValuePatternId: int = 10002
_UIA_SelectionItemPatternId: int = 10010
_UIA_ExpandCollapsePatternId: int = 10005

_UIA_AutomationIdPropertyId: int = 30011
_UIA_NamePropertyId: int = 30005
_UIA_ControlTypePropertyId: int = 30003
_UIA_BoundingRectanglePropertyId: int = 30001
_UIA_IsEnabledPropertyId: int = 30010
_UIA_IsOffscreenPropertyId: int = 30022
_UIA_ValueValuePropertyId: int = 30045
_UIA_IsInvokePatternAvailablePropertyId: int = 30020
_UIA_IsValuePatternAvailablePropertyId: int = 30043
_UIA_IsSelectionItemPatternAvailablePropertyId: int = 30030
_UIA_IsExpandCollapsePatternAvailablePropertyId: int = 30018

_TreeScope_Subtree: int = 0x7

# CLSID / ProgID for the CUIAutomation COM object
_CLSID_CUIAutomation = "{FF48DBA4-60EF-4201-AA87-54103EEF594E}"


# ---------------------------------------------------------------------------
# UIAElement
# ---------------------------------------------------------------------------


@dataclass
class UIAElement:
    """
    A snapshot of a Windows UI Automation element.

    All fields are plain Python types — no live COM references are held.
    This makes elements safe to pass across threads and to serialize.
    """

    automation_id: str
    name: str
    control_type: int
    bounding_rect: Rect | None
    is_enabled: bool
    is_visible: bool         # True when element is NOT off-screen
    value: str | None
    children: list[UIAElement] = field(default_factory=list)
    supports_invoke: bool = False
    supports_value: bool = False
    supports_selection: bool = False
    supports_expand_collapse: bool = False

    # Keep a reference to the raw COM element so action methods can use it.
    # Not serialized / compared.
    _raw: Any = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# COM factory helper
# ---------------------------------------------------------------------------


def _default_automation_factory() -> Any:
    """Create the real IUIAutomation COM object via comtypes."""
    import comtypes.client  # noqa: PLC0415

    return comtypes.client.CreateObject(
        _CLSID_CUIAutomation,
        interface=None,  # returns IDispatch; cast later
    )


# ---------------------------------------------------------------------------
# UIAAdapter
# ---------------------------------------------------------------------------


class UIAAdapter:
    """
    Thin wrapper over IUIAutomation for element discovery and action dispatch.

    Parameters
    ----------
    settings:
        NexusSettings instance — uses ``source.uia_timeout_ms``.
    _automation_factory:
        Optional callable that returns an IUIAutomation-compatible object.
        When ``None`` the real ``comtypes`` COM object is created on first use.
        Pass a mock here in unit tests.
    """

    def __init__(
        self,
        settings: NexusSettings,
        *,
        _automation_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._timeout_ms: int = settings.source.uia_timeout_ms
        self._factory: Callable[[], Any] = (
            _automation_factory or _default_automation_factory
        )
        self._automation: Any | None = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the UIA COM stack is reachable."""
        return self._get_automation() is not None

    # ------------------------------------------------------------------
    # Element discovery
    # ------------------------------------------------------------------

    def get_elements(
        self,
        window_handle: int,
        timeout_ms: int | None = None,
    ) -> list[UIAElement] | None:
        """
        Return all UI Automation elements under the given window handle.

        Parameters
        ----------
        window_handle:
            HWND of the target window.
        timeout_ms:
            Override instance timeout for this call.

        Returns
        -------
        list[UIAElement] on success, ``None`` on failure.
        """
        auto = self._get_automation()
        if auto is None:
            return None
        timeout = timeout_ms if timeout_ms is not None else self._timeout_ms

        def _work() -> list[UIAElement]:
            root_elem = auto.ElementFromHandle(window_handle)
            condition = auto.CreateTrueCondition()
            raw_collection = root_elem.FindAll(_TreeScope_Subtree, condition)
            elements: list[UIAElement] = []
            length = raw_collection.Length
            for i in range(length):
                raw = raw_collection.GetElement(i)
                elem = self._raw_to_element(raw)
                if elem is not None:
                    elements.append(elem)
            return elements

        return self._run_with_timeout(_work, timeout)

    def find_by_name(self, name: str) -> UIAElement | None:
        """Return the first element whose ``Name`` property matches *name*."""
        auto = self._get_automation()
        if auto is None:
            return None
        try:
            condition = auto.CreatePropertyCondition(
                _UIA_NamePropertyId, name
            )
            root = auto.GetRootElement()
            raw = root.FindFirst(_TreeScope_Subtree, condition)
            if raw is None:
                return None
            return self._raw_to_element(raw)
        except Exception as exc:
            _log.debug("uia_find_by_name_failed", name=name, error=str(exc))
            return None

    def find_by_automation_id(self, automation_id: str) -> UIAElement | None:
        """Return the first element whose ``AutomationId`` matches."""
        auto = self._get_automation()
        if auto is None:
            return None
        try:
            condition = auto.CreatePropertyCondition(
                _UIA_AutomationIdPropertyId, automation_id
            )
            root = auto.GetRootElement()
            raw = root.FindFirst(_TreeScope_Subtree, condition)
            if raw is None:
                return None
            return self._raw_to_element(raw)
        except Exception as exc:
            _log.debug(
                "uia_find_by_automation_id_failed",
                automation_id=automation_id,
                error=str(exc),
            )
            return None

    def get_value(self, element: UIAElement) -> str | None:
        """Return the current Value of *element*, or None on failure."""
        if element._raw is None:
            return None
        try:
            pattern = element._raw.GetCurrentPattern(_UIA_ValuePatternId)
            return str(pattern.CurrentValue)
        except Exception as exc:
            _log.debug("uia_get_value_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Action methods
    # ------------------------------------------------------------------

    def invoke(self, element: UIAElement) -> bool:
        """
        Call ``InvokePattern.Invoke()`` on *element*.

        Returns True on success, False if the pattern is unavailable or if
        any COM exception occurs.
        """
        if element._raw is None:
            return False
        try:
            pattern = element._raw.GetCurrentPattern(_UIA_InvokePatternId)
            pattern.Invoke()
            _log.debug("uia_invoke_ok", name=element.name)
            return True
        except Exception as exc:
            _log.debug("uia_invoke_failed", name=element.name, error=str(exc))
            return False

    def set_value(self, element: UIAElement, value: str) -> bool:
        """
        Call ``ValuePattern.SetValue(value)`` on *element*.

        Returns True on success, False on any COM exception.
        """
        if element._raw is None:
            return False
        try:
            pattern = element._raw.GetCurrentPattern(_UIA_ValuePatternId)
            pattern.SetValue(value)
            _log.debug("uia_set_value_ok", name=element.name, value=value)
            return True
        except Exception as exc:
            _log.debug("uia_set_value_failed", name=element.name, error=str(exc))
            return False

    def select(self, element: UIAElement) -> bool:
        """
        Call ``SelectionItemPattern.Select()`` on *element*.

        Returns True on success, False on any COM exception.
        """
        if element._raw is None:
            return False
        try:
            pattern = element._raw.GetCurrentPattern(
                _UIA_SelectionItemPatternId
            )
            pattern.Select()
            _log.debug("uia_select_ok", name=element.name)
            return True
        except Exception as exc:
            _log.debug("uia_select_failed", name=element.name, error=str(exc))
            return False

    def expand(self, element: UIAElement) -> bool:
        """
        Call ``ExpandCollapsePattern.Expand()`` on *element*.

        Returns True on success, False on any COM exception.
        """
        if element._raw is None:
            return False
        try:
            pattern = element._raw.GetCurrentPattern(
                _UIA_ExpandCollapsePatternId
            )
            pattern.Expand()
            _log.debug("uia_expand_ok", name=element.name)
            return True
        except Exception as exc:
            _log.debug("uia_expand_failed", name=element.name, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_automation(self) -> Any | None:
        if self._automation is not None:
            return self._automation
        try:
            self._automation = self._factory()
        except Exception as exc:
            _log.debug("uia_automation_init_failed", error=str(exc))
            return None
        return self._automation

    def _raw_to_element(self, raw: Any) -> UIAElement | None:
        """Convert a raw COM element to a UIAElement dataclass."""
        try:
            def _prop(pid: int) -> Any:
                return raw.GetCurrentPropertyValue(pid)

            auto_id = str(_prop(_UIA_AutomationIdPropertyId) or "")
            name = str(_prop(_UIA_NamePropertyId) or "")
            control_type = int(_prop(_UIA_ControlTypePropertyId) or 0)
            is_enabled = bool(_prop(_UIA_IsEnabledPropertyId))
            is_offscreen = bool(_prop(_UIA_IsOffscreenPropertyId))
            raw_value = _prop(_UIA_ValueValuePropertyId)
            value = str(raw_value) if raw_value not in (None, "") else None

            rect_raw = _prop(_UIA_BoundingRectanglePropertyId)
            bounding_rect = self._parse_rect(rect_raw)

            supports_invoke = bool(
                _prop(_UIA_IsInvokePatternAvailablePropertyId)
            )
            supports_value = bool(
                _prop(_UIA_IsValuePatternAvailablePropertyId)
            )
            supports_selection = bool(
                _prop(_UIA_IsSelectionItemPatternAvailablePropertyId)
            )
            supports_expand = bool(
                _prop(_UIA_IsExpandCollapsePatternAvailablePropertyId)
            )

            return UIAElement(
                automation_id=auto_id,
                name=name,
                control_type=control_type,
                bounding_rect=bounding_rect,
                is_enabled=is_enabled,
                is_visible=not is_offscreen,
                value=value,
                children=[],
                supports_invoke=supports_invoke,
                supports_value=supports_value,
                supports_selection=supports_selection,
                supports_expand_collapse=supports_expand,
                _raw=raw,
            )
        except Exception as exc:
            _log.debug("uia_raw_to_element_failed", error=str(exc))
            return None

    @staticmethod
    def _parse_rect(rect_raw: Any) -> Rect | None:
        """
        Convert a raw UIA bounding-rectangle value to a Rect.

        UIA returns a ``RECT`` struct or a sequence ``(left, top, width,
        height)`` depending on the COM binding.  We handle both.
        """
        if rect_raw is None:
            return None
        try:
            # comtypes / win32 RECT struct (left, top, right, bottom)
            if hasattr(rect_raw, "left"):
                left = int(rect_raw.left)
                top = int(rect_raw.top)
                w = max(0, int(rect_raw.right) - left)
                h = max(0, int(rect_raw.bottom) - top)
                return Rect(left, top, w, h)
            # Sequence: some bindings return (left, top, width, height)
            seq = list(rect_raw)
            if len(seq) == 4:
                return Rect(int(seq[0]), int(seq[1]), int(seq[2]), int(seq[3]))
        except Exception:
            pass
        return None

    def _run_with_timeout(
        self,
        fn: Callable[[], Any],
        timeout_ms: int,
    ) -> Any | None:
        """
        Execute *fn* in a thread pool and return its result.

        Returns None if the call exceeds *timeout_ms* or raises.
        """
        timeout_s = timeout_ms / 1000.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(fn)
            try:
                return future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                _log.warning(
                    "uia_timeout",
                    timeout_ms=timeout_ms,
                )
                return None
            except Exception as exc:
                _log.debug("uia_run_failed", error=str(exc))
                return None
