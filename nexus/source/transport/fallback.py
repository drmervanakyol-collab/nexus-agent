"""
nexus/source/transport/fallback.py
Mouse and keyboard transports — low-level OS input fallbacks.

These are used when a native UIA/DOM action is unavailable or fails.
Both transports are async and wrap synchronous OS API calls via
``asyncio.to_thread`` so they don't block the event loop.

Pluggable click / type functions
---------------------------------
Both classes accept optional ``_click_fn`` / ``_type_fn`` parameters so
that unit tests can inject synchronous callables without touching the OS
input stack.  Production code uses the real pyautogui / pywin32 backends
(lazy-imported so the module loads cleanly on machines without those libs).
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable

from nexus.infra.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default OS-level implementations (lazy imports)
# ---------------------------------------------------------------------------


def _focus_window_at(x: int, y: int) -> None:
    """
    Bring the window at screen coordinates (x, y) to the foreground.

    Uses Win32 WindowFromPoint + SetForegroundWindow.  Silently ignored
    when the API is unavailable (non-Windows or missing ctypes).
    """
    try:
        import ctypes  # noqa: PLC0415
        import ctypes.wintypes  # noqa: PLC0415
        import time  # noqa: PLC0415

        pt = ctypes.wintypes.POINT(x, y)
        hwnd = ctypes.windll.user32.WindowFromPoint(pt)
        if hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            time.sleep(0.05)  # allow window manager to process focus change
    except Exception:  # noqa: BLE001
        pass


def _default_mouse_click(x: int, y: int) -> None:
    """
    Bring the target window to foreground, then click at (x, y).

    Focuses the window under the cursor before dispatching input so that
    SendInput / pyautogui events land on the correct target even when another
    application currently holds focus.

    Tries pyautogui first; falls back to pywin32.
    Raises RuntimeError if neither is available.
    """
    _focus_window_at(x, y)

    try:
        import pyautogui  # noqa: PLC0415

        pyautogui.click(x, y)
    except ImportError:
        try:
            import win32api  # noqa: PLC0415
            import win32con  # noqa: PLC0415

            win32api.SetCursorPos((x, y))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
        except ImportError as exc:
            raise RuntimeError(
                "MouseTransport requires pyautogui or pywin32"
            ) from exc


def _default_keyboard_type(text: str) -> None:
    """
    Type *text* using the OS keyboard.

    Tries pyautogui first; raises RuntimeError if unavailable.
    """
    try:
        import pyautogui  # noqa: PLC0415

        pyautogui.typewrite(text, interval=0.02)
    except ImportError as exc:
        raise RuntimeError(
            "KeyboardTransport requires pyautogui"
        ) from exc


def _default_keyboard_press(key: str) -> None:
    """
    Press a single key or hotkey combination using the OS keyboard.

    Examples: "win", "enter", "ctrl+c", "alt+f4"
    """
    try:
        import pyautogui  # noqa: PLC0415

        if "+" in key:
            parts = [p.strip() for p in key.split("+")]
            pyautogui.hotkey(*parts)
        else:
            pyautogui.press(key)
    except ImportError as exc:
        raise RuntimeError(
            "KeyboardTransport requires pyautogui"
        ) from exc


# ---------------------------------------------------------------------------
# MouseTransport
# ---------------------------------------------------------------------------


class MouseTransport:
    """
    Delivers click actions via the OS mouse API.

    Parameters
    ----------
    _click_fn:
        Sync callable ``(x: int, y: int) -> None``.  Defaults to the
        real pyautogui / win32api backend.  Inject a mock in tests.
    """

    def __init__(
        self,
        *,
        _click_fn: Callable[[int, int], None] | None = None,
    ) -> None:
        self._click_fn: Callable[[int, int], None] = (
            _click_fn or _default_mouse_click
        )

    async def click(self, x: int, y: int) -> bool:
        """
        Click at screen coordinates (x, y).

        Returns True on success, False on any OS-level error.
        """
        try:
            await asyncio.to_thread(self._click_fn, x, y)
            _log.debug("mouse_click_ok", x=x, y=y)
            return True
        except Exception as exc:
            _log.debug("mouse_click_failed", x=x, y=y, error=str(exc))
            return False


# ---------------------------------------------------------------------------
# KeyboardTransport
# ---------------------------------------------------------------------------


class KeyboardTransport:
    """
    Delivers text input via the OS keyboard API.

    Parameters
    ----------
    _type_fn:
        Sync callable ``(text: str) -> None``.  Defaults to the real
        pyautogui backend.  Inject a mock in tests.
    """

    def __init__(
        self,
        *,
        _type_fn: Callable[[str], None] | None = None,
        _press_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._type_fn: Callable[[str], None] = (
            _type_fn or _default_keyboard_type
        )
        self._press_fn: Callable[[str], None] = (
            _press_fn or _default_keyboard_press
        )

    async def type_text(self, text: str) -> bool:
        """
        Type *text* using OS keyboard events.

        Returns True on success, False on any OS-level error.
        """
        try:
            await asyncio.to_thread(self._type_fn, text)
            _log.debug("keyboard_type_ok", length=len(text))
            return True
        except Exception as exc:
            _log.debug("keyboard_type_failed", error=str(exc))
            return False

    async def press_key(self, key: str) -> bool:
        """
        Press a single key or hotkey combination.

        Examples: "win", "enter", "ctrl+c", "alt+f4"
        Returns True on success, False on any OS-level error.
        """
        try:
            await asyncio.to_thread(self._press_fn, key)
            _log.debug("keyboard_press_ok", key=key)
            return True
        except Exception as exc:
            _log.debug("keyboard_press_failed", key=key, error=str(exc))
            return False
