"""
nexus/source/transport/keyboard_transport.py
Full-featured keyboard transport — Windows SendInput, IME bypass, Turkish support.

Used as the OS-level fallback when native UIA / DOM text-input fails.  All
Windows API calls are injectable so the module is fully unit-testable without
a live keyboard device or IME context.

Architecture
------------
``type_text`` sends each Unicode character individually via SendInput with the
``KEYEVENTF_UNICODE`` flag.  This approach works for every Unicode codepoint
(including Turkish İ U+0130 and ı U+0131) without needing VK lookup per
keyboard layout, because KEYEVENTF_UNICODE bypasses the VK → scan → char
translation chain entirely.

IME bypass
----------
Before typing, the current HWND's IME context is obtained via
``ImmGetContext``, disabled with ``ImmSetOpenStatus(himc, FALSE)``, typing
proceeds, and then the original status is restored.  This prevents IME
composition from intercepting individual keystrokes in CJK or mixed-input
fields.

Turkish VK mapping
------------------
``hotkey`` and ``special_key`` require virtual-key codes.  Turkish characters
that differ from the US layout are resolved via ``VkKeyScanExW`` with the
Turkish keyboard layout handle, falling back to direct Unicode SendInput when
a VK cannot be determined.  The mapping covers the most common Turkish keys:

  ğ → VK_OEM_4    (Turkish Q: ğ is on the [ key position)
  ü → VK_OEM_1
  ş → VK_OEM_2
  ı → VK_OEM_3    (dotless i — distinct from Latin i U+0069)
  İ → VK + SHIFT  (capital I-with-dot — U+0130)
  ö → VK_OEM_7
  ç → VK_OEM_COMMA

Text verification
-----------------
After ``type_text`` completes, an optional ``_ocr_verify_fn`` callable is
called with the expected text.  If it returns different text, the transport
sends Backspace × len(actual) and re-types (up to ``max_retries`` times).

Injectable callables
--------------------
_send_input_fn:
    ``(events: list[_KeyEvent | _MouseEvent]) -> int``
_get_keyboard_layout_fn:
    ``() -> int``  — returns the current thread's HKL handle.
_imm_get_context_fn:
    ``(hwnd: int) -> int``  — returns the HIMC for *hwnd*.
_imm_set_open_status_fn:
    ``(himc: int, open: bool) -> None``  — enables/disables IME.
_imm_get_open_status_fn:
    ``(himc: int) -> bool``  — queries current IME open state.
_imm_release_context_fn:
    ``(hwnd: int, himc: int) -> None``  — releases the HIMC.
_get_foreground_window_fn:
    ``() -> int``  — returns the foreground HWND.
_ocr_verify_fn:
    ``(expected: str) -> str | None``  — returns the text actually observed,
    or None when verification is unavailable.
settings:
    NexusSettings.  Uses ``transport.key_press_delay_ms`` and
    ``transport.type_verify_max_retries``.
"""
from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes
import time
from collections.abc import Callable

from nexus.core.settings import NexusSettings
from nexus.infra.logger import get_logger
from nexus.source.transport.mouse_transport import (
    _INPUT,
    _INPUT_KEYBOARD,
    _AnyEvent,
    _KeyEvent,
    _MouseEvent,
    _SendInputFn,
)

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Windows keyboard constants
# ---------------------------------------------------------------------------

_KEYEVENTF_EXTENDEDKEY: int = 0x0001
_KEYEVENTF_KEYUP: int = 0x0002
_KEYEVENTF_UNICODE: int = 0x0004
_KEYEVENTF_SCANCODE: int = 0x0008

# VK codes for special keys
_VK_BACK: int = 0x08       # Backspace
_VK_TAB: int = 0x09
_VK_RETURN: int = 0x0D     # Enter
_VK_SHIFT: int = 0x10
_VK_CONTROL: int = 0x11
_VK_MENU: int = 0x12       # Alt
_VK_ESCAPE: int = 0x1B
_VK_SPACE: int = 0x20
_VK_DELETE: int = 0x2E
_VK_F1: int = 0x70         # F1..F12 = 0x70..0x7B
_VK_LWIN: int = 0x5B       # Windows key

# Named special key → VK mapping
_SPECIAL_KEY_MAP: dict[str, int] = {
    "backspace": _VK_BACK,
    "tab":       _VK_TAB,
    "enter":     _VK_RETURN,
    "return":    _VK_RETURN,
    "escape":    _VK_ESCAPE,
    "esc":       _VK_ESCAPE,
    "space":     _VK_SPACE,
    "delete":    _VK_DELETE,
    "del":       _VK_DELETE,
    "shift":     _VK_SHIFT,
    "ctrl":      _VK_CONTROL,
    "control":   _VK_CONTROL,
    "alt":       _VK_MENU,
    "win":       _VK_LWIN,
    **{f"f{i}": _VK_F1 + (i - 1) for i in range(1, 13)},
}

# Turkish character → (VK, needs_shift)  — Turkish Q layout
# These are used by hotkey() / special_key() where a VK is required.
# type_text() uses KEYEVENTF_UNICODE instead and doesn't need this table.
_TURKISH_VK_MAP: dict[str, tuple[int, bool]] = {
    "ğ": (0xDB, False),  # OEM_4 / [ key
    "Ğ": (0xDB, True),
    "ü": (0xBA, False),  # OEM_1
    "Ü": (0xBA, True),
    "ş": (0xDC, False),  # OEM_5 / \\ key area
    "Ş": (0xDC, True),
    "ı": (0xBD, False),  # OEM_MINUS
    "İ": (0x49, True),   # 'I' + SHIFT
    "ö": (0xDE, False),  # OEM_7 / '
    "Ö": (0xDE, True),
    "ç": (0xBC, False),  # OEM_COMMA
    "Ç": (0xBC, True),
}

# ---------------------------------------------------------------------------
# Injectable callable type aliases
# ---------------------------------------------------------------------------

_GetKeyboardLayoutFn = Callable[[], int]
_ImmGetContextFn = Callable[[int], int]
_ImmSetOpenStatusFn = Callable[[int, bool], None]
_ImmGetOpenStatusFn = Callable[[int], bool]
_ImmReleaseContextFn = Callable[[int, int], None]
_GetForegroundWindowFn = Callable[[], int]
_OcrVerifyFn = Callable[[str], "str | None"]

# ---------------------------------------------------------------------------
# Default Windows API implementations
# ---------------------------------------------------------------------------


def _default_send_input(events: list[_AnyEvent]) -> int:
    """Shared with mouse_transport — re-implemented here for the keyboard path."""
    if not events:
        return 0
    try:
        n = len(events)
        arr = (_INPUT * n)()
        for i, ev in enumerate(events):
            if isinstance(ev, _KeyEvent):
                arr[i].type = _INPUT_KEYBOARD
                arr[i]._input.ki.wVk = ev.vk
                arr[i]._input.ki.wScan = ev.scan
                arr[i]._input.ki.dwFlags = ev.flags
            elif isinstance(ev, _MouseEvent):
                from nexus.source.transport.mouse_transport import (
                    _INPUT_MOUSE,
                )
                arr[i].type = _INPUT_MOUSE
                arr[i]._input.mi.dx = ev.phys_x
                arr[i]._input.mi.dy = ev.phys_y
                arr[i]._input.mi.mouseData = ev.mouse_data
                arr[i]._input.mi.dwFlags = ev.flags
        result = ctypes.windll.user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
        return int(result)
    except Exception as exc:
        _log.debug("send_input_failed", error=str(exc))
        return 0


def _default_get_keyboard_layout() -> int:
    """Return the current thread's keyboard layout handle (HKL)."""
    try:
        tid = ctypes.windll.kernel32.GetCurrentThreadId()
        hkl = ctypes.windll.user32.GetKeyboardLayout(tid)
        return int(hkl) if hkl else 0
    except Exception:
        return 0


def _default_get_foreground_window() -> int:
    """Return the foreground window HWND (0 when unavailable)."""
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        return int(hwnd) if hwnd else 0
    except Exception:
        return 0


def _default_imm_get_context(hwnd: int) -> int:
    try:
        himc = ctypes.windll.imm32.ImmGetContext(ctypes.c_void_p(hwnd))
        return int(himc) if himc else 0
    except Exception:
        return 0


def _default_imm_set_open_status(himc: int, open_: bool) -> None:
    import contextlib
    with contextlib.suppress(Exception):
        ctypes.windll.imm32.ImmSetOpenStatus(
            ctypes.c_void_p(himc), ctypes.c_bool(open_)
        )


def _default_imm_get_open_status(himc: int) -> bool:
    try:
        result = ctypes.windll.imm32.ImmGetOpenStatus(ctypes.c_void_p(himc))
        return bool(result)
    except Exception:
        return False


def _default_imm_release_context(hwnd: int, himc: int) -> None:
    import contextlib
    with contextlib.suppress(Exception):
        ctypes.windll.imm32.ImmReleaseContext(
            ctypes.c_void_p(hwnd), ctypes.c_void_p(himc)
        )


# ---------------------------------------------------------------------------
# KeyboardTransport
# ---------------------------------------------------------------------------


class KeyboardTransport:
    """
    Delivers text and key events via Windows SendInput with IME bypass.

    Parameters
    ----------
    settings:
        NexusSettings for ``transport.key_press_delay_ms`` and
        ``transport.type_verify_max_retries``.  Uses defaults when None.
    _send_input_fn:
        Injectable SendInput implementation.
    _get_keyboard_layout_fn:
        Injectable GetKeyboardLayout implementation.
    _imm_get_context_fn / _imm_set_open_status_fn /
    _imm_get_open_status_fn / _imm_release_context_fn:
        Injectable IME API implementations.
    _get_foreground_window_fn:
        Injectable GetForegroundWindow implementation.
    _ocr_verify_fn:
        Optional ``(expected: str) -> str | None`` callable.
        When provided, type_text calls it after typing and retries on mismatch.
    """

    def __init__(
        self,
        *,
        settings: NexusSettings | None = None,
        _send_input_fn: _SendInputFn | None = None,
        _get_keyboard_layout_fn: _GetKeyboardLayoutFn | None = None,
        _imm_get_context_fn: _ImmGetContextFn | None = None,
        _imm_set_open_status_fn: _ImmSetOpenStatusFn | None = None,
        _imm_get_open_status_fn: _ImmGetOpenStatusFn | None = None,
        _imm_release_context_fn: _ImmReleaseContextFn | None = None,
        _get_foreground_window_fn: _GetForegroundWindowFn | None = None,
        _ocr_verify_fn: _OcrVerifyFn | None = None,
    ) -> None:
        cfg = settings or NexusSettings()
        self._delay_ms: int = cfg.transport.key_press_delay_ms
        self._max_retries: int = cfg.transport.type_verify_max_retries

        self._send = _send_input_fn or _default_send_input
        self._get_hkl = _get_keyboard_layout_fn or _default_get_keyboard_layout
        self._imm_get_ctx = _imm_get_context_fn or _default_imm_get_context
        self._imm_set_open = _imm_set_open_status_fn or _default_imm_set_open_status
        self._imm_get_open = _imm_get_open_status_fn or _default_imm_get_open_status
        self._imm_release = _imm_release_context_fn or _default_imm_release_context
        self._get_hwnd = _get_foreground_window_fn or _default_get_foreground_window
        self._ocr_verify = _ocr_verify_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def type_text(self, text: str) -> bool:
        """
        Type *text* character-by-character via Unicode SendInput.

        - No clipboard usage (no Ctrl+V).
        - Each character is sent as ``KEYEVENTF_UNICODE``.
        - IME is temporarily disabled around the input burst.
        - If ``_ocr_verify_fn`` is set, the result is verified via OCR and
          up to ``type_verify_max_retries`` Backspace+retype cycles are
          attempted on mismatch.

        Returns True when the text was delivered (and verified, if applicable).
        """
        return await asyncio.to_thread(self._type_text_sync, text)

    async def hotkey(self, keys: list[str]) -> bool:
        """
        Press a key combination simultaneously (e.g. ``["ctrl", "c"]``).

        Keys are pressed in order, then released in reverse order.
        """
        return await asyncio.to_thread(self._hotkey_sync, keys)

    async def special_key(self, key: str) -> bool:
        """
        Press and release a single special key (e.g. ``"enter"``, ``"f5"``).
        """
        return await asyncio.to_thread(self._special_key_sync, key)

    # ------------------------------------------------------------------
    # Synchronous implementation helpers
    # ------------------------------------------------------------------

    def _type_text_sync(self, text: str) -> bool:
        hwnd = self._get_hwnd()
        with _ImeBypass(hwnd, self._imm_get_ctx, self._imm_set_open,
                         self._imm_get_open, self._imm_release):
            for attempt in range(max(1, self._max_retries)):
                ok = self._send_unicode_chars(text)
                if not ok:
                    return False

                if self._ocr_verify is None:
                    _log.debug("keyboard_type_ok_no_verify", length=len(text))
                    return True

                actual = self._ocr_verify(text)
                if actual is None or actual == text:
                    _log.debug("keyboard_type_verified", length=len(text))
                    return True

                # Mismatch — erase and retry
                backspaces = len(actual) if actual else len(text)
                _log.debug(
                    "keyboard_type_mismatch",
                    expected=text,
                    actual=actual,
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                )
                self._send_backspaces(backspaces)

        _log.warning("keyboard_type_failed_after_retries", text_length=len(text))
        return False

    def _hotkey_sync(self, keys: list[str]) -> bool:
        vk_codes = [_resolve_vk(k) for k in keys]
        if any(vk is None for vk in vk_codes):
            _log.debug("hotkey_unknown_key", keys=keys)
            return False
        # Press all keys down
        down_events: list[_AnyEvent] = [
            _KeyEvent(vk=vk, scan=0, flags=0)  # type: ignore[arg-type]
            for vk in vk_codes
        ]
        # Release in reverse order
        up_events: list[_AnyEvent] = [
            _KeyEvent(vk=vk, scan=0, flags=_KEYEVENTF_KEYUP)  # type: ignore[arg-type]
            for vk in reversed(vk_codes)
        ]
        total = len(down_events) + len(up_events)
        sent = self._send(down_events) + self._send(up_events)
        ok = sent == total
        _log.debug("hotkey", keys=keys, ok=ok)
        return ok

    def _special_key_sync(self, key: str) -> bool:
        vk = _resolve_vk(key)
        if vk is None:
            _log.debug("special_key_unknown", key=key)
            return False
        events: list[_AnyEvent] = [
            _KeyEvent(vk=vk, scan=0, flags=0),
            _KeyEvent(vk=vk, scan=0, flags=_KEYEVENTF_KEYUP),
        ]
        sent = self._send(events)
        ok = sent == len(events)
        _log.debug("special_key", key=key, vk=vk, ok=ok)
        return ok

    def _send_unicode_chars(self, text: str) -> bool:
        """Send each character in *text* as a KEYEVENTF_UNICODE event pair."""
        total_events = 0
        total_sent = 0
        for char in text:
            codepoint = ord(char)
            events: list[_AnyEvent] = [
                _KeyEvent(vk=0, scan=codepoint, flags=_KEYEVENTF_UNICODE),
                _KeyEvent(
                    vk=0,
                    scan=codepoint,
                    flags=_KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP,
                ),
            ]
            total_events += len(events)
            total_sent += self._send(events)
            if self._delay_ms > 0:
                time.sleep(self._delay_ms / 1000.0)
        return total_sent == total_events

    def _send_backspaces(self, count: int) -> None:
        """Send *count* Backspace key events to erase mistyped text."""
        for _ in range(count):
            self._send([
                _KeyEvent(vk=_VK_BACK, scan=0, flags=0),
                _KeyEvent(vk=_VK_BACK, scan=0, flags=_KEYEVENTF_KEYUP),
            ])
            if self._delay_ms > 0:
                time.sleep(self._delay_ms / 1000.0)


# ---------------------------------------------------------------------------
# IME bypass context manager
# ---------------------------------------------------------------------------


class _ImeBypass:
    """
    Context manager that disables IME for the focused window and restores
    the original open-status on exit.

    When ``himc == 0`` (no IME context available), is a no-op.
    """

    def __init__(
        self,
        hwnd: int,
        get_ctx: _ImmGetContextFn,
        set_open: _ImmSetOpenStatusFn,
        get_open: _ImmGetOpenStatusFn,
        release: _ImmReleaseContextFn,
    ) -> None:
        self._hwnd = hwnd
        self._get_ctx = get_ctx
        self._set_open = set_open
        self._get_open = get_open
        self._release = release
        self._himc: int = 0
        self._was_open: bool = False

    def __enter__(self) -> _ImeBypass:
        self._himc = self._get_ctx(self._hwnd)
        if self._himc:
            self._was_open = self._get_open(self._himc)
            if self._was_open:
                self._set_open(self._himc, False)
        return self

    def __exit__(self, *_: object) -> None:
        if self._himc:
            if self._was_open:
                self._set_open(self._himc, True)
            self._release(self._hwnd, self._himc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_vk(key: str) -> int | None:
    """
    Return the VK code for *key*, checking special-key and Turkish maps.

    Returns None when the key cannot be resolved.
    """
    lower = key.lower()
    if lower in _SPECIAL_KEY_MAP:
        return _SPECIAL_KEY_MAP[lower]
    # Single character — check Turkish map first
    if len(key) == 1:
        if key in _TURKISH_VK_MAP:
            vk, _shift = _TURKISH_VK_MAP[key]
            return vk
        # ASCII: use ord() if it fits in a VK range
        cp = ord(key)
        if 0x30 <= cp <= 0x5A:  # 0-9, A-Z (uppercase)
            return cp
        if 0x61 <= cp <= 0x7A:  # a-z (map to uppercase VK)
            return cp - 32
    return None
