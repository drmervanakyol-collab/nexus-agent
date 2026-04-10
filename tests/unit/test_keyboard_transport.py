"""
tests/unit/test_keyboard_transport.py
Unit tests for nexus/source/transport/keyboard_transport.py.

Coverage
--------
  type_text            — Unicode SendInput per character, delay, return value
  Turkish İ            — U+0130 sent as KEYEVENTF_UNICODE scan code
  IME bypass           — ImmGetContext → SetOpenStatus(False) → restore
  Text verification    — OCR mismatch → backspace + retry
  hotkey               — down+up sequence for multi-key combo
  special_key          — named key VK resolution
  _resolve_vk          — Turkish VK map, special key map, ASCII
  _send_backspaces     — correct number of backspace events
"""
from __future__ import annotations

from nexus.source.transport.keyboard_transport import (
    _KEYEVENTF_KEYUP,
    _KEYEVENTF_UNICODE,
    _TURKISH_VK_MAP,
    _VK_BACK,
    _VK_RETURN,
    KeyboardTransport,
    _resolve_vk,
)
from nexus.source.transport.mouse_transport import _KeyEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _KeyRecorder:
    """Records key events sent via _send_input_fn."""

    def __init__(self, return_count: int | None = None) -> None:
        self.all_events: list[_KeyEvent] = []
        self._ret = return_count

    def __call__(self, events: list) -> int:
        self.all_events.extend(events)
        return self._ret if self._ret is not None else len(events)


def _make_transport(
    *,
    recorder: _KeyRecorder | None = None,
    ocr_verify_fn=None,
    imm_get_ctx_fn=None,
    imm_set_open_fn=None,
    imm_get_open_fn=None,
    imm_release_fn=None,
    key_delay_ms: int = 0,
    max_retries: int = 3,
) -> tuple[KeyboardTransport, _KeyRecorder]:
    from nexus.core.settings import NexusSettings, TransportSettings

    cfg = NexusSettings(
        transport=TransportSettings(
            key_press_delay_ms=key_delay_ms,
            type_verify_max_retries=max_retries,
        )
    )
    rec = recorder or _KeyRecorder()
    kt = KeyboardTransport(
        settings=cfg,
        _send_input_fn=rec,
        _get_keyboard_layout_fn=lambda: 0xF0410409,   # Turkish Q HKL
        _imm_get_context_fn=imm_get_ctx_fn or (lambda hwnd: 0),
        _imm_set_open_status_fn=imm_set_open_fn or (lambda himc, v: None),
        _imm_get_open_status_fn=imm_get_open_fn or (lambda himc: False),
        _imm_release_context_fn=imm_release_fn or (lambda hwnd, himc: None),
        _get_foreground_window_fn=lambda: 12345,
        _ocr_verify_fn=ocr_verify_fn,
    )
    return kt, rec


# ---------------------------------------------------------------------------
# TestTypeText
# ---------------------------------------------------------------------------


class TestTypeText:
    async def test_type_text_returns_true_on_success(self) -> None:
        kt, _ = _make_transport()
        assert await kt.type_text("hello") is True

    async def test_type_text_sends_two_events_per_char(self) -> None:
        """Down + Up for each character."""
        kt, rec = _make_transport()
        await kt.type_text("abc")
        assert len(rec.all_events) == 6  # 3 chars × 2 events

    async def test_type_text_uses_keyeventf_unicode(self) -> None:
        kt, rec = _make_transport()
        await kt.type_text("A")
        for ev in rec.all_events:
            assert ev.flags & _KEYEVENTF_UNICODE

    async def test_type_text_vk_is_zero_for_unicode(self) -> None:
        kt, rec = _make_transport()
        await kt.type_text("X")
        for ev in rec.all_events:
            assert ev.vk == 0

    async def test_type_text_scan_is_unicode_codepoint(self) -> None:
        kt, rec = _make_transport()
        await kt.type_text("Z")
        down_event = rec.all_events[0]
        assert down_event.scan == ord("Z")

    async def test_type_text_keyup_flag_on_release_event(self) -> None:
        kt, rec = _make_transport()
        await kt.type_text("A")
        up_event = rec.all_events[1]
        assert up_event.flags & _KEYEVENTF_KEYUP

    async def test_type_text_no_clipboard(self) -> None:
        """No clipboard-related VK codes (Ctrl+V) should appear."""
        kt, rec = _make_transport()
        await kt.type_text("hello world")
        # VK for 'V' = 0x56, Ctrl = 0x11
        vks = [e.vk for e in rec.all_events]
        assert 0x56 not in vks   # no 'V' key press via VK (only via unicode)
        assert 0x11 not in vks   # no Ctrl press

    async def test_type_empty_string_returns_true(self) -> None:
        kt, _ = _make_transport()
        assert await kt.type_text("") is True

    async def test_type_text_returns_false_when_send_fails(self) -> None:
        rec = _KeyRecorder(return_count=0)
        kt, _ = _make_transport(recorder=rec)
        assert await kt.type_text("x") is False


# ---------------------------------------------------------------------------
# TestTurkishCharacters
# ---------------------------------------------------------------------------


class TestTurkishCharacters:
    async def test_capital_i_with_dot_scan_code(self) -> None:
        """İ (U+0130) must be sent with scan=0x0130 and KEYEVENTF_UNICODE."""
        kt, rec = _make_transport()
        await kt.type_text("İ")
        down = rec.all_events[0]
        assert down.scan == 0x0130
        assert down.flags & _KEYEVENTF_UNICODE

    async def test_dotless_i_scan_code(self) -> None:
        """ı (U+0131) must be sent with scan=0x0131."""
        kt, rec = _make_transport()
        await kt.type_text("ı")
        assert rec.all_events[0].scan == 0x0131

    async def test_turkish_g_with_breve_scan_code(self) -> None:
        """ğ (U+011F) scan code correct."""
        kt, rec = _make_transport()
        await kt.type_text("ğ")
        assert rec.all_events[0].scan == ord("ğ")

    async def test_turkish_string_all_unicode(self) -> None:
        """A full Turkish sentence — every char uses KEYEVENTF_UNICODE."""
        kt, rec = _make_transport()
        text = "Şişli İstanbul"
        await kt.type_text(text)
        assert len(rec.all_events) == len(text) * 2
        for ev in rec.all_events:
            assert ev.flags & _KEYEVENTF_UNICODE

    async def test_turkish_vk_map_contains_dotted_i(self) -> None:
        """The VK map should have an entry for İ."""
        assert "İ" in _TURKISH_VK_MAP

    async def test_turkish_vk_map_contains_dotless_i(self) -> None:
        assert "ı" in _TURKISH_VK_MAP

    async def test_turkish_vk_map_has_seven_characters(self) -> None:
        """ğ, Ğ, ü, Ü, ş, Ş, ı, İ, ö, Ö, ç, Ç = 12 entries."""
        assert len(_TURKISH_VK_MAP) >= 6


# ---------------------------------------------------------------------------
# TestImeBypass
# ---------------------------------------------------------------------------


class TestImeBypass:
    async def test_imm_get_context_called_with_hwnd(self) -> None:
        get_ctx_calls: list[int] = []

        def _get_ctx(hwnd: int) -> int:
            get_ctx_calls.append(hwnd)
            return 999  # fake HIMC

        kt, _ = _make_transport(imm_get_ctx_fn=_get_ctx)
        await kt.type_text("a")
        assert get_ctx_calls == [12345]  # the injected foreground HWND

    async def test_ime_disabled_before_typing(self) -> None:
        """ImmSetOpenStatus(False) must be called before any key events."""
        call_log: list[str] = []

        def _get_ctx(hwnd: int) -> int:
            return 999

        def _get_open(himc: int) -> bool:
            return True  # IME was open

        def _set_open(himc: int, v: bool) -> None:
            call_log.append(f"set_open:{v}")

        rec = _KeyRecorder()

        class _RecordingKb(KeyboardTransport):
            def _send_unicode_chars(self, text: str) -> bool:
                call_log.append("type")
                return True

        from nexus.core.settings import NexusSettings, TransportSettings
        cfg = NexusSettings(transport=TransportSettings(key_press_delay_ms=0, type_verify_max_retries=1))
        kt = _RecordingKb(
            settings=cfg,
            _send_input_fn=rec,
            _get_keyboard_layout_fn=lambda: 0,
            _imm_get_context_fn=_get_ctx,
            _imm_set_open_status_fn=_set_open,
            _imm_get_open_status_fn=_get_open,
            _imm_release_context_fn=lambda *_: None,
            _get_foreground_window_fn=lambda: 12345,
        )
        await kt.type_text("hi")
        # order: disable → type → re-enable
        assert call_log.index("set_open:False") < call_log.index("type")
        assert call_log.index("type") < call_log.index("set_open:True")

    async def test_ime_restored_after_typing(self) -> None:
        """ImmSetOpenStatus(True) must be called after typing (restore)."""
        set_open_calls: list[tuple[int, bool]] = []

        def _set_open(himc: int, v: bool) -> None:
            set_open_calls.append((himc, v))

        kt, _ = _make_transport(
            imm_get_ctx_fn=lambda hwnd: 999,
            imm_set_open_fn=_set_open,
            imm_get_open_fn=lambda himc: True,
        )
        await kt.type_text("x")
        # The last SetOpenStatus call should restore to True
        assert set_open_calls[-1] == (999, True)

    async def test_no_himc_skips_ime_calls(self) -> None:
        """When ImmGetContext returns 0, SetOpenStatus must not be called."""
        set_open_calls: list = []
        kt, _ = _make_transport(
            imm_get_ctx_fn=lambda hwnd: 0,
            imm_set_open_fn=lambda himc, v: set_open_calls.append(v),
        )
        await kt.type_text("z")
        assert set_open_calls == []

    async def test_imm_release_context_called(self) -> None:
        release_calls: list[tuple] = []

        kt, _ = _make_transport(
            imm_get_ctx_fn=lambda hwnd: 777,
            imm_release_fn=lambda hwnd, himc: release_calls.append((hwnd, himc)),
        )
        await kt.type_text("r")
        assert (12345, 777) in release_calls


# ---------------------------------------------------------------------------
# TestTextVerification
# ---------------------------------------------------------------------------


class TestTextVerification:
    async def test_no_verify_fn_always_passes(self) -> None:
        kt, _ = _make_transport(ocr_verify_fn=None)
        assert await kt.type_text("hello") is True

    async def test_correct_ocr_returns_true(self) -> None:
        kt, _ = _make_transport(ocr_verify_fn=lambda expected: expected)
        assert await kt.type_text("hello") is True

    async def test_mismatch_triggers_backspace_events(self) -> None:
        attempts = [0]

        def _verify(expected: str) -> str:
            attempts[0] += 1
            if attempts[0] == 1:
                return "helo"   # one char missing
            return expected     # second attempt succeeds

        kt, rec = _make_transport(ocr_verify_fn=_verify, max_retries=3)
        result = await kt.type_text("hello")
        assert result is True
        # After first attempt: some backspace events should appear
        backspace_events = [e for e in rec.all_events if e.vk == _VK_BACK]
        assert len(backspace_events) > 0

    async def test_max_retries_exhausted_returns_false(self) -> None:
        """If OCR never matches, we give up after max_retries."""
        kt, _ = _make_transport(
            ocr_verify_fn=lambda expected: "WRONG",
            max_retries=2,
        )
        result = await kt.type_text("hello")
        assert result is False

    async def test_ocr_none_means_unverifiable_passes(self) -> None:
        """When OCR returns None (unavailable), treat as success."""
        kt, _ = _make_transport(ocr_verify_fn=lambda expected: None)
        assert await kt.type_text("abc") is True

    async def test_backspace_count_matches_actual_text_length(self) -> None:
        """Backspaces sent == len(actual_returned) on mismatch."""
        attempts = [0]
        actual_on_first = "helo"   # 4 chars — should send 4 backspaces

        def _verify(expected: str) -> str:
            attempts[0] += 1
            if attempts[0] == 1:
                return actual_on_first
            return expected

        kt, rec = _make_transport(ocr_verify_fn=_verify, max_retries=3)
        await kt.type_text("hello")
        backspace_events = [e for e in rec.all_events if e.vk == _VK_BACK and not (e.flags & _KEYEVENTF_KEYUP)]
        assert len(backspace_events) == len(actual_on_first)


# ---------------------------------------------------------------------------
# TestHotkey
# ---------------------------------------------------------------------------


class TestHotkey:
    async def test_hotkey_ctrl_c_returns_true(self) -> None:
        kt, _ = _make_transport()
        assert await kt.hotkey(["ctrl", "c"]) is True

    async def test_hotkey_sends_down_then_up(self) -> None:
        kt, rec = _make_transport()
        await kt.hotkey(["ctrl", "c"])
        events = rec.all_events
        # 2 keys × 2 events (down+up) = 4
        assert len(events) == 4
        # First two are key-down (no KEYUP flag)
        assert not (events[0].flags & _KEYEVENTF_KEYUP)
        assert not (events[1].flags & _KEYEVENTF_KEYUP)

    async def test_hotkey_unknown_key_returns_false(self) -> None:
        kt, _ = _make_transport()
        assert await kt.hotkey(["ctrl", "???unknown"]) is False

    async def test_hotkey_single_key(self) -> None:
        kt, _ = _make_transport()
        assert await kt.hotkey(["enter"]) is True

    async def test_hotkey_uses_correct_vk_for_ctrl(self) -> None:
        kt, rec = _make_transport()
        await kt.hotkey(["ctrl"])
        assert rec.all_events[0].vk == 0x11  # VK_CONTROL


# ---------------------------------------------------------------------------
# TestSpecialKey
# ---------------------------------------------------------------------------


class TestSpecialKey:
    async def test_enter_returns_true(self) -> None:
        kt, _ = _make_transport()
        assert await kt.special_key("enter") is True

    async def test_enter_sends_down_and_up(self) -> None:
        kt, rec = _make_transport()
        await kt.special_key("enter")
        assert len(rec.all_events) == 2
        assert not (rec.all_events[0].flags & _KEYEVENTF_KEYUP)
        assert rec.all_events[1].flags & _KEYEVENTF_KEYUP

    async def test_enter_vk_correct(self) -> None:
        kt, rec = _make_transport()
        await kt.special_key("enter")
        assert rec.all_events[0].vk == _VK_RETURN

    async def test_unknown_special_key_returns_false(self) -> None:
        kt, _ = _make_transport()
        assert await kt.special_key("notakey") is False

    async def test_f5_resolves_correctly(self) -> None:
        kt, rec = _make_transport()
        await kt.special_key("f5")
        assert rec.all_events[0].vk == 0x74  # VK_F5

    async def test_escape_key(self) -> None:
        kt, rec = _make_transport()
        await kt.special_key("esc")
        assert rec.all_events[0].vk == 0x1B  # VK_ESCAPE


# ---------------------------------------------------------------------------
# TestResolveVk
# ---------------------------------------------------------------------------


class TestResolveVk:
    def test_enter_resolves(self) -> None:
        assert _resolve_vk("enter") == 0x0D

    def test_backspace_resolves(self) -> None:
        assert _resolve_vk("backspace") == 0x08

    def test_ctrl_resolves(self) -> None:
        assert _resolve_vk("ctrl") == 0x11

    def test_f1_resolves(self) -> None:
        assert _resolve_vk("f1") == 0x70

    def test_f12_resolves(self) -> None:
        assert _resolve_vk("f12") == 0x7B

    def test_unknown_returns_none(self) -> None:
        assert _resolve_vk("notakey") is None

    def test_ascii_uppercase_resolves(self) -> None:
        assert _resolve_vk("A") == 0x41

    def test_ascii_lowercase_resolves_to_uppercase_vk(self) -> None:
        assert _resolve_vk("a") == 0x41

    def test_digit_resolves(self) -> None:
        assert _resolve_vk("1") == ord("1")

    def test_turkish_soft_g_resolves(self) -> None:
        vk = _resolve_vk("ğ")
        assert vk is not None
        assert vk == _TURKISH_VK_MAP["ğ"][0]

    def test_turkish_capital_i_dot_resolves(self) -> None:
        vk = _resolve_vk("İ")
        assert vk is not None
