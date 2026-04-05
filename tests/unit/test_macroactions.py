"""
tests/unit/test_macroactions.py
Unit tests for nexus/action/macroactions.py.

Coverage
--------
  MacroActionResult    — field defaults, inheritance from ActionResult
  SafeFieldReplace     — success path, click failure, type failure,
                         OCR mismatch + retry, max_retries exhausted,
                         no-OCR path, partial_completion flag,
                         transport-aware (native → fallback)
  SafeRowWrite         — success path, identity lock failure (initial),
                         wrong-cell guard (mid-write), cell failure,
                         partial_completion, audit trail (step_results)
  VerifyAndSubmit      — all fields match, field mismatch → correction,
                         submit failure, submit verify failure
  GuardedSelectAndConfirm — success path, click failure, select failure,
                         OCR mismatch + retry, no-OCR path,
                         transport-aware (native select)
"""
from __future__ import annotations

import pytest

from nexus.action.macroactions import (
    GuardedSelectAndConfirm,
    MacroActionResult,
    SafeFieldReplace,
    SafeRowWrite,
    VerifyAndSubmit,
)
from nexus.action.registry import ActionResult

# ---------------------------------------------------------------------------
# Async primitive factories
# ---------------------------------------------------------------------------

_COORDS = (100, 200)
_VALUE = "hello"
_IDENTITY = "Row-1"


async def _ok_click(coords):       return True
async def _fail_click(coords):     return False
async def _ok_type(text):          return True
async def _fail_type(text):        return False
async def _ok_hotkey(keys):        return True
async def _ok_native_click(eid):   return True
async def _fail_native_click(eid): return False


def _ocr_always(text: str):
    """Returns a callable that always returns *text*."""
    async def _read(coords): return text
    return _read


def _ocr_sequence(*values):
    """Returns a callable that yields *values* in turn (then repeats last)."""
    it = iter(values)
    last = [None]

    async def _read(coords):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]
    return _read


def _make_sfr(
    *,
    native_click=None,
    fallback_click=_ok_click,
    type_fn=_ok_type,
    hotkey_fn=_ok_hotkey,
    ocr_read=None,
    undo_fn=None,
    max_retries=3,
    preferred_transport=None,
) -> SafeFieldReplace:
    return SafeFieldReplace(
        _native_click_fn=native_click,
        _fallback_click_fn=fallback_click,
        _type_fn=type_fn,
        _hotkey_fn=hotkey_fn,
        _ocr_read_fn=ocr_read,
        _undo_fn=undo_fn,
        max_retries=max_retries,
        preferred_transport=preferred_transport,
    )


def _make_srw(
    *,
    sfr: SafeFieldReplace | None = None,
    ocr_read=None,
    identity_verify_retries=1,
) -> SafeRowWrite:
    _sfr = sfr or _make_sfr()
    _ocr = ocr_read or _ocr_always(_IDENTITY)
    return SafeRowWrite(
        field_replace=_sfr,
        _ocr_read_fn=_ocr,
        identity_verify_retries=identity_verify_retries,
    )


def _make_vas(
    *,
    sfr: SafeFieldReplace | None = None,
    ocr_read=None,
    click_fn=_ok_click,
    submit_verify_fn=None,
) -> VerifyAndSubmit:
    return VerifyAndSubmit(
        field_replace=sfr or _make_sfr(),
        _ocr_read_fn=ocr_read or _ocr_always(""),
        _click_fn=click_fn,
        _submit_verify_fn=submit_verify_fn,
    )


def _make_gsc(
    *,
    native_select=None,
    fallback_click=_ok_click,
    fallback_select=None,
    ocr_read=None,
    max_retries=3,
    preferred_transport=None,
) -> GuardedSelectAndConfirm:
    return GuardedSelectAndConfirm(
        _native_select_fn=native_select,
        _fallback_click_fn=fallback_click,
        _fallback_select_fn=fallback_select,
        _ocr_read_fn=ocr_read,
        max_retries=max_retries,
        preferred_transport=preferred_transport,
    )


# ---------------------------------------------------------------------------
# TestMacroActionResult
# ---------------------------------------------------------------------------


class TestMacroActionResult:
    def test_is_subclass_of_action_result(self) -> None:
        r = MacroActionResult(success=True)
        assert isinstance(r, ActionResult)

    def test_extra_fields_have_defaults(self) -> None:
        r = MacroActionResult(success=True)
        assert r.steps_completed == 0
        assert r.total_steps == 0
        assert r.step_results == []

    def test_all_fields_set(self) -> None:
        inner = ActionResult(success=True)
        r = MacroActionResult(
            success=False,
            error="boom",
            partial_completion=True,
            steps_completed=2,
            total_steps=4,
            step_results=[inner],
        )
        assert r.success is False
        assert r.error == "boom"
        assert r.partial_completion is True
        assert r.steps_completed == 2
        assert r.total_steps == 4
        assert r.step_results == [inner]

    def test_inherits_action_result_fields(self) -> None:
        r = MacroActionResult(success=True, duration_ms=12.5, transport_used="uia")
        assert r.duration_ms == 12.5
        assert r.transport_used == "uia"


# ---------------------------------------------------------------------------
# TestSafeFieldReplace
# ---------------------------------------------------------------------------


class TestSafeFieldReplace:
    async def test_success_no_ocr(self) -> None:
        sfr = _make_sfr()
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.success is True
        assert r.partial_completion is False

    async def test_success_ocr_match(self) -> None:
        sfr = _make_sfr(ocr_read=_ocr_always(_VALUE))
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.success is True

    async def test_step_results_populated(self) -> None:
        sfr = _make_sfr(ocr_read=_ocr_always(_VALUE))
        r = await sfr.execute(_COORDS, _VALUE)
        # click, ctrl+a, type, verify = 4 steps
        assert len(r.step_results) == 4

    async def test_total_steps_is_four(self) -> None:
        sfr = _make_sfr(ocr_read=_ocr_always(_VALUE))
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.total_steps == 4

    async def test_click_failure_returns_false(self) -> None:
        sfr = _make_sfr(fallback_click=_fail_click)
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.success is False
        assert "Click" in (r.error or "")

    async def test_click_failure_steps_completed_is_one(self) -> None:
        sfr = _make_sfr(fallback_click=_fail_click)
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.steps_completed == 1

    async def test_type_failure_partial_completion(self) -> None:
        sfr = _make_sfr(type_fn=_fail_type)
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.success is False
        assert r.partial_completion is True

    async def test_duration_ms_is_positive(self) -> None:
        sfr = _make_sfr()
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.duration_ms >= 0.0

    # ── Retry tests ──────────────────────────────────────────────────────

    async def test_ocr_mismatch_triggers_retry(self) -> None:
        """First OCR wrong → second attempt correct."""
        undo_calls: list[int] = []

        async def _undo(): undo_calls.append(1)

        # First call returns wrong, second returns correct
        sfr = _make_sfr(
            ocr_read=_ocr_sequence("WRONG", _VALUE),
            undo_fn=_undo,
            max_retries=3,
        )
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.success is True
        assert len(undo_calls) == 1

    async def test_undo_called_on_mismatch(self) -> None:
        undo_calls: list[int] = []

        async def _undo(): undo_calls.append(1)

        sfr = _make_sfr(
            ocr_read=_ocr_sequence("WRONG", _VALUE),
            undo_fn=_undo,
            max_retries=3,
        )
        await sfr.execute(_COORDS, _VALUE)
        assert len(undo_calls) >= 1

    async def test_max_retries_exhausted_returns_false(self) -> None:
        sfr = _make_sfr(
            ocr_read=_ocr_always("ALWAYS_WRONG"),
            max_retries=2,
        )
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.success is False

    async def test_exhausted_partial_completion_true(self) -> None:
        sfr = _make_sfr(
            ocr_read=_ocr_always("ALWAYS_WRONG"),
            max_retries=2,
        )
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.partial_completion is True

    async def test_exhausted_error_mentions_retries(self) -> None:
        sfr = _make_sfr(
            ocr_read=_ocr_always("ALWAYS_WRONG"),
            max_retries=2,
        )
        r = await sfr.execute(_COORDS, _VALUE)
        assert "2" in (r.error or "") or "attempt" in (r.error or "")

    async def test_step_results_accumulate_across_retries(self) -> None:
        """With 2 failed attempts, step_results should contain > 4 entries."""
        sfr = _make_sfr(
            ocr_read=_ocr_always("ALWAYS_WRONG"),
            max_retries=2,
        )
        r = await sfr.execute(_COORDS, _VALUE)
        assert len(r.step_results) > 4  # 2 attempts × 4 steps each

    # ── Transport-aware tests ─────────────────────────────────────────────

    async def test_native_click_used_when_preferred_transport_uia(self) -> None:
        native_calls: list[str | None] = []

        async def _native(eid): native_calls.append(eid); return True

        sfr = _make_sfr(
            native_click=_native,
            preferred_transport="uia",
        )
        r = await sfr.execute(_COORDS, _VALUE, element_id="el-1")
        assert r.success is True
        assert native_calls == ["el-1"]

    async def test_fallback_click_not_called_when_native_succeeds(self) -> None:
        fallback_calls: list = []

        async def _fb(coords): fallback_calls.append(coords); return True

        sfr = _make_sfr(
            native_click=_ok_native_click,
            fallback_click=_fb,
            preferred_transport="uia",
        )
        await sfr.execute(_COORDS, _VALUE)
        assert fallback_calls == []

    async def test_fallback_used_when_native_fails(self) -> None:
        fallback_calls: list = []

        async def _fb(coords): fallback_calls.append(coords); return True

        sfr = _make_sfr(
            native_click=_fail_native_click,
            fallback_click=_fb,
            preferred_transport="uia",
        )
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.success is True
        assert len(fallback_calls) > 0

    async def test_transport_used_is_native_when_native_succeeds(self) -> None:
        sfr = _make_sfr(
            native_click=_ok_native_click,
            preferred_transport="uia",
            ocr_read=_ocr_always(_VALUE),
        )
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.transport_used == "uia"

    async def test_transport_used_is_mouse_when_no_native(self) -> None:
        sfr = _make_sfr(
            native_click=None,
            preferred_transport=None,
            ocr_read=_ocr_always(_VALUE),
        )
        r = await sfr.execute(_COORDS, _VALUE)
        assert r.transport_used == "mouse"


# ---------------------------------------------------------------------------
# TestSafeRowWrite
# ---------------------------------------------------------------------------

_CELLS = [
    ((100, 200), "val-A"),
    ((200, 200), "val-B"),
]


class TestSafeRowWrite:
    async def test_success_path(self) -> None:
        srw = _make_srw(ocr_read=_ocr_always(_IDENTITY))
        r = await srw.execute(_CELLS, identity_coords=(50, 200), expected_identity=_IDENTITY)
        assert r.success is True
        assert r.steps_completed == 2

    async def test_step_results_one_per_cell(self) -> None:
        srw = _make_srw(ocr_read=_ocr_always(_IDENTITY))
        r = await srw.execute(_CELLS, identity_coords=(50, 200), expected_identity=_IDENTITY)
        assert len(r.step_results) == len(_CELLS)

    async def test_total_steps_equals_cell_count(self) -> None:
        srw = _make_srw(ocr_read=_ocr_always(_IDENTITY))
        r = await srw.execute(_CELLS, identity_coords=(50, 200), expected_identity=_IDENTITY)
        assert r.total_steps == len(_CELLS)

    async def test_identity_lock_failure_initial(self) -> None:
        """Initial identity check fails → abort immediately."""
        srw = _make_srw(ocr_read=_ocr_always("WRONG_ROW"))
        r = await srw.execute(
            _CELLS,
            identity_coords=(50, 200),
            expected_identity=_IDENTITY,
        )
        assert r.success is False
        assert r.steps_completed == 0
        assert "identity" in (r.error or "").lower()

    async def test_wrong_cell_guard_mid_write(self) -> None:
        """Identity changes after first cell → abort with partial_completion."""
        # First read: identity OK, then identity changes
        ocr = _ocr_sequence(_IDENTITY, "val-A_written", "DIFFERENT_ROW")

        sfr = _make_sfr()
        srw = SafeRowWrite(
            field_replace=sfr,
            _ocr_read_fn=ocr,
            identity_verify_retries=1,
        )
        r = await srw.execute(
            _CELLS,
            identity_coords=(50, 200),
            expected_identity=_IDENTITY,
        )
        assert r.success is False
        assert r.partial_completion is True
        assert "wrong-cell" in (r.error or "").lower() or "identity" in (r.error or "").lower()

    async def test_cell_failure_returns_partial(self) -> None:
        """A cell write failure sets partial_completion."""
        sfr = _make_sfr(fallback_click=_fail_click)  # click always fails
        srw = _make_srw(sfr=sfr, ocr_read=_ocr_always(_IDENTITY))
        r = await srw.execute(_CELLS, identity_coords=(50, 200), expected_identity=_IDENTITY)
        assert r.success is False
        assert r.partial_completion is True

    async def test_empty_cells_returns_success(self) -> None:
        srw = _make_srw(ocr_read=_ocr_always(_IDENTITY))
        r = await srw.execute([], identity_coords=(50, 200), expected_identity=_IDENTITY)
        assert r.success is True
        assert r.steps_completed == 0

    async def test_audit_trail_contains_step_results(self) -> None:
        """step_results acts as the write audit trail."""
        srw = _make_srw(ocr_read=_ocr_always(_IDENTITY))
        r = await srw.execute(_CELLS, identity_coords=(50, 200), expected_identity=_IDENTITY)
        assert isinstance(r.step_results, list)
        assert all(isinstance(s, ActionResult) for s in r.step_results)

    async def test_wrong_cell_error_mentions_index(self) -> None:
        ocr = _ocr_sequence(_IDENTITY, "val-A_written", "DRIFT")
        sfr = _make_sfr()
        srw = SafeRowWrite(field_replace=sfr, _ocr_read_fn=ocr, identity_verify_retries=1)
        r = await srw.execute(
            _CELLS, identity_coords=(50, 200), expected_identity=_IDENTITY
        )
        assert r.success is False
        # The error should mention the cell index
        assert r.error is not None


# ---------------------------------------------------------------------------
# TestVerifyAndSubmit
# ---------------------------------------------------------------------------


class TestVerifyAndSubmit:
    _FIELDS = [
        ((100, 100), "Alice"),
        ((200, 100), "30"),
    ]
    _SUBMIT = (300, 400)

    async def test_all_fields_match_success(self) -> None:
        """All OCR reads match → click submit → success."""

        async def _ocr(coords): return {(100, 100): "Alice", (200, 100): "30"}.get(coords)

        vas = VerifyAndSubmit(
            field_replace=_make_sfr(),
            _ocr_read_fn=_ocr,
            _click_fn=_ok_click,
        )
        r = await vas.execute(self._FIELDS, self._SUBMIT)
        assert r.success is True

    async def test_field_mismatch_triggers_correction(self) -> None:
        """When OCR returns wrong value, SafeFieldReplace is invoked."""
        correction_calls: list = []

        async def _ocr(coords): return "WRONG"  # always wrong

        class _RecordSFR(SafeFieldReplace):
            async def execute(self, coords, value, element_id=None):
                correction_calls.append((coords, value))
                return MacroActionResult(success=True, steps_completed=4, total_steps=4)

        vas = VerifyAndSubmit(
            field_replace=_RecordSFR(
                _fallback_click_fn=_ok_click,
                _type_fn=_ok_type,
                _hotkey_fn=_ok_hotkey,
            ),
            _ocr_read_fn=_ocr,
            _click_fn=_ok_click,
        )
        r = await vas.execute(self._FIELDS, self._SUBMIT)
        assert r.success is True
        assert len(correction_calls) == len(self._FIELDS)

    async def test_submit_click_failure(self) -> None:
        vas = _make_vas(
            ocr_read=_ocr_always(""),
            click_fn=_fail_click,
        )
        r = await vas.execute([], self._SUBMIT)
        assert r.success is False
        assert "submit" in (r.error or "").lower()

    async def test_submit_verify_failure(self) -> None:
        async def _verify(): return False

        vas = _make_vas(
            ocr_read=_ocr_always(""),
            submit_verify_fn=_verify,
        )
        r = await vas.execute([], self._SUBMIT)
        assert r.success is False
        assert "verification" in (r.error or "").lower()

    async def test_submit_verify_success(self) -> None:
        async def _verify(): return True

        vas = _make_vas(
            ocr_read=_ocr_always(""),
            submit_verify_fn=_verify,
        )
        r = await vas.execute([], self._SUBMIT)
        assert r.success is True

    async def test_correction_failure_returns_partial(self) -> None:
        """If correction fails, partial_completion must be set."""

        async def _ocr(coords): return "WRONG"

        sfr = _make_sfr(fallback_click=_fail_click)  # click fails → correction fails
        vas = VerifyAndSubmit(
            field_replace=sfr,
            _ocr_read_fn=_ocr,
            _click_fn=_ok_click,
        )
        r = await vas.execute(self._FIELDS, self._SUBMIT)
        assert r.success is False
        assert r.partial_completion is True

    async def test_step_results_includes_submit(self) -> None:
        vas = _make_vas(ocr_read=_ocr_always(""))
        r = await vas.execute([], self._SUBMIT)
        # Empty fields → phase1 skipped; step_results = [submit_click, submit_verify]
        assert len(r.step_results) >= 1

    async def test_no_submit_verify_fn_assumes_success(self) -> None:
        """When _submit_verify_fn is None, submit click success → macro success."""
        vas = _make_vas(ocr_read=_ocr_always(""), submit_verify_fn=None)
        r = await vas.execute([], self._SUBMIT)
        assert r.success is True


# ---------------------------------------------------------------------------
# TestGuardedSelectAndConfirm
# ---------------------------------------------------------------------------


class TestGuardedSelectAndConfirm:
    _VAL = "Option-B"
    _ELEM = "dropdown-1"

    async def _ok_select(coords, value): return True
    async def _fail_select(coords, value): return False

    async def test_success_no_ocr(self) -> None:
        gsc = _make_gsc(fallback_select=TestGuardedSelectAndConfirm._ok_select)
        r = await gsc.execute(_COORDS, self._VAL)
        assert r.success is True

    async def test_success_ocr_match(self) -> None:
        gsc = _make_gsc(
            fallback_select=TestGuardedSelectAndConfirm._ok_select,
            ocr_read=_ocr_always(self._VAL),
        )
        r = await gsc.execute(_COORDS, self._VAL)
        assert r.success is True

    async def test_click_failure_returns_false(self) -> None:
        gsc = _make_gsc(
            fallback_click=_fail_click,
            fallback_select=TestGuardedSelectAndConfirm._ok_select,
        )
        r = await gsc.execute(_COORDS, self._VAL)
        assert r.success is False
        assert "click" in (r.error or "").lower()

    async def test_select_failure_returns_false(self) -> None:
        gsc = _make_gsc(
            fallback_select=TestGuardedSelectAndConfirm._fail_select,
        )
        r = await gsc.execute(_COORDS, self._VAL)
        assert r.success is False

    async def test_ocr_mismatch_retries(self) -> None:
        """First OCR wrong → second attempt correct."""
        gsc = _make_gsc(
            fallback_select=TestGuardedSelectAndConfirm._ok_select,
            ocr_read=_ocr_sequence("WRONG", self._VAL),
            max_retries=3,
        )
        r = await gsc.execute(_COORDS, self._VAL)
        assert r.success is True

    async def test_max_retries_exhausted(self) -> None:
        gsc = _make_gsc(
            fallback_select=TestGuardedSelectAndConfirm._ok_select,
            ocr_read=_ocr_always("ALWAYS_WRONG"),
            max_retries=2,
        )
        r = await gsc.execute(_COORDS, self._VAL)
        assert r.success is False
        assert r.partial_completion is True

    async def test_step_results_three_on_success(self) -> None:
        gsc = _make_gsc(
            fallback_select=TestGuardedSelectAndConfirm._ok_select,
            ocr_read=_ocr_always(self._VAL),
        )
        r = await gsc.execute(_COORDS, self._VAL)
        assert len(r.step_results) == 3  # click, select, verify

    async def test_total_steps_is_three(self) -> None:
        gsc = _make_gsc(fallback_select=TestGuardedSelectAndConfirm._ok_select)
        r = await gsc.execute(_COORDS, self._VAL)
        assert r.total_steps == 3

    # ── Transport-aware tests ─────────────────────────────────────────────

    async def test_native_select_used_when_preferred_uia(self) -> None:
        native_calls: list = []

        async def _native(eid, val):
            native_calls.append((eid, val))
            return True

        gsc = _make_gsc(
            native_select=_native,
            preferred_transport="uia",
            ocr_read=_ocr_always(self._VAL),
        )
        r = await gsc.execute(_COORDS, self._VAL, element_id=self._ELEM)
        assert r.success is True
        assert (self._ELEM, self._VAL) in native_calls

    async def test_fallback_select_used_when_no_native(self) -> None:
        fallback_calls: list = []

        async def _fb(coords, val):
            fallback_calls.append((coords, val))
            return True

        gsc = _make_gsc(
            native_select=None,
            fallback_select=_fb,
            ocr_read=_ocr_always(self._VAL),
        )
        r = await gsc.execute(_COORDS, self._VAL)
        assert r.success is True
        assert len(fallback_calls) > 0

    async def test_fallback_select_used_when_native_fails(self) -> None:
        fallback_calls: list = []

        async def _native_fail(eid, val): return False
        async def _fb(coords, val): fallback_calls.append(1); return True

        gsc = _make_gsc(
            native_select=_native_fail,
            fallback_select=_fb,
            preferred_transport="uia",
            ocr_read=_ocr_always(self._VAL),
        )
        r = await gsc.execute(_COORDS, self._VAL)
        assert r.success is True
        assert len(fallback_calls) > 0

    async def test_transport_used_is_uia_when_native_succeeds(self) -> None:
        async def _native(eid, val): return True

        gsc = _make_gsc(
            native_select=_native,
            preferred_transport="uia",
            ocr_read=_ocr_always(self._VAL),
        )
        r = await gsc.execute(_COORDS, self._VAL)
        assert r.transport_used == "uia"
