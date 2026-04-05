"""
nexus/action/macroactions.py
Macro-actions — composite, transport-aware, retry-safe multi-step operations.

Each macro-action wraps several primitive operations (click, type, hotkey,
OCR-read) into a single audited unit.  All Windows / UIA / DOM primitives are
injectable so every macro is fully unit-testable without real UI.

Transport-aware execution
--------------------------
Every macro accepts separate *native* and *fallback* callable pairs:

  _native_click_fn  — UIA-invoke or DOM-click (preferred when available)
  _fallback_click_fn — mouse transport (used when native is absent or fails)

The macro tries the native path first.  On failure (returns False or raises)
it falls back to the OS-level transport and records which transport was used.

MacroActionResult
-----------------
Extends ActionResult with:
  steps_completed — number of steps that ran to completion (not necessarily
                    all successful; see step_results for per-step detail).
  total_steps     — expected number of steps for a single success path.
  step_results    — individual ActionResult for every primitive step run,
                    including retry attempts.

Macros
------
SafeFieldReplace     — click → Ctrl+A → type → OCR-verify → Ctrl+Z+retry
SafeRowWrite         — row-identity lock → SafeFieldReplace per cell
                       → wrong-cell guard → write audit trail
VerifyAndSubmit      — read all fields → compare → correct → submit → verify
GuardedSelectAndConfirm — dropdown click → select → OCR-verify
"""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from nexus.action.registry import ActionResult
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Injectable callable type aliases
# ---------------------------------------------------------------------------

# Native transport (UIA / DOM) — receives optional element_id
_NativeClickFn = Callable[[str | None], Awaitable[bool]]
_NativeSelectFn = Callable[[str | None, str], Awaitable[bool]]

# OS-level transport (mouse / keyboard)
_FallbackClickFn = Callable[[tuple[int, int]], Awaitable[bool]]
_FallbackSelectFn = Callable[[tuple[int, int], str], Awaitable[bool]]

# Keyboard
_TypeFn = Callable[[str], Awaitable[bool]]
_HotkeyFn = Callable[[list[str]], Awaitable[bool]]
_UndoFn = Callable[[], Awaitable[bool]]

# OCR / read
_OcrReadFn = Callable[[tuple[int, int] | None], Awaitable[str | None]]

# ---------------------------------------------------------------------------
# MacroActionResult
# ---------------------------------------------------------------------------

_NATIVE_TRANSPORTS: frozenset[str] = frozenset({"uia", "dom"})


@dataclass
class MacroActionResult(ActionResult):
    """
    Result of a composite macro-action.

    Attributes
    ----------
    steps_completed:
        Number of primitive steps that were executed (includes retries).
    total_steps:
        Planned number of steps for the single-attempt success path.
    step_results:
        Individual :class:`~nexus.action.registry.ActionResult` for every
        primitive step, in execution order.
    """

    steps_completed: int = 0
    total_steps: int = 0
    step_results: list[ActionResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SafeFieldReplace
# ---------------------------------------------------------------------------


class SafeFieldReplace:
    """
    Replace the value of a single UI field safely.

    Steps per attempt:
      1. Click the field (native → mouse fallback).
      2. Ctrl+A — select all existing content.
      3. Type *new_value* via keyboard transport.
      4. OCR-read the field and compare to *new_value*.
         ─ If match → success.
         ─ If mismatch → Ctrl+Z (undo) and retry up to *max_retries* times.

    Parameters
    ----------
    _native_click_fn:
        Async ``(element_id: str | None) -> bool``.
        Used when *preferred_transport* is a native transport and the
        callable is provided.  On False, falls back to mouse.
    _fallback_click_fn:
        Async ``(coords: tuple[int, int]) -> bool``.  Mouse transport.
    _type_fn:
        Async ``(text: str) -> bool``.  Keyboard transport.
    _hotkey_fn:
        Async ``(keys: list[str]) -> bool``.  For Ctrl+A, Ctrl+Z.
    _ocr_read_fn:
        Async ``(coords: tuple[int, int] | None) -> str | None``.
        When None, verification is skipped (always assumed correct).
    _undo_fn:
        Async ``() -> bool``.  Undo shortcut (Ctrl+Z).
        When None, ``_hotkey_fn(["ctrl", "z"])`` is used as fallback.
    max_retries:
        Maximum type-and-verify attempts (default 3).
    preferred_transport:
        Caller hint for which transport to prefer.
    """

    def __init__(
        self,
        *,
        _native_click_fn: _NativeClickFn | None = None,
        _fallback_click_fn: _FallbackClickFn,
        _type_fn: _TypeFn,
        _hotkey_fn: _HotkeyFn,
        _ocr_read_fn: _OcrReadFn | None = None,
        _undo_fn: _UndoFn | None = None,
        max_retries: int = 3,
        preferred_transport: str | None = None,
    ) -> None:
        self._native_click = _native_click_fn
        self._fallback_click = _fallback_click_fn
        self._type = _type_fn
        self._hotkey = _hotkey_fn
        self._ocr_read = _ocr_read_fn
        self._undo = _undo_fn
        self._max_retries = max(1, max_retries)
        self._preferred_transport = preferred_transport
        # planned steps per single attempt (click, select-all, type, verify)
        self._planned_steps = 4

    async def execute(
        self,
        coordinates: tuple[int, int],
        new_value: str,
        element_id: str | None = None,
    ) -> MacroActionResult:
        """
        Execute a safe field-replace at *coordinates*.

        Parameters
        ----------
        coordinates:
            Logical screen (x, y) of the field.
        new_value:
            The value to write.
        element_id:
            Optional element ID passed to the native click fn.
        """
        t0 = time.monotonic()
        step_results: list[ActionResult] = []

        for attempt in range(self._max_retries):
            # ── Step 1: Click ─────────────────────────────────────────
            click_ok, click_transport = await self._transport_click(
                coordinates, element_id
            )
            step_results.append(
                ActionResult(success=click_ok, transport_used=click_transport)
            )
            if not click_ok:
                return _make_macro(
                    success=False,
                    error="Click failed on field",
                    steps_completed=len(step_results),
                    total_steps=self._planned_steps,
                    step_results=step_results,
                    t0=t0,
                )

            # ── Step 2: Ctrl+A ────────────────────────────────────────
            select_ok = await self._hotkey(["ctrl", "a"])
            step_results.append(ActionResult(success=select_ok))

            # ── Step 3: Type new value ────────────────────────────────
            type_ok = await self._type(new_value)
            step_results.append(
                ActionResult(success=type_ok, transport_used="keyboard")
            )
            if not type_ok:
                return _make_macro(
                    success=False,
                    partial_completion=True,
                    error="Type failed during field replace",
                    steps_completed=len(step_results),
                    total_steps=self._planned_steps,
                    step_results=step_results,
                    t0=t0,
                )

            # ── Step 4: OCR verify ────────────────────────────────────
            if self._ocr_read is None:
                # No verifier — trust the type was correct
                step_results.append(ActionResult(success=True))
                _log.debug("safe_field_replace_ok_no_verify", value=new_value)
                return _make_macro(
                    success=True,
                    steps_completed=len(step_results),
                    total_steps=self._planned_steps,
                    step_results=step_results,
                    transport_used=click_transport,
                    t0=t0,
                )

            actual = await self._ocr_read(coordinates)
            match = actual == new_value
            step_results.append(ActionResult(success=match))

            if match:
                _log.debug(
                    "safe_field_replace_ok", value=new_value, attempt=attempt + 1
                )
                return _make_macro(
                    success=True,
                    steps_completed=len(step_results),
                    total_steps=self._planned_steps,
                    step_results=step_results,
                    transport_used=click_transport,
                    t0=t0,
                )

            # Mismatch — undo and prepare for retry
            _log.debug(
                "safe_field_replace_mismatch",
                expected=new_value,
                actual=actual,
                attempt=attempt + 1,
                max_retries=self._max_retries,
            )
            if attempt < self._max_retries - 1:
                await self._do_undo()

        return _make_macro(
            success=False,
            partial_completion=True,
            error=(
                f"Field value mismatch after {self._max_retries} "
                f"attempt(s): expected {new_value!r}"
            ),
            steps_completed=len(step_results),
            total_steps=self._planned_steps,
            step_results=step_results,
            t0=t0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _transport_click(
        self,
        coords: tuple[int, int],
        element_id: str | None,
    ) -> tuple[bool, str]:
        """Try native click; fall back to mouse.  Returns (ok, transport_name)."""
        if (
            self._preferred_transport in _NATIVE_TRANSPORTS
            and self._native_click is not None
        ):
            try:
                ok = await self._native_click(element_id)
            except Exception as exc:
                _log.debug("native_click_failed", error=str(exc))
                ok = False
            if ok:
                return True, self._preferred_transport or "uia"

        ok = await self._fallback_click(coords)
        return ok, "mouse"

    async def _do_undo(self) -> None:
        if self._undo is not None:
            await self._undo()
        else:
            await self._hotkey(["ctrl", "z"])


# ---------------------------------------------------------------------------
# SafeRowWrite
# ---------------------------------------------------------------------------


class SafeRowWrite:
    """
    Write multiple cells in a single spreadsheet row safely.

    Workflow
    --------
    1. **Row-identity lock** — OCR the identity cell (*identity_coords*) and
       store the anchor value.  If it doesn't match *expected_identity*, abort
       immediately (wrong-row guard).
    2. **Per-cell SafeFieldReplace** — for every ``(coords, value)`` pair.
    3. **Wrong-cell guard** — before each cell write, re-check the identity
       cell.  If the anchor has changed (e.g. scroll/selection drift), abort.
    4. **Write audit trail** — accumulated per-step results in ``step_results``.

    Parameters
    ----------
    field_replace:
        Pre-configured :class:`SafeFieldReplace` used for each cell.
    _ocr_read_fn:
        Async ``(coords: tuple[int, int] | None) -> str | None``.
        Used for identity-lock reads.
    identity_verify_retries:
        How many times to re-read the identity cell before declaring a
        wrong-cell condition (allows for transient OCR noise).
    """

    def __init__(
        self,
        *,
        field_replace: SafeFieldReplace,
        _ocr_read_fn: _OcrReadFn,
        identity_verify_retries: int = 1,
    ) -> None:
        self._field_replace = field_replace
        self._ocr_read = _ocr_read_fn
        self._identity_retries = max(1, identity_verify_retries)

    async def execute(
        self,
        cells: list[tuple[tuple[int, int], str]],
        *,
        identity_coords: tuple[int, int],
        expected_identity: str,
    ) -> MacroActionResult:
        """
        Write *cells* to a single row.

        Parameters
        ----------
        cells:
            List of ``(coordinates, value)`` pairs — one per cell to write.
        identity_coords:
            Screen coordinates of the identity cell (e.g. the first column).
        expected_identity:
            The OCR text that must be present in *identity_coords* before and
            between cell writes.
        """
        t0 = time.monotonic()
        step_results: list[ActionResult] = []
        total_steps = len(cells)

        # ── Initial identity lock ─────────────────────────────────────────
        if not await self._check_identity(identity_coords, expected_identity):
            return _make_macro(
                success=False,
                error=(
                    f"Row identity mismatch: expected {expected_identity!r} "
                    "at identity cell (initial check)."
                ),
                steps_completed=0,
                total_steps=total_steps,
                step_results=step_results,
                t0=t0,
            )

        # ── Per-cell write ────────────────────────────────────────────────
        for idx, (coords, value) in enumerate(cells):
            # Wrong-cell guard: re-verify identity before every cell write
            if idx > 0 and not await self._check_identity(
                identity_coords, expected_identity
            ):
                return _make_macro(
                    success=False,
                    partial_completion=True,
                    error=(
                        f"Wrong-cell guard triggered at cell index {idx}: "
                        "row identity changed during write."
                    ),
                    steps_completed=idx,
                    total_steps=total_steps,
                    step_results=step_results,
                    t0=t0,
                )

            cell_result = await self._field_replace.execute(coords, value)
            step_results.append(cell_result)

            if not cell_result.success:
                return _make_macro(
                    success=False,
                    partial_completion=True,
                    error=(
                        f"Cell {idx} write failed: "
                        f"{cell_result.error or 'unknown error'}"
                    ),
                    steps_completed=idx,
                    total_steps=total_steps,
                    step_results=step_results,
                    t0=t0,
                )

        _log.debug(
            "safe_row_write_ok",
            cells=len(cells),
            identity=expected_identity,
        )
        return _make_macro(
            success=True,
            steps_completed=len(cells),
            total_steps=total_steps,
            step_results=step_results,
            t0=t0,
        )

    async def _check_identity(
        self,
        coords: tuple[int, int],
        expected: str,
    ) -> bool:
        """Read the identity cell and compare to *expected*."""
        for _ in range(self._identity_retries):
            actual = await self._ocr_read(coords)
            if actual == expected:
                return True
        _log.debug(
            "identity_check_failed",
            expected=expected,
            actual=actual,  # type: ignore[possibly-undefined]
        )
        return False


# ---------------------------------------------------------------------------
# VerifyAndSubmit
# ---------------------------------------------------------------------------


class VerifyAndSubmit:
    """
    Verify all form fields, correct any mismatches, then submit.

    Steps
    -----
    1. OCR-read each field.
    2. Compare to the expected value.
    3. If mismatch → :class:`SafeFieldReplace` to correct.
    4. Click the submit button.
    5. OCR-read to confirm submission success.

    Parameters
    ----------
    field_replace:
        Pre-configured :class:`SafeFieldReplace` for step 3.
    _ocr_read_fn:
        Async ``(coords: tuple[int, int] | None) -> str | None``.
    _click_fn:
        Async ``(coords: tuple[int, int]) -> bool``.  Used to click submit.
    _submit_verify_fn:
        Optional async ``() -> bool``.  Called after submit to verify success.
        When None, the macro assumes submit succeeded if the click returned True.
    """

    def __init__(
        self,
        *,
        field_replace: SafeFieldReplace,
        _ocr_read_fn: _OcrReadFn,
        _click_fn: _FallbackClickFn,
        _submit_verify_fn: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        self._field_replace = field_replace
        self._ocr_read = _ocr_read_fn
        self._click = _click_fn
        self._submit_verify = _submit_verify_fn

    async def execute(
        self,
        fields: list[tuple[tuple[int, int], str]],
        submit_coords: tuple[int, int],
    ) -> MacroActionResult:
        """
        Verify *fields*, correct any mismatches, then click *submit_coords*.

        Parameters
        ----------
        fields:
            List of ``(coordinates, expected_value)`` pairs.
        submit_coords:
            Coordinates of the submit / OK button.
        """
        t0 = time.monotonic()
        step_results: list[ActionResult] = []
        # total: one step per field (read+verify or correct) + 1 submit + 1 confirm
        total_steps = len(fields) + 2

        # ── Phase 1: Verify and correct each field ────────────────────────
        for idx, (coords, expected) in enumerate(fields):
            actual = await self._ocr_read(coords)
            match = actual == expected
            step_results.append(ActionResult(success=match))

            if not match:
                _log.debug(
                    "verify_and_submit_mismatch",
                    field_idx=idx,
                    expected=expected,
                    actual=actual,
                )
                correction = await self._field_replace.execute(coords, expected)
                step_results.append(correction)
                if not correction.success:
                    return _make_macro(
                        success=False,
                        partial_completion=True,
                        error=(
                            f"Field {idx} correction failed: "
                            f"{correction.error or 'unknown'}"
                        ),
                        steps_completed=idx,
                        total_steps=total_steps,
                        step_results=step_results,
                        t0=t0,
                    )

        # ── Phase 2: Submit ───────────────────────────────────────────────
        submit_ok = await self._click(submit_coords)
        step_results.append(
            ActionResult(success=submit_ok, transport_used="mouse")
        )
        if not submit_ok:
            return _make_macro(
                success=False,
                partial_completion=True,
                error="Submit click failed",
                steps_completed=len(fields),
                total_steps=total_steps,
                step_results=step_results,
                t0=t0,
            )

        # ── Phase 3: Verify submission ────────────────────────────────────
        if self._submit_verify is not None:
            verified = await self._submit_verify()
        else:
            verified = True
        step_results.append(ActionResult(success=verified))

        if not verified:
            return _make_macro(
                success=False,
                partial_completion=True,
                error="Submit verification failed",
                steps_completed=len(fields) + 1,
                total_steps=total_steps,
                step_results=step_results,
                t0=t0,
            )

        _log.debug("verify_and_submit_ok", fields=len(fields))
        return _make_macro(
            success=True,
            steps_completed=total_steps,
            total_steps=total_steps,
            step_results=step_results,
            t0=t0,
        )


# ---------------------------------------------------------------------------
# GuardedSelectAndConfirm
# ---------------------------------------------------------------------------


class GuardedSelectAndConfirm:
    """
    Select an option from a dropdown or list and verify the selection.

    Steps
    -----
    1. Click the dropdown (native → mouse fallback).
    2. Select *value* via native SelectionItem or fallback (click/type).
    3. OCR-verify the visible selection equals *value*.
    4. If wrong → retry up to *max_retries* times.

    Parameters
    ----------
    _native_select_fn:
        Async ``(element_id: str | None, value: str) -> bool``.
        Uses UIA SelectionItem pattern when available.
    _fallback_click_fn:
        Async ``(coords: tuple[int, int]) -> bool``.
    _fallback_select_fn:
        Async ``(coords: tuple[int, int], value: str) -> bool``.
        Fallback selection (e.g. type the option text + Enter).
    _ocr_read_fn:
        Async ``(coords: tuple[int, int] | None) -> str | None``.
    max_retries:
        Maximum select-and-verify attempts.
    preferred_transport:
        ``"uia"`` or ``"dom"`` to use the native select path first.
    """

    def __init__(
        self,
        *,
        _native_select_fn: _NativeSelectFn | None = None,
        _fallback_click_fn: _FallbackClickFn,
        _fallback_select_fn: _FallbackSelectFn | None = None,
        _ocr_read_fn: _OcrReadFn | None = None,
        max_retries: int = 3,
        preferred_transport: str | None = None,
    ) -> None:
        self._native_select = _native_select_fn
        self._fallback_click = _fallback_click_fn
        self._fallback_select = _fallback_select_fn
        self._ocr_read = _ocr_read_fn
        self._max_retries = max(1, max_retries)
        self._preferred_transport = preferred_transport
        self._planned_steps = 3  # click, select, verify

    async def execute(
        self,
        coordinates: tuple[int, int],
        value: str,
        element_id: str | None = None,
    ) -> MacroActionResult:
        """
        Select *value* at *coordinates* and verify the result.

        Parameters
        ----------
        coordinates:
            Logical screen coordinates of the dropdown / select element.
        value:
            The option string to select.
        element_id:
            Optional element ID for native UIA selection.
        """
        t0 = time.monotonic()
        step_results: list[ActionResult] = []

        for attempt in range(self._max_retries):
            # ── Step 1: Click dropdown ────────────────────────────────
            click_ok = await self._fallback_click(coordinates)
            step_results.append(
                ActionResult(success=click_ok, transport_used="mouse")
            )
            if not click_ok:
                return _make_macro(
                    success=False,
                    error="Dropdown click failed",
                    steps_completed=len(step_results),
                    total_steps=self._planned_steps,
                    step_results=step_results,
                    t0=t0,
                )

            # ── Step 2: Select value ──────────────────────────────────
            select_ok, select_transport = await self._transport_select(
                coordinates, element_id, value
            )
            step_results.append(
                ActionResult(success=select_ok, transport_used=select_transport)
            )
            if not select_ok:
                return _make_macro(
                    success=False,
                    error=f"Select failed for value {value!r}",
                    steps_completed=len(step_results),
                    total_steps=self._planned_steps,
                    step_results=step_results,
                    t0=t0,
                )

            # ── Step 3: Verify ────────────────────────────────────────
            if self._ocr_read is None:
                step_results.append(ActionResult(success=True))
                return _make_macro(
                    success=True,
                    steps_completed=len(step_results),
                    total_steps=self._planned_steps,
                    step_results=step_results,
                    transport_used=select_transport,
                    t0=t0,
                )

            actual = await self._ocr_read(coordinates)
            match = actual == value
            step_results.append(ActionResult(success=match))

            if match:
                _log.debug(
                    "guarded_select_ok", value=value, attempt=attempt + 1
                )
                return _make_macro(
                    success=True,
                    steps_completed=len(step_results),
                    total_steps=self._planned_steps,
                    step_results=step_results,
                    transport_used=select_transport,
                    t0=t0,
                )

            _log.debug(
                "guarded_select_mismatch",
                expected=value,
                actual=actual,
                attempt=attempt + 1,
            )

        return _make_macro(
            success=False,
            partial_completion=True,
            error=(
                f"Select verification failed after {self._max_retries} "
                f"attempt(s): expected {value!r}"
            ),
            steps_completed=len(step_results),
            total_steps=self._planned_steps,
            step_results=step_results,
            t0=t0,
        )

    async def _transport_select(
        self,
        coords: tuple[int, int],
        element_id: str | None,
        value: str,
    ) -> tuple[bool, str]:
        """Try native select; fall back to click/type."""
        if (
            self._preferred_transport in _NATIVE_TRANSPORTS
            and self._native_select is not None
        ):
            try:
                ok = await self._native_select(element_id, value)
            except Exception as exc:
                _log.debug("native_select_failed", error=str(exc))
                ok = False
            if ok:
                return True, self._preferred_transport or "uia"

        if self._fallback_select is not None:
            ok = await self._fallback_select(coords, value)
            return ok, "mouse"

        return False, "mouse"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_macro(
    *,
    success: bool,
    steps_completed: int,
    total_steps: int,
    step_results: list[ActionResult],
    t0: float,
    error: str | None = None,
    partial_completion: bool = False,
    transport_used: str | None = None,
) -> MacroActionResult:
    """Build a MacroActionResult, computing duration from *t0*."""
    duration_ms = (time.monotonic() - t0) * 1000
    return MacroActionResult(
        success=success,
        duration_ms=round(duration_ms, 3),
        error=error,
        partial_completion=partial_completion,
        steps_completed=steps_completed,
        total_steps=total_steps,
        step_results=list(step_results),
        transport_used=transport_used,
    )
