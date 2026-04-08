"""
nexus/skills/spreadsheet/safety.py
Spreadsheet safety guard — protects against wrong-cell writes, formula
overwrites, identity mismatches, and Excel autocorrect surprises.

SpreadsheetSafetyGuard
-----------------------
Five guards, all synchronous:

verify_correct_cell(expected_column)
  Read the current cell reference from the Name Box via UIAAdapter.
  Extract the column letters and compare to *expected_column*.
  Returns False when the cell cannot be read (UIA unavailable, empty).

check_formula_protection(cell)
  Return True when *cell* (a cell value string) begins with '=',
  indicating the cell contains a formula that must not be overwritten
  without deliberate operator confirmation.

row_identity_lock(identifier)
  Read the current row number from the Name Box, then read the value of
  column-A for that row via the injectable ``_get_cell_value_fn``.
  Returns True only when that value equals *identifier* — confirming
  the operator is on the expected row.

check_calculation_mode()
  Delegate to the injectable ``_get_calc_mode_fn``.
  Returns ``"automatic"`` or ``"manual"``.
  Default implementation always returns ``"automatic"`` (safe default).

bypass_autocorrect(value)
  Excel silently converts certain text strings: fractions ("1/2" →
  January 2nd), leading-zero numbers ("007" → 7), scientific notation
  ("1e5" → 100000).  Prefixing with an apostrophe forces Excel to treat
  the value as plain text and suppresses autocorrect.
  Returns "'" + value when *value* matches a known autocorrect pattern,
  otherwise returns *value* unchanged.

Injectable callables
--------------------
_get_current_cell_fn : () -> str | None
    Read the Name Box value.  Default uses UIAAdapter.
_get_cell_value_fn   : (cell_ref: str) -> str | None
    Read a cell value by reference (e.g. "A5").  Default uses UIAAdapter.
_get_calc_mode_fn    : () -> str
    Return "automatic" or "manual".  Default returns "automatic".
"""
from __future__ import annotations

import re
from collections.abc import Callable

from nexus.infra.logger import get_logger
from nexus.source.uia.adapter import UIAAdapter

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Excel autocorrect detection patterns
# ---------------------------------------------------------------------------

_AUTOCORRECT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d+/\d+$"),           # fractions  : 1/2 → Jan 2
    re.compile(r"^\d{1,2}-\d{1,2}$"),  # date-like  : 3-5 → Mar 5
    re.compile(r"^\d+[eE][+-]?\d+$"),  # scientific : 1e5 → 100000
    re.compile(r"^0\d+$"),             # lead-zero  : 007 → 7
)

# Automation IDs for the Excel Name Box (tried in order)
_NAME_BOX_AIDS: tuple[str, ...] = ("Box", "NameBox")


# ---------------------------------------------------------------------------
# SpreadsheetSafetyGuard
# ---------------------------------------------------------------------------


class SpreadsheetSafetyGuard:
    """
    Guards for spreadsheet write operations.

    Parameters
    ----------
    uia:
        UIAAdapter used by the default cell-reading implementations.
    _get_current_cell_fn:
        Injectable ``() -> str | None``.  Default reads the Name Box via UIA.
    _get_cell_value_fn:
        Injectable ``(cell_ref: str) -> str | None``.  Default uses UIA.
    _get_calc_mode_fn:
        Injectable ``() -> str``.  Default returns ``"automatic"``.
    """

    def __init__(
        self,
        uia: UIAAdapter,
        *,
        _get_current_cell_fn: Callable[[], str | None] | None = None,
        _get_cell_value_fn: Callable[[str], str | None] | None = None,
        _get_calc_mode_fn: Callable[[], str] | None = None,
    ) -> None:
        self._uia = uia
        self._get_current_cell: Callable[[], str | None] = (
            _get_current_cell_fn or self._uia_get_current_cell
        )
        self._get_cell_value: Callable[[str], str | None] = (
            _get_cell_value_fn or self._uia_get_cell_value
        )
        self._get_calc_mode: Callable[[], str] = (
            _get_calc_mode_fn or (lambda: "automatic")
        )

    # ------------------------------------------------------------------
    # Public guards
    # ------------------------------------------------------------------

    def verify_correct_cell(self, expected_column: str) -> bool:
        """
        Confirm the active cell is in *expected_column*.

        Reads the Name Box (e.g. "B5") and extracts the column letters.
        Returns False when the cell reference cannot be determined.

        Parameters
        ----------
        expected_column:
            Column label to match, case-insensitive (e.g. ``"B"``).
        """
        current = self._get_current_cell()
        if not current:
            _log.debug("verify_correct_cell_no_ref")
            return False
        col = "".join(c for c in current if c.isalpha()).upper()
        expected = expected_column.upper()
        ok = col == expected
        _log.debug(
            "verify_correct_cell",
            current=current,
            col=col,
            expected=expected,
            ok=ok,
        )
        return ok

    def check_formula_protection(self, cell: str) -> bool:
        """
        Return True when *cell* contains a formula (starts with ``'='``).

        Parameters
        ----------
        cell:
            The current content of the cell to inspect (not a reference).
        """
        protected = cell.startswith("=")
        _log.debug("check_formula_protection", cell=cell[:40], protected=protected)
        return protected

    def row_identity_lock(self, identifier: str) -> bool:
        """
        Verify the current row's identifier matches *identifier*.

        Reads the row number from the Name Box, reads column-A of that row
        via ``_get_cell_value_fn``, and compares to *identifier*.

        Returns False when the current cell cannot be determined or when
        the identifier value does not match.

        Parameters
        ----------
        identifier:
            Expected value in column A of the current row.
        """
        current = self._get_current_cell()
        if not current:
            _log.debug("row_identity_lock_no_ref")
            return False
        row_num = "".join(c for c in current if c.isdigit())
        if not row_num:
            _log.debug("row_identity_lock_no_row", current=current)
            return False
        id_cell = f"A{row_num}"
        value = self._get_cell_value(id_cell)
        ok = value == identifier
        _log.debug(
            "row_identity_lock",
            id_cell=id_cell,
            value=value,
            expected=identifier,
            ok=ok,
        )
        return ok

    def check_calculation_mode(self) -> str:
        """
        Return the current Excel calculation mode.

        Returns
        -------
        ``"automatic"`` or ``"manual"``.
        """
        mode = self._get_calc_mode()
        _log.debug("check_calculation_mode", mode=mode)
        return mode

    def bypass_autocorrect(self, value: str) -> str:
        """
        Prefix *value* with ``'`` to suppress Excel autocorrect when needed.

        Patterns that trigger autocorrect and receive the prefix:
          - Fractions  : ``"1/2"``   →  ``"'1/2"``
          - Date-like  : ``"3-5"``   →  ``"'3-5"``
          - Scientific : ``"1e5"``   →  ``"'1e5"``
          - Lead-zero  : ``"007"``   →  ``"'007"``

        Any other value is returned unchanged.

        Parameters
        ----------
        value:
            The string that would be typed into the spreadsheet cell.
        """
        for pattern in _AUTOCORRECT_PATTERNS:
            if pattern.match(value):
                _log.debug("bypass_autocorrect_applied", value=value)
                return "'" + value
        return value

    # ------------------------------------------------------------------
    # Default UIA implementations (used when no injectable is provided)
    # ------------------------------------------------------------------

    def _uia_get_current_cell(self) -> str | None:
        """Read the active cell reference from the Excel Name Box via UIA."""
        for aid in _NAME_BOX_AIDS:
            elem = self._uia.find_by_automation_id(aid)
            if elem is not None:
                return self._uia.get_value(elem)
        elem = self._uia.find_by_name("Name Box")
        if elem is not None:
            return self._uia.get_value(elem)
        _log.debug("uia_name_box_not_found")
        return None

    def _uia_get_cell_value(self, cell_ref: str) -> str | None:
        """Read a cell value by reference via UIA find_by_name."""
        elem = self._uia.find_by_name(cell_ref)
        if elem is None:
            return None
        return self._uia.get_value(elem)
