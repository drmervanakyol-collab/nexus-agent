"""
nexus/skills/pdf/extractor.py
PDF field and table extractor — structured data from DocumentContent.

PDFExtractor
------------
Extracts labelled fields and tabular data from a DocumentContent object
produced by PDFReader (or FileAdapter directly).

extract_field(field_name, content) -> str | None
  Scans every page of *content* for a pattern matching *field_name*
  followed by a colon and its value.  Recognised patterns (case-
  insensitive field name):

    "Field Name: value"    — inline colon-separated
    "Field Name\nvalue"    — label on one line, value on the next

  Returns the stripped value string on the first match, or None.

extract_table(content, table_index) -> list[list[str]]
  Primary path — structured tables:
    When ``content.tables`` is non-empty and *table_index* is in range,
    return ``content.tables[table_index]`` directly.

  Secondary path — text heuristic:
    When no structured tables are available, scan the combined page text
    for lines that look like table rows (separated by two or more spaces,
    or by tab characters).  Each such line becomes a row; cells are the
    space/tab-delimited tokens.  Returns all parsed rows as a single
    ``list[list[str]]`` (index 0 only; further indexes return ``[]``).

  Returns an empty list when the index is out of range or no table-like
  content is found.
"""
from __future__ import annotations

import re

from nexus.infra.logger import get_logger
from nexus.source.file.adapter import DocumentContent

_log = get_logger(__name__)

# Regex patterns for field extraction
# Pattern A: "Field Name: value" on a single line (value to end-of-line)
_INLINE_PATTERN = re.compile(
    r"^(?P<label>.+?)\s*:\s*(?P<value>.+)$",
    re.MULTILINE,
)

# Minimum number of spaces between tokens to count as a table column separator
_TABLE_MIN_GAP: int = 2
_TABLE_RE = re.compile(r" {2,}|\t")   # 2+ spaces or a tab


class PDFExtractor:
    """
    Extracts structured data (fields and tables) from DocumentContent.

    No UIA, keyboard, or file I/O is performed — extraction is a pure
    computation over text content.
    """

    def extract_field(
        self,
        field_name: str,
        content: DocumentContent,
    ) -> str | None:
        """
        Find the value associated with *field_name* in *content*.

        Parameters
        ----------
        field_name:
            The label to search for (e.g. ``"Invoice Number"``).
            Matching is case-insensitive.
        content:
            DocumentContent whose pages are scanned.

        Returns
        -------
        The trimmed value string on the first match, or None.
        """
        needle = field_name.strip().lower()
        full_text = "\n".join(content.pages)

        # Pattern A: inline colon — "Field Name: value"
        for match in _INLINE_PATTERN.finditer(full_text):
            if match.group("label").strip().lower() == needle:
                value = match.group("value").strip()
                _log.debug(
                    "extract_field_inline",
                    field=field_name,
                    value=value[:80],
                )
                return value

        # Pattern B: label then newline then value
        lines = full_text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().lower() == needle and i + 1 < len(lines):
                value = lines[i + 1].strip()
                if value:
                    _log.debug(
                        "extract_field_next_line",
                        field=field_name,
                        value=value[:80],
                    )
                    return value

        _log.debug("extract_field_not_found", field=field_name)
        return None

    def extract_table(
        self,
        content: DocumentContent,
        table_index: int = 0,
    ) -> list[list[str]]:
        """
        Return a table from *content* at *table_index*.

        Parameters
        ----------
        content:
            DocumentContent produced by FileAdapter or PDFReader.
        table_index:
            Zero-based index of the table to return.

        Returns
        -------
        A list of rows, each a list of cell strings.  Empty list when
        the index is out of range or no table is found.
        """
        # Primary path: structured tables from FileAdapter (XLSX / tagged PDF)
        if content.tables and table_index < len(content.tables):
            table = content.tables[table_index]
            _log.debug(
                "extract_table_structured",
                index=table_index,
                rows=len(table),
            )
            return [list(row) for row in table]

        # Secondary path: heuristic from text content (index 0 only)
        if table_index != 0:
            _log.debug(
                "extract_table_index_out_of_range",
                index=table_index,
            )
            return []

        rows = _parse_text_table("\n".join(content.pages))
        _log.debug(
            "extract_table_heuristic",
            rows=len(rows),
        )
        return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_text_table(text: str) -> list[list[str]]:
    """
    Heuristically extract table rows from plain text.

    A line is treated as a table row when it contains at least two cells
    separated by two or more spaces or a tab character.
    """
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p.strip() for p in _TABLE_RE.split(stripped) if p.strip()]
        if len(parts) >= 2:  # noqa: PLR2004
            rows.append(parts)
    return rows
