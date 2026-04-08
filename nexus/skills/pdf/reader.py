"""
nexus/skills/pdf/reader.py
PDF reader skill — path-aware document content extraction.

PDFReader.read(file_path, current_frame, *, task_id)
----------------------------------------------------
Selects the extraction path based on what is provided:

PATH 1 — file_path given:
  1. Call FileAdapter.extract(file_path).
     - source_type ``"pdf_text"``  → return directly (deterministic text layer).
     - source_type ``"pdf_ocr"``   → return directly (OCR already applied).
  2. If FileAdapter returns None:
     - Call _is_encrypted_fn(file_path).
     - Encrypted → request operator password via HITLManager.request(),
       then return None (re-opening with password is left to the caller).
     - Not encrypted → extraction failed for another reason → return None.

PATH 2 — current_frame given (no file_path):
  Scroll-and-scan: apply the injectable _ocr_frame_fn to the frame and
  return a DocumentContent with source_type ``"pdf_ocr"`` and a single
  page containing the OCR'd text.

PATH 3 — both None:
  Return None immediately.

Injectable callables
--------------------
_is_encrypted_fn : (path: Path) -> bool
    Check whether a PDF is password-protected without full extraction.
    Default uses pypdf (lazy import).
_ocr_frame_fn : (frame: Frame) -> str
    Extract text from a single screen frame via OCR.
    Default returns an empty string (safe, testable default).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from nexus.capture.frame import Frame
from nexus.core.hitl_manager import HITLManager, HITLRequest
from nexus.infra.logger import get_logger
from nexus.source.file.adapter import DocumentContent, FileAdapter

_log = get_logger(__name__)

_HITL_QUESTION = (
    "The PDF is password-protected. "
    "Please enter the password to continue, or skip."
)
_HITL_OPTIONS = ["Enter password manually", "Skip this file"]


# ---------------------------------------------------------------------------
# Default injectable implementations
# ---------------------------------------------------------------------------


def _default_is_encrypted(path: Path) -> bool:
    """Return True when the PDF at *path* is password-protected."""
    try:
        import pypdf  # noqa: PLC0415

        reader = pypdf.PdfReader(str(path))
        return bool(reader.is_encrypted)
    except Exception as exc:
        _log.debug("is_encrypted_check_failed", path=str(path), error=str(exc))
        return False


def _default_ocr_frame(_frame: Frame) -> str:
    """Stub OCR frame extractor — returns empty string (safe default)."""
    return ""


# ---------------------------------------------------------------------------
# PDFReader
# ---------------------------------------------------------------------------


class PDFReader:
    """
    Path-aware PDF content extractor.

    Parameters
    ----------
    file_adapter:
        FileAdapter for structured PDF/XLSX extraction.
    hitl_manager:
        HITLManager used to request operator action on encrypted PDFs.
    _is_encrypted_fn:
        Sync ``(path: Path) -> bool``.  Default uses pypdf.
    _ocr_frame_fn:
        Sync ``(frame: Frame) -> str``.  Used for scroll-and-scan path.
        Default returns ``""``.
    """

    def __init__(
        self,
        file_adapter: FileAdapter,
        hitl_manager: HITLManager,
        *,
        _is_encrypted_fn: Callable[[Path], bool] | None = None,
        _ocr_frame_fn: Callable[[Frame], str] | None = None,
    ) -> None:
        self._file_adapter = file_adapter
        self._hitl = hitl_manager
        self._is_encrypted: Callable[[Path], bool] = (
            _is_encrypted_fn or _default_is_encrypted
        )
        self._ocr_frame: Callable[[Frame], str] = (
            _ocr_frame_fn or _default_ocr_frame
        )

    async def read(
        self,
        file_path: str | Path | None = None,
        current_frame: Frame | None = None,
        *,
        task_id: str = "",
    ) -> DocumentContent | None:
        """
        Extract PDF content via the appropriate path.

        Parameters
        ----------
        file_path:
            Path to a PDF file.  When provided, FileAdapter is used.
        current_frame:
            Screen frame to OCR when no file_path is available (scroll-
            and-scan path).  Ignored when file_path is provided.
        task_id:
            Task identifier forwarded to HITLManager on encrypted PDFs.

        Returns
        -------
        DocumentContent on success, None on failure or operator skip.
        """
        # ---- PATH 1: file path ------------------------------------------
        if file_path is not None:
            return await self._read_from_file(Path(file_path), task_id)

        # ---- PATH 2: frame OCR ------------------------------------------
        if current_frame is not None:
            return self._read_from_frame(current_frame)

        # ---- PATH 3: nothing provided -----------------------------------
        _log.debug("pdf_reader_no_input")
        return None

    # ------------------------------------------------------------------
    # Internal path implementations
    # ------------------------------------------------------------------

    async def _read_from_file(
        self, path: Path, task_id: str
    ) -> DocumentContent | None:
        content = self._file_adapter.extract(path)

        if content is not None:
            _log.debug(
                "pdf_read_ok",
                path=str(path),
                source_type=content.source_type,
            )
            return content

        # FileAdapter returned None — check for encryption
        if self._is_encrypted(path):
            _log.warning(
                "pdf_encrypted_hitl_requested",
                path=str(path),
                task_id=task_id,
            )
            await self._hitl.request(
                HITLRequest(
                    task_id=task_id or "pdf_read",
                    question=_HITL_QUESTION,
                    options=_HITL_OPTIONS,
                    default_index=1,
                    context={"path": str(path)},
                )
            )
            return None

        _log.debug("pdf_read_failed_non_encrypted", path=str(path))
        return None

    def _read_from_frame(self, frame: Frame) -> DocumentContent:
        text = self._ocr_frame(frame)
        _log.debug("pdf_frame_ocr_ok", text_length=len(text))
        return DocumentContent(
            source_type="pdf_ocr",
            pages=[text],
            tables=[],
            metadata={"frame_sequence": getattr(frame, "sequence_number", 0)},
            extraction_confidence=0.85,
        )
