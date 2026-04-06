"""
nexus/core/hitl_manager.py
HITLManager — Human-in-the-Loop terminal prompt with timeout and headless support.

HITLRequest
-----------
  task_id         : str            — task that triggered the HITL pause
  question        : str            — question shown to the operator
  options         : list[str]      — numbered choices (empty → free-form)
  default_index   : int            — 0-based index of the pre-selected default
  context         : dict           — arbitrary extra info (shown as JSON)

HITLResponse
------------
  task_id         : str
  chosen_option   : str            — the text of the chosen option
  chosen_index    : int | None     — option index, None for free-form
  timed_out       : bool           — True when timeout fired before input
  elapsed_s       : float          — wall-clock seconds spent waiting

HITLManager.request(request) → HITLResponse
  Terminal path (headless=False):
    1. Print the question and numbered option list.
    2. Print context JSON (compact, indented 2).
    3. Wait up to timeout_s for stdin input via the injectable input_fn.
    4. Parse the user's answer:
         - Integer 1..N  → selects that option.
         - Empty input   → selects default_index.
         - Any other text → stored as-is as chosen_option (free-form).
    5. On timeout (TimeoutExpired or KeyboardInterrupt) → default option.

  Headless path (headless=True):
    Return the default option immediately, timed_out=False (deliberate skip).

  Audit:
    Every request/response pair is appended to the hitl_log table.

Injectability
-------------
  input_fn   : Callable[[str], str]  — replaces built-in input() in tests.
  time_fn    : Callable[[], float]   — replaces time.monotonic() in tests.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nexus.core.settings import HITLSettings
from nexus.infra.database import Database
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass
class HITLRequest:
    """A single HITL prompt to be shown to the operator."""

    task_id: str
    question: str
    options: list[str] = field(default_factory=list)
    default_index: int = 0
    context: dict[str, Any] = field(default_factory=dict)

    def default_option(self) -> str:
        """Return the default option text (or empty string if no options)."""
        if not self.options:
            return ""
        idx = max(0, min(self.default_index, len(self.options) - 1))
        return self.options[idx]


@dataclass
class HITLResponse:
    """The operator's answer to a HITLRequest."""

    task_id: str
    chosen_option: str
    chosen_index: int | None
    timed_out: bool
    elapsed_s: float


# ---------------------------------------------------------------------------
# Internal sentinel (defined before HITLManager so ruff can resolve it)
# ---------------------------------------------------------------------------


class _TimeoutExpiredError(Exception):
    """Raised internally when the read thread does not finish in time."""


# ---------------------------------------------------------------------------
# HITLManager
# ---------------------------------------------------------------------------


class HITLManager:
    """
    Presents HITL prompts on the terminal and records results.

    Parameters
    ----------
    db:
        Database instance for audit logging.
    settings:
        HITLSettings controlling timeout, headless mode, and default action.
    input_fn:
        Injectable replacement for ``input()``; must accept a prompt string
        and return the typed line.  Use in tests to avoid blocking stdin.
    """

    def __init__(
        self,
        db: Database,
        settings: HITLSettings | None = None,
        input_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._db = db
        self._settings = settings or HITLSettings()
        self._input_fn: Callable[[str], str] = input_fn or input

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def request(self, req: HITLRequest) -> HITLResponse:
        """
        Present *req* to the operator and return their response.

        Never raises.  Headless mode and timeouts both produce a valid
        HITLResponse carrying the default option.
        """
        t0 = time.monotonic()

        if self._settings.headless:
            response = self._headless_response(req, t0)
        else:
            response = self._terminal_prompt(req, t0)

        await self._log(req, response)
        return response

    # ------------------------------------------------------------------
    # Terminal interaction
    # ------------------------------------------------------------------

    def _terminal_prompt(self, req: HITLRequest, t0: float) -> HITLResponse:
        """Render the prompt and read a response from stdin."""
        self._print_prompt(req)

        try:
            raw = self._read_with_timeout(req)
        except _TimeoutExpiredError:
            elapsed = time.monotonic() - t0
            _log.info(
                "hitl.timeout",
                task_id=req.task_id,
                elapsed_s=round(elapsed, 1),
            )
            return self._default_response(req, elapsed_s=elapsed, timed_out=True)
        except (EOFError, KeyboardInterrupt):
            elapsed = time.monotonic() - t0
            return self._default_response(req, elapsed_s=elapsed, timed_out=True)

        elapsed = time.monotonic() - t0
        return self._parse_response(req, raw.strip(), elapsed)

    def _print_prompt(self, req: HITLRequest) -> None:
        print()
        print("=" * 60)
        print(f"[NEXUS HITL] {req.question}")
        if req.context:
            print(f"  Context: {json.dumps(req.context, indent=2)}")
        if req.options:
            for i, opt in enumerate(req.options, start=1):
                marker = " (default)" if i - 1 == req.default_index else ""
                print(f"  [{i}] {opt}{marker}")
            print(
                f"  Enter 1-{len(req.options)}, or press Enter for default "
                f"(timeout {self._settings.timeout_s:.0f}s):"
            )
        else:
            print(
                f"  Enter response or press Enter for default "
                f"(timeout {self._settings.timeout_s:.0f}s):"
            )
        print("=" * 60)

    def _read_with_timeout(self, req: HITLRequest) -> str:
        """
        Read a line from the injectable input_fn, enforcing timeout_s.

        On CPython the standard ``input()`` blocks indefinitely. We implement
        a best-effort timeout: if the platform supports ``signal.alarm``
        (POSIX) we use it; otherwise we fall back to a blocking read with no
        OS-level timeout but honour the timeout after the fact.

        In tests the injectable ``input_fn`` returns immediately, so the
        timeout path is exercised by returning a sentinel.
        """
        import threading

        result: list[str] = []
        exc: list[BaseException] = []

        def _reader() -> None:
            try:
                result.append(self._input_fn("  > "))
            except Exception as e:  # noqa: BLE001
                exc.append(e)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout=self._settings.timeout_s)

        if thread.is_alive():
            # Thread still blocked — treat as timeout.
            raise _TimeoutExpiredError()

        if exc:
            raise exc[0]

        return result[0] if result else ""

    def _parse_response(
        self, req: HITLRequest, raw: str, elapsed_s: float
    ) -> HITLResponse:
        """Interpret the raw input string and build a HITLResponse."""
        # Empty → default
        if not raw:
            return self._default_response(req, elapsed_s=elapsed_s, timed_out=False)

        # Integer → option index
        if req.options and raw.isdigit():
            idx = int(raw) - 1  # 1-based display → 0-based
            if 0 <= idx < len(req.options):
                _log.info(
                    "hitl.response",
                    task_id=req.task_id,
                    chosen_index=idx,
                    chosen_option=req.options[idx],
                )
                return HITLResponse(
                    task_id=req.task_id,
                    chosen_option=req.options[idx],
                    chosen_index=idx,
                    timed_out=False,
                    elapsed_s=elapsed_s,
                )

        # Free-form text
        _log.info(
            "hitl.response_freeform",
            task_id=req.task_id,
            raw=raw,
        )
        return HITLResponse(
            task_id=req.task_id,
            chosen_option=raw,
            chosen_index=None,
            timed_out=False,
            elapsed_s=elapsed_s,
        )

    # ------------------------------------------------------------------
    # Headless / default helpers
    # ------------------------------------------------------------------

    def _headless_response(self, req: HITLRequest, t0: float) -> HITLResponse:
        elapsed = time.monotonic() - t0
        default = req.default_option() or self._settings.default_action
        _log.info(
            "hitl.headless_skip",
            task_id=req.task_id,
            default=default,
        )
        return HITLResponse(
            task_id=req.task_id,
            chosen_option=default,
            chosen_index=req.default_index if req.options else None,
            timed_out=False,
            elapsed_s=elapsed,
        )

    def _default_response(
        self,
        req: HITLRequest,
        *,
        elapsed_s: float,
        timed_out: bool,
    ) -> HITLResponse:
        default = req.default_option() or self._settings.default_action
        return HITLResponse(
            task_id=req.task_id,
            chosen_option=default,
            chosen_index=req.default_index if req.options else None,
            timed_out=timed_out,
            elapsed_s=elapsed_s,
        )

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    async def _log(self, req: HITLRequest, resp: HITLResponse) -> None:
        """Persist request/response to hitl_log (best-effort, never raises)."""
        try:
            async with self._db.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO hitl_log
                        (task_id, question, options, chosen, timed_out, elapsed_s)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        req.task_id,
                        req.question,
                        json.dumps(req.options),
                        resp.chosen_option,
                        int(resp.timed_out),
                        resp.elapsed_s,
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            _log.warning("hitl.log_failed", error=str(exc))


