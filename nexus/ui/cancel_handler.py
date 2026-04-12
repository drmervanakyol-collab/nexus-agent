"""
nexus/ui/cancel_handler.py
Graceful cancellation handler for Nexus Agent.

CancelHandler
-------------
Intercepts SIGINT (Ctrl+C) and routes it through a two-stage policy:

  First Ctrl+C  → graceful cancel
    Sets a cancel flag on the installed executor, prints a notice,
    and calls _cancel_fn (default: executor.cancel()).

  Second Ctrl+C → force quit
    If a second SIGINT arrives before the graceful cancel completes,
    calls _force_quit_fn (default: sys.exit(1)).

install(executor)
  Register the SIGINT handler.  Stores a reference to the executor so
  the cancel flag can be set.  Safe to call multiple times — each call
  replaces the previous handler.

uninstall()
  Restore the original SIGINT handler.

Injectable callables
--------------------
_cancel_fn    : () -> None   — called on first Ctrl+C
_force_quit_fn: () -> None   — called on second Ctrl+C
_print_fn     : (text: str) -> None
_signal_fn    : (sig, handler) -> None   — signal.signal wrapper
"""
from __future__ import annotations

import signal
import sys
from collections.abc import Callable
from typing import Any

from nexus.infra.logger import get_logger

_log = get_logger(__name__)


class CancelHandler:
    """
    Two-stage Ctrl+C handler.

    Parameters
    ----------
    _cancel_fn:
        Called on the first SIGINT.  Receives no arguments.
        Default: call ``executor.cancel()`` if the executor has that method,
        otherwise set ``executor.cancelled = True``.
    _force_quit_fn:
        Called on the second SIGINT.  Default: ``sys.exit(1)``.
    _print_fn:
        Output callable.  Default: ``print``.
    _signal_fn:
        ``signal.signal`` replacement for testing.
        Signature: ``(sig: int, handler: Callable) -> None``.
    """

    def __init__(
        self,
        *,
        _cancel_fn: Callable[[], None] | None = None,
        _force_quit_fn: Callable[[], None] | None = None,
        _print_fn: Callable[[str], None] | None = None,
        _signal_fn: Callable[[int, Any], Any] | None = None,
    ) -> None:
        self._cancel_fn = _cancel_fn
        self._force_quit_fn = _force_quit_fn or (lambda: sys.exit(1))
        self._print = _print_fn or print
        self._signal = _signal_fn or signal.signal

        self._executor: Any = None
        self._cancel_requested: bool = False
        self._original_handler: Any = signal.SIG_DFL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install(self, executor: Any) -> None:
        """
        Register the two-stage SIGINT handler.

        Parameters
        ----------
        executor:
            The running TaskExecutor (or any object with a
            ``cancel()`` method or a ``cancelled`` attribute).
        """
        self._executor = executor
        self._cancel_requested = False
        self._original_handler = self._signal(signal.SIGINT, self._handle_sigint)
        _log.debug("cancel_handler_installed")

    def uninstall(self) -> None:
        """Restore the original SIGINT handler."""
        self._signal(signal.SIGINT, self._original_handler)
        self._executor = None
        self._cancel_requested = False
        _log.debug("cancel_handler_uninstalled")

    @property
    def cancel_requested(self) -> bool:
        """True after the first Ctrl+C has been received."""
        return self._cancel_requested

    # ------------------------------------------------------------------
    # SIGINT handler
    # ------------------------------------------------------------------

    def _handle_sigint(self, _signum: int, frame: Any) -> None:
        if not self._cancel_requested:
            # First Ctrl+C — graceful cancel
            self._cancel_requested = True
            self._print(
                "\n  [!] İptal isteği alındı — görev durduruluyor..."
                "\n  (Zorla çıkmak için tekrar Ctrl+C)"
            )
            _log.info("graceful_cancel_requested")
            self._do_cancel()
        else:
            # Second Ctrl+C — force quit
            self._print("\n  [✗] Zorla çıkılıyor.")
            _log.warning("force_quit_requested")
            self._force_quit_fn()

    def _do_cancel(self) -> None:
        """Invoke the cancel callable or fall back to executor attribute."""
        if self._cancel_fn is not None:
            self._cancel_fn()
            return
        if self._executor is None:
            return
        if callable(getattr(self._executor, "cancel", None)):
            self._executor.cancel()
        else:
            # Fallback: set a flag attribute
            self._executor.cancelled = True
