"""
tests/integration/test_block1.py
Blok 1 Integration Tests — Faz 16

TEST 1 — UIA read + native action
TEST 2 — UIA action fail → mouse fallback
TEST 3 — DOM read + native action
TEST 4 — Visual source → direct mouse
TEST 5 — PDF text path (OCR not called)
TEST 6 — Transport audit (3 kayıt)
TEST 7 — Full source chain: UIA None + DOM None → visual

Her test birden fazla katmanın birlikte çalışmasını doğrular.
Real uygulama sınıfları kullanılır; OS/COM/CDP çağrıları inject edilir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nexus.core.settings import NexusSettings
from nexus.core.types import Rect
from nexus.source.resolver import SourcePriorityResolver, SourceResult
from nexus.source.transport.fallback import KeyboardTransport, MouseTransport
from nexus.source.transport.resolver import ActionSpec, TransportResolver


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _settings(*, prefer_native: bool = True) -> NexusSettings:
    return NexusSettings.model_validate(
        {"transport": {"prefer_native_action": prefer_native}}
    )


def _element_with_rect(x: int = 10, y: int = 20, w: int = 100, h: int = 40) -> Any:
    elem = MagicMock()
    elem.bounding_rect = Rect(x, y, w, h)
    return elem


async def _noop_audit(result: Any, spec: Any) -> None:
    pass


# ---------------------------------------------------------------------------
# TEST 1 — UIA read + native action
# ---------------------------------------------------------------------------


class TestUiaReadAndNativeAction:
    """
    SourcePriorityResolver → UIA source → TransportResolver.execute() →
    UIA invoker is called and method_used == 'uia'.
    """

    async def test_uia_source_resolves_and_invoke_is_called(self) -> None:
        # 1. SourcePriorityResolver: UIA probe returns data
        uia_data = {"element_name": "Submit", "control_type": 50000}
        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=lambda ctx: uia_data,
        )
        source_result = resolver.resolve({"intent": "read", "description": "Submit"})

        assert source_result.source_type == "uia"
        assert source_result.data == uia_data

        # 2. TransportResolver: invoke() is called for UIA source + click
        invoke_calls: list[Any] = []

        def uia_invoker(elem: Any) -> bool:
            invoke_calls.append(elem)
            return True

        transport = TransportResolver(
            _settings(),
            _uia_invoker=uia_invoker,
            _audit_writer=_noop_audit,
        )
        element = _element_with_rect()
        transport_result = await transport.execute(
            ActionSpec(action_type="click", task_id="t1"),
            source_result,
            element,
        )

        assert transport_result.method_used == "uia"
        assert transport_result.success is True
        assert transport_result.fallback_used is False
        assert len(invoke_calls) == 1
        assert invoke_calls[0] is element

    async def test_uia_source_confidence_is_highest(self) -> None:
        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=lambda ctx: {"data": True},
        )
        result = resolver.resolve({})
        assert result.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TEST 2 — UIA action fail → mouse fallback
# ---------------------------------------------------------------------------


class TestUiaFailMouseFallback:
    """
    UIA invoker returns False → TransportResolver falls back to mouse.
    fallback_used must be True; method_used must be 'mouse'.
    """

    async def test_uia_invoke_false_triggers_mouse_fallback(self) -> None:
        # UIA source via resolver
        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=lambda ctx: {"element": "btn"},
        )
        source = resolver.resolve({})
        assert source.source_type == "uia"

        # TransportResolver: invoker fails → mouse
        mouse_clicks: list[tuple[int, int]] = []
        mouse = MouseTransport(_click_fn=lambda x, y: mouse_clicks.append((x, y)))

        transport = TransportResolver(
            _settings(),
            _uia_invoker=lambda elem: False,  # always fails
            _mouse_transport=mouse,
            _audit_writer=_noop_audit,
        )
        # Rect(10,20,100,40) → centre (60, 40)
        element = _element_with_rect(x=10, y=20, w=100, h=40)
        result = await transport.execute(
            ActionSpec(action_type="click", task_id="t2"),
            source,
            element,
        )

        assert result.method_used == "mouse"
        assert result.success is True
        assert result.fallback_used is True
        assert mouse_clicks == [(60, 40)]

    async def test_fallback_used_flag_is_true(self) -> None:
        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=lambda ctx: "data",
        )
        source = resolver.resolve({})

        mouse = MouseTransport(_click_fn=lambda x, y: None)
        transport = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: False,
            _mouse_transport=mouse,
            _audit_writer=_noop_audit,
        )
        result = await transport.execute(
            ActionSpec(action_type="click"),
            source,
            _element_with_rect(),
        )
        assert result.fallback_used is True


# ---------------------------------------------------------------------------
# TEST 3 — DOM read + native action
# ---------------------------------------------------------------------------


class TestDomReadAndNativeAction:
    """
    SourcePriorityResolver → DOM source → TransportResolver →
    dom_clicker is called, method_used == 'dom'.
    """

    async def test_dom_source_click_calls_dom_clicker(self) -> None:
        # UIA fails → DOM succeeds
        dom_data = {"tag": "button", "text": "Login"}
        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=lambda ctx: None,
            _dom_probe=lambda ctx: dom_data,
        )
        source = resolver.resolve({"description": "Login button"})
        assert source.source_type == "dom"

        click_calls: list[Any] = []

        async def dom_clicker(elem: Any) -> bool:
            click_calls.append(elem)
            return True

        transport = TransportResolver(
            _settings(),
            _dom_clicker=dom_clicker,
            _audit_writer=_noop_audit,
        )
        element = _element_with_rect()
        result = await transport.execute(
            ActionSpec(action_type="click", task_id="t3"),
            source,
            element,
        )

        assert result.method_used == "dom"
        assert result.success is True
        assert result.fallback_used is False
        assert click_calls[0] is element

    async def test_dom_type_calls_dom_typer(self) -> None:
        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=lambda ctx: None,
            _dom_probe=lambda ctx: {"tag": "input"},
        )
        source = resolver.resolve({})

        typed: list[tuple[Any, str]] = []

        async def dom_typer(elem: Any, text: str) -> bool:
            typed.append((elem, text))
            return True

        transport = TransportResolver(
            _settings(),
            _dom_typer=dom_typer,
            _audit_writer=_noop_audit,
        )
        result = await transport.execute(
            ActionSpec(action_type="type", text="nexus@test.com"),
            source,
            _element_with_rect(),
        )

        assert result.method_used == "dom"
        assert typed[0][1] == "nexus@test.com"


# ---------------------------------------------------------------------------
# TEST 4 — Visual source → direct mouse
# ---------------------------------------------------------------------------


class TestVisualSourceDirectMouse:
    """
    All source probes return None → SourcePriorityResolver falls to visual.
    TransportResolver routes click directly to mouse without any native attempt.
    Native callables must NOT be called.
    """

    async def test_visual_source_uses_mouse_directly(self) -> None:
        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=lambda ctx: None,
            _dom_probe=lambda ctx: None,
            _file_probe=lambda ctx: None,
            _visual_probe=lambda ctx: {"screenshot": "frame_42"},
        )
        source = resolver.resolve({})
        assert source.source_type == "visual"

        invoke_calls: list[Any] = []
        dom_click_calls: list[Any] = []
        mouse_clicks: list[tuple[int, int]] = []
        mouse = MouseTransport(_click_fn=lambda x, y: mouse_clicks.append((x, y)))

        transport = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: invoke_calls.append(e) or True,
            _dom_clicker=lambda e: dom_click_calls.append(e),
            _mouse_transport=mouse,
            _audit_writer=_noop_audit,
        )
        result = await transport.execute(
            ActionSpec(action_type="click", task_id="t4"),
            source,
            _element_with_rect(x=0, y=0, w=200, h=50),
        )

        assert result.method_used == "mouse"
        assert result.fallback_used is False  # went direct, not via fallback
        assert invoke_calls == []            # UIA never touched
        assert dom_click_calls == []         # DOM never touched
        assert len(mouse_clicks) == 1
        assert mouse_clicks[0] == (100, 25)  # centre of (0,0,200,50)

    async def test_visual_source_type_uses_keyboard_directly(self) -> None:
        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=lambda ctx: None,
            _dom_probe=lambda ctx: None,
            _file_probe=lambda ctx: None,
        )
        source = resolver.resolve({})
        assert source.source_type == "visual"

        typed: list[str] = []
        kb = KeyboardTransport(_type_fn=lambda t: typed.append(t))

        transport = TransportResolver(
            _settings(),
            _keyboard_transport=kb,
            _audit_writer=_noop_audit,
        )
        result = await transport.execute(
            ActionSpec(action_type="type", text="direct_keyboard"),
            source,
            None,
        )

        assert result.method_used == "keyboard"
        assert typed == ["direct_keyboard"]


# ---------------------------------------------------------------------------
# TEST 5 — PDF text path
# ---------------------------------------------------------------------------


class TestPdfTextPath:
    """
    FileAdapter with a mock PDF that has text → source_type='pdf_text'.
    OCR reader must never be called when text is present.
    """

    def test_text_pdf_gives_pdf_text_source_type(self) -> None:
        from nexus.source.file.adapter import FileAdapter

        ocr_called: list[Any] = []

        # Mock PDF reader: page with text
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is the extracted text."

        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page, mock_page]

        adapter = FileAdapter(
            _pdf_reader_factory=lambda p: mock_reader,
            _ocr_reader_factory=lambda p: ocr_called.append(p) or [],
        )

        result = adapter.extract("report.pdf")

        assert result is not None
        assert result.source_type == "pdf_text"
        assert len(result.pages) == 2
        assert result.extraction_confidence == pytest.approx(1.0)
        assert ocr_called == []  # OCR never triggered

    def test_text_pdf_pages_contain_extracted_text(self) -> None:
        from nexus.source.file.adapter import FileAdapter

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "  Hello World  "

        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]

        adapter = FileAdapter(_pdf_reader_factory=lambda p: mock_reader)
        result = adapter.extract("doc.pdf")

        assert result is not None
        assert result.pages == ["Hello World"]  # strip() applied

    def test_encrypted_pdf_returns_none_ocr_not_called(self) -> None:
        from nexus.source.file.adapter import FileAdapter

        ocr_called: list[Any] = []
        mock_reader = MagicMock()
        mock_reader.is_encrypted = True

        adapter = FileAdapter(
            _pdf_reader_factory=lambda p: mock_reader,
            _ocr_reader_factory=lambda p: ocr_called.append(p) or [],
        )
        result = adapter.extract("locked.pdf")

        assert result is None
        assert ocr_called == []


# ---------------------------------------------------------------------------
# TEST 6 — Transport audit (3 kayıt)
# ---------------------------------------------------------------------------


class TestTransportAudit:
    """
    Execute 3 actions via TransportResolver (UIA success, DOM success,
    mouse fallback).  Each result is written to the transport_audit table
    via TransportAuditRepository.  Assert 3 rows in the DB.
    """

    @pytest.fixture
    async def db(self, tmp_path: Path):  # type: ignore[type-arg]
        from nexus.infra.database import Database

        d = Database(str(tmp_path / "block1_audit.db"))
        await d.init()
        return d

    async def test_three_executions_produce_three_audit_rows(
        self, db: Any
    ) -> None:
        from nexus.infra.repositories import (
            ActionRepository,
            TaskRepository,
            TransportAuditRepository,
        )

        task_repo = TaskRepository()
        action_repo = ActionRepository()
        audit_repo = TransportAuditRepository()

        # Seed a task + action row so FK constraint is satisfied
        async with db.connection() as conn:
            await task_repo.create(conn, id="blk1-task", goal="blok1 audit test")
            await action_repo.create(
                conn, id="blk1-act", task_id="blk1-task", type="click"
            )

        # Build an audit_writer that writes to the real DB
        async def audit_writer(result: Any, spec: ActionSpec) -> None:
            async with db.connection() as conn:
                await audit_repo.record(
                    conn,
                    task_id=spec.task_id,
                    action_id=spec.action_id,
                    attempted_transport=result.method_used,
                    fallback_used=result.fallback_used,
                    success=result.success,
                    latency_ms=result.latency_ms,
                )
                await conn.commit()

        mouse = MouseTransport(_click_fn=lambda x, y: None)
        kb = KeyboardTransport(_type_fn=lambda t: None)

        # Transport 1: UIA click → success
        t1 = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: True,
            _audit_writer=audit_writer,
        )
        await t1.execute(
            ActionSpec(action_type="click", task_id="blk1-task", action_id="blk1-act"),
            SourceResult("uia", {}, 1.0, 1.0),
            _element_with_rect(),
        )

        # Transport 2: DOM type → success
        async def dom_typer(e: Any, t: str) -> bool:
            return True

        t2 = TransportResolver(
            _settings(),
            _dom_typer=dom_typer,
            _audit_writer=audit_writer,
        )
        await t2.execute(
            ActionSpec(action_type="type", text="hello", task_id="blk1-task", action_id="blk1-act"),
            SourceResult("dom", {}, 0.95, 2.0),
            _element_with_rect(),
        )

        # Transport 3: UIA invoke fails → mouse fallback
        t3 = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: False,
            _mouse_transport=mouse,
            _audit_writer=audit_writer,
        )
        await t3.execute(
            ActionSpec(action_type="click", task_id="blk1-task", action_id="blk1-act"),
            SourceResult("uia", {}, 1.0, 1.5),
            _element_with_rect(),
        )

        # Assert 3 rows in DB
        async with db.connection() as conn:
            rows = await audit_repo.list_for_task(conn, "blk1-task")

        assert len(rows) == 3

        methods = [r.attempted_transport for r in rows]
        assert "uia" in methods
        assert "dom" in methods
        assert "mouse" in methods

        # Third row (mouse fallback) has fallback_used=True
        mouse_row = next(r for r in rows if r.attempted_transport == "mouse")
        assert mouse_row.fallback_used is True
        assert mouse_row.success is True

    async def test_audit_rows_have_positive_latency(self, db: Any) -> None:
        from nexus.infra.repositories import (
            ActionRepository,
            TaskRepository,
            TransportAuditRepository,
        )

        task_repo = TaskRepository()
        action_repo = ActionRepository()
        audit_repo = TransportAuditRepository()

        async with db.connection() as conn:
            await task_repo.create(conn, id="lat-task", goal="latency test")
            await action_repo.create(conn, id="lat-act", task_id="lat-task", type="click")

        async def audit_writer(result: Any, spec: ActionSpec) -> None:
            async with db.connection() as conn:
                await audit_repo.record(
                    conn,
                    task_id=spec.task_id,
                    action_id=spec.action_id,
                    attempted_transport=result.method_used,
                    fallback_used=result.fallback_used,
                    success=result.success,
                    latency_ms=result.latency_ms,
                )
                await conn.commit()

        transport = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: True,
            _audit_writer=audit_writer,
        )
        await transport.execute(
            ActionSpec(action_type="click", task_id="lat-task", action_id="lat-act"),
            SourceResult("uia", {}, 1.0, 1.0),
            _element_with_rect(),
        )

        async with db.connection() as conn:
            rows = await audit_repo.list_for_task(conn, "lat-task")

        assert rows[0].latency_ms >= 0.0


# ---------------------------------------------------------------------------
# TEST 7 — Full source chain: UIA None + DOM None → visual
# ---------------------------------------------------------------------------


class TestFullSourceChain:
    """
    End-to-end: SourcePriorityResolver with all concrete sources failing
    must resolve to visual and return a non-None SourceResult.
    """

    def test_uia_none_dom_none_resolves_to_visual(self) -> None:
        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=lambda ctx: None,
            _dom_probe=lambda ctx: None,
            _file_probe=lambda ctx: None,
            # _visual_probe uses default sentinel
        )
        result = resolver.resolve({"intent": "read", "description": "anything"})

        assert result.source_type == "visual"
        assert result.data is not None
        assert result.confidence == pytest.approx(0.70)

    def test_fallback_chain_order_preserved(self) -> None:
        """UIA fails → DOM tried → DOM fails → file tried → file fails → visual."""
        call_order: list[str] = []

        def uia(ctx: Any) -> Any:
            call_order.append("uia")
            return None

        def dom(ctx: Any) -> Any:
            call_order.append("dom")
            return None

        def file_probe(ctx: Any) -> Any:
            call_order.append("file")
            return None

        def visual(ctx: Any) -> Any:
            call_order.append("visual")
            return {"visual": True}

        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=uia,
            _dom_probe=dom,
            _file_probe=file_probe,
            _visual_probe=visual,
        )
        result = resolver.resolve({})

        assert call_order == ["uia", "dom", "file", "visual"]
        assert result.source_type == "visual"

    def test_prefer_source_changes_attempt_order(self) -> None:
        """prefer_source='dom' → DOM tried first even when UIA would succeed."""
        attempt_order: list[str] = []

        def uia(ctx: Any) -> Any:
            attempt_order.append("uia")
            return "uia_data"  # would succeed

        def dom(ctx: Any) -> Any:
            attempt_order.append("dom")
            return "dom_data"  # also succeeds

        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=uia,
            _dom_probe=dom,
        )
        result = resolver.resolve({}, prefer_source="dom")

        assert result.source_type == "dom"
        assert attempt_order[0] == "dom"
        assert "uia" not in attempt_order  # short-circuited after DOM success

    async def test_visual_source_end_to_end_with_transport(self) -> None:
        """Full chain: resolver → visual → transport → mouse."""
        resolver = SourcePriorityResolver(
            _settings(),
            _uia_probe=lambda ctx: None,
            _dom_probe=lambda ctx: None,
            _file_probe=lambda ctx: None,
        )
        source = resolver.resolve({})
        assert source.source_type == "visual"

        mouse_clicks: list[tuple[int, int]] = []
        mouse = MouseTransport(_click_fn=lambda x, y: mouse_clicks.append((x, y)))

        transport = TransportResolver(
            _settings(),
            _mouse_transport=mouse,
            _audit_writer=_noop_audit,
        )
        result = await transport.execute(
            ActionSpec(action_type="click"),
            source,
            _element_with_rect(x=50, y=100, w=80, h=30),
        )

        assert result.method_used == "mouse"
        assert result.success is True
        # centre of (50,100,80,30) → (90, 115)
        assert mouse_clicks == [(90, 115)]
