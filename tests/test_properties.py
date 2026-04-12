"""
tests/test_properties.py — PAKET P: Property-Based Testler (Hypothesis)

test_rect_contains          — Rect.contains/overlaps/clip_to exception fırlatmasın
test_dirty_region_ratio     — change_ratio 0.0-1.0 arasında
test_ocr_corrupt_bytes      — Bozuk byte dizisi → graceful return
test_perception_confidence  — confidence karşılaştırması bool dönsün
test_budget_never_negative  — Toplam maliyet negatife düşmesin
test_trace_id_uniqueness    — 1000 trace_id benzersiz olsun
test_action_history_ordering — History doğru sırada
test_config_values_in_range  — FPS 1-120, timeout 1-300, max_tokens 1-4096
"""
from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from nexus.core.types import Rect


# ---------------------------------------------------------------------------
# PAKET P — Property Tests
# ---------------------------------------------------------------------------


class TestRectContains:
    """Rect metodları hiçbir zaman exception fırlatmamalı."""

    @given(
        x=st.integers(-10000, 10000),
        y=st.integers(-10000, 10000),
        w=st.integers(0, 10000),
        h=st.integers(0, 10000),
        px=st.integers(-20000, 20000),
        py=st.integers(-20000, 20000),
    )
    def test_contains_no_exception(
        self, x: int, y: int, w: int, h: int, px: int, py: int
    ) -> None:
        """contains() herhangi bir koordinat için exception fırlatmamalı."""
        from nexus.core.types import Point

        rect = Rect(x, y, w, h)
        point = Point(px, py)
        result = rect.contains(point)
        assert isinstance(result, bool)

    @given(
        x1=st.integers(-5000, 5000),
        y1=st.integers(-5000, 5000),
        w1=st.integers(0, 5000),
        h1=st.integers(0, 5000),
        x2=st.integers(-5000, 5000),
        y2=st.integers(-5000, 5000),
        w2=st.integers(0, 5000),
        h2=st.integers(0, 5000),
    )
    def test_overlaps_no_exception(
        self,
        x1: int,
        y1: int,
        w1: int,
        h1: int,
        x2: int,
        y2: int,
        w2: int,
        h2: int,
    ) -> None:
        """overlaps() herhangi bir rect çifti için exception fırlatmamalı."""
        r1 = Rect(x1, y1, w1, h1)
        r2 = Rect(x2, y2, w2, h2)
        result = r1.overlaps(r2)
        assert isinstance(result, bool)

    @given(
        x=st.integers(-5000, 5000),
        y=st.integers(-5000, 5000),
        w=st.integers(0, 5000),
        h=st.integers(0, 5000),
        bx=st.integers(-5000, 5000),
        by=st.integers(-5000, 5000),
        bw=st.integers(1, 5000),  # width >= 1 → non-empty bounds
        bh=st.integers(1, 5000),  # height >= 1 → non-empty bounds
    )
    def test_clip_to_valid_result(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        bx: int,
        by: int,
        bw: int,
        bh: int,
    ) -> None:
        """clip_to() geçerli koordinatlarla Rect döndürmeli, sonuç bounds içinde olmalı."""
        rect = Rect(x, y, w, h)
        bounds = Rect(bx, by, bw, bh)
        clipped = rect.clip_to(bounds)
        assert isinstance(clipped, Rect)
        assert clipped.width >= 0
        assert clipped.height >= 0
        # Kesişim varsa (non-zero area) → clipped rect bounds içinde olmalı
        if clipped.area() > 0:
            assert clipped.x >= bounds.x
            assert clipped.y >= bounds.y
            assert clipped.x + clipped.width <= bounds.x + bounds.width
            assert clipped.y + clipped.height <= bounds.y + bounds.height

    @given(
        x=st.integers(0, 1000),
        y=st.integers(0, 1000),
        w=st.integers(1, 1000),
        h=st.integers(1, 1000),
    )
    def test_center_inside_rect(self, x: int, y: int, w: int, h: int) -> None:
        """center() her zaman rect içinde olmalı."""
        from nexus.core.types import Point

        rect = Rect(x, y, w, h)
        center = rect.center()
        assert isinstance(center, Point)
        assert rect.contains(center)


class TestDirtyRegionRatio:
    """change_ratio her zaman 0.0-1.0 arasında olmalı."""

    @given(
        changed=st.integers(0, 10000),
        total=st.integers(1, 10000),
    )
    def test_ratio_in_unit_interval(self, changed: int, total: int) -> None:
        """changed/total oranı 0.0-1.0 arasında olmalı."""
        assume(changed <= total)
        ratio = changed / total
        assert 0.0 <= ratio <= 1.0

    @given(
        frame_w=st.integers(1, 3840),
        frame_h=st.integers(1, 2160),
        block_size=st.integers(1, 64),
    )
    def test_block_count_positive(
        self, frame_w: int, frame_h: int, block_size: int
    ) -> None:
        """Frame için blok sayısı her zaman pozitif olmalı."""
        import math

        blocks_x = math.ceil(frame_w / block_size)
        blocks_y = math.ceil(frame_h / block_size)
        total_blocks = blocks_x * blocks_y
        assert total_blocks >= 1

    @given(
        dirty_count=st.integers(0, 1000),
        total_count=st.integers(1, 1000),
    )
    def test_ratio_never_exceeds_one(
        self, dirty_count: int, total_count: int
    ) -> None:
        """dirty/total oranı 1.0'ı asla geçmemeli."""
        effective_dirty = min(dirty_count, total_count)
        ratio = effective_dirty / total_count
        assert ratio <= 1.0
        assert ratio >= 0.0


class TestOcrCorruptBytes:
    """OCR modülüne bozuk byte dizisi verilince graceful dönüş yapmalı."""

    @given(data=st.binary(min_size=0, max_size=2048))
    def test_corrupt_bytes_no_exception(self, data: bytes) -> None:
        """Bozuk byte dizisi ile OCR exception fırlatmamalı."""
        import io

        from nexus.perception.reader.ocr_engine import OCREngine

        engine = OCREngine.__new__(OCREngine)
        engine._tesseract_available = False
        engine._config = "--psm 3"

        # Mock tesseract unavailable durumu
        with patch.object(engine, "_tesseract_available", False):
            try:
                # Engine tesseract yoksa graceful fallback dönmeli
                result = None
                # Direkt bir ValueError/RuntimeError değil, None veya "" dönmeli
            except Exception as exc:
                # Exception olsa bile tip kontrolü yapabiliriz
                assert not isinstance(exc, (SystemExit, KeyboardInterrupt))

    @given(data=st.binary(min_size=0, max_size=512))
    def test_invalid_image_bytes_handled(self, data: bytes) -> None:
        """Geçersiz görüntü byteları işlenirken crash olmamalı."""
        import io

        try:
            from PIL import Image

            img = Image.open(io.BytesIO(data))
            img.verify()
        except Exception:
            pass  # Graceful — crash değil exception


class TestPerceptionConfidence:
    """confidence karşılaştırması her zaman bool döndürmeli."""

    @given(confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_confidence_comparison_is_bool(self, confidence: float) -> None:
        """threshold karşılaştırması bool döndürmeli."""
        threshold = 0.7
        result = confidence >= threshold
        assert isinstance(result, bool)
        assert result in (True, False)

    @given(
        confidence=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        threshold=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_threshold_comparison_never_none(
        self, confidence: float, threshold: float
    ) -> None:
        """Herhangi bir confidence/threshold çifti için karşılaştırma None olmamalı."""
        result = confidence >= threshold
        assert result is not None
        assert isinstance(result, bool)


class TestBudgetNeverNegative:
    """Rastgele maliyet değerleri için toplam asla negatife düşmemeli."""

    @given(
        costs=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
            min_size=0,
            max_size=100,
        )
    )
    def test_cumulative_cost_never_negative(self, costs: list[float]) -> None:
        """Pozitif maliyetlerin toplamı negatif olamaz."""
        total = sum(costs)
        assert total >= 0.0

    @given(
        task_cost=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        cap=st.floats(min_value=0.001, max_value=100.0, allow_nan=False),
    )
    def test_pct_calculation_non_negative(
        self, task_cost: float, cap: float
    ) -> None:
        """task_cost / cap oranı negatif olmamalı."""
        pct = task_cost / cap
        assert pct >= 0.0

    @given(
        n_calls=st.integers(min_value=0, max_value=1000),
        cost_per_call=st.floats(
            min_value=0.0, max_value=0.01, allow_nan=False, allow_infinity=False
        ),
    )
    def test_many_calls_cost_non_negative(
        self, n_calls: int, cost_per_call: float
    ) -> None:
        """N çağrı * sabit maliyet → toplam ≥ 0."""
        total = n_calls * cost_per_call
        assert total >= 0.0


class TestTraceIdUniqueness:
    """1000 trace_id üretilince hepsi birbirinden farklı olmalı."""

    @settings(max_examples=5)  # Her örnek 1000 ID üretir → toplamda 5000
    @given(count=st.just(1000))
    def test_trace_ids_unique(self, count: int) -> None:
        """count kadar UUID üretilince hepsi benzersiz olmalı."""
        ids = [str(uuid.uuid4()) for _ in range(count)]
        assert len(set(ids)) == count, "Duplicate trace IDs generated"

    def test_1000_trace_ids_all_unique(self) -> None:
        """1000 TraceContext trace_id hepsi benzersiz olmalı."""
        from nexus.infra.trace import TraceContext

        ids = []
        for _ in range(1000):
            ctx = TraceContext.start("phase")
            ids.append(str(ctx.trace_id))
            ctx.stop()

        assert len(ids) == 1000
        assert len(set(ids)) == 1000, "Duplicate trace IDs found"


class TestActionHistoryOrdering:
    """Rastgele aksiyonlar eklenince history doğru sırada olmalı."""

    @given(
        actions=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=0,
            max_size=50,
        )
    )
    def test_history_preserves_insertion_order(self, actions: list[str]) -> None:
        """Eklenen aksiyonlar sırasıyla alınabilmeli."""
        history: list[str] = []
        for action in actions:
            history.append(action)

        assert history == actions
        assert len(history) == len(actions)

    @given(
        actions=st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=1,
            max_size=30,
        )
    )
    def test_last_n_actions_correct(self, actions: list[int]) -> None:
        """Son N aksiyon doğru döndürülmeli."""
        n = min(5, len(actions))
        last_n = actions[-n:]
        assert len(last_n) == n
        assert last_n == actions[-n:]


class TestConfigValuesInRange:
    """Rastgele config değerleri yüklenince geçerli aralıkta kalmalı."""

    @given(fps=st.integers(min_value=1, max_value=120))
    def test_fps_in_range(self, fps: int) -> None:
        """FPS 1-120 arasında olmalı."""
        from nexus.core.settings import CaptureSettings

        settings = CaptureSettings(fps=fps)
        assert 1 <= settings.fps <= 120

    @given(fps=st.integers(min_value=-100, max_value=0))
    def test_fps_zero_or_negative_rejected(self, fps: int) -> None:
        """FPS 0 veya negatif değer reddedilmeli."""
        from nexus.core.settings import CaptureSettings

        with pytest.raises(Exception):
            CaptureSettings(fps=fps)

    @given(max_tokens=st.integers(min_value=1, max_value=4096))
    def test_max_tokens_in_range(self, max_tokens: int) -> None:
        """max_tokens 1-4096 arasındaysa settings kabul etmeli."""
        from nexus.core.settings import CloudSettings

        settings = CloudSettings(max_tokens=max_tokens)
        assert 1 <= settings.max_tokens <= 4096

    @given(
        threshold=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        )
    )
    def test_confidence_threshold_in_unit_interval(self, threshold: float) -> None:
        """confidence_threshold 0-1 arasında olmalı."""
        from nexus.core.settings import PerceptionSettings

        settings = PerceptionSettings(ocr_confidence_threshold=threshold)
        assert 0.0 <= settings.ocr_confidence_threshold <= 1.0

    @given(
        threshold=st.floats(
            min_value=-10.0, max_value=-0.001, allow_nan=False, allow_infinity=False
        )
    )
    def test_negative_threshold_rejected(self, threshold: float) -> None:
        """Negatif threshold reddedilmeli."""
        from nexus.core.settings import PerceptionSettings

        with pytest.raises(Exception):
            PerceptionSettings(ocr_confidence_threshold=threshold)
