"""
tests/unit/test_memory.py
Unit tests for nexus/memory — FingerprintStore and CorrectionStore.

Coverage
--------
  UIFingerprint helpers
    - compute_layout_hash: stable, order-independent
    - compute_element_signature: stable, order-independent

  FingerprintStore
    - save() / find_similar() basic round-trip
    - find_similar() bumps seen_count on every hit
    - find_similar() returns None when no layout_hash match
    - record_outcome() success: strategy appended, preferred_transport learned
    - record_outcome() failure: failure_pattern appended, preferred demoted
    - preferred_transport learning: most successful transport wins
    - record_outcome() on unknown fingerprint_id: no crash (warning logged)
    - evict_stale(): rows older than N days removed
    - evict_stale(): recent rows kept
    - LRU eviction: max_rows cap enforced on save()
    - max_rows cap enforced after record_outcome()
    - confidence_boost incremented on success, decremented on failure
    - count() reflects DB state

  CorrectionRecord
    - hash_context(): stable, order-independent
    - hash_context(): different dicts → different hashes

  CorrectionStore
    - save() / find_applicable() basic round-trip
    - find_applicable() with fingerprint_id: specific first, global second
    - find_applicable() no match → empty list
    - increment_apply_count() increments correctly
    - count() reflects DB state
    - transport_correction field preserved
    - new_correction() factory generates unique IDs
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from nexus.infra.database import Database
from nexus.memory import (
    CorrectionRecord,
    CorrectionStore,
    FingerprintStore,
    UIFingerprint,
    new_correction,
    new_fingerprint,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "memory_test.db"))
    await database.init()
    return database


@pytest_asyncio.fixture
async def store(db):
    return FingerprintStore(db, max_rows=10)


@pytest_asyncio.fixture
async def cstore(db):
    return CorrectionStore(db)


# ---------------------------------------------------------------------------
# UIFingerprint helpers
# ---------------------------------------------------------------------------


class TestUIFingerprintHelpers:
    def test_layout_hash_stable(self):
        elements = [
            {"x": 0, "y": 0, "width": 100, "height": 30},
            {"x": 0, "y": 40, "width": 100, "height": 30},
        ]
        h1 = UIFingerprint.compute_layout_hash(elements)
        h2 = UIFingerprint.compute_layout_hash(list(reversed(elements)))
        assert h1 == h2

    def test_layout_hash_different_on_different_positions(self):
        a = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        b = [{"x": 5, "y": 5, "width": 10, "height": 10}]
        assert UIFingerprint.compute_layout_hash(a) != UIFingerprint.compute_layout_hash(b)  # noqa: E501

    def test_element_signature_stable(self):
        elements = [
            {"type": "button", "role": "submit"},
            {"type": "input", "role": "text"},
        ]
        s1 = UIFingerprint.compute_element_signature(elements)
        s2 = UIFingerprint.compute_element_signature(list(reversed(elements)))
        assert s1 == s2

    def test_element_signature_differs_on_type_change(self):
        a = [{"type": "button", "role": "submit"}]
        b = [{"type": "link", "role": "submit"}]
        ha = UIFingerprint.compute_element_signature(a)
        hb = UIFingerprint.compute_element_signature(b)
        assert ha != hb


# ---------------------------------------------------------------------------
# FingerprintStore — basic CRUD
# ---------------------------------------------------------------------------


class TestFingerprintStoreBasic:
    @pytest.mark.asyncio
    async def test_save_and_find_similar(self, store):
        fp = new_fingerprint("hash-A", "sig-A", app_name="MyApp")
        await store.save(fp)
        found = await store.find_similar("hash-A", "sig-A")
        assert found is not None
        assert found.id == fp.id
        assert found.app_name == "MyApp"

    @pytest.mark.asyncio
    async def test_find_similar_returns_none_on_no_match(self, store):
        result = await store.find_similar("no-such-hash", "no-sig")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_similar_bumps_seen_count(self, store):
        fp = new_fingerprint("hash-B", "sig-B")
        await store.save(fp)
        await store.find_similar("hash-B", "sig-B")
        found = await store.find_similar("hash-B", "sig-B")
        assert found is not None
        assert found.seen_count >= 3  # initial=1, +1 each find_similar

    @pytest.mark.asyncio
    async def test_count_reflects_saved_rows(self, store):
        assert await store.count() == 0
        await store.save(new_fingerprint("h1", "s1"))
        await store.save(new_fingerprint("h2", "s2"))
        assert await store.count() == 2


# ---------------------------------------------------------------------------
# FingerprintStore — transport learning
# ---------------------------------------------------------------------------


class TestTransportLearning:
    @pytest.mark.asyncio
    async def test_preferred_transport_learned_on_success(self, store):
        fp = new_fingerprint("hash-T", "sig-T")
        await store.save(fp)

        await store.record_outcome(fp.id, True, "uia", "click")
        await store.record_outcome(fp.id, True, "uia", "click")
        await store.record_outcome(fp.id, True, "dom", "click")

        found = await store.find_similar("hash-T", "sig-T")
        assert found is not None
        assert found.preferred_transport == "uia"  # 2 wins vs 1 win

    @pytest.mark.asyncio
    async def test_preferred_transport_demoted_on_failure(self, store):
        fp = new_fingerprint("hash-U", "sig-U")
        await store.save(fp)

        # Give UIA one win, then fail twice → score goes to 0 → demoted
        await store.record_outcome(fp.id, True, "uia", "click")
        await store.record_outcome(fp.id, False, "uia", "click")
        await store.record_outcome(fp.id, False, "uia", "click")
        # Give DOM one win
        await store.record_outcome(fp.id, True, "dom", "click")

        found = await store.find_similar("hash-U", "sig-U")
        assert found is not None
        # UIA score <= 0, DOM score = 1 → DOM preferred
        assert found.preferred_transport == "dom"

    @pytest.mark.asyncio
    async def test_success_strategies_recorded(self, store):
        fp = new_fingerprint("hash-V", "sig-V")
        await store.save(fp)
        await store.record_outcome(fp.id, True, "uia", "invoke+verify")

        found = await store.find_similar("hash-V", "sig-V")
        assert found is not None
        # strategy should appear in successful_strategies (excluding tally sentinel)
        strategies = [
            s for s in found.successful_strategies
            if not s.startswith("__tally__")
        ]
        assert "invoke+verify" in strategies

    @pytest.mark.asyncio
    async def test_failure_patterns_recorded(self, store):
        fp = new_fingerprint("hash-W", "sig-W")
        await store.save(fp)
        await store.record_outcome(fp.id, False, "mouse", "click+timeout")

        found = await store.find_similar("hash-W", "sig-W")
        assert found is not None
        assert "click+timeout" in found.failure_patterns

    @pytest.mark.asyncio
    async def test_confidence_boost_increases_on_success(self, store):
        fp = new_fingerprint("hash-X", "sig-X")
        await store.save(fp)
        assert fp.confidence_boost == 0.0

        await store.record_outcome(fp.id, True, "uia", "click")
        found = await store.find_similar("hash-X", "sig-X")
        assert found is not None
        assert found.confidence_boost > 0.0

    @pytest.mark.asyncio
    async def test_confidence_boost_decreases_on_failure(self, store):
        fp = new_fingerprint("hash-Y", "sig-Y", preferred_transport="uia")
        # Set initial boost to 0.5 by saving custom FP
        fp2 = UIFingerprint(
            id=fp.id,
            layout_hash=fp.layout_hash,
            element_signature=fp.element_signature,
            confidence_boost=0.5,
        )
        await store.save(fp2)
        await store.record_outcome(fp.id, False, "uia", "click")
        found = await store.find_similar("hash-Y", "sig-Y")
        assert found is not None
        assert found.confidence_boost < 0.5

    @pytest.mark.asyncio
    async def test_record_outcome_unknown_id_no_crash(self, store):
        # Should log a warning and return gracefully
        await store.record_outcome("nonexistent-id", True, "uia", "click")


# ---------------------------------------------------------------------------
# FingerprintStore — eviction
# ---------------------------------------------------------------------------


class TestEviction:
    @pytest.mark.asyncio
    async def test_evict_stale_removes_old_rows(self, db, tmp_path):
        store = FingerprintStore(db, max_rows=1000)
        fp = new_fingerprint("old-hash", "old-sig")
        await store.save(fp)

        # Force last_seen into the past directly
        async with db.connection() as conn:
            await conn.execute(
                "UPDATE ui_fingerprints SET last_seen = '2000-01-01T00:00:00.000Z' WHERE id = ?",  # noqa: E501
                (fp.id,),
            )

        deleted = await store.evict_stale(days=1)
        assert deleted >= 1
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_evict_stale_keeps_recent_rows(self, store):
        fp = new_fingerprint("recent-hash", "recent-sig")
        await store.save(fp)
        deleted = await store.evict_stale(days=30)
        assert deleted == 0
        assert await store.count() >= 1

    @pytest.mark.asyncio
    async def test_lru_eviction_on_max_rows(self, db):
        store = FingerprintStore(db, max_rows=3)

        # Save 5 fingerprints with slightly different timestamps
        for i in range(5):
            fp = new_fingerprint(f"hash-{i}", f"sig-{i}")
            await store.save(fp)

        # Only max_rows should remain
        assert await store.count() <= 3

    @pytest.mark.asyncio
    async def test_max_rows_not_exceeded(self, db):
        store = FingerprintStore(db, max_rows=5)
        for i in range(10):
            await store.save(new_fingerprint(f"h{i}", f"s{i}"))
        assert await store.count() <= 5


# ---------------------------------------------------------------------------
# CorrectionRecord helpers
# ---------------------------------------------------------------------------


class TestCorrectionRecordHelpers:
    def test_hash_context_stable(self):
        ctx = {"action": "click", "element": "submit-btn", "step": 3}
        h1 = CorrectionRecord.hash_context(ctx)
        h2 = CorrectionRecord.hash_context(
            {"step": 3, "element": "submit-btn", "action": "click"}
        )
        assert h1 == h2

    def test_hash_context_differs_on_different_values(self):
        a = {"action": "click"}
        b = {"action": "type"}
        assert CorrectionRecord.hash_context(a) != CorrectionRecord.hash_context(b)

    def test_new_correction_unique_ids(self):
        c1 = new_correction("hash", {"action": "click"}, {"action": "type"})
        c2 = new_correction("hash", {"action": "click"}, {"action": "type"})
        assert c1.id != c2.id


# ---------------------------------------------------------------------------
# CorrectionStore — basic CRUD
# ---------------------------------------------------------------------------


class TestCorrectionStoreBasic:
    @pytest.mark.asyncio
    async def test_save_and_find_applicable(self, cstore):
        ctx_hash = CorrectionRecord.hash_context({"action": "click", "step": 1})
        rec = new_correction(
            ctx_hash,
            wrong_action={"action": "click", "target": "cancel"},
            correct_action={"action": "click", "target": "submit"},
        )
        await cstore.save(rec)

        found = await cstore.find_applicable(ctx_hash)
        assert len(found) == 1
        assert found[0].id == rec.id
        assert found[0].correct_action["target"] == "submit"

    @pytest.mark.asyncio
    async def test_find_applicable_no_match(self, cstore):
        result = await cstore.find_applicable("nonexistent-hash")
        assert result == []

    @pytest.mark.asyncio
    async def test_fingerprint_specific_returned_first(self, db, cstore):
        ctx_hash = CorrectionRecord.hash_context({"step": 2})

        # Create and persist a real fingerprint so the FK constraint is satisfied.
        fp_store = FingerprintStore(db, max_rows=100)
        fp = new_fingerprint("hash-fk", "sig-fk")
        await fp_store.save(fp)
        fp_id = fp.id

        global_rec = new_correction(ctx_hash, {"a": 1}, {"a": 2})
        specific_rec = new_correction(
            ctx_hash, {"a": 1}, {"a": 3}, fingerprint_id=fp_id
        )

        await cstore.save(global_rec)
        await cstore.save(specific_rec)

        found = await cstore.find_applicable(ctx_hash, fingerprint_id=fp_id)
        assert len(found) == 2
        # Fingerprint-specific should come first
        assert found[0].fingerprint_id == fp_id

    @pytest.mark.asyncio
    async def test_increment_apply_count(self, cstore):
        ctx_hash = CorrectionRecord.hash_context({"x": 1})
        rec = new_correction(ctx_hash, {"a": 1}, {"a": 2})
        await cstore.save(rec)

        await cstore.increment_apply_count(rec.id)
        await cstore.increment_apply_count(rec.id)

        found = await cstore.find_applicable(ctx_hash)
        assert found[0].apply_count == 2

    @pytest.mark.asyncio
    async def test_transport_correction_preserved(self, cstore):
        ctx_hash = CorrectionRecord.hash_context({"ctx": "transport"})
        rec = new_correction(
            ctx_hash,
            wrong_action={"action": "click"},
            correct_action={"action": "click"},
            transport_correction="dom",
        )
        await cstore.save(rec)

        found = await cstore.find_applicable(ctx_hash)
        assert found[0].transport_correction == "dom"

    @pytest.mark.asyncio
    async def test_count_reflects_saved_records(self, cstore):
        assert await cstore.count() == 0
        await cstore.save(new_correction("h1", {"a": 1}, {"a": 2}))
        await cstore.save(new_correction("h2", {"b": 1}, {"b": 2}))
        assert await cstore.count() == 2

    @pytest.mark.asyncio
    async def test_find_returns_highest_apply_count_first(self, cstore):
        ctx_hash = CorrectionRecord.hash_context({"order": "test"})
        low = new_correction(ctx_hash, {"a": 1}, {"a": 2})
        high = new_correction(ctx_hash, {"a": 1}, {"a": 3})

        await cstore.save(low)
        await cstore.save(high)
        await cstore.increment_apply_count(high.id)
        await cstore.increment_apply_count(high.id)

        found = await cstore.find_applicable(ctx_hash)
        assert found[0].id == high.id  # highest apply_count first
