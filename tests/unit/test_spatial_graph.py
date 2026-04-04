"""
tests/unit/test_spatial_graph.py
Unit tests for nexus/perception/spatial_graph.py — Faz 28.

Sections:
  1.  SpatialRelation / SpatialNode value objects
  2.  RelationType enum completeness
  3.  Geometry helpers (_contains, _bbox_distance, _overlap_confidence)
  4.  SpatialGraph construction
  5.  SpatialGraph — max_node pruning
  6.  find_by_text — exact (substring) mode
  7.  find_by_text — fuzzy mode
  8.  find_by_type
  9.  find_by_affordance
  10. find_best_target
  11. LABELED_BY relation
  12. CONTAINS / CONTAINED_BY relations
  13. ABOVE / BELOW relations
  14. ADJACENT_LEFT / ADJACENT_RIGHT relations
  15. get_context_around
  16. to_summary_dict — JSON-serializable
  17. Error: mismatched element / label lengths
"""
from __future__ import annotations

import json
import math
import uuid
from typing import Any

import pytest

from nexus.core.types import ElementId, Rect
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.matcher.matcher import Affordance, SemanticLabel
from nexus.perception.spatial_graph import (
    RelationType,
    SpatialGraph,
    SpatialNode,
    SpatialRelation,
    _bbox_distance,
    _contains,
    _overlap_confidence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _el(
    x: int,
    y: int,
    w: int = 80,
    h: int = 30,
    el_type: ElementType = ElementType.BUTTON,
) -> UIElement:
    return UIElement(
        id=ElementId(str(uuid.uuid4())),
        element_type=el_type,
        bounding_box=Rect(x, y, w, h),
        confidence=0.9,
        is_visible=True,
        is_occluded=False,
        occlusion_ratio=0.0,
        z_order_estimate=0,
    )


def _sem(
    el: UIElement,
    label: str = "Button",
    affordance: Affordance = Affordance.CLICKABLE,
    confidence: float = 0.9,
    is_destructive: bool = False,
) -> SemanticLabel:
    return SemanticLabel(
        element_id=el.id,
        primary_label=label,
        secondary_labels=(el.element_type.name.capitalize(),),
        confidence=confidence,
        affordance=affordance,
        is_destructive=is_destructive,
    )


def _graph(
    specs: list[tuple[UIElement, SemanticLabel, str]],
    max_nodes: int = 500,
) -> SpatialGraph:
    els = [s[0] for s in specs]
    sems = [s[1] for s in specs]
    texts = {s[0].id: s[2] for s in specs}
    return SpatialGraph(els, sems, texts, max_nodes=max_nodes)


# ---------------------------------------------------------------------------
# Section 1 — SpatialRelation / SpatialNode value objects
# ---------------------------------------------------------------------------


class TestValueObjects:
    def test_spatial_relation_frozen(self):
        sr = SpatialRelation(
            relation_type=RelationType.ABOVE,
            target_id=ElementId("x"),
            confidence=0.9,
            distance=10.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            sr.confidence = 0.5  # type: ignore[misc]

    def test_spatial_node_id_property(self):
        el = _el(0, 0)
        sem = _sem(el)
        node = SpatialNode(element=el, semantic=sem, text="hello")
        assert node.id == el.id

    def test_spatial_node_mutable_relations(self):
        el = _el(0, 0)
        node = SpatialNode(element=el, semantic=_sem(el), text="")
        node.relations.append(
            SpatialRelation(RelationType.ABOVE, ElementId("y"), 0.8, 5.0)
        )
        assert len(node.relations) == 1


# ---------------------------------------------------------------------------
# Section 2 — RelationType enum completeness
# ---------------------------------------------------------------------------


class TestRelationTypeEnum:
    _REQUIRED = {
        "CONTAINS", "CONTAINED_BY", "ADJACENT_LEFT", "ADJACENT_RIGHT",
        "ABOVE", "BELOW", "LABELED_BY", "LABELS", "OVERLAPS", "PART_OF",
    }

    def test_all_required_members(self):
        assert self._REQUIRED <= {m.name for m in RelationType}


# ---------------------------------------------------------------------------
# Section 3 — Geometry helpers
# ---------------------------------------------------------------------------


class TestGeometryHelpers:
    def test_contains_full(self):
        outer = Rect(0, 0, 200, 100)
        inner = Rect(10, 10, 50, 30)
        assert _contains(outer, inner) is True

    def test_contains_false_partial(self):
        a = Rect(0, 0, 50, 50)
        b = Rect(30, 30, 50, 50)
        assert _contains(a, b) is False

    def test_bbox_distance_same(self):
        r = Rect(0, 0, 10, 10)
        assert _bbox_distance(r, r) == pytest.approx(0.0)

    def test_bbox_distance_horizontal(self):
        a = Rect(0, 0, 10, 10)   # centre (5, 5)
        b = Rect(15, 0, 10, 10)  # centre (20, 5)
        assert _bbox_distance(a, b) == pytest.approx(15.0)

    def test_overlap_confidence_identical(self):
        r = Rect(0, 0, 10, 10)
        assert _overlap_confidence(r, r) == pytest.approx(1.0)

    def test_overlap_confidence_no_overlap(self):
        a = Rect(0, 0, 10, 10)
        b = Rect(20, 0, 10, 10)
        assert _overlap_confidence(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Section 4 — SpatialGraph construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_graph(self):
        g = SpatialGraph([], [], {})
        assert g.nodes == []

    def test_single_node(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el), "hello")])
        assert len(g.nodes) == 1

    def test_node_text_set(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el), "Kaydet")])
        node = g.get_node(el.id)
        assert node is not None
        assert node.text == "Kaydet"

    def test_missing_text_empty_string(self):
        el = _el(0, 0)
        els = [el]
        sems = [_sem(el)]
        g = SpatialGraph(els, sems, {})  # no text provided
        node = g.get_node(el.id)
        assert node is not None
        assert node.text == ""

    def test_mismatched_lengths_raises(self):
        el = _el(0, 0)
        with pytest.raises(ValueError, match="equal length"):
            SpatialGraph([el], [], {})


# ---------------------------------------------------------------------------
# Section 5 — max_node pruning
# ---------------------------------------------------------------------------


class TestMaxNodePruning:
    def test_prune_to_max_nodes(self):
        # Create 10 elements with varying confidence
        specs = []
        for i in range(10):
            el = _el(i * 10, 0)
            sem = _sem(el, confidence=i * 0.1)  # 0.0, 0.1, ..., 0.9
            specs.append((el, sem, f"text_{i}"))
        g = _graph(specs, max_nodes=5)
        assert len(g.nodes) == 5

    def test_pruned_keeps_highest_confidence(self):
        specs = []
        for i in range(6):
            el = _el(i * 10, 0)
            conf = 0.1 * (i + 1)  # 0.1 to 0.6
            sem = _sem(el, label=f"node_{i}", confidence=conf)
            specs.append((el, sem, f"text_{i}"))
        g = _graph(specs, max_nodes=3)
        # Should keep the 3 highest confidence nodes (conf=0.4, 0.5, 0.6)
        kept_confs = sorted([n.semantic.confidence for n in g.nodes], reverse=True)
        assert kept_confs[0] == pytest.approx(0.6)
        assert kept_confs[1] == pytest.approx(0.5)
        assert kept_confs[2] == pytest.approx(0.4)

    def test_within_limit_no_pruning(self):
        specs = [(e := _el(i * 10, 0), _sem(e), "") for i in range(3)]
        g = _graph(specs, max_nodes=10)
        assert len(g.nodes) == 3


# ---------------------------------------------------------------------------
# Section 6 — find_by_text (exact / substring)
# ---------------------------------------------------------------------------


class TestFindByTextExact:
    def test_exact_match_found(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el, "Kaydet"), "Kaydet")])
        results = g.find_by_text("Kaydet")
        assert len(results) == 1
        assert results[0].id == el.id

    def test_substring_match(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el), "Kaydet butonu")])
        results = g.find_by_text("kaydet")  # case-insensitive substring
        assert any(r.id == el.id for r in results)

    def test_label_match(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el, label="Kaydet"), "")])
        results = g.find_by_text("Kaydet")
        assert any(r.id == el.id for r in results)

    def test_no_match_returns_empty(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el), "Tamam")])
        assert g.find_by_text("xyz") == []

    def test_empty_query_returns_empty(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el), "anything")])
        assert g.find_by_text("") == []


# ---------------------------------------------------------------------------
# Section 7 — find_by_text (fuzzy)
# ---------------------------------------------------------------------------


class TestFindByTextFuzzy:
    def test_fuzzy_near_match(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el, label="Kaydet"), "Kaydet")])
        # "Kaydt" is close enough to "Kaydet"
        results = g.find_by_text("Kaydt", fuzzy=True)
        assert any(r.id == el.id for r in results)

    def test_fuzzy_exact_still_found(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el), "Submit")])
        results = g.find_by_text("Submit", fuzzy=True)
        assert any(r.id == el.id for r in results)

    def test_fuzzy_very_different_not_found(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el), "A")])
        results = g.find_by_text("ZZZZZZZZZ", fuzzy=True)
        assert results == []

    def test_fuzzy_empty_query_returns_empty(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el), "hello")])
        assert g.find_by_text("", fuzzy=True) == []


# ---------------------------------------------------------------------------
# Section 8 — find_by_type
# ---------------------------------------------------------------------------


class TestFindByType:
    def test_finds_correct_type(self):
        btn = _el(0, 0, el_type=ElementType.BUTTON)
        inp = _el(100, 0, el_type=ElementType.INPUT)
        g = _graph([
            (btn, _sem(btn), ""),
            (inp, _sem(inp, affordance=Affordance.TYPEABLE), ""),
        ])
        buttons = g.find_by_type(ElementType.BUTTON)
        assert len(buttons) == 1
        assert buttons[0].id == btn.id

    def test_no_match_returns_empty(self):
        el = _el(0, 0, el_type=ElementType.BUTTON)
        g = _graph([(el, _sem(el), "")])
        assert g.find_by_type(ElementType.DIALOG) == []

    def test_multiple_same_type(self):
        els = [_el(i * 100, 0, el_type=ElementType.BUTTON) for i in range(3)]
        specs = [(e, _sem(e), "") for e in els]
        g = _graph(specs)
        assert len(g.find_by_type(ElementType.BUTTON)) == 3


# ---------------------------------------------------------------------------
# Section 9 — find_by_affordance
# ---------------------------------------------------------------------------


class TestFindByAffordance:
    def test_finds_clickable(self):
        btn = _el(0, 0, el_type=ElementType.BUTTON)
        inp = _el(100, 0, el_type=ElementType.INPUT)
        g = _graph([
            (btn, _sem(btn, affordance=Affordance.CLICKABLE), ""),
            (inp, _sem(inp, affordance=Affordance.TYPEABLE), ""),
        ])
        clickables = g.find_by_affordance(Affordance.CLICKABLE)
        assert len(clickables) == 1
        assert clickables[0].id == btn.id

    def test_typeable_found(self):
        inp = _el(0, 0, el_type=ElementType.INPUT)
        g = _graph([(inp, _sem(inp, affordance=Affordance.TYPEABLE), "")])
        assert len(g.find_by_affordance(Affordance.TYPEABLE)) == 1

    def test_not_found_returns_empty(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el, affordance=Affordance.CLICKABLE), "")])
        assert g.find_by_affordance(Affordance.SCROLLABLE) == []


# ---------------------------------------------------------------------------
# Section 10 — find_best_target
# ---------------------------------------------------------------------------


class TestFindBestTarget:
    def test_returns_best_text_match(self):
        save = _el(0, 0)
        cancel = _el(100, 0)
        g = _graph([
            (save,   _sem(save,   label="Kaydet"), "Kaydet"),
            (cancel, _sem(cancel, label="İptal"),  "İptal"),
        ])
        result = g.find_best_target("Kaydet")
        assert result is not None
        assert result.id == save.id

    def test_best_target_prefers_clickable(self):
        lbl = _el(0, 0, el_type=ElementType.LABEL)
        btn = _el(0, 50)
        # Both have same text but btn is CLICKABLE
        g = _graph([
            (lbl, _sem(lbl, label="Save", affordance=Affordance.READ_ONLY), "Save"),
            (btn, _sem(btn, label="Save", affordance=Affordance.CLICKABLE), "Save"),
        ])
        result = g.find_best_target("Save")
        assert result is not None
        assert result.id == btn.id

    def test_empty_description_returns_none(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el), "hello")])
        assert g.find_best_target("") is None

    def test_empty_graph_returns_none(self):
        g = SpatialGraph([], [], {})
        assert g.find_best_target("anything") is None

    def test_finds_by_label_when_no_text(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el, label="Sil"), "")])
        result = g.find_best_target("Sil")
        assert result is not None
        assert result.id == el.id


# ---------------------------------------------------------------------------
# Section 11 — LABELED_BY relation
# ---------------------------------------------------------------------------


class TestLabeledByRelation:
    def test_label_adjacent_to_input_creates_labeled_by(self):
        """
        A LABEL element just to the left of an INPUT should produce a
        LABELED_BY relation on the INPUT and a LABELS relation on the LABEL.
        """
        label_el = _el(10, 10, w=60, h=20, el_type=ElementType.LABEL)
        input_el = _el(80, 10, w=150, h=24, el_type=ElementType.INPUT)
        label_sem = _sem(label_el, "E-posta", Affordance.READ_ONLY)
        input_sem = _sem(input_el, "Input", Affordance.TYPEABLE)
        g = _graph([
            (label_el, label_sem, "E-posta"),
            (input_el, input_sem, ""),
        ])
        input_node = g.get_node(input_el.id)
        assert input_node is not None
        rel_types = {r.relation_type for r in input_node.relations}
        assert RelationType.LABELED_BY in rel_types

    def test_labels_relation_on_label_node(self):
        label_el = _el(10, 10, w=60, h=20, el_type=ElementType.LABEL)
        input_el = _el(80, 10, w=150, h=24, el_type=ElementType.INPUT)
        g = _graph([
            (label_el, _sem(label_el, affordance=Affordance.READ_ONLY), "Name"),
            (input_el, _sem(input_el, affordance=Affordance.TYPEABLE), ""),
        ])
        label_node = g.get_node(label_el.id)
        assert label_node is not None
        rel_types = {r.relation_type for r in label_node.relations}
        assert RelationType.LABELS in rel_types

    def test_far_label_no_labeled_by(self):
        label_el = _el(10, 10, w=60, h=20, el_type=ElementType.LABEL)
        input_el = _el(500, 10, w=150, h=24, el_type=ElementType.INPUT)
        g = _graph([
            (label_el, _sem(label_el, affordance=Affordance.READ_ONLY), "Far"),
            (input_el, _sem(input_el, affordance=Affordance.TYPEABLE), ""),
        ])
        input_node = g.get_node(input_el.id)
        assert input_node is not None
        rel_types = {r.relation_type for r in input_node.relations}
        assert RelationType.LABELED_BY not in rel_types


# ---------------------------------------------------------------------------
# Section 12 — CONTAINS / CONTAINED_BY
# ---------------------------------------------------------------------------


class TestContainmentRelations:
    def test_panel_contains_button(self):
        panel = _el(0, 0, w=300, h=200, el_type=ElementType.PANEL)
        btn = _el(50, 50, w=80, h=30)
        g = _graph([
            (panel, _sem(panel), ""),
            (btn, _sem(btn), ""),
        ])
        panel_node = g.get_node(panel.id)
        assert panel_node is not None
        rel_types = {r.relation_type for r in panel_node.relations}
        assert RelationType.CONTAINS in rel_types

    def test_button_contained_by_panel(self):
        panel = _el(0, 0, w=300, h=200, el_type=ElementType.PANEL)
        btn = _el(50, 50, w=80, h=30)
        g = _graph([
            (panel, _sem(panel), ""),
            (btn, _sem(btn), ""),
        ])
        btn_node = g.get_node(btn.id)
        assert btn_node is not None
        rel_types = {r.relation_type for r in btn_node.relations}
        assert RelationType.CONTAINED_BY in rel_types


# ---------------------------------------------------------------------------
# Section 13 — ABOVE / BELOW
# ---------------------------------------------------------------------------


class TestAboveBelowRelations:
    def test_above_relation(self):
        top = _el(50, 10, w=80, h=30)
        bot = _el(50, 80, w=80, h=30)
        g = _graph([
            (top, _sem(top), ""),
            (bot, _sem(bot), ""),
        ])
        top_node = g.get_node(top.id)
        assert top_node is not None
        rel_types = {r.relation_type for r in top_node.relations}
        assert RelationType.ABOVE in rel_types

    def test_below_relation(self):
        top = _el(50, 10, w=80, h=30)
        bot = _el(50, 80, w=80, h=30)
        g = _graph([
            (top, _sem(top), ""),
            (bot, _sem(bot), ""),
        ])
        bot_node = g.get_node(bot.id)
        assert bot_node is not None
        rel_types = {r.relation_type for r in bot_node.relations}
        assert RelationType.BELOW in rel_types


# ---------------------------------------------------------------------------
# Section 14 — ADJACENT_LEFT / ADJACENT_RIGHT
# ---------------------------------------------------------------------------


class TestAdjacentRelations:
    def test_adjacent_right(self):
        left_el = _el(10, 50, w=80, h=30)
        right_el = _el(100, 50, w=80, h=30)  # gap = 10px
        g = _graph([
            (left_el, _sem(left_el), ""),
            (right_el, _sem(right_el), ""),
        ])
        left_node = g.get_node(left_el.id)
        assert left_node is not None
        rel_types = {r.relation_type for r in left_node.relations}
        assert RelationType.ADJACENT_RIGHT in rel_types

    def test_adjacent_left(self):
        left_el = _el(10, 50, w=80, h=30)
        right_el = _el(100, 50, w=80, h=30)
        g = _graph([
            (left_el, _sem(left_el), ""),
            (right_el, _sem(right_el), ""),
        ])
        right_node = g.get_node(right_el.id)
        assert right_node is not None
        rel_types = {r.relation_type for r in right_node.relations}
        assert RelationType.ADJACENT_LEFT in rel_types


# ---------------------------------------------------------------------------
# Section 15 — get_context_around
# ---------------------------------------------------------------------------


class TestGetContextAround:
    def test_returns_nearby_nodes(self):
        anchor = _el(100, 100, w=40, h=20)
        near = _el(120, 100, w=40, h=20)    # close
        far = _el(600, 600, w=40, h=20)     # far
        g = _graph([
            (anchor, _sem(anchor), ""),
            (near,   _sem(near),   ""),
            (far,    _sem(far),    ""),
        ])
        context = g.get_context_around(anchor.id, radius=100)
        ids = {n.id for n in context}
        assert near.id in ids
        assert far.id not in ids
        assert anchor.id not in ids  # anchor excluded

    def test_unknown_id_returns_empty(self):
        g = SpatialGraph([], [], {})
        result = g.get_context_around(ElementId("nonexistent"), 50)
        assert result == []

    def test_radius_zero_returns_empty(self):
        anchor = _el(0, 0)
        other = _el(1, 0)  # 1px away
        g = _graph([
            (anchor, _sem(anchor), ""),
            (other,  _sem(other),  ""),
        ])
        result = g.get_context_around(anchor.id, radius=0)
        assert result == []


# ---------------------------------------------------------------------------
# Section 16 — to_summary_dict — JSON-serializable
# ---------------------------------------------------------------------------


class TestToSummaryDict:
    def test_json_serializable(self):
        specs = [
            (_el(i * 10, 0), _sem(_el(i * 10, 0), f"node_{i}"), f"text_{i}")
            for i in range(3)
        ]
        # Use shared el references
        els = [_el(i * 10, 0) for i in range(3)]
        sems = [_sem(e, f"node_{i}") for i, e in enumerate(els)]
        texts = {e.id: f"text_{i}" for i, e in enumerate(els)}
        g = SpatialGraph(els, sems, texts)
        d = g.to_summary_dict()
        # Must not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_summary_contains_node_count(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el), "hi")])
        d = g.to_summary_dict()
        assert d["node_count"] == 1

    def test_summary_nodes_list(self):
        el = _el(0, 0)
        g = _graph([(el, _sem(el, label="Kaydet"), "Kaydet")])
        d = g.to_summary_dict()
        assert "nodes" in d
        assert len(d["nodes"]) == 1
        node_d = d["nodes"][0]
        assert node_d["label"] == "Kaydet"
        assert node_d["text"] == "Kaydet"
        assert "bbox" in node_d
        assert isinstance(node_d["bbox"], list)
        assert len(node_d["bbox"]) == 4

    def test_summary_text_truncated_to_80(self):
        el = _el(0, 0)
        long_text = "x" * 200
        g = _graph([(el, _sem(el), long_text)])
        d = g.to_summary_dict()
        assert len(d["nodes"][0]["text"]) <= 80

    def test_empty_graph_summary(self):
        g = SpatialGraph([], [], {})
        d = g.to_summary_dict()
        assert d["node_count"] == 0
        assert d["nodes"] == []
        # Must be JSON-serializable
        json.dumps(d)


# ---------------------------------------------------------------------------
# Section 17 — Error: mismatched lengths
# ---------------------------------------------------------------------------


class TestMismatchedLengths:
    def test_more_elements_than_labels(self):
        el1 = _el(0, 0)
        el2 = _el(100, 0)
        with pytest.raises(ValueError):
            SpatialGraph([el1, el2], [_sem(el1)], {})

    def test_more_labels_than_elements(self):
        el1 = _el(0, 0)
        with pytest.raises(ValueError):
            SpatialGraph([el1], [_sem(el1), _sem(_el(100, 0))], {})
