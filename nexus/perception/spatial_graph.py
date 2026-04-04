"""
nexus/perception/spatial_graph.py
Spatial Graph — a queryable graph of UI elements with spatial relations.

Nodes and relations
-------------------
  SpatialNode     : element + semantic label + OCR text + computed relations
  SpatialRelation : directed edge (type, target_id, confidence, distance)

RelationType
------------
  CONTAINS        — this node's bbox fully encloses another's
  CONTAINED_BY    — this node's bbox is fully inside another's
  ADJACENT_LEFT   — this node is immediately left of another (same row)
  ADJACENT_RIGHT  — this node is immediately right of another (same row)
  ABOVE           — this node is visually above another
  BELOW           — this node is visually below another
  LABELED_BY      — another LABEL element serves as this element's caption
  LABELS          — this LABEL element captions another element
  OVERLAPS        — bboxes intersect without full containment
  PART_OF         — this node is a visual component of a larger container

Construction
------------
  SpatialGraph(elements, semantic_labels, element_texts, max_nodes)
    1. Pair elements with their SemanticLabels (by matching element_id).
    2. If len > max_nodes: prune lowest-confidence nodes (keep top-N).
    3. For every ordered pair (a, b): compute applicable relations.
    4. Index nodes by id, type, affordance for O(1) / O(k) lookup.

Query API
---------
  find_by_text(text, fuzzy)   → list[SpatialNode]   substring or fuzzy ratio
  find_by_type(element_type)  → list[SpatialNode]
  find_by_affordance(aff)     → list[SpatialNode]
  find_best_target(desc)      → SpatialNode | None  scored text + label match
  get_context_around(id, r)   → list[SpatialNode]   nodes within radius r px
  to_summary_dict()           → dict                JSON-serializable summary

Spatial thresholds
------------------
  ADJACENCY_GAP_PX  = 80   max gap (pixels) between bbox edges to be adjacent
  ROW_TOLERANCE_PX  = 20   vertical overlap tolerance for "same row"
  CONTAIN_SLACK_PX  = 4    tolerance for containment check
  FUZZY_THRESHOLD   = 0.6  minimum difflib ratio for fuzzy matches
"""
from __future__ import annotations

import difflib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from nexus.core.types import ElementId
from nexus.infra.logger import get_logger
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.matcher.matcher import Affordance, SemanticLabel

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Spatial thresholds
# ---------------------------------------------------------------------------

_ADJACENCY_GAP_PX: int = 80
_ROW_TOLERANCE_PX: int = 20
_CONTAIN_SLACK_PX: int = 4
_FUZZY_THRESHOLD: float = 0.6
_LABEL_PROXIMITY_PX: int = 120   # max gap for LABELED_BY relation

# ---------------------------------------------------------------------------
# RelationType
# ---------------------------------------------------------------------------


class RelationType(Enum):
    CONTAINS      = auto()
    CONTAINED_BY  = auto()
    ADJACENT_LEFT  = auto()
    ADJACENT_RIGHT = auto()
    ABOVE         = auto()
    BELOW         = auto()
    LABELED_BY    = auto()
    LABELS        = auto()
    OVERLAPS      = auto()
    PART_OF       = auto()


# ---------------------------------------------------------------------------
# SpatialRelation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialRelation:
    """
    A directed spatial relationship from one node to another.

    Attributes
    ----------
    relation_type:
        The geometric or semantic relationship.
    target_id:
        ElementId of the target SpatialNode.
    confidence:
        Geometric confidence of this relation in [0.0, 1.0].
    distance:
        Euclidean distance (pixels) between the two bbox centres.
    """

    relation_type: RelationType
    target_id: ElementId
    confidence: float
    distance: float


# ---------------------------------------------------------------------------
# SpatialNode
# ---------------------------------------------------------------------------


@dataclass
class SpatialNode:
    """
    A single node in the spatial graph.

    Attributes
    ----------
    element:
        The underlying UIElement (bounding box, type, etc.).
    semantic:
        The SemanticLabel assigned by the Matcher.
    text:
        OCR text for this element (empty string if none).
    relations:
        Directed edges from this node to others.
    """

    element: UIElement
    semantic: SemanticLabel
    text: str
    relations: list[SpatialRelation] = field(default_factory=list)

    @property
    def id(self) -> ElementId:
        return self.element.id


# ---------------------------------------------------------------------------
# SpatialGraph
# ---------------------------------------------------------------------------


class SpatialGraph:
    """
    A queryable spatial graph of UI elements.

    Parameters
    ----------
    elements:
        Detected UI elements (from Locator).
    semantic_labels:
        SemanticLabels in the same order as *elements* (from Matcher).
    element_texts:
        OCR text per element ID (from Reader).  Missing IDs → empty string.
    max_nodes:
        Maximum number of nodes to keep.  When ``len(elements) > max_nodes``
        the lowest-confidence elements are pruned first.
        Default: 500 (matches PerceptionSettings.max_graph_nodes).
    """

    def __init__(
        self,
        elements: Sequence[UIElement],
        semantic_labels: Sequence[SemanticLabel],
        element_texts: dict[ElementId, str],
        max_nodes: int = 500,
    ) -> None:
        if len(elements) != len(semantic_labels):
            raise ValueError(
                f"elements ({len(elements)}) and semantic_labels "
                f"({len(semantic_labels)}) must have equal length"
            )

        # Pair and optionally prune
        paired = list(zip(elements, semantic_labels, strict=True))
        if len(paired) > max_nodes:
            _log.warning(
                "spatial_graph_pruning",
                original=len(paired),
                max_nodes=max_nodes,
            )
            paired.sort(key=lambda p: p[1].confidence, reverse=True)
            paired = paired[:max_nodes]

        # Build nodes
        self._nodes: dict[ElementId, SpatialNode] = {}
        for el, sem in paired:
            text = element_texts.get(el.id, "")
            self._nodes[el.id] = SpatialNode(element=el, semantic=sem, text=text)

        # Compute relations
        self._compute_relations()

        # Build secondary indexes for fast queries
        self._by_type: dict[ElementType, list[SpatialNode]] = {}
        self._by_affordance: dict[Affordance, list[SpatialNode]] = {}
        for node in self._nodes.values():
            self._by_type.setdefault(node.element.element_type, []).append(node)
            self._by_affordance.setdefault(node.semantic.affordance, []).append(node)

        _log.debug(
            "spatial_graph_built",
            nodes=len(self._nodes),
            max_nodes=max_nodes,
        )

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> list[SpatialNode]:
        """All nodes in the graph (order not guaranteed)."""
        return list(self._nodes.values())

    def find_by_text(
        self,
        text: str,
        fuzzy: bool = False,
    ) -> list[SpatialNode]:
        """
        Find nodes whose OCR text or primary label matches *text*.

        Parameters
        ----------
        text:
            Query string.
        fuzzy:
            When True use ``difflib`` similarity (>= FUZZY_THRESHOLD).
            When False perform a case-insensitive substring search.
        """
        if not text:
            return []
        needle = text.lower()
        results: list[SpatialNode] = []
        for node in self._nodes.values():
            haystack_text = node.text.lower()
            haystack_label = node.semantic.primary_label.lower()
            if fuzzy:
                text_score = difflib.SequenceMatcher(
                    None, needle, haystack_text
                ).ratio()
                label_score = difflib.SequenceMatcher(
                    None, needle, haystack_label
                ).ratio()
                if max(text_score, label_score) >= _FUZZY_THRESHOLD:
                    results.append(node)
            else:
                if needle in haystack_text or needle in haystack_label:
                    results.append(node)
        return results

    def find_by_type(self, element_type: ElementType) -> list[SpatialNode]:
        """Return all nodes of the given element type."""
        return list(self._by_type.get(element_type, []))

    def find_by_affordance(self, affordance: Affordance) -> list[SpatialNode]:
        """Return all nodes with the given affordance."""
        return list(self._by_affordance.get(affordance, []))

    def find_best_target(self, description: str) -> SpatialNode | None:
        """
        Find the node that best matches *description*.

        Scoring:
          - Text match score  (difflib ratio against node.text)
          - Label match score (difflib ratio against primary_label)
          - Affordance bonus  (+0.1 for CLICKABLE / TYPEABLE)
          - Confidence weight (node semantic confidence)

        Returns the highest-scoring node, or None when the graph is empty.
        """
        if not self._nodes or not description:
            return None

        needle = description.lower()
        best_node: SpatialNode | None = None
        best_score: float = -1.0

        for node in self._nodes.values():
            text_score = difflib.SequenceMatcher(
                None, needle, node.text.lower()
            ).ratio()
            label_score = difflib.SequenceMatcher(
                None, needle, node.semantic.primary_label.lower()
            ).ratio()
            # Also check secondary labels
            sec_score = max(
                (
                    difflib.SequenceMatcher(None, needle, s.lower()).ratio()
                    for s in node.semantic.secondary_labels
                ),
                default=0.0,
            )
            base = max(text_score, label_score, sec_score)
            aff_bonus = (
                0.10
                if node.semantic.affordance
                in (Affordance.CLICKABLE, Affordance.TYPEABLE)
                else 0.0
            )
            score = base * node.semantic.confidence + aff_bonus
            if score > best_score:
                best_score = score
                best_node = node

        return best_node

    def get_context_around(
        self,
        element_id: ElementId,
        radius: float,
    ) -> list[SpatialNode]:
        """
        Return all nodes whose bbox centre lies within *radius* pixels of
        the bbox centre of *element_id*.  The anchor node itself is excluded.
        """
        anchor = self._nodes.get(element_id)
        if anchor is None:
            return []
        ax, ay = _centre(anchor.element)
        result: list[SpatialNode] = []
        for node in self._nodes.values():
            if node.id == element_id:
                continue
            nx, ny = _centre(node.element)
            if math.hypot(nx - ax, ny - ay) <= radius:
                result.append(node)
        return result

    def to_summary_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable summary suitable for cloud prompt injection.

        Structure::

            {
              "node_count": int,
              "nodes": [
                {
                  "id": str,
                  "type": str,
                  "affordance": str,
                  "label": str,
                  "text": str,          # first 80 chars
                  "is_destructive": bool,
                  "confidence": float,
                  "bbox": [x, y, w, h],
                  "relations": [
                    {"type": str, "target_id": str, "confidence": float}
                  ]
                },
                ...
              ]
            }
        """
        nodes_out = []
        for node in self._nodes.values():
            bb = node.element.bounding_box
            nodes_out.append({
                "id": str(node.id),
                "type": node.element.element_type.name,
                "affordance": node.semantic.affordance.name,
                "label": node.semantic.primary_label,
                "text": node.text[:80],
                "is_destructive": node.semantic.is_destructive,
                "confidence": round(node.semantic.confidence, 3),
                "bbox": [bb.x, bb.y, bb.width, bb.height],
                "relations": [
                    {
                        "type": r.relation_type.name,
                        "target_id": str(r.target_id),
                        "confidence": round(r.confidence, 3),
                    }
                    for r in node.relations
                ],
            })
        return {
            "node_count": len(self._nodes),
            "nodes": nodes_out,
        }

    # ------------------------------------------------------------------
    # Internal: relation computation
    # ------------------------------------------------------------------

    def _compute_relations(self) -> None:
        """Compute all spatial relations for every ordered pair (a, b)."""
        node_list = list(self._nodes.values())
        n = len(node_list)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a = node_list[i]
                b = node_list[j]
                _add_relations(a, b)

    # ------------------------------------------------------------------
    # Node access (for tests / internal use)
    # ------------------------------------------------------------------

    def get_node(self, element_id: ElementId) -> SpatialNode | None:
        """Return the node with the given ID, or None."""
        return self._nodes.get(element_id)


# ---------------------------------------------------------------------------
# Relation-building helpers
# ---------------------------------------------------------------------------


def _add_relations(a: SpatialNode, b: SpatialNode) -> None:
    """
    Compute geometric relations from *a* to *b* and append them to
    ``a.relations``.  Each call covers exactly one directed pair.
    """
    ba = a.element.bounding_box
    bb = b.element.bounding_box
    dist = _bbox_distance(ba, bb)

    # CONTAINS / CONTAINED_BY
    if _contains(ba, bb):
        a.relations.append(SpatialRelation(RelationType.CONTAINS, b.id, 0.95, dist))
        return  # skip other relations for strict containment
    if _contains(bb, ba):
        a.relations.append(SpatialRelation(RelationType.CONTAINED_BY, b.id, 0.95, dist))
        return

    # OVERLAPS (partial intersection without containment)
    if ba.overlaps(bb):
        conf = _overlap_confidence(ba, bb)
        a.relations.append(SpatialRelation(RelationType.OVERLAPS, b.id, conf, dist))
        return

    # Directional relations (non-overlapping boxes)
    a_cx, a_cy = _centre(a.element)
    b_cx, b_cy = _centre(b.element)

    # ABOVE / BELOW — vertical displacement dominates
    v_gap = bb.y - (ba.y + ba.height)    # positive → b is below a

    # ADJACENT_LEFT / ADJACENT_RIGHT — on the same row, side by side
    same_row = abs(a_cy - b_cy) <= _ROW_TOLERANCE_PX + max(ba.height, bb.height) // 2
    right_gap = bb.x - (ba.x + ba.width)     # > 0 when b is right of a
    left_gap = ba.x - (bb.x + bb.width)      # > 0 when b is left of a
    if same_row:
        if 0 <= right_gap <= _ADJACENCY_GAP_PX:
            conf = max(0.5, 1.0 - right_gap / _ADJACENCY_GAP_PX)
            a.relations.append(
                SpatialRelation(RelationType.ADJACENT_RIGHT, b.id, conf, dist)
            )
        elif 0 <= left_gap <= _ADJACENCY_GAP_PX:
            conf = max(0.5, 1.0 - left_gap / _ADJACENCY_GAP_PX)
            a.relations.append(
                SpatialRelation(RelationType.ADJACENT_LEFT, b.id, conf, dist)
            )

    # ABOVE / BELOW — b is below a
    if v_gap > 0 and v_gap <= _ADJACENCY_GAP_PX * 2:
        conf = max(0.5, 1.0 - v_gap / (_ADJACENCY_GAP_PX * 2))
        a.relations.append(SpatialRelation(RelationType.ABOVE, b.id, conf, dist))
    elif v_gap < 0:
        u_gap = ba.y - (bb.y + bb.height)  # a is below b
        if 0 < u_gap <= _ADJACENCY_GAP_PX * 2:
            conf = max(0.5, 1.0 - u_gap / (_ADJACENCY_GAP_PX * 2))
            a.relations.append(SpatialRelation(RelationType.BELOW, b.id, conf, dist))

    # LABELED_BY / LABELS — LABEL element adjacent to INPUT / interactive
    _try_label_relation(a, b, dist)


def _try_label_relation(a: SpatialNode, b: SpatialNode, dist: float) -> None:
    """
    Add LABELED_BY to *a* when *b* is a LABEL-type element that is close and
    to the left of / above *a*.
    Add LABELS to *a* when *a* is a LABEL-type near an interactive element *b*.
    """
    _interactive = frozenset([
        ElementType.INPUT, ElementType.BUTTON, ElementType.CHECKBOX,
        ElementType.RADIO, ElementType.DROPDOWN,
    ])

    if dist > _LABEL_PROXIMITY_PX:
        return

    if (
        b.element.element_type is ElementType.LABEL
        and a.element.element_type in _interactive
    ):
        # b labels a — add LABELED_BY on a
        conf = max(0.5, 1.0 - dist / _LABEL_PROXIMITY_PX)
        a.relations.append(
            SpatialRelation(RelationType.LABELED_BY, b.id, conf, dist)
        )

    if (
        a.element.element_type is ElementType.LABEL
        and b.element.element_type in _interactive
    ):
        conf = max(0.5, 1.0 - dist / _LABEL_PROXIMITY_PX)
        a.relations.append(
            SpatialRelation(RelationType.LABELS, b.id, conf, dist)
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _centre(el: UIElement) -> tuple[float, float]:
    bb = el.bounding_box
    return (bb.x + bb.width / 2, bb.y + bb.height / 2)


def _contains(outer: Any, inner: Any) -> bool:
    """Return True when *outer* Rect fully encloses *inner* (with slack)."""
    s = _CONTAIN_SLACK_PX
    return bool(
        outer.x - s <= inner.x
        and inner.x + inner.width <= outer.x + outer.width + s
        and outer.y - s <= inner.y
        and inner.y + inner.height <= outer.y + outer.height + s
    )


def _bbox_distance(a: Any, b: Any) -> float:
    """Euclidean distance between the centres of two Rects."""
    ax = a.x + a.width / 2
    ay = a.y + a.height / 2
    bx = b.x + b.width / 2
    by = b.y + b.height / 2
    return math.hypot(bx - ax, by - ay)


def _overlap_confidence(a: Any, b: Any) -> float:
    """Confidence for an OVERLAPS relation: intersection / min_area."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    min_area = min(a.area(), b.area())
    return float(inter) / min_area if min_area > 0 else 0.0
