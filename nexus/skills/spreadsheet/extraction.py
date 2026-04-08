"""
nexus/skills/spreadsheet/extraction.py
Spreadsheet table extractor — converts perception output to structured rows.

SpreadsheetExtractor
--------------------
Extracts tabular data from a PerceptionResult by analysing the SpatialGraph
nodes' bounding boxes and OCR text.

extract_table(perception, has_header) -> list[dict]
  Algorithm:
    1. Collect all SpatialNodes that carry non-empty OCR text.
    2. Group nodes into row buckets using a Y-coordinate tolerance
       (_ROW_TOLERANCE_PX = 20 px).  Nodes whose vertical centres
       differ by less than the tolerance are placed in the same row.
    3. Sort row buckets by their mean Y-coordinate (top → bottom).
    4. Within each row, sort nodes by their X-coordinate (left → right).
    5. When has_header=True the first row becomes column keys and the
       remaining rows become data dicts.  When has_header=False or there
       is only one row, integer-indexed keys ("col_0", "col_1", …) are
       used and every row is returned as a data row.
    6. Rows with fewer cells than the header are padded with ``""``; rows
       with more cells are truncated to the header width.

Returns an empty list when the SpatialGraph has no text nodes or when
all detected nodes fall into a single row that is also the header.
"""
from __future__ import annotations

from nexus.infra.logger import get_logger
from nexus.perception.orchestrator import PerceptionResult
from nexus.perception.spatial_graph import SpatialNode

_log = get_logger(__name__)

_ROW_TOLERANCE_PX: int = 20


class SpreadsheetExtractor:
    """
    Extracts tabular data from a PerceptionResult's SpatialGraph.

    No UIA or keyboard transport is needed — extraction is a pure
    computation over visual perception output.
    """

    def extract_table(
        self,
        perception: PerceptionResult,
        has_header: bool = True,
    ) -> list[dict[str, str]]:
        """
        Parse the SpatialGraph into a list of row dictionaries.

        Parameters
        ----------
        perception:
            Latest PerceptionResult containing a populated SpatialGraph.
        has_header:
            When True (default), the topmost row is treated as column
            headers and all subsequent rows are returned as dicts keyed
            by those headers.  When False, keys are ``"col_0"``,
            ``"col_1"``, etc.

        Returns
        -------
        A list of dicts, one per data row (header row excluded when
        ``has_header=True``).  Returns an empty list when no text is
        found or when only a header row is present.
        """
        nodes = [n for n in perception.spatial_graph.nodes if n.text.strip()]
        if not nodes:
            _log.debug("extract_table_no_text_nodes")
            return []

        rows = _group_into_rows(nodes)
        if not rows:
            return []

        # Build column headers
        if has_header and len(rows) >= 1:
            header_cells = [n.text.strip() for n in rows[0]]
            data_rows = rows[1:]
        else:
            col_count = max(len(r) for r in rows)
            header_cells = [f"col_{i}" for i in range(col_count)]
            data_rows = rows

        if not data_rows:
            _log.debug("extract_table_header_only")
            return []

        result: list[dict[str, str]] = []
        n_cols = len(header_cells)
        for row in data_rows:
            cells = [n.text.strip() for n in row]
            # Pad or truncate to header width
            cells = (cells + [""] * n_cols)[:n_cols]
            result.append(dict(zip(header_cells, cells, strict=False)))

        _log.debug(
            "extract_table_ok",
            rows=len(result),
            cols=n_cols,
        )
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _node_cy(node: SpatialNode) -> int:
    """Return the vertical centre of a node's bounding box."""
    bb = node.element.bounding_box
    return bb.y + bb.height // 2


def _node_cx(node: SpatialNode) -> int:
    """Return the horizontal centre of a node's bounding box."""
    bb = node.element.bounding_box
    return bb.x + bb.width // 2


def _group_into_rows(nodes: list[SpatialNode]) -> list[list[SpatialNode]]:
    """
    Cluster *nodes* into horizontal rows by Y-centre proximity.

    Each cluster contains nodes whose vertical centres are within
    _ROW_TOLERANCE_PX of the cluster's representative Y value.
    Clusters are returned sorted top-to-bottom; nodes within each
    cluster are sorted left-to-right.
    """
    sorted_nodes = sorted(nodes, key=_node_cy)
    buckets: list[list[SpatialNode]] = []
    bucket_ys: list[float] = []

    for node in sorted_nodes:
        cy = _node_cy(node)
        placed = False
        for i, mean_y in enumerate(bucket_ys):
            if abs(cy - mean_y) <= _ROW_TOLERANCE_PX:
                buckets[i].append(node)
                # Update running mean
                bucket_ys[i] = (
                    sum(_node_cy(n) for n in buckets[i]) / len(buckets[i])
                )
                placed = True
                break
        if not placed:
            buckets.append([node])
            bucket_ys.append(float(cy))

    # Sort rows top-to-bottom by paired Y value, cells left-to-right
    sorted_pairs = sorted(
        zip(bucket_ys, buckets, strict=False), key=lambda p: p[0]
    )
    sorted_buckets = [b for _, b in sorted_pairs]
    for bucket in sorted_buckets:
        bucket.sort(key=_node_cx)

    return sorted_buckets
