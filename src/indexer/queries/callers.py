"""
callers.py - Caller traversal query helpers.

Provides the query implementation for tracing inbound call relationships
from a root symbol and attaching edge confidence metadata to visited nodes.

Public API:
    query_callers(store, qualified_name, project, depth) -> TraceQueryResult | None
"""

from indexer.store import BFSResult, Store

from .models import TraceEntry, TraceQueryResult
from .util import node_row_to_symbol_ref


def query_callers(
    store: Store,
    qualified_name: str,
    project: str | None = None,
    depth: int = 3,
) -> TraceQueryResult | None:
    # Verify the node exists
    node = store.get_node_by_qn(qualified_name, project=project)
    if node is None:
        return None

    res = store.bfs_callers(
        qualified_name,
        project=project,
        max_depth=depth,
        max_nodes=200,
        edge_types=["CALLS"],
    )

    if res is None:
        return None

    return TraceQueryResult(
        root=node_row_to_symbol_ref(node),
        direction="callers",
        visited=tuple(
            TraceEntry(
                symbol=node_row_to_symbol_ref(v),
                hop=depth,
                confidence=_build_confidence_map(res).get(v.id),
                strategy=None,
            )
            for v, depth in res.visited
        ),
        confidence_map=_build_confidence_map(res),
    )


def _build_confidence_map(result: BFSResult) -> dict[int, float]:
    """
    Build a dict mapping node_id → highest confidence of any edge to it.

    Used by trace_callers() to display confidence scores alongside each
    caller. When multiple edges point to the same node (via different
    intermediate nodes), takes the maximum confidence.

    Args:
        result: BFSResult from store.bfs_callers()

    Returns:
        Dict mapping SymbolRef.qualified_name → float confidence score.
        Missing IDs default to 0.0 at the call site.
    """
    conf_map: dict[int, float] = {}
    for edge in result.edges:
        conf = edge.properties.get("confidence", 0.0)
        if not isinstance(conf, (int, float)):
            conf = 0.0
        # edge.source_id is the caller; we want to annotate the caller node
        current = conf_map.get(edge.source_id, 0.0)
        if conf > current:
            conf_map[edge.source_id] = float(conf)
    return conf_map
