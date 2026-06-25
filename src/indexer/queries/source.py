"""
source.py - Source retrieval query helpers.

Loads one symbol's source text and direct caller/callee relationships,
including conservative truncation safeguards for large source bodies.

Public API:
    query_source(store, qualified_name, project) -> SourceQueryResult | None
"""

from indexer.store import Store

from .models import RelatedSymbol, SourceQueryResult, SymbolRef
from .util import node_row_to_symbol_ref

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum source bytes returned by get_source() for a single node.
# Prevents a 5,000-line generated file from flooding the context.
MAX_SOURCE_BYTES: int = 32_768  # 32 KB

# Maximum number of caller/callee QNs shown in get_source() output.
MAX_RELATED_SHOWN: int = 20


def query_source(
    store: Store,
    qualified_name: str,
    project: str | None = None,
) -> SourceQueryResult | None:
    node = store.get_node_by_qn(qualified_name, project=project)
    if node is None:
        return None

    source = _maybe_truncate(node.source)

    callers_result = store.bfs_callers(
        qualified_name,
        project=project,
        max_depth=1,
        max_nodes=MAX_RELATED_SHOWN + 1,
        edge_types=["CALLS"],
    )
    callees_result = store.bfs_callees(
        qualified_name,
        project=project,
        max_depth=1,
        max_nodes=MAX_RELATED_SHOWN + 1,
        edge_types=["CALLS"],
    )

    callers = callers_result.visited if callers_result else []
    callees = callees_result.visited if callees_result else []

    return SourceQueryResult(
        symbol=SymbolRef(
            qualified_name=node.qualified_name,
            label=node.label,
            file_path=node.file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=node.signature,
        ),
        source=source,
        callers=tuple(
            RelatedSymbol(
                symbol=node_row_to_symbol_ref(c),
                relationship="caller",
                confidence=1.0,  # BFS is deterministic
                strategy="bfs",
            )
            for c, _ in callers[:MAX_RELATED_SHOWN]
        ),
        callees=tuple(
            RelatedSymbol(
                symbol=node_row_to_symbol_ref(c),
                relationship="callee",
                confidence=1.0,  # BFS is deterministic
                strategy="bfs",
            )
            for c, _ in callees[:MAX_RELATED_SHOWN]
        ),
    )


def _maybe_truncate(source: str) -> str:
    """
    Truncate source text at MAX_SOURCE_BYTES, appending a notice.

    The notice is appended on a new line:
        \n# [source truncated at 32768 bytes — call get_source() with
        #  a more specific node QN to see remaining content]

    Args:
        source: raw source text

    Returns:
        Source text, possibly truncated.
    """
    encoded = source.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_SOURCE_BYTES:
        return source
    truncated = encoded[:MAX_SOURCE_BYTES].decode("utf-8", errors="replace")
    # Snap to last newline to avoid cutting mid-line
    last_nl = truncated.rfind("\n")
    if last_nl > MAX_SOURCE_BYTES // 2:
        truncated = truncated[:last_nl]
    return (
        truncated + f"\n# [source truncated at {MAX_SOURCE_BYTES} bytes — "
        f"use a more specific QN to see the full body]"
    )
