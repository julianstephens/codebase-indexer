"""
tools.py — Agent-facing tool functions for the repo knowledge graph.

These are the three functions the agent calls during a vulnerability
remediation session:

    get_source(db_path, qualified_name, project)
        Return the full source of a node + its callers + its callees.
        This is the primary tool: the agent calls it after reading the
        skeleton to fetch the actual code it needs to understand or fix.

    search(db_path, query, project, label, limit)
        Full-text search across name, signature, and source.
        The agent calls this when the skeleton doesn't give enough
        information to identify the right node by QN.

    trace_callers(db_path, qualified_name, project, depth)
        BFS up the call graph and return the blast radius for a node.
        The agent calls this to understand what will break if it changes
        a function.

Each function returns a human-readable string formatted for direct
inclusion in an LLM conversation. They do not return JSON or structured
objects — the agent reads the text and reasons about it.

All functions open the database in read-only mode, perform their query,
and close the connection. They never modify the database.

Error handling:
    Every function catches all exceptions and returns a human-readable
    error string rather than raising. This ensures a tool call failure
    surfaces as a recoverable message in the agent session rather than
    crashing the tool loop.

Token awareness:
    get_source() and trace_callers() include token estimates in their
    output so the agent can make informed decisions about how many more
    tool calls to make before its context fills up.
"""

import logging
from pathlib import Path

from .context import estimate_tokens
from .store import BFSResult, NodeRow, SearchParams, Store, open_path_readonly

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum source bytes returned by get_source() for a single node.
# Prevents a 5,000-line generated file from flooding the context.
MAX_SOURCE_BYTES: int = 32_768  # 32 KB

# Maximum number of caller/callee QNs shown in get_source() output.
MAX_RELATED_SHOWN: int = 20

# Separator line used between sections in tool output.
_SEP: str = "─" * 60


# ---------------------------------------------------------------------------
# get_source
# ---------------------------------------------------------------------------


def get_source(
    db_path: str,
    qualified_name: str,
    project: str | None = None,
) -> str:
    """
    Return the full source of a node plus its callers and callees.

    This is the primary tool for the agent. After reading the skeleton,
    the agent identifies a relevant node by its QN and calls this
    function to read the actual code.

    Output format:

        ────────────────────────────────────────────────────────────
        # src/payments/service.py  lines 12-45  [Function]
        # src.payments.service.charge
        ────────────────────────────────────────────────────────────

        def charge(user: User, amount_cents: int, currency: str) -> Payment:
            ...full source body...

        ────────────────────────────────────────────────────────────
        # called by (2):
            src.payments.views.checkout                   views.py:88
            src.orders.processor.complete_order   processor.py:142

        # calls (3):
            src.payments.models.Payment.save      models.py:34
            src.payments.stripe_client.charge     stripe_client.py:67
            src.auth.models.User.is_active        models.py:19

        # ~210 tokens
        ────────────────────────────────────────────────────────────

    If the source exceeds MAX_SOURCE_BYTES it is truncated with a
    visible notice so the agent knows the output was cut.

    Args:
        db_path:        absolute path to the .db file
        qualified_name: exact QN of the node, e.g.
                        "src.payments.service.charge"
        project:        optional project filter. If None, searches all
                        projects (QNs are globally unique within a project).

    Returns:
        Formatted string ready for inclusion in the agent conversation.
        Returns a "node not found" message if the QN does not exist.
        Returns an error string if the database cannot be opened.

    Examples:
        >>> print(get_source("/cache/my-app.db", "src.payments.service.charge"))
        ─────────────────────...
        # src/payments/service.py  lines 12-45  [Function]
        ...
    """
    try:
        if not Path(db_path).exists():
            return f"Error: database not found: {db_path}"

        store = open_path_readonly(db_path)
        try:
            return _get_source_impl(store, qualified_name, project)
        finally:
            store.close()
    except Exception as exc:
        logger.exception("get_source failed for %r", qualified_name)
        return f"Error: get_source failed: {exc}"


def _get_source_impl(
    store: Store,
    qualified_name: str,
    project: str | None,
) -> str:
    """
    Core implementation of get_source(), operating on an open Store.

    Args:
        store:          open Store (read-only)
        qualified_name: exact QN to look up
        project:        optional project filter

    Returns:
        Formatted output string.
    """
    node = store.get_node_by_qn(qualified_name, project=project)
    if node is None:
        return (
            f"Node not found: {qualified_name!r}\n\n"
            f'Hint: use search("{_bare_name(qualified_name)}") to find '
            f"similarly named nodes."
        )

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

    parts: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────
    parts.append(_SEP)
    parts.append(
        f"# {node.file_path}  lines {node.start_line}-{node.end_line}"
        f"  [{node.label}]"
    )
    parts.append(f"# {node.qualified_name}")
    parts.append(_SEP)
    parts.append("")

    # ── Source ──────────────────────────────────────────────────────────
    parts.append(source)
    parts.append("")

    # ── Related nodes ────────────────────────────────────────────────────
    parts.append(_SEP)

    caller_lines = _format_related(callers, "called by")
    parts.append(caller_lines)
    parts.append("")

    callee_lines = _format_related(callees, "calls")
    parts.append(callee_lines)

    # ── Token estimate ───────────────────────────────────────────────────
    body = "\n".join(parts)
    token_est = estimate_tokens(body)
    parts.append(f"\n# ~{token_est} tokens")
    parts.append(_SEP)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def search(
    db_path: str,
    query: str,
    project: str | None = None,
    label: str | None = None,
    limit: int = 10,
) -> str:
    """
    Full-text search across node names, signatures, and source bodies.

    Uses SQLite FTS5 to find nodes matching the query string. Results
    are ranked by relevance (FTS5 BM25 rank).

    Output format:

        # search: "sql injection"  (4 results)
        ────────────────────────────────────────────────────────────
        src.payments.repository.find_payment
          Function  src/payments/repository.py:23
          def find_payment(payment_id: str) -> Payment | None:

        src.orders.repository.list_orders
          Function  src/orders/repository.py:45
          def list_orders(user_id: str, status: str) -> list[Order]:
        ...

    Args:
        db_path:  absolute path to the .db file
        query:    FTS5 query string. Supports:
                    - bare words:  "sql injection"
                    - AND/OR/NOT:  "charge AND stripe"
                    - prefix:      "charg*"
                    - phrase:      '"charge user"'
        project:  optional project filter. None = all projects.
        label:    optional label filter, e.g. "Function" or "Class".
        limit:    maximum number of results to return. Defaults to 10.
                  Capped at 50.

    Returns:
        Formatted search results string. Returns "No results" if the
        query matches nothing. Returns an error string if the database
        cannot be opened.

    Examples:
        >>> print(search("/cache/my-app.db", "sql injection"))
        # search: "sql injection"  (2 results)
        ─────────────────────...
    """
    try:
        if not Path(db_path).exists():
            return f"Error: database not found: {db_path}"

        limit = min(limit, 50)

        store = open_path_readonly(db_path)
        try:
            return _search_impl(store, query, project, label, limit)
        finally:
            store.close()

    except Exception as exc:
        logger.exception("search failed for query %r", query)
        return f"Error: search failed: {exc}"


def _search_impl(
    store: Store,
    query: str,
    project: str | None,
    label: str | None,
    limit: int,
) -> str:
    """
    Core implementation of search(), operating on an open Store.

    Args:
        store:   open Store (read-only)
        query:   FTS5 query string
        project: optional project filter
        label:   optional label filter
        limit:   maximum results

    Returns:
        Formatted search results string.
    """
    params = SearchParams(
        project=project,
        label=label,
        fts_query=query,
        limit=limit,
    )
    result = store.search_nodes(params)

    if result.total == 0:
        return (
            f"# search: {query!r}  (0 results)\n\n"
            f"No nodes matched. Try:\n"
            f'  - shorter query: search("{_bare_name(query)}")\n'
            f'  - different terms: search("payment process")\n'
            f"  - remove label filter"
            if label
            else "No nodes matched. Try a shorter or broader query."
        )

    parts: list[str] = []
    shown = len(result.rows)
    total_note = (
        f"showing {shown} of {result.total}"
        if result.total > shown
        else str(result.total)
    )
    parts.append(f"# search: {query!r}  ({total_note} results)")
    parts.append(_SEP)

    for node in result.rows:
        parts.append(_format_search_hit(node))
        parts.append("")

    if result.total > shown:
        parts.append(
            f"# {result.total - shown} more results — narrow your query or "
            f'use search("{query}", limit={min(limit * 2, 50)})'
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# trace_callers
# ---------------------------------------------------------------------------


def trace_callers(
    db_path: str,
    qualified_name: str,
    project: str | None = None,
    depth: int = 3,
) -> str:
    """
    BFS up the call graph and return the blast radius for a node.

    The agent calls this to understand what other code will be affected
    by a change to the given node. Essential for vulnerability
    remediation: before patching a function, the agent should understand
    all callers to verify the fix is safe.

    Output format:

        # trace_callers: src.payments.service.charge  (depth=3)
        # 5 callers found across 2 hops
        ────────────────────────────────────────────────────────────

        hop 1 — direct callers (2)
          src.payments.views.checkout                   confidence=0.95
          src.orders.processor.complete_order           confidence=0.85

        hop 2 — indirect callers (3)
          src.api.v1.orders.create_order                confidence=0.95
          src.api.v2.payments.charge_card               confidence=0.85
          src.admin.billing.retry_failed_charge         confidence=0.40

        # blast radius: 5 nodes across 2 hops
        # ~180 tokens
        ────────────────────────────────────────────────────────────

    Args:
        db_path:        absolute path to the .db file
        qualified_name: exact QN of the node to trace from
        project:        optional project filter
        depth:          maximum BFS hops. Defaults to 3. Capped at 10.

    Returns:
        Formatted blast-radius string. Returns a "node not found"
        message if the QN does not exist. Returns an error string if
        the database cannot be opened.

    Examples:
        >>> print(trace_callers("/cache/my-app.db",
        ...                     "src.payments.service.charge", depth=2))
        # trace_callers: src.payments.service.charge  (depth=2)
        ...
    """
    try:
        if not Path(db_path).exists():
            return f"Error: database not found: {db_path}"

        depth = min(max(1, depth), 10)

        store = open_path_readonly(db_path)
        try:
            return _trace_callers_impl(store, qualified_name, project, depth)
        finally:
            store.close()

    except Exception as exc:
        logger.exception("trace_callers failed for %r", qualified_name)
        return f"Error: trace_callers failed: {exc}"


def _trace_callers_impl(
    store: Store,
    qualified_name: str,
    project: str | None,
    depth: int,
) -> str:
    """
    Core implementation of trace_callers(), operating on an open Store.

    Args:
        store:          open Store (read-only)
        qualified_name: exact QN to start BFS from
        project:        optional project filter
        depth:          maximum BFS hops

    Returns:
        Formatted blast-radius string.
    """
    # Verify the node exists
    node = store.get_node_by_qn(qualified_name, project=project)
    if node is None:
        return (
            f"Node not found: {qualified_name!r}\n\n"
            f'Hint: use search("{_bare_name(qualified_name)}") to find '
            f"similarly named nodes."
        )

    result = store.bfs_callers(
        qualified_name,
        project=project,
        max_depth=depth,
        max_nodes=200,
        edge_types=["CALLS"],
    )

    parts: list[str] = []
    parts.append(f"# trace_callers: {qualified_name}  (depth={depth})")

    if result is None or not result.visited:
        parts.append("# 0 callers found")
        parts.append(_SEP)
        parts.append("(no callers — this node is not called by any indexed code)")
        parts.append(_SEP)
        return "\n".join(parts)

    total_callers = len(result.visited)
    max_hop = max(hop for _, hop in result.visited)
    hop_label = "hop" if max_hop == 1 else "hops"
    parts.append(f"# {total_callers} callers found across {max_hop} {hop_label}")
    parts.append(_SEP)

    # Group by hop depth
    by_hop: dict[int, list[NodeRow]] = {}
    for caller_node, hop in result.visited:
        by_hop.setdefault(hop, []).append(caller_node)

    # Build edge confidence map: target_id → confidence
    confidence_map = _build_confidence_map(result)

    for hop in sorted(by_hop.keys()):
        hop_nodes = by_hop[hop]
        label = "direct callers" if hop == 1 else "indirect callers"
        parts.append(f"\nhop {hop} — {label} ({len(hop_nodes)})")

        for caller in sorted(hop_nodes, key=lambda n: n.qualified_name):
            conf = confidence_map.get(caller.id, 0.0)
            conf_str = f"confidence={conf:.2f}" if conf > 0 else ""
            qn_display = _format_qn(caller.qualified_name, width=60)
            parts.append(f"  {qn_display}  {conf_str}")

    parts.append("")
    parts.append(
        f"# blast radius: {total_callers} node{'s' if total_callers != 1 else ''}"
        f" across {max_hop} hop{'s' if max_hop != 1 else ''}"
    )

    body = "\n".join(parts)
    token_est = estimate_tokens(body)
    parts.append(f"# ~{token_est} tokens")
    parts.append(_SEP)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------


def _format_related(
    visited: list[tuple[NodeRow, int]],
    direction: str,
) -> str:
    """
    Format the callers or callees section of get_source() output.

    Produces lines like:
        # called by (2):
            src.payments.views.checkout                   views.py:88
            src.orders.processor.complete_order   processor.py:142

    Args:
        visited:   list of (NodeRow, hop_depth) from bfs_callers/callees.
                   Only hop=1 nodes are shown (direct relationships only).
        direction: "called by" or "calls"

    Returns:
        Formatted multi-line string.
    """
    direct = [n for n, hop in visited if hop == 1]
    count = len(direct)

    if count == 0:
        return f"# {direction} (0):\n    (none)"

    lines = [f"# {direction} ({count}):"]
    shown = direct[:MAX_RELATED_SHOWN]
    for node in sorted(shown, key=lambda n: n.qualified_name):
        loc = f"{Path(node.file_path).name}:{node.start_line}"
        qn_display = _format_qn(node.qualified_name, width=52)
        lines.append(f"    {qn_display}  {loc}")

    if count > MAX_RELATED_SHOWN:
        lines.append(f"    ... and {count - MAX_RELATED_SHOWN} more")

    return "\n".join(lines)


def _format_search_hit(node: NodeRow) -> str:
    """
    Format a single search result node for the search() output.

    Produces lines like:
        src.payments.repository.find_payment
          Function  src/payments/repository.py:23
          def find_payment(payment_id: str) -> Payment | None:

    Args:
        node: NodeRow from search_nodes()

    Returns:
        Multi-line string for this search hit.
    """
    sig = _truncate_sig(node.signature, 80)
    return (
        f"{node.qualified_name}\n"
        f"  {node.label}  {node.file_path}:{node.start_line}\n"
        f"  {sig}"
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
        Dict mapping NodeRow.id → float confidence score.
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


def _format_qn(qn: str, width: int) -> str:
    """
    Left-justify a qualified name in a field of the given width.

    Truncates from the left (keeping the tail) if the QN is longer
    than width, replacing the truncated prefix with "...". This keeps
    the most specific (rightmost) part of the name visible.

    Args:
        qn:    qualified name string
        width: desired field width in characters

    Returns:
        String of exactly `width` characters (padded or truncated).

    Examples:
        >>> _format_qn("src.payments.service.charge", 30)
        'src.payments.service.charge   '
        >>> _format_qn("src.very.long.module.path.charge", 20)
        '...module.path.charge'
    """
    if len(qn) <= width:
        return qn.ljust(width)
    # Truncate from the left
    tail = qn[-(width - 3) :]
    return f"...{tail}"


def _truncate_sig(signature: str, max_len: int) -> str:
    """
    Truncate a signature to at most max_len characters.

    Appends "..." if truncation occurs. Attempts to cut at a comma or
    space boundary within 20 chars of the limit.

    Args:
        signature: single-line signature string
        max_len:   maximum character length including "..."

    Returns:
        Signature of length <= max_len.
    """
    if len(signature) <= max_len:
        return signature
    cut = max_len - 3
    for i in range(cut, max(cut - 20, 0), -1):
        if i < len(signature) and signature[i] in (",", " "):
            return signature[:i] + "..."
    return signature[:cut] + "..."


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


def _bare_name(qn: str) -> str:
    """
    Extract the final component of a qualified name for hint messages.

    Args:
        qn: qualified name string, e.g. "src.payments.service.charge"

    Returns:
        Last component, e.g. "charge". Returns the input unchanged if
        there are no dots.

    Examples:
        >>> _bare_name("src.payments.service.charge")
        'charge'
        >>> _bare_name("charge")
        'charge'
    """
    return qn.rsplit(".", 1)[-1] if "." in qn else qn
