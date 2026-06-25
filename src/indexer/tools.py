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

from indexer.queries.callers import query_callers
from indexer.queries.models import SearchQueryResult
from indexer.queries.search import query_search
from indexer.queries.source import query_source
from indexer.renderers import render_node_not_found, render_trace_text
from indexer.renderers.text import render_search_text, render_source_text

from .store import open_path_readonly

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
            res = query_source(store, qualified_name, project)
        finally:
            store.close()

        if res is None:
            return render_node_not_found(qualified_name)
        return render_source_text(res)
    except Exception as exc:
        logger.exception("get_source failed for %r", qualified_name)
        return f"Error: get_source failed: {exc}"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def search(
    db_path: str,
    query: str,
    file_pattern: str | None = None,
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
        file_pattern: optional glob pattern to filter results by file path.
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
            res = query_search(store, query, file_pattern, project, label, limit)
        finally:
            store.close()

        if not res:
            return render_search_text(
                SearchQueryResult(query=query, matches=tuple(), total=0)
            )
        return render_search_text(res)
    except Exception as exc:
        logger.exception("search failed for query %r", query)
        return f"Error: search failed: {exc}"


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
            res = query_callers(store, qualified_name, project, depth)
        finally:
            store.close()

        if res is None:
            return render_node_not_found(qualified_name)

        return render_trace_text(res)

    except Exception as exc:
        logger.exception("trace_callers failed for %r", qualified_name)
        return f"Error: trace_callers failed: {exc}"
