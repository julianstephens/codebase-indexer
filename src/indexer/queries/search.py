"""
search.py - Full-text symbol search query helpers.

Wraps store-level FTS search with query-layer error handling and converts
store rows into typed query result models.

Public API:
    query_search(store, query, project, label, limit) -> SearchQueryResult | None
"""

from indexer.errors import SearchQueryError
from indexer.store import SearchParams, Store

from .models import SearchQueryResult
from .util import node_row_to_symbol_ref


def query_search(
    store: Store,
    query: str,
    file_pattern: str | None = None,
    project: str | None = None,
    label: str | None = None,
    limit: int = 10,
) -> SearchQueryResult | None:
    """
    Performs a search query on the codebase.

    Args:
        store: The Store instance to perform the search on.
        query: The search query string.
        file_pattern: Optional file pattern to filter the search.
        project: Optional project name to filter the search.
        label: Optional label to filter the search.
        limit: The maximum number of results to return.

    Returns:
        A SearchQueryResult instance containing the search results, or
        None if no results are found.

    Raises:
        SearchQueryError: If there is an error during the search query.
    """
    params = SearchParams(
        project=project,
        label=label,
        fts_query=query,
        file_pattern=file_pattern,
        limit=limit,
    )
    try:
        res = store.search_nodes(params)
    except Exception as e:
        raise SearchQueryError(query) from e

    if res.total == 0:
        return None

    return SearchQueryResult(
        query=query,
        matches=tuple(node_row_to_symbol_ref(node) for node in res.rows),
        total=res.total,
    )
