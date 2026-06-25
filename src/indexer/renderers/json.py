"""
json.py — JSON-compatible renderers for structured indexer query results.

Converts query result dataclasses into dictionaries containing only JSON-native
values. The CLI is responsible for serializing the returned dictionaries.

Public API:
    render_source_json(result)   — source and direct relationships
    render_search_json(result)   — ranked symbol search results
    render_trace_json(result)    — graph traversal results
"""

import json

from ..context import estimate_tokens
from ..queries.models import (
    RelatedSymbol,
    SearchQueryResult,
    SourceQueryResult,
    SymbolRef,
    TraceEntry,
    TraceQueryResult,
)
from ._common import truncate_source

FORMAT_VERSION: int = 1


def render_source_json(result: SourceQueryResult) -> dict[str, object]:
    """
    Render one symbol source result as JSON-native values.

    Args:
        result: structured source query result.

    Returns:
        JSON-compatible source result dictionary.
    """
    source, truncated = truncate_source(result.source)
    return _with_estimated_tokens(
        {
            "format_version": FORMAT_VERSION,
            "kind": "source",
            "symbol": _symbol_to_dict(result.symbol),
            "source": source,
            "source_truncated": truncated,
            "callers": [_related_to_dict(item) for item in result.callers],
            "callees": [_related_to_dict(item) for item in result.callees],
        }
    )


def render_search_json(result: SearchQueryResult) -> dict[str, object]:
    """
    Render symbol search results as JSON-native values.

    Args:
        result: structured search query result.

    Returns:
        JSON-compatible search result dictionary.
    """
    return _with_estimated_tokens(
        {
            "format_version": FORMAT_VERSION,
            "kind": "search",
            "query": result.query,
            "total": result.total,
            "returned": len(result.matches),
            "matches": [_symbol_to_dict(symbol) for symbol in result.matches],
        }
    )


def render_trace_json(result: TraceQueryResult) -> dict[str, object]:
    """
    Render graph traversal results as JSON-native values.

    Args:
        result: structured caller or callee traversal result.

    Returns:
        JSON-compatible traversal result dictionary.
    """
    return _with_estimated_tokens(
        {
            "format_version": FORMAT_VERSION,
            "kind": "trace",
            "direction": result.direction,
            "root": _symbol_to_dict(result.root),
            "total": len(result.visited),
            "visited": [
                {
                    "hop": entry.hop,
                    "symbol": _symbol_to_dict(entry.symbol),
                }
                for entry in result.visited
            ],
        }
    )


def _symbol_to_dict(symbol: SymbolRef) -> dict[str, object]:
    """
    Convert a symbol reference to JSON-native values.

    Args:
        symbol: structured symbol reference.

    Returns:
        JSON-compatible symbol dictionary.
    """
    return {
        "qualified_name": symbol.qualified_name,
        "label": symbol.label,
        "file_path": symbol.file_path,
        "start_line": symbol.start_line,
        "end_line": symbol.end_line,
        "signature": symbol.signature,
    }


def _related_to_dict(related: RelatedSymbol) -> dict[str, object]:
    """
    Convert a direct relationship to JSON-native values.

    Args:
        related: structured direct relationship.

    Returns:
        JSON-compatible relationship dictionary.
    """
    return {
        "relationship": related.relationship,
        "confidence": related.confidence,
        "strategy": related.strategy,
        "symbol": _symbol_to_dict(related.symbol),
    }


def _trace_entry_to_dict(entry: TraceEntry) -> dict[str, object]:
    """
    Convert a traversal entry to JSON-native values.

    Args:
        entry: structured graph traversal entry.

    Returns:
        JSON-compatible traversal entry dictionary.
    """
    return {
        "hop": entry.hop,
        "confidence": entry.confidence,
        "strategy": entry.strategy,
        "symbol": _symbol_to_dict(entry.symbol),
    }


def _with_estimated_tokens(
    payload: dict[str, object],
) -> dict[str, object]:
    """
    Add an approximate serialized token count to a payload.

    The estimate excludes the estimated_tokens field itself.

    Args:
        payload: JSON-compatible result dictionary.

    Returns:
        Copy of payload with estimated_tokens added.
    """
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        **payload,
        "estimated_tokens": estimate_tokens(serialized),
    }
