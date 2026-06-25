"""
text.py — Human-readable renderers for structured indexer query results.

Converts query result dataclasses into the text format consumed by agents and
printed by the CLI. Rendering is deterministic and does not access the store.

Public API:
    render_source_text(result)   — source body plus direct callers/callees
    render_search_text(result)   — ranked symbol search results
    render_trace_text(result)    — graph results grouped by hop
"""

from pathlib import Path

from indexer.context import estimate_tokens
from indexer.queries.models import (
    RelatedSymbol,
    SearchQueryResult,
    SourceQueryResult,
    SymbolRef,
    TraceQueryResult,
)

from .common import (
    MAX_RELATED_SHOWN,
    MAX_SEARCH_SIGNATURE_LEN,
    MAX_SOURCE_BYTES,
    format_qualified_name,
    truncate_signature,
    truncate_source,
)

_SEP: str = "─" * 60


def render_source_text(result: SourceQueryResult) -> str:
    """
    Render one symbol's source plus its direct callers and callees.

    Args:
        result: structured source query result.

    Returns:
        Human-readable source result suitable for direct agent inclusion.
    """
    symbol = result.symbol
    source, truncated = truncate_source(result.source)
    if truncated:
        source += (
            f"\n# [source truncated at {MAX_SOURCE_BYTES} bytes — "
            "use a more specific QN to see the full body]"
        )

    parts = [
        _SEP,
        (
            f"# {symbol.file_path}  lines {symbol.start_line}-{symbol.end_line}"
            f"  [{symbol.label}]"
        ),
        f"# {symbol.qualified_name}",
        _SEP,
        "",
        source,
        "",
        _SEP,
        _format_related(result.callers, "called by"),
        "",
        _format_related(result.callees, "calls"),
    ]
    body = "\n".join(parts)
    parts.extend([f"\n# ~{estimate_tokens(body)} tokens", _SEP])
    return "\n".join(parts)


def render_search_text(result: SearchQueryResult) -> str:
    """
    Render ranked symbol search results.

    Args:
        result: structured search query result.

    Returns:
        Human-readable search result with pagination guidance.
    """
    if result.total == 0:
        return _render_empty_search(result)

    shown = len(result.matches)
    total_note = (
        f"showing {shown} of {result.total}"
        if result.total > shown
        else str(result.total)
    )
    parts = [
        f"# search: {result.query!r}  ({total_note} results)",
        _SEP,
    ]

    for symbol in result.matches:
        parts.extend([_format_search_hit(symbol), ""])

    if result.total > shown:
        remaining = result.total - shown
        parts.append(
            f"# {remaining} more result"
            f"{'s' if remaining != 1 else ''} — "
            "narrow the query or request a higher result limit"
        )

    return "\n".join(parts)


def render_trace_text(result: TraceQueryResult) -> str:
    """
    Render graph traversal results grouped by hop depth.

    Args:
        result: structured caller or callee traversal result.

    Returns:
        Human-readable traversal summary grouped by graph distance.
    """
    noun = result.direction
    command = f"trace_{result.direction}"
    parts = [f"# {command}: {result.root.qualified_name}"]

    if not result.visited:
        message = (
            "(no callers — this node is not called by any indexed code)"
            if result.direction == "callers"
            else "(no callees — this node calls no indexed code)"
        )
        parts.extend(
            [
                f"# 0 {noun} found",
                _SEP,
                message,
                _SEP,
            ]
        )
        return "\n".join(parts)

    total = len(result.visited)
    max_hop = max(entry.hop for entry in result.visited)
    hop_label = "hop" if max_hop == 1 else "hops"

    parts.extend(
        [
            f"# {total} {noun} found across {max_hop} {hop_label}",
            _SEP,
        ]
    )

    by_hop: dict[int, list[SymbolRef]] = {}
    for entry in result.visited:
        by_hop.setdefault(entry.hop, []).append(entry.symbol)

    for hop in sorted(by_hop):
        symbols = by_hop[hop]
        prefix = "direct" if hop == 1 else "indirect"
        parts.append(f"\nhop {hop} — {prefix} {noun} ({len(symbols)})")

        for symbol in sorted(
            symbols,
            key=lambda item: item.qualified_name,
        ):
            qualified_name = format_qualified_name(
                symbol.qualified_name,
                width=60,
            )
            location = f"{symbol.file_path}:{symbol.start_line}"
            parts.append(f"  {qualified_name}  {location}")

    summary = "blast radius" if result.direction == "callers" else "dependency set"
    parts.extend(
        [
            "",
            f"# {summary}: {total} "
            f"node{'s' if total != 1 else ''} "
            f"across {max_hop} "
            f"hop{'s' if max_hop != 1 else ''}",
        ]
    )

    body = "\n".join(parts)
    parts.extend(
        [
            f"# ~{estimate_tokens(body)} tokens",
            _SEP,
        ]
    )
    return "\n".join(parts)


def render_node_not_found(qualified_name: str) -> str:
    """
    Render a response for a qualified name that is not in the index.

    Args:
        qualified_name: qualified name requested by the caller.

    Returns:
        Human-readable response with a suggested search query.
    """
    bare_name = qualified_name.rsplit(".", 1)[-1]
    return (
        f"Node not found: {qualified_name!r}\n\n"
        f'Hint: use search("{bare_name}") to find similarly named nodes.'
    )


def _format_related(
    related: tuple[RelatedSymbol, ...],
    direction: str,
) -> str:
    """
    Format one direct relationship section.

    Args:
        related: related symbols for one direction.
        direction: section title, such as "called by" or "calls".

    Returns:
        Formatted multi-line relationship section.
    """
    if not related:
        return f"# {direction} (0):\n    (none)"

    lines = [f"# {direction} ({len(related)}):"]
    shown = sorted(
        related[:MAX_RELATED_SHOWN],
        key=lambda item: item.symbol.qualified_name,
    )
    for relation in shown:
        symbol = relation.symbol
        location = f"{Path(symbol.file_path).name}:{symbol.start_line}"
        qualified_name = format_qualified_name(
            symbol.qualified_name,
            width=52,
        )
        lines.append(f"    {qualified_name}  {location}")

    if len(related) > MAX_RELATED_SHOWN:
        lines.append(f"    ... and {len(related) - MAX_RELATED_SHOWN} more")
    return "\n".join(lines)


def _format_search_hit(symbol: SymbolRef) -> str:
    """
    Format one search result.

    Args:
        symbol: symbol returned by a search query.

    Returns:
        Three-line representation of the symbol and source location.
    """
    signature = truncate_signature(
        symbol.signature,
        MAX_SEARCH_SIGNATURE_LEN,
    )
    return (
        f"{symbol.qualified_name}\n"
        f"  {symbol.label}  {symbol.file_path}:{symbol.start_line}\n"
        f"  {signature}"
    )


def _render_empty_search(result: SearchQueryResult) -> str:
    """
    Render an empty search result with recovery guidance.

    Args:
        result: empty structured search result.

    Returns:
        Human-readable empty-search response.
    """
    return (
        f"# search: {result.query!r}  (0 results)\n\n"
        "No nodes matched. Try a shorter or broader query."
    )
