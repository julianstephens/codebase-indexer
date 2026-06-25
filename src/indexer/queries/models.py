"""
models.py - Dataclasses for query-layer request and response payloads.

Defines shared symbol references and structured results returned by source,
search, and trace query functions.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SymbolRef:
    """
    Represents a reference to a symbol in the codebase.

    Attributes:
        qualified_name: The fully qualified name of the symbol.
        label: The label of the symbol.
        file_path: The file path where the symbol is defined.
        start_line: The starting line number of the symbol definition.
        end_line: The ending line number of the symbol definition.
        signature: The signature of the symbol.
    """

    qualified_name: str
    label: str
    file_path: str
    start_line: int
    end_line: int
    signature: str


@dataclass(frozen=True)
class RelatedSymbol:
    """
    Represents a symbol related to another symbol in the codebase.

    Attributes:
        symbol: The related symbol.
        relationship: The type of relationship ("caller" or "callee").
        confidence: The confidence level of the relationship.
        strategy: The strategy used to determine the relationship.
    """

    symbol: SymbolRef
    relationship: Literal["caller", "callee"]
    confidence: float | None
    strategy: str | None


@dataclass(frozen=True)
class SourceQueryResult:
    """
    Represents the result of a source query in the codebase.

    Attributes:
        symbol: The symbol being queried.
        source: The source code of the symbol.
        callers: A tuple of related symbols that call the queried symbol.
        callees: A tuple of related symbols that are called by the queried symbol.
    """

    symbol: SymbolRef
    source: str
    callers: tuple[RelatedSymbol, ...]
    callees: tuple[RelatedSymbol, ...]


@dataclass(frozen=True)
class SearchQueryResult:
    """
    Represents the result of a search query in the codebase.

    Attributes:
        query: The search query string.
        matches: A tuple of symbols that match the search query.
        total: The total number of matches.
    """

    query: str
    matches: tuple[SymbolRef, ...]
    total: int


@dataclass(frozen=True)
class TraceEntry:
    """
    One symbol reached during a graph traversal.

    Attributes:
        symbol: symbol reached by the traversal.
        hop: 1-based graph distance from the root.
        confidence: confidence assigned to the traversed edge.
        strategy: resolution strategy that produced the edge.
    """

    symbol: SymbolRef
    hop: int
    confidence: float | None = None
    strategy: str | None = None


@dataclass(frozen=True)
class TraceQueryResult:
    """
    Represents the result of a trace query in the codebase.

    Attributes:
        root: The root symbol of the trace query.
        direction: The direction of the trace ("callers" or "callees").
        visited: A tuple of visited symbols and their respective depths.
    """

    root: SymbolRef
    direction: Literal["callers", "callees"]
    visited: tuple[TraceEntry, ...]
    confidence_map: dict[int, float] | None = None
