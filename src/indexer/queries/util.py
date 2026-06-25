"""
util.py - Shared query-layer conversion utilities.

Contains lightweight helpers used to normalize store rows into query
dataclass shapes.
"""

from indexer.store import NodeRow

from .models import SymbolRef


def node_row_to_symbol_ref(node_row: NodeRow) -> SymbolRef:
    return SymbolRef(
        qualified_name=node_row.qualified_name,
        label=node_row.label,
        file_path=node_row.file_path,
        start_line=node_row.start_line,
        end_line=node_row.end_line,
        signature=node_row.signature,
    )
