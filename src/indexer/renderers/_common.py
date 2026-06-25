"""
_common.py — Shared helpers for structured query result renderers.

Contains formatting and truncation helpers used by both human-readable and
JSON-compatible renderers. No helper in this module accesses the graph store.
"""

MAX_SOURCE_BYTES: int = 32_768
MAX_RELATED_SHOWN: int = 20
MAX_SEARCH_SIGNATURE_LEN: int = 80


def truncate_source(source: str) -> tuple[str, bool]:
    """
    Truncate source text at MAX_SOURCE_BYTES.

    The output snaps to the final complete line when possible.

    Args:
        source: raw source text.

    Returns:
        Tuple of rendered source and whether truncation occurred.
    """
    encoded = source.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_SOURCE_BYTES:
        return source, False

    truncated = encoded[:MAX_SOURCE_BYTES].decode(
        "utf-8",
        errors="replace",
    )
    last_newline = truncated.rfind("\n")
    if last_newline > MAX_SOURCE_BYTES // 2:
        truncated = truncated[:last_newline]
    return truncated, True


def truncate_signature(signature: str, max_len: int) -> str:
    """
    Truncate a signature to at most max_len characters.

    Attempts to cut at a comma or space near the requested limit.

    Args:
        signature: single-line signature.
        max_len: maximum output length, including the ellipsis.

    Returns:
        Original or truncated signature.
    """
    if len(signature) <= max_len:
        return signature

    cut = max(max_len - 3, 0)
    for index in range(cut, max(cut - 20, 0), -1):
        if index < len(signature) and signature[index] in {",", " "}:
            return signature[:index] + "..."
    return signature[:cut] + "..."


def format_qualified_name(qualified_name: str, width: int) -> str:
    """
    Fit a qualified name into a fixed-width field.

    Long names are truncated from the left so the most specific portion
    remains visible.

    Args:
        qualified_name: full dotted symbol name.
        width: output width in characters.

    Returns:
        Padded or left-truncated qualified name.
    """
    if len(qualified_name) <= width:
        return qualified_name.ljust(width)
    return "..." + qualified_name[-(width - 3) :]
