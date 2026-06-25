"""
context.py — Skeleton renderer and token budget manager.

Produces the text representation of the repository that is sent to the
agent at the start of each session. The agent reads this skeleton once,
identifies which nodes are relevant to the task, then fetches full
source on demand via tools.py.

Skeleton format:

    # <project> — <N> files, <N> nodes
    # schema: Function=120 Class=30 Method=85 Interface=5 Type=8 File=12

    ### src/payments/service.py
    # imports: stripe, src.payments.models, src.auth.models
    def charge(user: User, amount_cents: int, currency: str) -> Payment:
    # src.payments.service.charge
    def refund(payment: Payment) -> bool:  # src.payments.service.refund

    ### src/payments/models.py
    class Payment(BaseModel):  # src.payments.models.Payment
        def save(self) -> None:  # src.payments.models.Payment.save
        def to_dict(self) -> dict:  # src.payments.models.Payment.to_dict

Token budget logic:
    The renderer operates in one of four modes, chosen automatically
    based on the estimated token count of the full skeleton vs the
    configured budget:

        skeleton  — file headers + all signatures + QN comments
                    Used when full skeleton fits in the token budget.

        compact   — skeleton but signatures truncated at 80 chars,
                    QN comment omitted. Used when skeleton is slightly
                    over budget.

        summary   — one line per file showing path, counts, token estimate.
                    Used when the repo is large.

        deps      — import edges only, no code at all.
                    Used when even the summary exceeds the budget.

    The mode is chosen by build_context() using _choose_mode(). Callers
    can also request a mode explicitly.

Public API:
    build_context(db_path, project, token_budget, mode)  →  str
        Main entry point. Returns the context string for the agent.

    build_skeleton(db_path, project)  →  str
        Render the full skeleton regardless of token budget.

    estimate_tokens(text)  →  int
        Rough token count: len(text) // 4.

    MODES  — frozenset of valid mode strings.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DatabaseNotFoundError, InvalidContextError
from .store import SearchParams, Store, open_path_readonly

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid rendering modes.
MODES: frozenset[str] = frozenset({"skeleton", "compact", "summary", "deps"})

# Default token budget. Fits comfortably in a 16k-token context window
# alongside a system prompt and a few tool call results.
DEFAULT_TOKEN_BUDGET: int = 8_000

# Tokens per character (rough approximation).
# GPT-4 / Claude average ~4 chars per token for source code.
CHARS_PER_TOKEN: int = 4

# Maximum number of imports shown per file in the skeleton header.
# Longer import lists are truncated with " ...".
MAX_IMPORTS_SHOWN: int = 8

# Maximum signature line length in compact mode.
COMPACT_MAX_SIG_LEN: int = 80

# Indentation used for methods under their class.
METHOD_INDENT: str = "    "


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _SkeletonEntry:
    """
    One line of the skeleton: a single node's representation.

    Attributes:
        qn:         qualified name
        signature:  single-line signature string
        label:      node label (Function, Class, Method, etc.)
        parent:     enclosing class short name, or ""
        start_line: 1-based source line (for ordering)
    """

    qn: str
    signature: str
    label: str
    parent: str
    start_line: int


@dataclass
class _FileBlock:
    """
    All skeleton data for one file, collected before rendering.

    Attributes:
        path:       repo-relative file path
        imports:    list of import strings (short module names)
        entries:    list of _SkeletonEntry in source order
    """

    path: str
    imports: list[str] = field(default_factory=list)
    entries: list[_SkeletonEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "MODES",
    "build_context",
    "build_skeleton",
    "estimate_tokens",
]


def build_context(
    db_path: str,
    project: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    mode: str | None = None,
) -> str:
    """
    Build the repository context string for the agent.

    Opens the database at db_path in read-only mode, queries the
    skeleton data, chooses a rendering mode, and returns the rendered
    string.

    Mode selection (when mode=None):
        1. Estimate tokens for full skeleton.
        2. If estimate <= token_budget              → "skeleton"
        3. If estimate <= token_budget * 2          → "compact"
        4. If estimate <= token_budget * 10         → "summary"
        5. Otherwise                                → "deps"

    When mode is given explicitly, it is used regardless of token count.
    Must be one of MODES, otherwise ValueError is raised.

    Args:
        db_path:       absolute path to the working .db file.
        project:       project name to render (must exist in the db).
        token_budget:  soft token limit for automatic mode selection.
                       Defaults to DEFAULT_TOKEN_BUDGET (8,000).
        mode:          explicit rendering mode, or None for auto.

    Returns:
        Rendered context string. Never empty — at minimum contains the
        project header line.

    Raises:
        FileNotFoundError: if db_path does not exist.
        ValueError:        if mode is not in MODES.

    Examples:
        >>> ctx = build_context("/cache/my-app.db", "my-app")
        >>> ctx.startswith("# my-app")
        True
        >>> "def charge" in ctx
        True
    """
    if mode is not None and mode not in MODES:
        raise InvalidContextError(
            message=f"Invalid mode {mode!r}. Must be one of {sorted(MODES)}"
        )

    if not Path(db_path).exists():
        raise DatabaseNotFoundError(db_path)

    store = open_path_readonly(db_path)
    try:
        blocks = _load_blocks(store, project)
        summary = store.get_schema_summary(project)

        if mode is None:
            full_text = _render_skeleton(blocks, project, summary)
            mode = _choose_mode(full_text, token_budget)

        return _render(blocks, project, summary, mode)
    finally:
        store.close()


def build_skeleton(db_path: str, project: str) -> str:
    """
    Render the full skeleton for a project regardless of token budget.

    Convenience wrapper over build_context() that always uses
    mode="skeleton". Useful for CLI output and debugging.

    Args:
        db_path: absolute path to the working .db file
        project: project name

    Returns:
        Full skeleton string.

    Raises:
        FileNotFoundError: if db_path does not exist.
    """
    return build_context(db_path, project, mode="skeleton")


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a string.

    Uses the rough approximation of 1 token per CHARS_PER_TOKEN (4)
    characters. This is accurate enough for budget decisions and avoids
    a dependency on a tokeniser library.

    Args:
        text: any string

    Returns:
        Non-negative integer token estimate.

    Examples:
        >>> estimate_tokens("def charge(): pass")
        4
        >>> estimate_tokens("")
        0
    """
    return max(0, len(text) // CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(
    blocks: list[_FileBlock],
    project: str,
    summary: dict,
    mode: str,
) -> str:
    """
    Dispatch to the appropriate renderer for the given mode.

    Args:
        blocks:  list of _FileBlock objects, one per file
        project: project name
        summary: dict from store.get_schema_summary()
        mode:    one of MODES

    Returns:
        Rendered string for the given mode.
    """
    if mode == "skeleton":
        return _render_skeleton(blocks, project, summary)
    if mode == "compact":
        return _render_compact(blocks, project, summary)
    if mode == "summary":
        return _render_summary(blocks, project, summary)
    if mode == "deps":
        return _render_deps(blocks, project, summary)
    return _render_skeleton(blocks, project, summary)


def _render_skeleton(
    blocks: list[_FileBlock],
    project: str,
    summary: dict,
) -> str:
    """
    Render the full skeleton: file headers + all signatures + QN comments.

    Format per file:

        ### src/payments/service.py
        # imports: stripe, src.payments.models, src.auth.models
        def charge(user: User, amount_cents: int, currency: str) -> Payment:
        # src.payments.service.charge
        def refund(payment: Payment) -> bool:  # src.payments.service.refund

        ### src/payments/models.py
        class Payment(BaseModel):  # src.payments.models.Payment
            def save(self) -> None:  # src.payments.models.Payment.save

    Methods are indented under their class. Classes immediately precede
    their methods. File nodes are listed as a single comment line with
    no signature.

    Args:
        blocks:  list of _FileBlock objects
        project: project name
        summary: schema summary dict

    Returns:
        Full skeleton string with a header and one section per file.
    """
    parts = [_render_header(project, summary, "skeleton")]

    for block in sorted(blocks, key=lambda b: b.path):
        if not block.entries and not block.imports:
            continue

        parts.append(f"\n### {block.path}")

        if block.imports:
            shown = block.imports[:MAX_IMPORTS_SHOWN]
            rest = len(block.imports) - len(shown)
            suffix = " ..." if rest > 0 else ""
            parts.append(f"# imports: {', '.join(shown)}{suffix}")

        # Group methods under their class
        classes_seen: set[str] = set()
        for entry in sorted(block.entries, key=lambda e: e.start_line):
            if entry.parent and entry.parent not in classes_seen:
                # Parent class not yet emitted (e.g. File node without class)
                pass

            if entry.label == "Class":
                classes_seen.add(entry.qn.split(".")[-1])
                parts.append(f"{entry.signature}  # {entry.qn}")
            elif entry.label == "File":
                parts.append(entry.signature)
            elif entry.parent:
                parts.append(f"{METHOD_INDENT}{entry.signature}  # {entry.qn}")
            else:
                parts.append(f"{entry.signature}  # {entry.qn}")

    return "\n".join(parts)


def _render_compact(
    blocks: list[_FileBlock],
    project: str,
    summary: dict,
) -> str:
    """
    Render a compact skeleton: signatures truncated at COMPACT_MAX_SIG_LEN,
    QN comments omitted, imports omitted.

    Used when the full skeleton slightly exceeds the token budget.
    Typically achieves ~40% token reduction compared to full skeleton.

    Format per file:

        ### src/payments/service.py
        def charge(user: User, amount_cents: int, currency:...
        def refund(payment: Payment) -> bool:
        class Payment(BaseModel):
            def save(self) -> None:

    Args:
        blocks:  list of _FileBlock objects
        project: project name
        summary: schema summary dict

    Returns:
        Compact skeleton string.
    """
    parts = [_render_header(project, summary, "compact")]

    for block in sorted(blocks, key=lambda b: b.path):
        if not block.entries:
            continue

        parts.append(f"\n### {block.path}")
        for entry in sorted(block.entries, key=lambda e: e.start_line):
            if entry.label == "File":
                continue
            sig = _truncate_sig(entry.signature, COMPACT_MAX_SIG_LEN)
            indent = METHOD_INDENT if entry.parent else ""
            parts.append(f"{indent}{sig}")

    return "\n".join(parts)


def _render_summary(
    blocks: list[_FileBlock],
    project: str,
    summary: dict,
) -> str:
    """
    Render a one-line-per-file summary: path, label counts, token estimate.

    Used when the repo is large and the skeleton would exceed the token
    budget even after compaction. Gives the agent a map of what exists
    and where without any code content.

    Format:

        # <project> — summary mode (N files)
        # Use search("query") or get_source("qn") to retrieve code.
        #
        # path                                      cls  fn  method  (~N tokens)
        src/payments/service.py                       0   2       0  (~85 tokens)
        src/payments/models.py                        1   0       3  (~120 tokens)
        ...

    Args:
        blocks:  list of _FileBlock objects
        project: project name
        summary: schema summary dict

    Returns:
        Summary string.
    """
    parts = [_render_header(project, summary, "summary")]
    parts.append('# Use search("query") or get_source("qn") to retrieve code.\n#')
    parts.append(f"# {'path':<60}  cls   fn  meth   est.tokens")
    parts.append(f"# {'-'*60}  ---  ---  ----  ----------")

    for block in sorted(blocks, key=lambda b: b.path):
        cls_count = sum(1 for e in block.entries if e.label == "Class")
        fn_count = sum(1 for e in block.entries if e.label == "Function")
        meth_count = sum(1 for e in block.entries if e.label == "Method")
        sig_text = "\n".join(e.signature for e in block.entries)
        tok_estimate = estimate_tokens(sig_text)
        parts.append(
            f"{block.path:<62}  {cls_count:>3}  {fn_count:>3}  {meth_count:>4}"
            f"  (~{tok_estimate} tokens)"
        )

    return "\n".join(parts)


def _render_deps(
    blocks: list[_FileBlock],
    project: str,
    summary: dict,
) -> str:
    """
    Render a pure dependency map: import edges only, no code.

    Used as a last resort when even the summary exceeds the token
    budget. Provides structural information at minimal token cost.

    Format:

        # <project> — deps mode
        # Use search("query") or get_source("qn") to retrieve code.

        # import graph
        src/payments/service.py -> stripe
        src/payments/service.py -> src.payments.models
        src/payments/models.py  -> src.db.base

    Args:
        blocks:  list of _FileBlock objects
        project: project name
        summary: schema summary dict

    Returns:
        Dependency map string.
    """
    parts = [_render_header(project, summary, "deps")]
    parts.append('# Use search("query") or get_source("qn") to retrieve code.\n')
    parts.append("# import graph")

    for block in sorted(blocks, key=lambda b: b.path):
        for imp in block.imports:
            parts.append(f"{block.path} -> {imp}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _render_header(
    project: str,
    summary: dict,
    mode: str,
) -> str:
    """
    Render the single-line header shown at the top of every context string.

    Format:
        # <project> — <N> files, <N> nodes  [<mode>]
        # schema: Function=120 Class=30 Method=85 Interface=5 Type=8 File=12

    Args:
        project: project name
        summary: dict from store.get_schema_summary() with keys
                 total_nodes, node_labels (list of {label, count})
        mode:    rendering mode string

    Returns:
        Two-line header string (no trailing newline).
    """
    total = summary.get("total_nodes", 0)
    file_count = sum(
        label["count"]
        for label in summary.get("node_labels", [])
        if label["label"] == "File"
    )
    label_parts = " ".join(
        f"{label['label']}={label['count']}"
        for label in sorted(summary.get("node_labels", []), key=lambda x: x["label"])
    )
    line1 = f"# {project} — {file_count} files, {total} nodes  [{mode}]"
    line2 = f"# schema: {label_parts}" if label_parts else "# schema: (empty)"
    return f"{line1}\n{line2}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_blocks(store: Store, project: str) -> list[_FileBlock]:
    """
    Query the store and assemble _FileBlock objects for every file.

    Uses store.iter_skeleton() to stream (file_path, signature, qn)
    rows, then groups them by file_path. For each file, also fetches
    the node's label and parent from the nodes table so methods can be
    indented under their class.

    Import strings are derived from the IMPORTS edges in the graph
    (edges of type IMPORTS where source is a file/module node), or from
    the properties column of Module nodes if present.

    The blocks are returned in arbitrary order — sorting is done in the
    renderer.

    Args:
        store:   open Store (read-only is fine)
        project: project name

    Returns:
        List of _FileBlock objects, one per file that has at least one
        node.
    """
    # Group skeleton rows by file
    file_map: dict[str, _FileBlock] = {}

    for file_path, signature, qn in store.iter_skeleton(project):
        if file_path not in file_map:
            file_map[file_path] = _FileBlock(path=file_path)

        # Fetch the node to get label and parent
        node = store.get_node_by_qn(qn, project=project)
        if node is None:
            continue

        # Parse import list from properties if available
        if file_path not in file_map or not file_map[file_path].imports:
            imports = _extract_imports_from_node(node)
            if imports:
                file_map[file_path].imports = imports

        file_map[file_path].entries.append(
            _SkeletonEntry(
                qn=qn,
                signature=signature,
                label=node.label,
                parent=node.properties.get("parent", ""),  # type: ignore
                start_line=node.start_line,
            )
        )

    # Also load File nodes for files that appear in the files table
    # but may not have extracted symbol nodes (config files, etc.)
    _load_file_nodes(store, project, file_map)

    return list(file_map.values())


def _load_file_nodes(
    store: Store,
    project: str,
    file_map: dict[str, _FileBlock],
) -> None:
    """
    Add File-label nodes to file_map for files with no other symbols.

    Queries nodes with label="File" and adds an entry for each one
    whose path is not already in file_map. This ensures fallback-
    extracted files (Dockerfiles, YAML configs, etc.) appear in the
    skeleton even though they have no function/class definitions.

    Mutates file_map in place.

    Args:
        store:    open Store
        project:  project name
        file_map: dict being built by _load_blocks()
    """

    result = store.search_nodes(
        SearchParams(
            project=project,
            label="File",
            limit=1000,
        )
    )
    for node in result.rows:
        if node.file_path not in file_map:
            # Skip files with no recognised language (dotfiles, config
            # files, etc.) — they add noise without aiding code navigation.
            props = node.properties or {}
            if props.get("reason") == "no_language":
                continue
            block = _FileBlock(path=node.file_path)
            block.entries.append(
                _SkeletonEntry(
                    qn=node.qualified_name,
                    signature=node.signature,
                    label="File",
                    parent="",
                    start_line=node.start_line,
                )
            )
            file_map[node.file_path] = block


def _extract_imports_from_node(node) -> list[str]:
    """
    Extract a short import list from a NodeRow's properties.

    The extractor stores import information in node.properties under
    the key "imports" as a list of module path strings. This function
    retrieves and shortens them to their last two dotted components for
    readability in the skeleton header.

    If no imports key is present, returns an empty list.

    Args:
        node: NodeRow from the store

    Returns:
        List of short import strings, e.g.
        ["stripe", "src.payments.models", "src.auth.models"]
    """
    raw = node.properties.get("imports", [])
    if not isinstance(raw, list):
        return []
    result = []
    for imp in raw:
        if isinstance(imp, str) and imp:
            # Shorten to last two components: "src.payments.models" → kept,
            # "django.contrib.auth.models" → "auth.models"
            parts = imp.split(".")
            short = ".".join(parts[-2:]) if len(parts) > 2 else imp
            result.append(short)
    return result


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------


def _choose_mode(full_skeleton: str, token_budget: int) -> str:
    """
    Choose a rendering mode based on full skeleton token count vs budget.

    Thresholds:
        <= budget * 1    → "skeleton"
        <= budget * 2    → "compact"
        <= budget * 10   → "summary"
        > budget * 10    → "deps"

    Args:
        full_skeleton: the rendered full skeleton string
        token_budget:  soft token limit from the caller

    Returns:
        One of: "skeleton" | "compact" | "summary" | "deps"

    Examples:
        >>> _choose_mode("short text", 8000)
        'skeleton'
    """
    tokens = estimate_tokens(full_skeleton)
    if tokens <= token_budget:
        return "skeleton"
    if tokens <= token_budget * 2:
        return "compact"
    if tokens <= token_budget * 10:
        return "summary"
    return "deps"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _truncate_sig(signature: str, max_len: int) -> str:
    """
    Truncate a signature line to at most max_len characters.

    Appends "..." if the line is longer than max_len. Truncation happens
    at a comma or space boundary where possible to avoid cutting in the
    middle of a type annotation.

    Args:
        signature: single-line signature string
        max_len:   maximum character length (including the "...")

    Returns:
        Signature string of length <= max_len. If the signature is
        already within the limit, returns it unchanged.

    Examples:
        >>> _truncate_sig("def charge(user: User, amount: int) -> Payment:", 30)
        'def charge(user: User, amount:...'
        >>> _truncate_sig("def short():", 80)
        'def short():'
    """
    if len(signature) <= max_len:
        return signature
    # Try to cut at a comma or space near the limit
    cut = max_len - 3
    for i in range(cut, max(cut - 20, 0), -1):
        if signature[i] in (", ", " ", ","):
            return signature[:i] + "..."
    return signature[:cut] + "..."


def _format_label_counts(summary: dict) -> str:
    """
    Format node label counts into a compact string for the header.

    Args:
        summary: dict from store.get_schema_summary()

    Returns:
        String like "Function=120 Class=30 Method=85"
        Empty string if no labels are present.

    Examples:
        >>> _format_label_counts({"node_labels": [
        ...     {"label": "Function", "count": 5},
        ...     {"label": "Class",    "count": 2},
        ... ]})
        'Class=2 Function=5'
    """
    labels = summary.get("node_labels", [])
    if not labels:
        return ""
    return " ".join(
        f"{label['label']}={label['count']}"
        for label in sorted(labels, key=lambda x: x["label"])
    )
