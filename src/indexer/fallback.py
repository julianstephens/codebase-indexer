"""
fallback.py — Whole-file extractor for unrecognised file types.

Used by extractor.py when detect_language() returns None or when the
tree-sitter extractor returns an empty result for a recognised language
(e.g. a file with only comments or imports, no definitions).

Produces a single NodeRecord with label="File" representing the entire
file. This ensures every file in the repo is reachable via get_source()
even when no symbols could be extracted from it.

Typical inputs:
    Dockerfile, docker-compose.yml, .env, Makefile, *.toml, *.yaml,
    *.json, *.md, *.txt, *.sql, *.proto, *.graphql, shell scripts with
    no function definitions, generated files.

The NodeRecord produced has:
    label       = "File"
    name        = the filename (e.g. "Dockerfile", "schema.sql")
    signature   = "# <repo-relative-path>"
    source      = full raw file content
    start_line  = 1
    end_line    = number of lines in the file
    parent      = "" (always top-level)
    language    = detected language or "unknown"
    properties  = {"fallback": True, "size_bytes": <int>,
                   "line_count": <int>, "reason": <str>}

Token budget note:
    File nodes are included in the skeleton as a single comment line
    (the signature). Their full source is only returned when the agent
    explicitly calls get_source() on the node's qualified_name. Large
    generated or binary-adjacent files (minified JS, lock files) are
    truncated at MAX_SOURCE_BYTES to prevent unbounded storage growth.

Public API:
    extract_fallback(path, source, reason="no_language")
        Main entry point. Always returns a list containing exactly one
        NodeRecord.

    should_skip(path, source)
        Returns True for files that should not be stored at all:
        lock files, minified bundles, binary-adjacent files, files
        exceeding MAX_SOURCE_BYTES after truncation is considered.

    truncate_source(source)
        Truncates source to MAX_SOURCE_BYTES at a line boundary,
        appending a truncation notice.
"""

from pathlib import Path

from .languages import EXTENSION_TO_LANG
from .treesitter import NodeRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Files larger than this are truncated before storage.
# 512 KB covers virtually all hand-written source files.
MAX_SOURCE_BYTES: int = 512 * 1024  # 512 KB

# Files larger than this are skipped entirely (no NodeRecord produced).
# Covers minified bundles, lock files with thousands of entries, etc.
MAX_FILE_BYTES: int = 2 * 1024 * 1024  # 2 MB

# Filename patterns that are always skipped.
# Matched against the lowercased filename (stem + suffix).
_SKIP_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "composer.lock",
        "cargo.lock",  # Rust; Cargo.toml is kept
        "gemfile.lock",
        "poetry.lock",
        "pipfile.lock",
        "flake.lock",  # Nix
        ".ds_store",
        "thumbs.db",
    }
)

# File extensions that are always skipped.
# Matched against the lowercased suffix.
_SKIP_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Compiled / binary
        ".pyc",
        ".pyo",
        ".pyd",
        ".class",
        ".jar",
        ".war",
        ".o",
        ".a",
        ".so",
        ".dll",
        ".exe",
        ".lib",
        ".wasm",
        # Media
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        # Archives
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".zst",
        ".7z",
        # Fonts
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        # Database / binary data
        ".db",
        ".sqlite",
        ".sqlite3",
        ".pkl",
        ".pickle",
        ".npy",
        ".npz",
        # Generated / minified (extension-based)
        ".min.js",  # note: matched on full suffix chain below
        ".min.css",
        ".map",  # source maps
    }
)

# Suffixes indicating minified content even without .min in the name.
# Checked via heuristic in should_skip().
_MINIFIED_INDICATORS: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".chunk.js",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_fallback(
    path: str,
    source: str,
    reason: str = "no_language",
) -> list[NodeRecord]:
    """
    Produce a single File NodeRecord for an unrecognised or empty file.

    Always returns a list containing exactly one NodeRecord. Never raises.
    If should_skip() returns True, returns an empty list instead.

    The qualified_name field is left empty ("") — the pipeline sets it
    via fqn.module() after calling this function.

    Args:
        path:   repo-relative file path, e.g. "Dockerfile" or
                "config/settings.yaml"
        source: full raw file content as a string
        reason: short string describing why fallback extraction was used.
                Stored in properties["reason"]. Common values:
                  "no_language"   — extension not in EXTENSION_TO_LANG
                  "no_definitions" — language recognised but no symbols found
                  "parse_error"   — tree-sitter raised on this file

    Returns:
        A list containing one NodeRecord with label="File", or an empty
        list if should_skip() returns True for this path/source.

    Examples:
        >>> records = extract_fallback("Dockerfile", source)
        >>> len(records)
        1
        >>> records[0].label
        'File'
        >>> records[0].name
        'Dockerfile'
        >>> records[0].properties["fallback"]
        True
    """
    if should_skip(path, source):
        return []

    name = _filename(path)
    language = _detect_language(path)
    line_count = _count_lines(source)
    size_bytes = len(source.encode("utf-8", errors="replace"))
    truncated_source, was_truncated = truncate_source(source)

    properties = {
        "fallback": True,
        "reason": reason,
        "line_count": line_count,
        "size_bytes": size_bytes,
        "was_truncated": was_truncated,
    }

    return [
        NodeRecord(
            file_path=path,
            label="File",
            name=name,
            signature=f"# {path}",
            source=truncated_source,
            start_line=1,
            end_line=line_count,
            parent="",
            language=language,
            properties=properties,
        )
    ]


def should_skip(path: str, source: str) -> bool:
    """
    Return True if this file should be excluded from the index entirely.

    A file is skipped when any of the following are true:
      1. Its lowercased filename matches _SKIP_FILENAMES exactly.
      2. Its lowercased extension matches _SKIP_EXTENSIONS.
      3. Its name ends with any suffix in _MINIFIED_INDICATORS.
      4. Its encoded size exceeds MAX_FILE_BYTES.
      5. It appears to be a minified file: a single line longer than
         5,000 characters (heuristic for minified JS/CSS).

    This function is also called by walker.py before reading file content
    for extensions in _SKIP_EXTENSIONS, so the source may be an empty
    string in that case — size check falls back to len(source).

    Args:
        path:   repo-relative file path
        source: full raw file content (may be empty string for
                extension-only checks from walker.py)

    Returns:
        True if the file should be excluded, False otherwise.

    Examples:
        >>> should_skip("package-lock.json", "")
        True
        >>> should_skip("src/utils.py", "def foo(): pass")
        False
        >>> should_skip("dist/bundle.min.js", "")
        True
        >>> should_skip("README.md", "# Hello")
        False
    """
    if _filename(path) in _SKIP_FILENAMES:
        return True
    if _extension(path) in _SKIP_EXTENSIONS:
        return True
    if any(_filename(path).endswith(suffix) for suffix in _MINIFIED_INDICATORS):
        return True
    if len(source.encode("utf-8", errors="replace")) > MAX_FILE_BYTES:
        return True
    return _is_minified(source)


def truncate_source(source: str) -> tuple[str, bool]:
    """
    Truncate source to MAX_SOURCE_BYTES at a line boundary.

    Finds the last newline before the byte limit and cuts there,
    then appends a human-readable truncation notice so the agent
    knows the content was cut.

    Args:
        source: raw file content string

    Returns:
        A (truncated_source, was_truncated) tuple.
        was_truncated is True if the content was shortened.
        If the source fits within MAX_SOURCE_BYTES, returns
        (source, False) unchanged.

    Examples:
        >>> short = "line1\nline2\n"
        >>> truncated, did_truncate = truncate_source(short)
        >>> did_truncate
        False
        >>> truncated == short
        True

        >>> long_source = "x\n" * 300_000   # ~600 KB
        >>> truncated, did_truncate = truncate_source(long_source)
        >>> did_truncate
        True
        >>> "[truncated]" in truncated
        True
        >>> len(truncated.encode()) <= MAX_SOURCE_BYTES + 200
        True
    """
    encoded = source.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_SOURCE_BYTES:
        return source, False

    # Slice to the byte limit, then back up to the last newline so we
    # don't cut in the middle of a multi-byte character or a line.
    cut = encoded[:MAX_SOURCE_BYTES]
    last_newline = cut.rfind(b"\n")
    if last_newline != -1:
        cut = cut[: last_newline + 1]

    truncated = cut.decode("utf-8", errors="replace")
    truncated += f"\n# [truncated: file exceeded {MAX_SOURCE_BYTES // 1024} KB]\n"
    return truncated, True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filename(path: str) -> str:
    """
    Return the lowercased filename (name + suffix) from a path.

    Args:
        path: any file path string

    Returns:
        Lowercased filename, e.g. "package-lock.json", "dockerfile"

    Examples:
        >>> _filename("config/package-lock.json")
        'package-lock.json'
        >>> _filename("Dockerfile")
        'dockerfile'
    """
    return Path(path).name.lower()


def _extension(path: str) -> str:
    """
    Return the lowercased file extension including the leading dot.

    For compound extensions like ".min.js", returns only the final
    suffix (".js"). Use _filename() to check compound suffixes.

    Args:
        path: any file path string

    Returns:
        Lowercased extension string, e.g. ".py", ".js", ".yaml".
        Returns "" if the file has no extension.

    Examples:
        >>> _extension("src/utils.py")
        '.py'
        >>> _extension("Makefile")
        ''
        >>> _extension("bundle.min.js")
        '.js'
    """
    suffix = Path(path).suffix
    return suffix.lower() if suffix else ""


def _detect_language(path: str) -> str:
    """
    Return the language name for the file, or "unknown" if unrecognised.

    Unlike languages.detect_language(), this function never returns None
    — it falls back to "unknown" so the NodeRecord always has a
    non-empty language field.

    Uses a small set of additional mappings for common non-code files
    that are worth labelling:
        .yaml / .yml  → "yaml"
        .json         → "json"
        .toml         → "toml"
        .md / .mdx    → "markdown"
        .sql          → "sql"
        .proto        → "protobuf"
        .graphql/.gql → "graphql"
        .env          → "dotenv"
        Dockerfile*   → "dockerfile"
        Makefile*     → "makefile"
        *.sh          → "bash"  (already in EXTENSION_TO_LANG but
                                 included here as belt-and-suspenders)

    Args:
        path: any file path string

    Returns:
        A language name string, never None or empty.

    Examples:
        >>> _detect_language("config/settings.yaml")
        'yaml'
        >>> _detect_language("Dockerfile")
        'dockerfile'
        >>> _detect_language("some.unknown.extension")
        'unknown'
    """
    ext = _extension(path)
    if ext in (".yaml", ".yml"):
        return "yaml"
    if ext == ".json":
        return "json"
    if ext == ".toml":
        return "toml"
    if ext in (".md", ".mdx"):
        return "markdown"
    if ext == ".sql":
        return "sql"
    if ext == ".proto":
        return "protobuf"
    if ext in (".graphql", ".gql"):
        return "graphql"
    if ext == ".env":
        return "dotenv"
    filename = _filename(path)
    if filename.startswith("dockerfile"):
        return "dockerfile"
    if filename.startswith("makefile"):
        return "makefile"
    return EXTENSION_TO_LANG.get(ext, "unknown")


def _count_lines(source: str) -> int:
    """
    Return the number of lines in source.

    An empty string has 0 lines. A string with no newline has 1 line.
    A trailing newline does not add an extra line.

    Args:
        source: raw file content

    Returns:
        Integer line count >= 0.

    Examples:
        >>> _count_lines("")
        0
        >>> _count_lines("hello")
        1
        >>> _count_lines("a\nb\nc")
        3
        >>> _count_lines("a\nb\n")
        2
    """
    if not source:
        return 0
    return len(source.splitlines())


def _is_minified(source: str) -> bool:
    """
    Heuristic check for minified content.

    A file is considered minified if it has at least one line longer
    than 5,000 characters. This catches minified JS/CSS that doesn't
    have .min in its filename.

    Only examines the first 10 lines for performance — minified files
    typically have their content on line 1.

    Args:
        source: raw file content

    Returns:
        True if any of the first 10 lines exceeds 5,000 characters.

    Examples:
        >>> _is_minified("short line\n")
        False
        >>> _is_minified("x" * 6000 + "\n")
        True
    """
    return any(len(line) > 5000 for line in source.splitlines()[:10])
