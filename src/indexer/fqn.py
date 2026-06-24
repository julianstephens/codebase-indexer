"""
fqn.py — Qualified name computation.

A qualified name (QN) is the globally unique address for every node in the
graph. It is used as the primary key for edge resolution, get_source()
lookups, and the skeleton renderer.

Format:
    <module_path>.<SymbolName>
    <module_path>.<ClassName>.<method_name>

Where module_path is derived from the repo-relative file path:
    src/payments/service.py  →  src.payments.service
    src/payments/__init__.py →  src.payments
    src/payments/index.ts    →  src.payments

Examples:
    compute("src/payments/service.py", "charge", parent=None)
        → "src.payments.service.charge"

    compute("src/auth/models.py", "get_full_name", parent="User")
        → "src.auth.models.User.get_full_name"

    module("src/payments/service.py")
        → "src.payments.service"

    folder("src/payments")
        → "src.payments"
"""

from pathlib import Path, PurePosixPath

from .errors import InvalidComputeArgumentsError

# Filename stems that are dropped from the module path because they add
# no information (the directory name already identifies the module).
_TRANSPARENT_STEMS = frozenset(
    {
        "__init__",  # Python package init
        "index",  # JS/TS barrel files
        "mod",  # Rust module root
        "main",  # common entry point — kept in QN to avoid collisions
    }
)


def compute(file_path: str, name: str, parent: str | None = None) -> str:
    """
    Compute the qualified name for a symbol.

    Args:
        file_path: repo-relative file path, e.g. "src/payments/service.py"
        name:      symbol short name, e.g. "charge"
        parent:    enclosing class/struct name for methods, e.g. "PaymentService"
                   Pass None for top-level definitions.

    Raises:
        InvalidComputeArgumentsError: if file_path, name, or parent are invalid or
        missing.

    Returns:
        Dotted qualified name string, e.g. "src.payments.service.charge"
        or "src.payments.service.PaymentService.charge"

    Examples:
        >>> compute("src/payments/service.py", "charge")
        'src.payments.service.charge'
        >>> compute("src/auth/models.py", "get_full_name", parent="User")
        'src.auth.models.User.get_full_name'
        >>> compute("src/payments/__init__.py", "setup")
        'src.payments.setup'
    """
    if not file_path:
        raise InvalidComputeArgumentsError(file_path=file_path)
    if not name:
        raise InvalidComputeArgumentsError(name=name)
    if parent is not None and not parent:
        raise InvalidComputeArgumentsError(parent=parent)

    mod = _path_to_module(file_path)
    segements = [mod, parent, name] if parent else [mod, name]
    return ".".join(s for s in segements if s)


def module(file_path: str) -> str:
    """
    Compute the module qualified name for a file (no symbol name appended).

    This is the QN used for Module and File nodes, and as the module_qn
    prefix during call resolution in registry.py.

    Args:
        file_path: repo-relative file path, e.g. "src/payments/service.py"

    Returns:
        Dotted module path string, e.g. "src.payments.service"
        Transparent stems (__init__, index, mod) collapse to the parent dir.

    Examples:
        >>> module("src/payments/service.py")
        'src.payments.service'
        >>> module("src/payments/__init__.py")
        'src.payments'
        >>> module("src/payments/index.ts")
        'src.payments'
        >>> module("auth.py")
        'auth'
    """
    return _path_to_module(file_path)


def folder(dir_path: str) -> str:
    """
    Compute the qualified name for a directory (Package/Folder node).

    Args:
        dir_path: repo-relative directory path, e.g. "src/payments"

    Returns:
        Dotted path string, e.g. "src.payments"

    Examples:
        >>> folder("src/payments")
        'src.payments'
        >>> folder("src")
        'src'
        >>> folder(".")
        ''
    """
    if not dir_path or dir_path == ".":
        return ""
    parts = [s for s in dir_path.replace("\\", "/").split("/") if s and s != "."]
    return ".".join(parts)


def from_path(file_path: str) -> str:
    """
    Derive a project name from an absolute or relative repository root path.

    Used by pipeline.py to set the project name when none is provided.
    Strips non-alphanumeric characters and normalises separators.

    Args:
        file_path: absolute or relative path to the repo root,
                   e.g. "/home/user/projects/my-app" or "../my-app"

    Returns:
        A safe, lowercase project name, e.g. "my-app"

    Examples:
        >>> from_path("/home/user/projects/my-app")
        'my-app'
        >>> from_path("/home/user/projects/my_app/")
        'my_app'
        >>> from_path(".")
        'unknown'
    """
    name = Path(file_path.rstrip("/\\")).name
    return name if name and name not in (".", "..") else "unknown"


def _path_to_module(file_path: str) -> str:
    """
    Convert a repo-relative file path to a dotted module path.

    Steps:
      1. Strip the file extension.
      2. Replace all path separators (/ and \\) with dots.
      3. If the final segment is a transparent stem (__init__, index, mod),
         drop it so the parent directory is the module address.
      4. Strip any leading dots.

    This is the shared core used by both module() and compute().

    Args:
        file_path: repo-relative file path with or without extension.

    Returns:
        Dotted module path string, never starts or ends with a dot.

    Examples:
        >>> _path_to_module("src/payments/service.py")
        'src.payments.service'
        >>> _path_to_module("src/payments/__init__.py")
        'src.payments'
        >>> _path_to_module("src\\auth\\models.py")
        'src.auth.models'
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    parts = [*list(p.parent.parts), p.stem]
    parts = [s for s in parts if s and s != "."]
    if parts and parts[-1] in _TRANSPARENT_STEMS:
        parts.pop()
    return ".".join(parts)
