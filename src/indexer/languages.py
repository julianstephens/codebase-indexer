"""
languages.py — Language detection and tree-sitter grammar configuration.

Two public data structures:

  EXTENSION_TO_LANG
      Maps file extensions to canonical language names.
      Used by detect_language() and walker.py to tag FileInfo objects.

  LANG_CONFIG
      Maps canonical language names to tree-sitter parser names and
      definition node type tables.

      Each entry in `definitions` maps a tree-sitter node type string
      to a (label, name_field) pair:
        label       — the graph node label: Function | Class | Method |
                      Interface | Type
        name_field  — the tree-sitter field name that holds the symbol's
                      short name (passed to node.child_by_field_name()).
                      None means the name must be extracted by a
                      language-specific fallback in treesitter.py.

One public function:

  detect_language(path)
      Returns the canonical language name for a file path, or None if
      the extension is not recognised.

Adding a new language:
  1. Add its extensions to EXTENSION_TO_LANG.
  2. Add a LANG_CONFIG entry. Find node type names by running the
     tree-sitter playground or inspecting the grammar's node-types.json:
       https://github.com/tree-sitter/tree-sitter-<lang>/blob/master/src/node-types.json
  3. Add any name_field=None entries to _CUSTOM_NAME_EXTRACTORS in
     treesitter.py if the name cannot be read from a single named field.
"""


# ---------------------------------------------------------------------------
# Extension → language
# ---------------------------------------------------------------------------
#
# Keys are lowercase extensions including the leading dot.
# Values are canonical language names that match keys in LANG_CONFIG.
#
# Multiple extensions can map to the same language (e.g. .js and .jsx
# both map to "javascript").
#
# Files with no entry here are passed to fallback.py, which stores the
# whole file as a single File node.

EXTENSION_TO_LANG: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",  # type stub files — valid Python syntax
    # JavaScript / TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",  # ES modules
    ".cjs": "javascript",  # CommonJS modules
    ".ts": "typescript",
    ".tsx": "typescript",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
    # Java
    ".java": "java",
    # C / C++
    ".c": "c",
    ".h": "c",  # treat headers as C; overridden by .hpp/.hxx
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    # Ruby
    ".rb": "ruby",
    # PHP
    ".php": "php",
    # C#
    ".cs": "c_sharp",
    # Shell
    ".sh": "bash",
    ".bash": "bash",
    # Kotlin
    ".kt": "kotlin",
    ".kts": "kotlin",
    # Swift
    ".swift": "swift",
    # Scala
    ".scala": "scala",
    # Lua
    ".lua": "lua",
    # Elixir
    ".ex": "elixir",
    ".exs": "elixir",
}


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------
#
# Structure:
#   LANG_CONFIG[language_name] = {
#       "parser":      str,           # name passed to get_parser()
#       "definitions": {
#           "<ts_node_type>": ("<Label>", "<name_field>" | None),
#           ...
#       }
#   }
#
# tree-sitter node type names are grammar-specific. When in doubt, check
# the grammar's node-types.json or use the tree-sitter playground.
#
# name_field=None signals that treesitter.py must use a custom extractor
# registered in _CUSTOM_NAME_EXTRACTORS for that (language, node_type).
#
# Labels must be one of:
#   Function | Class | Method | Interface | Type
#
# Methods are definitions that appear inside a class body. treesitter.py
# sets the `parent` field on the NodeRecord when recursing into class
# bodies, so the pipeline can distinguish methods from top-level functions
# even when both use the "Function" label.

LANG_CONFIG: dict[str, dict] = {
    # ── Python ───────────────────────────────────────────────────────────
    "python": {
        "parser": "python",
        "definitions": {
            "function_definition": ("Function", "name"),
            "async_function_definition": ("Function", "name"),
            "class_definition": ("Class", "name"),
            # decorated_definition wraps a function or class in decorators.
            # treesitter.py unwraps it and delegates to the inner node type.
            "decorated_definition": ("Function", None),
        },
    },
    # ── JavaScript ───────────────────────────────────────────────────────
    "javascript": {
        "parser": "javascript",
        "definitions": {
            "function_declaration": ("Function", "name"),
            "function_expression": ("Function", "name"),
            # Arrow functions are often anonymous; name_field=None means
            # treesitter.py looks at the parent assignment for the name.
            "arrow_function": ("Function", None),
            "class_declaration": ("Class", "name"),
            "class_expression": ("Class", "name"),
            "method_definition": ("Method", "name"),
            # export_statement wraps a declaration; treesitter.py unwraps.
            "export_statement": ("Function", None),
            "generator_function_declaration": ("Function", "name"),
        },
    },
    # ── TypeScript ───────────────────────────────────────────────────────
    "typescript": {
        "parser": "typescript",
        "definitions": {
            "function_declaration": ("Function", "name"),
            "function_signature": ("Function", "name"),
            "async_function_declaration": ("Function", "name"),
            "arrow_function": ("Function", None),
            "class_declaration": ("Class", "name"),
            "abstract_class_declaration": ("Class", "name"),
            "method_definition": ("Method", "name"),
            "method_signature": ("Method", "name"),
            "interface_declaration": ("Interface", "name"),
            "type_alias_declaration": ("Type", "name"),
            "enum_declaration": ("Type", "name"),
            "export_statement": ("Function", None),
        },
    },
    # ── Go ───────────────────────────────────────────────────────────────
    "go": {
        "parser": "go",
        "definitions": {
            "function_declaration": ("Function", "name"),
            # method_declaration has a receiver; treesitter.py extracts
            # the receiver type and sets it as the parent field.
            "method_declaration": ("Method", "name"),
            "type_declaration": ("Type", None),  # unwrap spec
        },
    },
    # ── Rust ─────────────────────────────────────────────────────────────
    "rust": {
        "parser": "rust",
        "definitions": {
            "function_item": ("Function", "name"),
            "struct_item": ("Class", "name"),
            "enum_item": ("Type", "name"),
            "trait_item": ("Interface", "name"),
            # impl_item has no name field directly; treesitter.py reads
            # the type being implemented from the "type" field.
            "impl_item": ("Class", None),
            "mod_item": ("Type", "name"),
        },
    },
    # ── Java ─────────────────────────────────────────────────────────────
    "java": {
        "parser": "java",
        "definitions": {
            "method_declaration": ("Function", "name"),
            "class_declaration": ("Class", "name"),
            "interface_declaration": ("Interface", "name"),
            "enum_declaration": ("Type", "name"),
            "record_declaration": ("Class", "name"),
            "constructor_declaration": ("Function", "name"),
            "annotation_type_declaration": ("Interface", "name"),
        },
    },
    # ── C ────────────────────────────────────────────────────────────────
    "c": {
        "parser": "c",
        "definitions": {
            # declarator is nested; treesitter.py walks to find the
            # innermost identifier.
            "function_definition": ("Function", None),
            "struct_specifier": ("Class", "name"),
            "enum_specifier": ("Type", "name"),
            "type_definition": ("Type", None),
        },
    },
    # ── C++ ──────────────────────────────────────────────────────────────
    "cpp": {
        "parser": "cpp",
        "definitions": {
            "function_definition": ("Function", None),
            "class_specifier": ("Class", "name"),
            "struct_specifier": ("Class", "name"),
            "namespace_definition": ("Type", "name"),
            "template_declaration": ("Function", None),
        },
    },
    # ── Ruby ─────────────────────────────────────────────────────────────
    "ruby": {
        "parser": "ruby",
        "definitions": {
            "method": ("Function", "name"),
            "singleton_method": ("Function", "name"),
            "class": ("Class", "name"),
            "module": ("Type", "name"),
        },
    },
    # ── PHP ──────────────────────────────────────────────────────────────
    "php": {
        "parser": "php",
        "definitions": {
            "function_definition": ("Function", "name"),
            "method_declaration": ("Method", "name"),
            "class_declaration": ("Class", "name"),
            "interface_declaration": ("Interface", "name"),
            "trait_declaration": ("Type", "name"),
            "enum_declaration": ("Type", "name"),
        },
    },
    # ── C# ───────────────────────────────────────────────────────────────
    "c_sharp": {
        "parser": "c_sharp",
        "definitions": {
            "method_declaration": ("Function", "name"),
            "class_declaration": ("Class", "name"),
            "interface_declaration": ("Interface", "name"),
            "struct_declaration": ("Class", "name"),
            "record_declaration": ("Class", "name"),
            "enum_declaration": ("Type", "name"),
            "constructor_declaration": ("Function", "name"),
            "local_function_statement": ("Function", "name"),
        },
    },
    # ── Bash ─────────────────────────────────────────────────────────────
    "bash": {
        "parser": "bash",
        "definitions": {
            "function_definition": ("Function", "name"),
        },
    },
    # ── Kotlin ───────────────────────────────────────────────────────────
    "kotlin": {
        "parser": "kotlin",
        "definitions": {
            "function_declaration": ("Function", "simple_identifier"),
            "class_declaration": ("Class", "simple_identifier"),
            "object_declaration": ("Class", "simple_identifier"),
            "interface_declaration": ("Interface", "simple_identifier"),
            "secondary_constructor": ("Function", None),
        },
    },
    # ── Swift ────────────────────────────────────────────────────────────
    "swift": {
        "parser": "swift",
        "definitions": {
            "function_declaration": ("Function", "simple_identifier"),
            "class_declaration": ("Class", "type_identifier"),
            "struct_declaration": ("Class", "type_identifier"),
            "protocol_declaration": ("Interface", "type_identifier"),
            "enum_declaration": ("Type", "type_identifier"),
            "init_declaration": ("Function", None),
        },
    },
    # ── Scala ────────────────────────────────────────────────────────────
    "scala": {
        "parser": "scala",
        "definitions": {
            "function_definition": ("Function", "identifier"),
            "class_definition": ("Class", "identifier"),
            "object_definition": ("Class", "identifier"),
            "trait_definition": ("Interface", "identifier"),
            "val_definition": ("Type", "identifier"),
        },
    },
    # ── Lua ──────────────────────────────────────────────────────────────
    "lua": {
        "parser": "lua",
        "definitions": {
            "function_declaration": ("Function", "name"),
            "local_function": ("Function", "name"),
            # Method-style: table.method = function(...)
            "assignment_statement": ("Function", None),
        },
    },
    # ── Elixir ───────────────────────────────────────────────────────────
    "elixir": {
        "parser": "elixir",
        "definitions": {
            "def": ("Function", "name"),
            "defp": ("Function", "name"),  # private
            "defmodule": ("Class", "name"),
            "defprotocol": ("Interface", "name"),
            "defmacro": ("Function", "name"),
        },
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_language(path: str) -> str | None:
    """
    Return the canonical language name for a file path, or None if the
    extension is not in EXTENSION_TO_LANG.

    Detection is based solely on the file extension. No content sniffing
    is performed — that is left to fallback.py for edge cases.

    Args:
        path: any file path string; only the suffix is examined.

    Returns:
        A language name matching a key in LANG_CONFIG, e.g. "python",
        "typescript", "go". Returns None for unrecognised extensions.

    Examples:
        >>> detect_language("src/auth/views.py")
        'python'
        >>> detect_language("src/api/handler.go")
        'go'
        >>> detect_language("Dockerfile")
        None
        >>> detect_language("README.md")
        None
    """
    if not path:
        return None
    ext = "." + path.rsplit(".", 1)[-1].lower()
    return EXTENSION_TO_LANG.get(ext)


def supported_extensions() -> list[str]:
    """
    Return a sorted list of all file extensions with tree-sitter support.

    Useful for walker.py to pre-filter files before passing them to the
    extractor, and for CLI help text.

    Returns:
        Sorted list of extension strings, e.g. [".bash", ".c", ".cc", ...]

    Examples:
        >>> ".py" in supported_extensions()
        True
        >>> ".md" in supported_extensions()
        False
    """
    return sorted(EXTENSION_TO_LANG.keys())


def supported_languages() -> list[str]:
    """
    Return a sorted list of canonical language names that have a LANG_CONFIG
    entry and can be parsed by tree-sitter.

    Returns:
        Sorted list of language name strings,
        e.g. ["bash", "c", "cpp", "c_sharp", ...]

    Examples:
        >>> "python" in supported_languages()
        True
        >>> "markdown" in supported_languages()
        False
    """
    return sorted(LANG_CONFIG.keys())
