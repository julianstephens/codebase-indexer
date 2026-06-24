# Supported Languages

The following languages are supported via tree-sitter parsers:

All parser packages for the languages below are included as runtime
dependencies in `codebase-indexer-py`, so standard installs (including pipx)
work without manual parser setup.

| Language | Extensions |
|---|---|
| Python | `.py` `.pyi` |
| TypeScript | `.ts` `.tsx` |
| JavaScript | `.js` `.jsx` `.mjs` `.cjs` |
| Go | `.go` |
| Rust | `.rs` |
| Java | `.java` |
| C | `.c` `.h` |
| C++ | `.cpp` `.cc` `.cxx` `.hpp` `.hxx` |
| C# | `.cs` |
| Ruby | `.rb` |
| PHP | `.php` |
| Kotlin | `.kt` `.kts` |
| Swift | `.swift` |
| Scala | `.scala` |
| Lua | `.lua` |
| Elixir | `.ex` `.exs` |
| Bash | `.sh` `.bash` |

## Unrecognised file types

Files with extensions not listed above (YAML, TOML, Dockerfile, SQL, Markdown, etc.) are stored as single `File` nodes so `get_source()` still works on them. To enable this behaviour, set `include_unknown_extensions=True` in `WalkConfig`.
