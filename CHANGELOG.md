# Changelog

## v1.0.3

- Promoted supported tree-sitter language parser packages to runtime
  dependencies so `indexer index` works across supported languages in
  production installs (including pipx)
- Updated installation and language docs to reflect runtime parser
  packaging and benchmark command output guidance

## v1.0.2

- Added benchmark matrix test infrastructure across language and repo-size
  combinations, including indexing runtime and token-usage metrics
- Improved benchmark output readability with a formatted summary table and
  clearer report-path visibility in `make benchmark`
- Fixed CLI packaging for pipx/PyPI installs by including the `cli` module in
  built artifacts
- Added `rich` as a runtime dependency required by the CLI entrypoint

## v1.0.1

- Implemented pass-6 v2 call resolution from `NodeRecord.properties` payloads
  (`imports`, `calls` / `call_sites`)
- Updated call-resolution pass to return edges plus `calls_resolved` and
  `calls_unresolved` statistics

## v1.0.0

- Initial release
