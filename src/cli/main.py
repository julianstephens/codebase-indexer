import json
import sys
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer
from rich import print
from rich.console import Console
from typer import Argument, Option, Typer

from indexer.context import build_context, build_skeleton
from indexer.pipeline import PipelineConfig
from indexer.pipeline import run as run_pipeline
from indexer.queries import query_callers, query_search, query_source
from indexer.renderers import (
    render_node_not_found,
    render_search_json,
    render_search_text,
    render_source_json,
    render_source_text,
    render_trace_json,
    render_trace_text,
)
from indexer.store import (
    DEFAULT_CACHE_DIR,
    default_db_path,
    open_path_readonly,
)

OutputFormat = Literal["text", "json"]

app = Typer(name="indexer", help="Codebase Indexer CLI")


def _require_db(
    project: str,
    db: str | None,
    cache_dir: str,
) -> str:
    """
    Resolve and validate the database path for a query command.

    Args:
        project: project name used to derive the default database path.
        db: optional explicit database path.
        cache_dir: directory containing project databases.
    Returns:
        Existing database path.
    Raises:
        typer.Exit: if the resolved database does not exist.
    """
    db_path = db if db else default_db_path(project, cache_dir)
    if not Path(db_path).exists():
        print(
            f"Error: database not found at {db_path!r}",
            file=sys.stderr,
        )
        raise typer.Exit(1)
    return db_path


def _write_result(
    text: str,
    json_result: dict[str, object],
    output_format: OutputFormat,
) -> None:
    """
    Write a rendered query result to standard output.

    Text output is printed without Rich markup processing.
    JSON output is serialized directly so it remains suitable for scripts and agent
    tools.

    Args:
        text: human-readable rendered result.
        json_result: JSON-compatible rendered result.
        output_format: requested output format.
    """
    if output_format == "json":
        sys.stdout.write(
            json.dumps(
                json_result,
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        return
    console = Console()
    console.print(
        text,
        markup=False,
        highlight=False,
    )


def _fail_node_not_found(qualified_name: str) -> NoReturn:
    """Print a node-not-found response and exit the command.

    Args:
        qualified_name: qualified name requested by the caller.
    Raises:
        typer.Exit: always, with exit status 1.
    """
    print(
        render_node_not_found(qualified_name),
        file=sys.stderr,
    )
    raise typer.Exit(1)


@app.command(name="index", help="Index a codebase")
def index(
    repo_path: Annotated[
        str,
        Argument(
            file_okay=False,
            dir_okay=True,
            exists=True,
            help="Path to the repository root",
        ),
    ] = ".",
    project: Annotated[
        str | None,
        Option(
            "--project",
            "-p",
            help="Project name (defaults to repo directory name)",
        ),
    ] = None,
    cache_dir: Annotated[
        str,
        Option(
            "--cache-dir",
            help="Directory for the working .db file",
            dir_okay=True,
            file_okay=False,
            exists=True,
        ),
    ] = DEFAULT_CACHE_DIR,
    workers: Annotated[
        int, Option("--workers", "-w", help="Number of parallel workers (0 = auto)")
    ] = 0,
    incremental: Annotated[
        bool, Option("--incremental/--no-incremental", help="Skip unchanged files")
    ] = True,
    export_artifact: Annotated[
        bool, Option("--export/--no-export", help="Write compressed .zst artifact")
    ] = True,
    verbose: Annotated[
        bool, Option("--verbose", "-v", help="Enable debug logging")
    ] = False,
) -> None:
    cfg = PipelineConfig(
        project=project or "",
        cache_dir=cache_dir,
        max_workers=workers,
        incremental=incremental,
        export_artifact=export_artifact,
        verbose=verbose,
    )
    try:
        result = run_pipeline(repo_path, cfg)
    except NotADirectoryError as exc:
        print(f"[red]Error:[/red] {repo_path!r} is not a directory")
        raise typer.Exit(1) from exc

    print(f"[bold]Project:[/bold]    {result.project}")
    print(f"[bold]DB:[/bold]         {result.db_path}")
    if result.artifact_path:
        print(f"[bold]Artifact:[/bold]   {result.artifact_path}")
    extracted = result.files_extracted
    unchanged = result.files_unchanged
    skipped = result.files_skipped
    print(f"Files:    {extracted} extracted, {unchanged} unchanged, {skipped} skipped")
    print(f"Nodes:      {result.nodes_total}")
    print(f"Edges:      {result.edges_total}")
    print(
        "Calls:      "
        f"{result.calls_discovered} discovered, "
        f"{result.calls_resolved} resolved, "
        f"{result.calls_unresolved} unresolved, "
        f"{result.calls_unsupported} unsupported"
    )
    if result.malformed_payloads:
        print(f"Malformed payloads: {result.malformed_payloads}")
    if result.relationship_unavailable_languages:
        unavailable = sorted(
            {
                lang
                for lang in result.relationship_unavailable_languages
                if lang != "unknown"
            }
        )
        if unavailable:
            print("Relationship unavailable: " + ", ".join(unavailable))
        if "unknown" in result.relationship_unavailable_languages:
            print("Relationship unavailable: unrecognized file types")
    print(f"Elapsed:    {result.elapsed_seconds:.2f}s")
    if result.errors:
        print(f"[red]Errors ({len(result.errors)}):[/red]")
        for path, msg in result.errors:
            print(f"  {path}: {msg}")


@app.command(
    name="skeleton",
    help="Print a skeleton of the codebase (file headers, imports, and signatures)",
)
def skeleton(
    project: Annotated[str, Argument(help="Project name")],
    db: Annotated[str | None, Option("--db", help="Path to the .db file")] = None,
    cache_dir: Annotated[
        str,
        Option(
            "--cache-dir",
            help="Directory for .db files",
            dir_okay=True,
            file_okay=False,
        ),
    ] = DEFAULT_CACHE_DIR,
    mode: Annotated[
        Literal["skeleton", "compact", "summary", "deps"] | None,
        Option(
            "--mode",
            "-m",
            help="Rendering mode: skeleton, compact, summary, deps (default: auto)",
        ),
    ] = None,
) -> None:
    db_path = _require_db(project or "", db, cache_dir)
    try:
        text = (
            build_skeleton(db_path, project)
            if mode is None
            else build_context(db_path, project, mode=mode)
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(1) from exc
    console = Console()
    console.print(text, markup=False, highlight=False)


@app.command(
    name="get-source",
    help="Get source code for a symbol",
)
def get_source(
    qualified_name: Annotated[
        str,
        Argument(help="Qualified name of the symbol, e.g. my_app.src.service.charge"),
    ],
    project: Annotated[
        str | None,
        Option("--project", "-p", help="Project name (required when db is not given)"),
    ] = None,
    db: Annotated[
        str | None,
        Option("--db", help="Path to the .db file"),
    ] = None,
    cache_dir: Annotated[
        str,
        Option(
            "--cache-dir",
            help="Directory for .db files",
            dir_okay=True,
            file_okay=False,
        ),
    ] = DEFAULT_CACHE_DIR,
    output_format: Annotated[
        OutputFormat,
        Option(
            "--format",
            "-f",
            help="Output format: text (default) or json",
        ),
    ] = "text",
) -> None:
    if not db and not project:
        print(
            "Error: provide --project or --db",
            file=sys.stderr,
        )
        raise typer.Exit(1)
    db_path = _require_db(project or "", db, cache_dir)
    store = open_path_readonly(db_path)
    try:
        result = query_source(
            store,
            qualified_name,
            project=project,
        )
    finally:
        store.close()
    if result is None:
        _fail_node_not_found(qualified_name)
    _write_result(
        render_source_text(result),
        render_source_json(result),
        output_format,
    )


@app.command(
    name="search",
    help="Search for symbols in the codebase",
)
def search(
    query: Annotated[
        str,
        Argument(help="Full-text search query"),
    ],
    project: Annotated[
        str | None,
        Option("--project", "-p", help="Filter by project name"),
    ] = None,
    label: Annotated[
        str | None,
        Option(
            "--label",
            "-l",
            help="Filter by label: Function, Class, Method, Interface, Type",
        ),
    ] = None,
    file_pattern: Annotated[
        str | None,
        Option(
            "--file",
            "-f",
            help="SQL LIKE pattern for file paths",
        ),
    ] = None,
    limit: Annotated[
        int,
        Option("--limit", "-n", help="Maximum number of results"),
    ] = 20,
    db: Annotated[
        str | None,
        Option("--db", help="Path to the .db file"),
    ] = None,
    cache_dir: Annotated[
        str,
        Option("--cache-dir", help="Directory for .db files"),
    ] = DEFAULT_CACHE_DIR,
    output_format: Annotated[
        OutputFormat,
        Option(
            "--format",
            "-f",
            help="Output format: text (default) or json",
        ),
    ] = "text",
) -> None:
    if not db and not project:
        print(
            "Error: provide --project or --db",
            file=sys.stderr,
        )
        raise typer.Exit(1)
    db_path = _require_db(project or "", db, cache_dir)
    store = open_path_readonly(db_path)
    try:
        result = query_search(
            store,
            query,
            project=project,
            label=label,
            file_pattern=file_pattern,
            limit=limit,
        )
    finally:
        store.close()
    if result is None:
        _fail_node_not_found(query)
    _write_result(
        render_search_text(result),
        render_search_json(result),
        output_format,
    )


@app.command(
    name="trace-callers",
    help="Trace direct and indirect callers of a symbol",
)
def trace_callers(
    qualified_name: Annotated[
        str,
        Argument(
            help=("Qualified name of the symbol, " "e.g. my_app.src.service.charge")
        ),
    ],
    project: Annotated[
        str | None,
        Option(
            "--project",
            "-p",
            help="Project name (required when db is not given)",
        ),
    ] = None,
    depth: Annotated[
        int,
        Option(
            "--depth",
            "-d",
            min=1,
            max=10,
            help="Maximum number of caller hops",
        ),
    ] = 3,
    db: Annotated[
        str | None,
        Option("--db", help="Path to the .db file"),
    ] = None,
    cache_dir: Annotated[
        str,
        Option(
            "--cache-dir",
            help="Directory for .db files",
            dir_okay=True,
            file_okay=False,
        ),
    ] = DEFAULT_CACHE_DIR,
    output_format: Annotated[
        OutputFormat,
        Option(
            "--format",
            help="Output format: text or json",
        ),
    ] = "text",
) -> None:
    """Trace callers of one indexed symbol.

    Args:
        qualified_name: exact qualified name of the starting symbol.
        project: optional project filter.
        depth: maximum caller traversal depth.
        db: optional explicit database path.
        cache_dir: directory containing project databases.
        output_format: text or JSON output.
    """
    if not db and not project:
        print(
            "Error: provide --project or --db",
            file=sys.stderr,
        )
        raise typer.Exit(1)
    db_path = _require_db(project or "", db, cache_dir)
    store = open_path_readonly(db_path)
    try:
        result = query_callers(
            store,
            qualified_name,
            project=project,
            depth=depth,
        )
    finally:
        store.close()
    if result is None:
        _fail_node_not_found(qualified_name)
    _write_result(
        render_trace_text(result),
        render_trace_json(result),
        output_format,
    )
