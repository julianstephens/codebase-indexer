import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from indexer.errors import (
    CorpusFileError,
    InvalidBenchmarkTaskError,
    InvalidRepositorySpecError,
)

from .runner import EvaluationScenario, ToolStep

SUPPORTED_TOOLS = {"get_source", "search", "trace_callers"}


@dataclass(frozen=True)
class RepositorySpec:
    repository_id: str
    path: Path
    revision: str
    project: str
    languages: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    repository_id: str
    description: str
    oracle_files: tuple[str, ...]
    indexed_scenario: EvaluationScenario


def load_repository_spec(path: str | Path) -> RepositorySpec:
    """Load and validate one repository TOML manifest."""
    path = _validate_path(path, "repository spec path", relative=False)

    try:
        with path.open("rb", encoding="utf-8") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise CorpusFileError(
            message=f"Failed to load repository spec TOML: {e}",
            path=path,
        ) from e

    spec = _parse_repository_spec(data)
    validate_repository_spec(spec)
    return spec


def load_benchmark_task(
    path: str | Path,
    *,
    repository: RepositorySpec | None = None,
) -> BenchmarkTask:
    """Load and validate one benchmark-task JSON manifest."""
    path = _validate_path(path, "benchmark task path", relative=False)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise CorpusFileError(
            message=f"Failed to load benchmark task JSON: {e}",
            path=path,
        ) from e

    task = _parse_benchmark_task(data)
    validate_benchmark_task(task, repository=repository)
    if repository is not None and task.repository_id != repository.repository_id:
        raise InvalidBenchmarkTaskError(
            task=path,
            message=(
                f"Benchmark task repository ID '{task.repository_id}' "
                f"does not match repository spec ID '{repository.repository_id}'"
            ),
        )
    return task


def validate_repository_spec(spec: RepositorySpec) -> None:
    """Validate an already-constructed repository specification."""
    if not spec.repository_id:
        raise InvalidRepositorySpecError(
            spec=spec.path,
            message="Repository ID cannot be empty",
        )
    try:
        _validate_path(spec.path, "repository path", relative=False)
    except CorpusFileError as e:
        raise InvalidRepositorySpecError(
            spec=spec.path,
            message=f"Invalid repository path: {e}",
        ) from e
    if not spec.path.exists():
        raise InvalidRepositorySpecError(
            spec=spec.path,
            message=f"Repository path does not exist: {spec.path}",
        )
    if not spec.path.is_dir():
        raise InvalidRepositorySpecError(
            spec=spec.path,
            message=f"Repository path is not a directory: {spec.path}",
        )
    if not _is_sha(spec.revision):
        raise InvalidRepositorySpecError(
            spec=spec.path,
            message=f"Revision must be a valid SHA-1 hash: {spec.revision}",
        )
    if not spec.project:
        raise InvalidRepositorySpecError(
            spec=spec.path,
            message="Project name cannot be empty",
        )
    if len(spec.languages) == 0:
        raise InvalidRepositorySpecError(
            spec=spec.path,
            message="At least one language must be specified",
        )
    if len(spec.exclude) != len(set(spec.exclude)):
        raise InvalidRepositorySpecError(
            spec=spec.path,
            message="Exclude list contains duplicate entries",
        )
    for exclude_path in spec.exclude:
        _validate_path(exclude_path, "exclude path", must_exist=False, relative=True)


def validate_benchmark_task(
    task: BenchmarkTask,
    *,
    repository: RepositorySpec | None = None,
) -> None:
    """Validate an already-constructed task and optional repository match."""
    if not task.task_id:
        raise InvalidBenchmarkTaskError(
            task=task.task_id,
            message="Benchmark task ID cannot be empty",
        )
    if not task.repository_id:
        raise InvalidBenchmarkTaskError(
            task=task.task_id,
            message="Benchmark task repository ID cannot be empty",
        )
    if not task.description:
        raise InvalidBenchmarkTaskError(
            task=task.task_id,
            message="Benchmark task description cannot be empty",
        )
    if len(task.indexed_scenario.steps) == 0:
        raise InvalidBenchmarkTaskError(
            task=task.task_id,
            message="Benchmark task must contain at least one tool step",
        )
    for tool in task.indexed_scenario.steps:
        if tool.name not in SUPPORTED_TOOLS:
            raise InvalidBenchmarkTaskError(
                task=task.task_id,
                message=f"Unsupported tool in benchmark task: {tool.name}",
            )
    for oracle_file in task.oracle_files:
        path = _validate_path(
            oracle_file, "oracle file", must_exist=True, relative=True
        )
        normalized_path = PurePosixPath(path).as_posix()
        if normalized_path != oracle_file:
            raise InvalidBenchmarkTaskError(
                task=task.task_id,
                message=(
                    f"Oracle file path must be normalized and relative: {oracle_file} "
                    f"(normalized: {normalized_path})"
                ),
            )
        if ".." in normalized_path.split("/"):
            raise InvalidBenchmarkTaskError(
                task=task.task_id,
                message=f"Oracle file path must not contain '..': {oracle_file}",
            )
    if repository is not None and task.repository_id != repository.repository_id:
        raise InvalidBenchmarkTaskError(
            task=task.task_id,
            message=(
                f"Benchmark task repository ID '{task.repository_id}' "
                f"does not match repository spec ID '{repository.repository_id}'"
            ),
        )


def _parse_benchmark_task(payload: Mapping[str, object]) -> BenchmarkTask:
    task_id = _require_string(payload.get("task_id"), "task_id")
    scenario = EvaluationScenario(
        scenario_id=f"{task_id}:indexed-scripted",
        steps=tuple(
            _parse_tool_step(step)
            for step in payload.get("indexed_steps", [])  # type: ignore
        ),
        description=_require_string(payload.get("description"), "description"),
    )
    return BenchmarkTask(
        task_id=_require_string(payload.get("task_id"), "task_id"),
        repository_id=_require_string(payload.get("repository_id"), "repository_id"),
        description=_require_string(payload.get("description"), "description"),
        oracle_files=_optional_string_list(payload.get("oracle_files"), "oracle_files"),
        indexed_scenario=scenario,
    )


def _parse_tool_step(payload: Mapping[str, object]) -> ToolStep:
    return ToolStep(
        name=_require_string(payload.get("name"), "name"),
        arguments=payload.get("arguments", {}),  # type: ignore
        purpose=_require_string(payload.get("purpose"), "purpose"),
        content_kind=_require_string(payload.get("content_kind"), "content_kind"),
    )


def _require_string(value: object, field_name: str) -> str:
    if not value:
        raise InvalidRepositorySpecError(
            spec=field_name,
            message=f"Field '{field_name}' cannot be empty",
        )
    if not isinstance(value, str):
        raise InvalidRepositorySpecError(
            spec=field_name,
            message=(
                f"Expected string for field '{field_name}', "
                f"got {type(value).__name__}"
            ),
        )
    return value


def _optional_string_list(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidRepositorySpecError(
            spec=field_name,
            message=(
                f"Expected list for field '{field_name}', "
                f"got {type(value).__name__}"
            ),
        )
    return tuple(
        _require_string(item, f"{field_name}[{i}]") for i, item in enumerate(value)
    )


def _validate_path(
    value: str | Path, field_name: str, must_exist=True, relative=True
) -> Path:
    if not value:
        raise CorpusFileError(message=f"{field_name} is empty", path="")
    if isinstance(value, str):
        value = Path(value)
    if must_exist and not value.exists():
        raise CorpusFileError(message=f"{field_name} does not exist", path=value)

    if relative and value.is_absolute():
        raise InvalidRepositorySpecError(
            spec=field_name,
            message=(
                f"Field '{field_name}' must be a relative path, "
                f"got absolute path: {value}"
            ),
        )
    return value


def _parse_repository_spec(payload: Mapping[str, object]) -> RepositorySpec:
    return RepositorySpec(
        repository_id=_require_string(payload.get("repository_id"), "repository_id"),
        path=Path(_require_string(payload.get("path"), "path")),
        revision=_require_string(payload.get("revision"), "revision"),
        project=_require_string(payload.get("project"), "project"),
        languages=_optional_string_list(payload.get("languages"), "languages"),
        exclude=_optional_string_list(payload.get("exclude"), "exclude"),
    )


def _is_sha(string):
    return bool(re.match(r"^[a-fA-F0-9]{40}$", string))
