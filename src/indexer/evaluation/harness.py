from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .corpus import BenchmarkTask
from .preparation import PreparedRepository
from .reporting import TokenSummary
from .runner import ScenarioResult
from .token_counter import TokenCounter

type PolicyName = Literal[
    "full-source",
    "oracle-source",
    "indexed-scripted",
]


@dataclass(frozen=True)
class PolicyRun:
    policy: PolicyName
    result: ScenarioResult
    summary: TokenSummary


@dataclass(frozen=True)
class TaskBenchmarkResult:
    repository_id: str
    revision: str
    task_id: str
    runs: tuple[PolicyRun, ...]


def run_task_benchmark(
    prepared: PreparedRepository,
    task: BenchmarkTask,
    *,
    counter: TokenCounter,
    policies: tuple[PolicyName, ...],
    results_dir: Path,
) -> TaskBenchmarkResult: ...
