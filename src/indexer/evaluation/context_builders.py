from dataclasses import dataclass

from .corpus import BenchmarkTask
from .preparation import PreparedRepository
from .token_counter import TokenCounter


@dataclass(frozen=True)
class RepositoryContext:
    text: str
    files: tuple[str, ...]
    byte_length: int
    tokens: int


def build_full_repository_context(
    prepared: PreparedRepository,
    counter: TokenCounter,
) -> RepositoryContext: ...


def build_oracle_context(
    prepared: PreparedRepository,
    task: BenchmarkTask,
    counter: TokenCounter,
) -> RepositoryContext: ...
