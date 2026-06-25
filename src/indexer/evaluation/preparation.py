from dataclasses import dataclass
from pathlib import Path

from .corpus import RepositorySpec


@dataclass(frozen=True)
class PreparedRepository:
    spec: RepositorySpec
    repository_root: Path
    actual_revision: str
    index_path: Path
    source_files: tuple[Path, ...]


def prepare_repository(
    spec: RepositorySpec,
    *,
    indexes_dir: Path,
    rebuild: bool = False,
) -> PreparedRepository: ...
