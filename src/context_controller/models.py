from dataclasses import dataclass


@dataclass(frozen=True)
class ContextItem:
    content_id: str
    kind: str
    content: str
    file_path: str | None
    qualified_name: str | None
    revision: str
    estimated_tokens: int
    selection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ContextBatch:
    items: tuple[ContextItem, ...]
    estimated_tokens: int
    request: str
    policy: str
