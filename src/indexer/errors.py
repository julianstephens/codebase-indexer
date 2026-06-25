from pathlib import Path


class IndexerError(Exception):
    """Base class for all indexer errors."""

    pass


class QualifiedNameError(IndexerError):
    """Raised when a qualified name is invalid."""

    pass


class InvalidComputeArgumentsError(QualifiedNameError):
    """Raised when the arguments to compute() are invalid."""

    def __init__(
        self,
        file_path: str | None = None,
        name: str | None = None,
        parent: str | None = None,
    ):
        if file_path:
            self.file_path = file_path
            message = f"Invalid file path for qualified name: {file_path}"
        elif name:
            self.name = name
            message = f"Invalid symbol name for qualified name: {name}"
        elif parent:
            self.parent = parent
            message = f"Invalid parent name for qualified name: {parent}"
        else:
            message = "Invalid arguments for qualified name computation."
        super().__init__(message)


class FileExtensionNotSupportedError(IndexerError):
    """Raised when the extension of a file is not supported."""

    def __init__(self, extension: str):
        self.extension = extension
        message = f"File extension not supported: {extension}"
        super().__init__(message)


class StoreError(IndexerError):
    """Raised when there is an error with the store."""

    pass


class FileNotFoundError(StoreError):
    """Raised when a file is not found in the store."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"File not found in store: {file_path}"
        super().__init__(message)


class InvalidNodeRecordError(StoreError):
    """Raised when a node record is invalid."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Invalid node record"
        super().__init__(message)


class StoreOperationError(StoreError):
    """Raised when there is an error during a store operation."""

    def __init__(self, op: str | None = None, message: str | None = None):
        if message is None:
            message = "Store operation failed"
        if op:
            message = f"{message}: {op}"
        super().__init__(message)


class ArtifactError(StoreError):
    """Raised when there is an error with the artifact."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Artifact error"
        super().__init__(message)


class StoreFileNotFoundError(ArtifactError):
    """Raised when the database file is not found."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"Database file not found: {file_path}"
        super().__init__(message)


class InvalidStoreFileError(ArtifactError):
    """Raised when the database file is invalid."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"Invalid database file: {file_path}"
        super().__init__(message)


class ArtifactNotFoundError(ArtifactError):
    """Raised when the artifact file is not found."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"Artifact file not found: {file_path}"
        super().__init__(message)


class MetadataNotFoundError(ArtifactError):
    """Raised when the metadata file is not found."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"Metadata file not found: {file_path}"
        super().__init__(message)


class InvalidArtifactError(ArtifactError):
    """Raised when the artifact file is invalid."""

    def __init__(self, file_path: str, message: str | None = None):
        self.file_path = file_path
        if message is None:
            message = f"Invalid artifact file: {file_path}"
        super().__init__(message)


class ContextError(IndexerError):
    """Raised when there is an error with the context."""

    pass


class DatabaseNotFoundError(ContextError):
    """Raised when the database file is not found."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        message = f"Database file not found: {db_path}"
        super().__init__(message)


class InvalidContextError(ContextError):
    """Raised when the context is invalid."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Invalid context"
        super().__init__(message)


class QueryError(IndexerError):
    """Raised when there is an error with a query."""

    pass


class SearchQueryError(QueryError):
    """Raised when there is an error with a search query."""

    def __init__(self, query: str, message: str | None = None):
        self.query = query
        if message is None:
            message = f"Search query failed: {query}"
        super().__init__(message)


class EvaluationError(IndexerError):
    """Raised when there is an error with evaluation."""

    pass


class EvaluationSerializationError(EvaluationError):
    """Raised when an evaluation record cannot be serialized or parsed."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Evaluation serialization error"
        super().__init__(message)


class UnsupportedTrajectoryEventError(EvaluationError):
    """Raised when an unsupported trajectory event is encountered."""

    def __init__(self, event_type: str):
        self.event_type = event_type
        message = f"Unsupported trajectory event type: {event_type}"
        super().__init__(message)


class TokenizationError(EvaluationError):
    """Raised when there is an error during tokenization."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Tokenization error"
        super().__init__(message)


class InvalidHeuristicError(TokenizationError):
    """Raised when the heuristic token counter is configured with invalid parameters."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Invalid heuristic token counter configuration"
        super().__init__(message)


class EvaluationReportingError(EvaluationError):
    """Raised when there is an error during reporting."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Reporting error"
        super().__init__(message)


class DuplicateToolCallError(EvaluationReportingError):
    """Raised when multiple ToolCall records use the same call ID."""

    def __init__(self, call_id: str):
        self.call_id = call_id
        super().__init__(f"Duplicate tool call ID: {call_id}")


class UnknownDeliveryCallError(EvaluationReportingError):
    """Raised when a delivery references a missing ToolCall."""

    def __init__(self, call_id: str):
        self.call_id = call_id
        super().__init__(f"Context delivery references unknown tool call: {call_id}")


class MixedTokenCounterError(EvaluationReportingError):
    """Raised when deliveries from different token counters are combined."""

    def __init__(self, counter_names: set[str]):
        self.counter_names = frozenset(counter_names)
        counters = ", ".join(sorted(counter_names))
        super().__init__(f"Scenario contains multiple token counters: {counters}")


class RepositoryPreparationError(EvaluationError):
    pass


class RevisionMismatchError(RepositoryPreparationError):
    pass


class RepositoryPathError(RepositoryPreparationError):
    pass


class CorpusError(EvaluationError):
    """Raised when a benchmark corpus definition is invalid."""


class CorpusFileError(CorpusError):
    """Raised when a corpus manifest cannot be read or decoded."""

    def __init__(self, path: str | Path, message: str | None = None):
        self.path = str(path)
        if message is None:
            message = f"Corpus file error: {path}"
        super().__init__(message)


class InvalidRepositorySpecError(CorpusError):
    """Raised when a repository specification is invalid."""

    def __init__(self, spec: str | Path, message: str | None = None):
        self.spec = str(spec)
        if message is None:
            message = f"Invalid repository specification: {spec}"
        super().__init__(message)


class InvalidBenchmarkTaskError(CorpusError):
    """Raised when a benchmark task is invalid."""

    def __init__(self, task: str | Path, message: str | None = None):
        self.task = str(task)
        if message is None:
            message = f"Invalid benchmark task: {task}"
        super().__init__(message)
