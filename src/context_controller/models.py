"""
models.py — Provider-neutral tool-call trajectory models.

Defines immutable records used to capture agent tool activity during a
software-engineering task. These models describe calls and results without
depending on a particular model provider or tool implementation.
"""

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

ToolStatus: TypeAlias = Literal[
    "success",
    "error",
    "cancelled",
]


@dataclass(frozen=True)
class ToolCall:
    """
    One tool invocation requested by an agent.

    A ToolCall records the request only. Execution outcome, returned content,
    timing, and token accounting are recorded separately in ToolResult.

    Attributes:
        call_id: unique identifier for this invocation.
        turn: 0-based agent turn in which the call was requested.
        sequence: global event sequence number within the task trajectory.
        name: provider-neutral tool name.
        arguments: normalized JSON-compatible tool arguments.
        purpose: optional agent-provided explanation of why the call is needed.
    """

    call_id: str
    turn: int
    sequence: int
    name: str
    arguments: dict[str, JsonValue] = field(default_factory=dict)
    purpose: str | None = None


@dataclass(frozen=True)
class ToolResult:
    """
    Execution result corresponding to one ToolCall.

    Attributes:
        call_id: identifier of the ToolCall that produced this result.
        sequence: global event sequence number within the task trajectory.
        status: execution outcome.
        output: textual output returned to the agent.
        error: error message when execution did not succeed.
        duration_ms: elapsed execution time in milliseconds.
        estimated_tokens: estimated tokens in the returned output.
        content_ids: stable context identifiers contained in the output.
    """

    call_id: str
    sequence: int
    status: ToolStatus
    output: str | None = None
    error: str | None = None
    duration_ms: float | None = None
    estimated_tokens: int | None = None
    content_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextItem:
    """
    A single piece of context content returned by a tool.

    Attributes:
        content_id: stable identifier for this content.
        kind: type of content, e.g. "repository_map", "search_hit", etc
        content: the actual content, e.g. a search hit snippet or symbol signature.
        file_path: optional file path associated with the content.
        qualified_name: optional fully-qualified symbol name associated with the
        content.
        revision: optional revision identifier associated with the content.
        estimated_tokens: estimated number of tokens in the content.
        selection_reasons: optional list of reasons why this content was selected for
        delivery.
    """

    content_id: str
    kind: Literal[
        "repository_map",
        "search_hit",
        "symbol_signature",
        "symbol_source",
        "caller_list",
        "callee_list",
        "test_output",
    ]
    content: str
    file_path: str | None
    qualified_name: str | None
    revision: str
    estimated_tokens: int
    selection_reasons: tuple[str, ...]


@dataclass
class ContextBatch:
    """
    A batch of context items returned by a tool.

    Attributes:
        items: list of context items in the batch.
        estimated_tokens: estimated number of tokens in the batch.
        request: the original request that generated this batch.
        policy: the policy applied to generate this batch.
    """

    items: list[ContextItem]
    estimated_tokens: int
    request: str
    policy: str


@dataclass
class ContextDelivery:
    """
    A delivery of context items to the agent.

    Attributes:
        turn: the turn number of the delivery.
        request: the original request that generated this delivery.
        item_ids: list of context item identifiers included in the delivery.
        logical_tokens: number of logical tokens in the delivery.
        novel_tokens: number of novel tokens in the delivery.
        repeated_tokens: number of repeated tokens in the delivery.
        cached_input_tokens: number of cached input tokens in the delivery.
        policy: the policy applied to generate this delivery.
    """

    turn: int
    request: str
    item_ids: list[str]
    logical_tokens: int
    novel_tokens: int
    repeated_tokens: int
    cached_input_tokens: int | None
    policy: str


@dataclass
class ContextSession:
    """
    A session of context interactions with the agent.

    Attributes:
        task_id: the identifier of the task associated with this session.
        model: the model used in this session.
        supplied_items: dictionary of context items supplied in this session.
        deliveries: list of context deliveries in this session.
        tool_calls: list of tool calls made in this session.
        token_budget: the token budget for this session.
    """

    task_id: str
    model: str
    supplied_items: dict[str, ContextItem]
    deliveries: list[ContextDelivery]
    tool_calls: list[ToolCall]
    token_budget: int
