"""
reporting.py - Aggregate and render evaluation scenario results.

Builds scenario-level and per-tool token summaries from raw trajectory events
and measured context deliveries. Reporting is read-only: it does not execute
tools, update ledgers, or write files.
"""

from collections import defaultdict
from dataclasses import dataclass

from indexer.errors import (
    DuplicateToolCallError,
    EvaluationReportingError,
    MixedTokenCounterError,
    UnknownDeliveryCallError,
)

from .models import ToolCall, ToolResult
from .runner import ScenarioResult


@dataclass(frozen=True)
class ToolSummary:
    """
    Contains aggregate measurements for one tool.

    Attributes:
        name: The provider-neutral tool name.
        calls: The total number of calls.
        successful_calls: The number of successful calls.
        failed_calls: The number of failed calls.
        delivered_results: The number of calls that produced measured output.
        total_tokens: The total tokens delivered by the tool.
        novel_tokens: The tokens delivered for the first time.
        repeated_tokens: The tokens repeated from an earlier delivery.
        duplication_rate: The repeated fraction of delivered tokens.
        total_duration_ms: The total recorded execution duration.
        average_duration_ms: The average duration across calls with timing data.
        average_tokens_per_delivery: The average tokens per measured delivery.
    """

    name: str
    calls: int
    successful_calls: int
    failed_calls: int
    delivered_results: int
    total_tokens: int
    novel_tokens: int
    repeated_tokens: int
    duplication_rate: float
    total_duration_ms: float
    average_duration_ms: float
    average_tokens_per_delivery: float


@dataclass(frozen=True)
class TokenSummary:
    """
    Contains aggregate measurements for one scenario execution.

    Attributes:
        scenario_id: The executed scenario identifier.
        counter_name: The token counter used for measured deliveries.
        completed_steps: The number of attempted scenario steps.
        successful_steps: The number of successful tool calls.
        failed_steps: The number of failed tool calls.
        delivered_results: The number of measured context deliveries.
        total_tokens: The total delivered context tokens.
        novel_tokens: The tokens delivered for the first time.
        repeated_tokens: The tokens repeated from earlier deliveries.
        duplication_rate: The repeated fraction of delivered tokens.
        elapsed_ms: The total scenario execution duration.
        tools: Per-tool summaries ordered by tool name.
    """

    scenario_id: str
    counter_name: str | None
    completed_steps: int
    successful_steps: int
    failed_steps: int
    delivered_results: int
    total_tokens: int
    novel_tokens: int
    repeated_tokens: int
    duplication_rate: float
    elapsed_ms: float
    tools: tuple[ToolSummary, ...]


@dataclass
class _MutableToolSummary:
    """
    Accumulates per-tool values while building a report.
    """

    calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    delivered_results: int = 0
    total_tokens: int = 0
    novel_tokens: int = 0
    repeated_tokens: int = 0
    total_duration_ms: float = 0.0
    measured_durations: int = 0


def summarize_result(result: ScenarioResult) -> TokenSummary:
    """
    Build aggregate token and tool metrics for a scenario result.

    Tool names are resolved from ToolCall records by call ID. Deliveries must
    reference calls present in the same result. All deliveries must use the
    same token counter.

    Args:
        result: The completed scenario result to summarize.

    Returns:
        Scenario-level and per-tool measurements.

    Raises:
        DuplicateToolCallError: If a call ID appears more than once.
        UnknownDeliveryCallError: If a delivery references an unknown call.
        MixedTokenCounterError: If deliveries use different counters.
    """
    calls_by_id: dict[str, ToolCall] = {}
    tool_totals: defaultdict[str, _MutableToolSummary] = defaultdict(
        _MutableToolSummary
    )

    for event in result.events:
        if isinstance(event, ToolCall):
            if event.call_id in calls_by_id:
                raise DuplicateToolCallError(event.call_id)

            calls_by_id[event.call_id] = event
            tool_totals[event.name].calls += 1
            continue

        if isinstance(event, ToolResult):
            call = calls_by_id.get(event.call_id)
            if call is None:
                continue

            totals = tool_totals[call.name]
            if event.status == "success":
                totals.successful_calls += 1
            else:
                totals.failed_calls += 1

            if event.duration_ms is not None:
                totals.total_duration_ms += event.duration_ms
                totals.measured_durations += 1

    counter_names = {delivery.counter_name for delivery in result.deliveries}
    if len(counter_names) > 1:
        raise MixedTokenCounterError(counter_names)

    for delivery in result.deliveries:
        call = calls_by_id.get(delivery.call_id)
        if call is None:
            raise UnknownDeliveryCallError(delivery.call_id)

        totals = tool_totals[call.name]
        totals.delivered_results += 1
        totals.total_tokens += delivery.tokens
        totals.novel_tokens += delivery.novel_tokens
        totals.repeated_tokens += delivery.repeated_tokens

    tools = tuple(
        _freeze_tool_summary(name, totals)
        for name, totals in sorted(tool_totals.items())
    )

    total_tokens = sum(delivery.tokens for delivery in result.deliveries)
    novel_tokens = sum(delivery.novel_tokens for delivery in result.deliveries)
    repeated_tokens = sum(delivery.repeated_tokens for delivery in result.deliveries)

    _validate_token_totals(
        total_tokens=total_tokens,
        novel_tokens=novel_tokens,
        repeated_tokens=repeated_tokens,
    )

    return TokenSummary(
        scenario_id=result.scenario_id,
        counter_name=next(iter(counter_names), None),
        completed_steps=result.completed_steps,
        successful_steps=result.successful_steps,
        failed_steps=result.failed_steps,
        delivered_results=len(result.deliveries),
        total_tokens=total_tokens,
        novel_tokens=novel_tokens,
        repeated_tokens=repeated_tokens,
        duplication_rate=_ratio(
            repeated_tokens,
            total_tokens,
        ),
        elapsed_ms=result.elapsed_ms,
        tools=tools,
    )


def summary_to_dict(summary: TokenSummary) -> dict[str, object]:
    """
    Convert a token summary to JSON-compatible values.

    Args:
        summary: The token summary to convert.

    Returns:
        A JSON-compatible summary dictionary.
    """
    return {
        "scenario_id": summary.scenario_id,
        "counter_name": summary.counter_name,
        "completed_steps": summary.completed_steps,
        "successful_steps": summary.successful_steps,
        "failed_steps": summary.failed_steps,
        "delivered_results": summary.delivered_results,
        "total_tokens": summary.total_tokens,
        "novel_tokens": summary.novel_tokens,
        "repeated_tokens": summary.repeated_tokens,
        "duplication_rate": summary.duplication_rate,
        "elapsed_ms": summary.elapsed_ms,
        "tools": [tool_summary_to_dict(tool) for tool in summary.tools],
    }


def tool_summary_to_dict(
    summary: ToolSummary,
) -> dict[str, object]:
    """
    Convert a per-tool summary to JSON-compatible values.

    Args:
        summary: The tool summary to convert.

    Returns:
        A JSON-compatible tool-summary dictionary.
    """
    return {
        "name": summary.name,
        "calls": summary.calls,
        "successful_calls": summary.successful_calls,
        "failed_calls": summary.failed_calls,
        "delivered_results": summary.delivered_results,
        "total_tokens": summary.total_tokens,
        "novel_tokens": summary.novel_tokens,
        "repeated_tokens": summary.repeated_tokens,
        "duplication_rate": summary.duplication_rate,
        "total_duration_ms": summary.total_duration_ms,
        "average_duration_ms": summary.average_duration_ms,
        "average_tokens_per_delivery": (summary.average_tokens_per_delivery),
    }


def render_summary_text(summary: TokenSummary) -> str:
    """
    Render a token summary as a plain-text report.

    Args:
        summary: The token summary to render.

    Returns:
        Human-readable report text.
    """
    counter = summary.counter_name or "none"
    lines = [
        f"# evaluation: {summary.scenario_id}",
        "",
        f"token counter:      {counter}",
        f"completed steps:    {summary.completed_steps}",
        f"successful steps:   {summary.successful_steps}",
        f"failed steps:       {summary.failed_steps}",
        f"delivered results:  {summary.delivered_results}",
        f"elapsed:            {summary.elapsed_ms:.2f} ms",
        "",
        f"total tokens:       {summary.total_tokens:,}",
        f"novel tokens:       {summary.novel_tokens:,}",
        f"repeated tokens:    {summary.repeated_tokens:,}",
        f"duplication rate:   {summary.duplication_rate:.2%}",
    ]

    if not summary.tools:
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "# tools",
            "",
            _render_tool_header(),
            _render_tool_separator(),
        ]
    )

    for tool in summary.tools:
        lines.append(_render_tool_row(tool))

    return "\n".join(lines)


def _freeze_tool_summary(
    name: str,
    totals: _MutableToolSummary,
) -> ToolSummary:
    """
    Convert one mutable accumulator to an immutable summary.

    Args:
        name: The tool name.
        totals: The accumulated measurements.

    Returns:
        Immutable per-tool summary.
    """
    _validate_token_totals(
        total_tokens=totals.total_tokens,
        novel_tokens=totals.novel_tokens,
        repeated_tokens=totals.repeated_tokens,
    )

    average_duration_ms = _ratio(
        totals.total_duration_ms,
        totals.measured_durations,
    )
    average_tokens = _ratio(
        totals.total_tokens,
        totals.delivered_results,
    )

    return ToolSummary(
        name=name,
        calls=totals.calls,
        successful_calls=totals.successful_calls,
        failed_calls=totals.failed_calls,
        delivered_results=totals.delivered_results,
        total_tokens=totals.total_tokens,
        novel_tokens=totals.novel_tokens,
        repeated_tokens=totals.repeated_tokens,
        duplication_rate=_ratio(
            totals.repeated_tokens,
            totals.total_tokens,
        ),
        total_duration_ms=totals.total_duration_ms,
        average_duration_ms=average_duration_ms,
        average_tokens_per_delivery=average_tokens,
    )


def _validate_token_totals(
    *,
    total_tokens: int,
    novel_tokens: int,
    repeated_tokens: int,
) -> None:
    """
    Validate aggregate token accounting.

    Args:
        total_tokens: Total delivered tokens.
        novel_tokens: Novel delivered tokens.
        repeated_tokens: Repeated delivered tokens.

    Raises:
        EvaluationReportingError: If aggregate totals are inconsistent.
    """
    if total_tokens != novel_tokens + repeated_tokens:
        raise EvaluationReportingError(
            message="Total tokens must equal novel tokens plus repeated tokens"
        )


def _ratio(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """
    Divide two values while handling a zero denominator.

    Args:
        numerator: The ratio numerator.
        denominator: The ratio denominator.

    Returns:
        The ratio, or zero when the denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _render_tool_header() -> str:
    """
    Render the per-tool table header.

    Returns:
        Plain-text header row.
    """
    return (
        f"{'tool':<24}"
        f"{'calls':>7}"
        f"{'ok':>7}"
        f"{'fail':>7}"
        f"{'tokens':>11}"
        f"{'novel':>11}"
        f"{'repeat':>11}"
        f"{'dup %':>9}"
        f"{'avg ms':>11}"
    )


def _render_tool_separator() -> str:
    """
    Render the per-tool table separator.

    Returns:
        Plain-text separator row.
    """
    return (
        f"{'-' * 24}"
        f"{'-' * 7}"
        f"{'-' * 7}"
        f"{'-' * 7}"
        f"{'-' * 11}"
        f"{'-' * 11}"
        f"{'-' * 11}"
        f"{'-' * 9}"
        f"{'-' * 11}"
    )


def _render_tool_row(summary: ToolSummary) -> str:
    """
    Render one row in the per-tool table.

    Args:
        summary: The tool summary to render.

    Returns:
        Plain-text table row.
    """
    return (
        f"{summary.name:<24}"
        f"{summary.calls:>7}"
        f"{summary.successful_calls:>7}"
        f"{summary.failed_calls:>7}"
        f"{summary.total_tokens:>11,}"
        f"{summary.novel_tokens:>11,}"
        f"{summary.repeated_tokens:>11,}"
        f"{summary.duplication_rate:>8.1%}"
        f"{summary.average_duration_ms:>11.2f}"
    )
