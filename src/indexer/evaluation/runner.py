"""
runner.py - Executes deterministic evaluation scenarios.

Coordinates scripted tool calls, records trajectory events, measures returned
context, and produces a complete scenario result. The runner is independent of
specific tools, repositories, token-counter implementations, and persistence
formats.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

from .ledger import TokenLedger
from .models import (
    ContextDelivery,
    JsonValue,
    ToolCall,
    ToolResult,
    TrajectoryEvent,
)
from .token_counter import TokenCounter

type ToolFunction = Callable[..., str]
type Clock = Callable[[], float]
type CallIdFactory = Callable[[int], str]


class EventSink(Protocol):
    """
    Receives evaluation records as a scenario executes.

    Implementations may persist records to JSONL, collect them in memory, or
    forward them to another reporting system.
    """

    def record_event(self, event: TrajectoryEvent) -> None:
        """
        Record one raw trajectory event.

        Args:
            event: The ToolCall or ToolResult to record.
        """
        ...

    def record_delivery(self, delivery: ContextDelivery) -> None:
        """
        Record one measured context delivery.

        Args:
            delivery: The context delivery to record.
        """
        ...


@dataclass(frozen=True)
class ToolStep:
    """
    Describes one tool invocation in a scripted scenario.

    Attributes:
        name: The registered tool name.
        arguments: The JSON-compatible arguments passed to the tool.
        purpose: The reason the step appears in the scenario.
        content_kind: The semantic kind assigned to successful output.
    """

    name: str
    arguments: dict[str, JsonValue] = field(default_factory=dict)
    purpose: str | None = None
    content_kind: str = "tool_output"


@dataclass(frozen=True)
class EvaluationScenario:
    """
    Describes a deterministic sequence of tool calls.

    Attributes:
        scenario_id: The stable identifier for the scenario.
        description: The behavior exercised by the scenario.
        steps: The ordered tool invocations.
    """

    scenario_id: str
    description: str
    steps: tuple[ToolStep, ...]


@dataclass(frozen=True)
class RunnerConfig:
    """
    Configures scenario execution behavior.

    Attributes:
        continue_on_error: Whether execution continues after a failed tool.
    """

    continue_on_error: bool = True


@dataclass(frozen=True)
class ScenarioResult:
    """
    Contains raw and measured output from one scenario execution.

    Attributes:
        scenario_id: The executed scenario identifier.
        events: The raw trajectory events in sequence order.
        deliveries: The measured successful context deliveries.
        elapsed_ms: The total scenario execution time in milliseconds.
        completed_steps: The number of steps that were attempted.
        successful_steps: The number of successful tool invocations.
        failed_steps: The number of failed tool invocations.
    """

    scenario_id: str
    events: tuple[TrajectoryEvent, ...]
    deliveries: tuple[ContextDelivery, ...]
    elapsed_ms: float
    completed_steps: int
    successful_steps: int
    failed_steps: int


class ToolNotRegisteredError(KeyError):
    """Raised when a scenario references an unknown tool."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Tool is not registered: {name}")


@dataclass(frozen=True)
class ToolRegistry:
    """
    Resolves provider-neutral tool names to callable functions.

    Attributes:
        tools: The registered tool functions keyed by name.
    """

    tools: Mapping[str, ToolFunction]

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        """
        Invoke one registered tool.

        Args:
            name: The registered tool name.
            arguments: The arguments passed to the function.

        Returns:
            The textual tool output.

        Raises:
            ToolNotRegisteredError: If the tool is not registered.
            TypeError: If the arguments do not match the function signature.
        """
        try:
            tool = self.tools[name]
        except KeyError as exc:
            raise ToolNotRegisteredError(name) from exc

        return tool(**dict(arguments))


@dataclass
class ScenarioRunner:
    """
    Executes scripted scenarios and records their token expenditure.

    Attributes:
        registry: The tools available to scenarios.
        counter: The counter used to measure successful tool output.
        config: Execution behavior.
        sink: Optional incremental event and delivery sink.
        clock: Monotonic clock used for duration measurement.
        call_id_factory: Function producing stable call IDs from step indexes.
    """

    registry: ToolRegistry
    counter: TokenCounter
    config: RunnerConfig = field(default_factory=RunnerConfig)
    sink: EventSink | None = None
    clock: Clock = perf_counter
    call_id_factory: CallIdFactory = lambda index: f"call-{index + 1:04d}"

    def run(self, scenario: EvaluationScenario) -> ScenarioResult:
        """
        Execute one evaluation scenario.

        Tool exceptions are converted into failed ToolResult records. When
        continue_on_error is false, execution stops after the first failure.

        Args:
            scenario: The deterministic scenario to execute.

        Returns:
            The complete raw and measured scenario result.
        """
        events: list[TrajectoryEvent] = []
        ledger = TokenLedger(counter=self.counter)
        successful_steps = 0
        failed_steps = 0

        started_at = self.clock()

        for step_index, step in enumerate(scenario.steps):
            call_id = self.call_id_factory(step_index)
            call_sequence = len(events)

            call = ToolCall(
                call_id=call_id,
                turn=step_index,
                sequence=call_sequence,
                name=step.name,
                arguments=dict(step.arguments),
                purpose=step.purpose,
            )
            self._record_event(events, call)

            result_started_at = self.clock()

            try:
                output = self.registry.invoke(
                    step.name,
                    step.arguments,
                )
            except Exception as exc:
                duration_ms = (self.clock() - result_started_at) * 1_000

                result = ToolResult(
                    call_id=call_id,
                    sequence=len(events),
                    status="error",
                    error=str(exc),
                    duration_ms=duration_ms,
                )
                failed_steps += 1
                self._record_event(events, result)

                if not self.config.continue_on_error:
                    break

                continue

            duration_ms = (self.clock() - result_started_at) * 1_000

            result = ToolResult(
                call_id=call_id,
                sequence=len(events),
                status="success",
                output=output,
                duration_ms=duration_ms,
            )
            successful_steps += 1
            self._record_event(events, result)

            delivery = ledger.record(
                result,
                kind=step.content_kind,
            )
            if delivery is not None and self.sink is not None:
                self.sink.record_delivery(delivery)

        elapsed_ms = (self.clock() - started_at) * 1_000

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            events=tuple(events),
            deliveries=tuple(ledger.deliveries),
            elapsed_ms=elapsed_ms,
            completed_steps=successful_steps + failed_steps,
            successful_steps=successful_steps,
            failed_steps=failed_steps,
        )

    def _record_event(
        self,
        events: list[TrajectoryEvent],
        event: TrajectoryEvent,
    ) -> None:
        """
        Store one event in memory and forward it to the optional sink.

        Args:
            events: The in-memory event collection.
            event: The event to record.
        """
        events.append(event)

        if self.sink is not None:
            self.sink.record_event(event)
