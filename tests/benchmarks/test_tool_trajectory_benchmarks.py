"""
tests/benchmarks/test_tool_trajectory_benchmarks.py

Exercises a deterministic repository-exploration trajectory using the real
indexer tools and evaluation runner.

Coverage:
  - Real search, source, and caller-trace tool execution
  - Raw trajectory event recording
  - Whole-delivery duplicate accounting
  - Scenario-level and per-tool token summaries
  - Deterministic token measurements across repeated runs
  - Machine-readable benchmark report output

These benchmarks do not invoke a model. They measure the context delivered by
a fixed sequence of repository tool calls.
"""

import json
from functools import partial
from pathlib import Path

import pytest

from indexer.evaluation.models import ToolCall, ToolResult
from indexer.evaluation.reporting import (
    TokenSummary,
    render_summary_text,
    summarize_result,
    summary_to_dict,
)
from indexer.evaluation.runner import (
    EvaluationScenario,
    ScenarioRunner,
    ToolRegistry,
    ToolStep,
)
from indexer.evaluation.token_counter import HeuristicTokenCounter
from indexer.store import open_path
from indexer.tools import get_source, search, trace_callers
from indexer.treesitter import NodeRecord

PROJECT = "trajectory-benchmark"

TARGET_QN = "trajectory_benchmark.src.payments.service.charge"
CHECKOUT_QN = "trajectory_benchmark.src.payments.views.checkout"
COMPLETE_QN = "trajectory_benchmark.src.orders.processor.complete"
REFUND_QN = "trajectory_benchmark.src.payments.service.refund"
SAVE_QN = "trajectory_benchmark.src.payments.models.Payment.save"


# ---------------------------------------------------------------------------
# Repository fixture
# ---------------------------------------------------------------------------


def _make_record(
    name: str,
    qualified_name: str,
    *,
    file_path: str,
    label: str = "Function",
    start_line: int = 1,
    end_line: int = 10,
    parent: str = "",
    signature: str = "",
    source: str = "",
) -> NodeRecord:
    """
    Build one node record for the benchmark repository.

    Args:
        name: The short symbol name.
        qualified_name: The fully qualified symbol name.
        file_path: The repository-relative source path.
        label: The graph node label.
        start_line: The first source line.
        end_line: The final source line.
        parent: The optional enclosing symbol name.
        signature: The symbol signature.
        source: The complete symbol source.

    Returns:
        The populated node record.
    """
    resolved_signature = signature or f"def {name}():"
    resolved_source = source or (f"{resolved_signature}\n" "    pass\n")

    return NodeRecord(
        label=label,
        name=name,
        qualified_name=qualified_name,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=resolved_signature,
        source=resolved_source,
        language="python",
        parent=parent,
        properties={},
    )


def _build_benchmark_db(tmp_path: Path) -> str:
    """
    Build the deterministic repository graph used by the benchmark.

    The relevant call graph is:

        checkout ─┐
                  ├──> charge ───> save
        complete ─┘           └──> refund

    Args:
        tmp_path: The pytest temporary directory.

    Returns:
        The path to the populated SQLite database.
    """
    db_path = str(tmp_path / "trajectory-benchmark.db")
    store = open_path(db_path)

    store.upsert_project(
        PROJECT,
        "/benchmark/repository",
        "python",
    )

    records = [
        _make_record(
            "charge",
            TARGET_QN,
            file_path="src/payments/service.py",
            start_line=1,
            end_line=19,
            signature=(
                "def charge("
                "user: User, "
                "amount_cents: int, "
                "currency: str"
                ") -> Payment:"
            ),
            source=(
                "def charge(\n"
                "    user: User,\n"
                "    amount_cents: int,\n"
                "    currency: str,\n"
                ") -> Payment:\n"
                '    """Charge a customer and save the payment."""\n'
                "    response = stripe_client.charge(\n"
                "        user.token,\n"
                "        amount_cents,\n"
                "        currency,\n"
                "    )\n"
                "    payment = Payment.from_response(response)\n"
                "    payment.save()\n"
                "    return payment\n"
            ),
        ),
        _make_record(
            "refund",
            REFUND_QN,
            file_path="src/payments/service.py",
            start_line=22,
            end_line=28,
            signature="def refund(payment: Payment) -> bool:",
            source=(
                "def refund(payment: Payment) -> bool:\n"
                "    return stripe_client.refund(payment.id)\n"
            ),
        ),
        _make_record(
            "checkout",
            CHECKOUT_QN,
            file_path="src/payments/views.py",
            start_line=1,
            end_line=12,
            signature=("def checkout(request: Request) -> Response:"),
            source=(
                "def checkout(request: Request) -> Response:\n"
                "    payment = charge(\n"
                "        request.user,\n"
                "        request.amount_cents,\n"
                '        "USD",\n'
                "    )\n"
                "    return Response(payment.id)\n"
            ),
        ),
        _make_record(
            "complete",
            COMPLETE_QN,
            file_path="src/orders/processor.py",
            start_line=5,
            end_line=15,
            signature=("def complete(order: Order) -> Payment:"),
            source=(
                "def complete(order: Order) -> Payment:\n"
                "    return charge(\n"
                "        order.user,\n"
                "        order.total_cents,\n"
                "        order.currency,\n"
                "    )\n"
            ),
        ),
        _make_record(
            "Payment",
            "trajectory_benchmark.src.payments.models.Payment",
            file_path="src/payments/models.py",
            label="Class",
            start_line=1,
            end_line=30,
            signature="class Payment:",
            source=(
                "class Payment:\n"
                "    id: str\n"
                "    amount_cents: int\n"
                "\n"
                "    def save(self) -> None:\n"
                "        database.save(self)\n"
            ),
        ),
        _make_record(
            "save",
            SAVE_QN,
            file_path="src/payments/models.py",
            label="Method",
            start_line=10,
            end_line=13,
            parent="Payment",
            signature="def save(self) -> None:",
            source=("def save(self) -> None:\n" "    database.save(self)\n"),
        ),
    ]

    store.begin()
    qualified_names = store.insert_nodes(
        records,
        PROJECT,
    )
    store.insert_edges(
        [
            (
                CHECKOUT_QN,
                TARGET_QN,
                "CALLS",
                {
                    "confidence": 0.98,
                    "strategy": "import_map",
                },
            ),
            (
                COMPLETE_QN,
                TARGET_QN,
                "CALLS",
                {
                    "confidence": 0.91,
                    "strategy": "import_map",
                },
            ),
            (
                TARGET_QN,
                SAVE_QN,
                "CALLS",
                {
                    "confidence": 0.95,
                    "strategy": "same_module",
                },
            ),
            (
                TARGET_QN,
                REFUND_QN,
                "CALLS",
                {
                    "confidence": 0.70,
                    "strategy": "same_module",
                },
            ),
        ],
        qualified_names,
        PROJECT,
    )
    store.insert_files(
        {
            "src/payments/service.py": (
                "def charge(...): pass\n" "def refund(...): pass\n"
            ),
            "src/payments/views.py": ("def checkout(...): pass\n"),
            "src/payments/models.py": ("class Payment: pass\n"),
            "src/orders/processor.py": ("def complete(...): pass\n"),
        },
        PROJECT,
        {
            "src/payments/service.py": "python",
            "src/payments/views.py": "python",
            "src/payments/models.py": "python",
            "src/orders/processor.py": "python",
        },
    )
    store.commit()
    store.close()

    return db_path


@pytest.fixture
def benchmark_db_path(tmp_path: Path) -> str:
    """
    Return the populated benchmark database path.

    Args:
        tmp_path: The pytest temporary directory.

    Returns:
        The benchmark SQLite path.
    """
    return _build_benchmark_db(tmp_path)


# ---------------------------------------------------------------------------
# Scenario and runner construction
# ---------------------------------------------------------------------------


def _build_scenario() -> EvaluationScenario:
    """
    Build the deterministic repository-exploration scenario.

    The repeated target-source request is intentional. It verifies that an
    identical complete delivery is counted as repeated context.

    Returns:
        The benchmark scenario.
    """
    return EvaluationScenario(
        scenario_id="search-source-repeat-trace",
        description=(
            "Locate a payment function, inspect it twice, trace its callers, "
            "and inspect one direct caller."
        ),
        steps=(
            ToolStep(
                name="search",
                arguments={
                    "query": "charge",
                    "limit": 10,
                },
                purpose="Locate the payment charge symbol.",
                content_kind="search_result",
            ),
            ToolStep(
                name="get_source",
                arguments={
                    "qualified_name": TARGET_QN,
                },
                purpose="Inspect the target implementation.",
                content_kind="symbol_source",
            ),
            ToolStep(
                name="get_source",
                arguments={
                    "qualified_name": TARGET_QN,
                },
                purpose=("Repeat the target request to measure duplicate delivery."),
                content_kind="symbol_source",
            ),
            ToolStep(
                name="trace_callers",
                arguments={
                    "qualified_name": TARGET_QN,
                    "depth": 2,
                },
                purpose="Measure the target blast radius.",
                content_kind="caller_trace",
            ),
            ToolStep(
                name="get_source",
                arguments={
                    "qualified_name": CHECKOUT_QN,
                },
                purpose="Inspect one direct caller.",
                content_kind="symbol_source",
            ),
        ),
    )


def _build_runner(db_path: str) -> ScenarioRunner:
    """
    Build a runner backed by the real indexer tools.

    Repository-level arguments are bound outside the scenario so its steps
    contain only task-relevant inputs.

    Args:
        db_path: The benchmark database path.

    Returns:
        The configured scenario runner.
    """
    registry = ToolRegistry(
        tools={
            "search": partial(
                search,
                db_path,
                project=PROJECT,
            ),
            "get_source": partial(
                get_source,
                db_path,
                project=PROJECT,
            ),
            "trace_callers": partial(
                trace_callers,
                db_path,
                project=PROJECT,
            ),
        }
    )

    return ScenarioRunner(
        registry=registry,
        counter=HeuristicTokenCounter(),
    )


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------


def _write_report(
    tmp_path: Path,
    payload: dict[str, object],
) -> Path:
    """
    Write one machine-readable benchmark report.

    Args:
        tmp_path: The pytest temporary directory.
        payload: The JSON-compatible report data.

    Returns:
        The report path.
    """
    report_path = tmp_path / "tool-trajectory-benchmark.json"
    report_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_real_tool_trajectory_records_token_costs(
    benchmark_db_path: str,
    tmp_path: Path,
) -> None:
    """
    Execute the scripted trajectory and validate its measured token costs.
    """
    runner = _build_runner(benchmark_db_path)
    scenario = _build_scenario()

    result = runner.run(scenario)
    summary = summarize_result(result)

    assert result.scenario_id == scenario.scenario_id
    assert result.completed_steps == len(scenario.steps)
    assert result.successful_steps == len(scenario.steps)
    assert result.failed_steps == 0

    assert len(result.events) == len(scenario.steps) * 2
    assert len(result.deliveries) == len(scenario.steps)

    calls = [event for event in result.events if isinstance(event, ToolCall)]
    results = [event for event in result.events if isinstance(event, ToolResult)]

    assert [call.name for call in calls] == [
        "search",
        "get_source",
        "get_source",
        "trace_callers",
        "get_source",
    ]
    assert all(event.status == "success" for event in results)

    assert [event.sequence for event in result.events] == list(
        range(len(result.events))
    )

    assert summary.counter_name == "heuristic-chars-4"
    assert summary.completed_steps == 5
    assert summary.successful_steps == 5
    assert summary.failed_steps == 0
    assert summary.delivered_results == 5

    assert summary.total_tokens > 0
    assert summary.novel_tokens > 0
    assert summary.repeated_tokens > 0
    assert summary.total_tokens == summary.novel_tokens + summary.repeated_tokens
    assert 0 < summary.duplication_rate < 1

    first_source = result.deliveries[1]
    repeated_source = result.deliveries[2]

    assert first_source.content_kind == "symbol_source"
    assert repeated_source.content_kind == "symbol_source"
    assert first_source.content_id == repeated_source.content_id
    assert first_source.tokens == repeated_source.tokens
    assert first_source.novel_tokens == first_source.tokens
    assert first_source.repeated_tokens == 0
    assert repeated_source.novel_tokens == 0
    assert repeated_source.repeated_tokens == repeated_source.tokens

    tools = {tool.name: tool for tool in summary.tools}

    assert set(tools) == {
        "search",
        "get_source",
        "trace_callers",
    }

    assert tools["search"].calls == 1
    assert tools["search"].successful_calls == 1
    assert tools["search"].failed_calls == 0
    assert tools["search"].total_tokens > 0

    assert tools["get_source"].calls == 3
    assert tools["get_source"].successful_calls == 3
    assert tools["get_source"].failed_calls == 0
    assert tools["get_source"].delivered_results == 3
    assert tools["get_source"].repeated_tokens > 0

    assert tools["trace_callers"].calls == 1
    assert tools["trace_callers"].successful_calls == 1
    assert tools["trace_callers"].failed_calls == 0
    assert tools["trace_callers"].total_tokens > 0

    payload = summary_to_dict(summary)
    report_path = _write_report(
        tmp_path,
        {
            "scenario": {
                "scenario_id": scenario.scenario_id,
                "description": scenario.description,
                "steps": len(scenario.steps),
            },
            "summary": payload,
        },
    )

    saved = json.loads(report_path.read_text(encoding="utf-8"))

    assert saved["scenario"]["scenario_id"] == scenario.scenario_id
    assert saved["summary"]["total_tokens"] == summary.total_tokens

    print()
    print(render_summary_text(summary))
    print(f"\nbenchmark report: {report_path}")


@pytest.mark.benchmark
def test_trajectory_token_measurements_are_deterministic(
    benchmark_db_path: str,
) -> None:
    """
    Verify repeated scenario runs produce identical token measurements.

    Timing values are intentionally excluded because runtime measurements are
    expected to vary between executions.
    """
    scenario = _build_scenario()

    first = _build_runner(benchmark_db_path).run(scenario)
    second = _build_runner(benchmark_db_path).run(scenario)

    first_summary = summarize_result(first)
    second_summary = summarize_result(second)

    assert first_summary.counter_name == second_summary.counter_name
    assert first_summary.total_tokens == second_summary.total_tokens
    assert first_summary.novel_tokens == second_summary.novel_tokens
    assert first_summary.repeated_tokens == second_summary.repeated_tokens
    assert first_summary.duplication_rate == second_summary.duplication_rate

    first_deliveries = [
        (
            item.call_id,
            item.content_id,
            item.content_kind,
            item.byte_length,
            item.tokens,
            item.novel_tokens,
            item.repeated_tokens,
        )
        for item in first.deliveries
    ]
    second_deliveries = [
        (
            item.call_id,
            item.content_id,
            item.content_kind,
            item.byte_length,
            item.tokens,
            item.novel_tokens,
            item.repeated_tokens,
        )
        for item in second.deliveries
    ]

    assert first_deliveries == second_deliveries

    first_tools = {
        item.name: (
            item.calls,
            item.successful_calls,
            item.failed_calls,
            item.delivered_results,
            item.total_tokens,
            item.novel_tokens,
            item.repeated_tokens,
            item.duplication_rate,
        )
        for item in first_summary.tools
    }
    second_tools = {
        item.name: (
            item.calls,
            item.successful_calls,
            item.failed_calls,
            item.delivered_results,
            item.total_tokens,
            item.novel_tokens,
            item.repeated_tokens,
            item.duplication_rate,
        )
        for item in second_summary.tools
    }

    assert first_tools == second_tools


@pytest.mark.benchmark
def test_repeated_source_delivery_has_expected_cost(
    benchmark_db_path: str,
) -> None:
    """
    Verify the deliberate repeat contributes only repeated tokens.

    This captures the current whole-delivery duplicate-accounting semantics.
    It does not attempt to detect partial overlap between different outputs.
    """
    runner = _build_runner(benchmark_db_path)
    result = runner.run(_build_scenario())

    first_source = result.deliveries[1]
    second_source = result.deliveries[2]

    assert first_source.content_id == second_source.content_id

    repeat_cost = second_source.repeated_tokens

    assert repeat_cost > 0
    assert second_source.tokens == repeat_cost
    assert second_source.novel_tokens == 0

    summary = summarize_result(result)

    assert summary.repeated_tokens >= repeat_cost
    assert summary.total_tokens >= repeat_cost
    assert summary.duplication_rate > 0


# ---------------------------------------------------------------------------
# Policy comparison fixtures
# ---------------------------------------------------------------------------


def _raw_repository_context() -> str:
    """
    Build the raw-source baseline delivered without repository tools.

    The baseline contains every source file relevant to the scripted task.
    File headings are included because an agent must be able to distinguish
    where each source unit came from.

    Returns:
        Concatenated repository source.
    """
    files = {
        "src/payments/service.py": (
            "def charge(\n"
            "    user: User,\n"
            "    amount_cents: int,\n"
            "    currency: str,\n"
            ") -> Payment:\n"
            '    """Charge a customer and save the payment."""\n'
            "    response = stripe_client.charge(\n"
            "        user.token,\n"
            "        amount_cents,\n"
            "        currency,\n"
            "    )\n"
            "    payment = Payment.from_response(response)\n"
            "    payment.save()\n"
            "    return payment\n"
            "\n"
            "\n"
            "def refund(payment: Payment) -> bool:\n"
            "    return stripe_client.refund(payment.id)\n"
        ),
        "src/payments/views.py": (
            "def checkout(request: Request) -> Response:\n"
            "    payment = charge(\n"
            "        request.user,\n"
            "        request.amount_cents,\n"
            '        "USD",\n'
            "    )\n"
            "    return Response(payment.id)\n"
        ),
        "src/payments/models.py": (
            "class Payment:\n"
            "    id: str\n"
            "    amount_cents: int\n"
            "\n"
            "    def save(self) -> None:\n"
            "        database.save(self)\n"
        ),
        "src/orders/processor.py": (
            "def complete(order: Order) -> Payment:\n"
            "    return charge(\n"
            "        order.user,\n"
            "        order.total_cents,\n"
            "        order.currency,\n"
            "    )\n"
        ),
    }

    sections = [
        f"# {file_path}\n\n{source.rstrip()}"
        for file_path, source in sorted(files.items())
    ]
    return "\n\n".join(sections) + "\n"


def _build_raw_source_scenario() -> EvaluationScenario:
    """
    Build the raw-source baseline scenario.

    Returns:
        A one-step scenario that delivers all relevant source.
    """
    return EvaluationScenario(
        scenario_id="raw-source",
        description=("Deliver every task-relevant source file as initial context."),
        steps=(
            ToolStep(
                name="initial_context",
                purpose="Supply the complete relevant source set.",
                content_kind="repository_source",
            ),
        ),
    )


def _build_raw_source_runner() -> ScenarioRunner:
    """
    Build the runner for the raw-source baseline.

    Returns:
        A runner whose sole tool returns the raw repository context.
    """

    def initial_context() -> str:
        return _raw_repository_context()

    return ScenarioRunner(
        registry=ToolRegistry(
            tools={
                "initial_context": initial_context,
            }
        ),
        counter=HeuristicTokenCounter(),
    )


def _build_no_repeat_scenario() -> EvaluationScenario:
    """
    Build the indexed trajectory without the duplicate source request.

    Returns:
        The optimized deterministic indexed scenario.
    """
    return EvaluationScenario(
        scenario_id="indexed-no-repeat",
        description=(
            "Locate the target, inspect it once, trace callers, and inspect "
            "one direct caller."
        ),
        steps=(
            ToolStep(
                name="search",
                arguments={
                    "query": "charge",
                    "limit": 10,
                },
                purpose="Locate the payment charge symbol.",
                content_kind="search_result",
            ),
            ToolStep(
                name="get_source",
                arguments={
                    "qualified_name": TARGET_QN,
                },
                purpose="Inspect the target implementation.",
                content_kind="symbol_source",
            ),
            ToolStep(
                name="trace_callers",
                arguments={
                    "qualified_name": TARGET_QN,
                    "depth": 2,
                },
                purpose="Measure the target blast radius.",
                content_kind="caller_trace",
            ),
            ToolStep(
                name="get_source",
                arguments={
                    "qualified_name": CHECKOUT_QN,
                },
                purpose="Inspect one direct caller.",
                content_kind="symbol_source",
            ),
        ),
    )


def _build_naive_indexed_scenario() -> EvaluationScenario:
    """
    Build the existing indexed scenario under an explicit policy ID.

    Returns:
        The indexed scenario containing one repeated source request.
    """
    original = _build_scenario()
    return EvaluationScenario(
        scenario_id="indexed-naive",
        description=original.description,
        steps=original.steps,
    )


def _policy_summary_payload(
    *,
    policy: str,
    summary: TokenSummary,
) -> dict[str, object]:
    """
    Convert one summary into a compact policy-comparison record.

    Args:
        policy: The policy identifier.
        summary: The scenario token summary.

    Returns:
        JSON-compatible policy measurements.
    """
    return {
        "policy": policy,
        "scenario_id": summary.scenario_id,
        "completed_steps": summary.completed_steps,
        "successful_steps": summary.successful_steps,
        "failed_steps": summary.failed_steps,
        "delivered_results": summary.delivered_results,
        "total_tokens": summary.total_tokens,
        "novel_tokens": summary.novel_tokens,
        "repeated_tokens": summary.repeated_tokens,
        "duplication_rate": summary.duplication_rate,
        "elapsed_ms": summary.elapsed_ms,
    }


def _policy_comparison_payload(
    raw_source: TokenSummary,
    indexed_naive: TokenSummary,
    indexed_no_repeat: TokenSummary,
) -> dict[str, object]:
    """
    Build a machine-readable comparison of all three policies.

    Positive token differences mean the left-hand indexed policy delivered
    more tokens than the comparison policy.

    Args:
        raw_source: Summary for the raw-source baseline.
        indexed_naive: Summary for indexed exploration with duplication.
        indexed_no_repeat: Summary for indexed exploration without duplication.

    Returns:
        JSON-compatible comparison data.
    """
    return {
        "policies": [
            _policy_summary_payload(
                policy="raw-source",
                summary=raw_source,
            ),
            _policy_summary_payload(
                policy="indexed-naive",
                summary=indexed_naive,
            ),
            _policy_summary_payload(
                policy="indexed-no-repeat",
                summary=indexed_no_repeat,
            ),
        ],
        "comparisons": {
            "indexed_naive_vs_raw": {
                "total_token_difference": (
                    indexed_naive.total_tokens - raw_source.total_tokens
                ),
                "total_token_ratio": (
                    indexed_naive.total_tokens / raw_source.total_tokens
                    if raw_source.total_tokens
                    else 0.0
                ),
            },
            "indexed_no_repeat_vs_raw": {
                "total_token_difference": (
                    indexed_no_repeat.total_tokens - raw_source.total_tokens
                ),
                "total_token_ratio": (
                    indexed_no_repeat.total_tokens / raw_source.total_tokens
                    if raw_source.total_tokens
                    else 0.0
                ),
            },
            "indexed_no_repeat_vs_naive": {
                "total_token_difference": (
                    indexed_no_repeat.total_tokens - indexed_naive.total_tokens
                ),
                "removed_tokens": (
                    indexed_naive.total_tokens - indexed_no_repeat.total_tokens
                ),
                "reduction_rate": (
                    (indexed_naive.total_tokens - indexed_no_repeat.total_tokens)
                    / indexed_naive.total_tokens
                    if indexed_naive.total_tokens
                    else 0.0
                ),
            },
        },
    }


def _render_policy_comparison(
    summaries: list[tuple[str, TokenSummary]],
) -> str:
    """
    Render a compact policy-comparison table.

    Args:
        summaries: Ordered policy names and summaries.

    Returns:
        Human-readable comparison text.
    """
    lines = [
        "# policy comparison",
        "",
        (
            f"{'policy':<24}"
            f"{'steps':>8}"
            f"{'tokens':>12}"
            f"{'novel':>12}"
            f"{'repeat':>12}"
            f"{'dup %':>10}"
        ),
        (
            f"{'-' * 24}"
            f"{'-' * 8}"
            f"{'-' * 12}"
            f"{'-' * 12}"
            f"{'-' * 12}"
            f"{'-' * 10}"
        ),
    ]

    for policy, summary in summaries:
        lines.append(
            f"{policy:<24}"
            f"{summary.completed_steps:>8}"
            f"{summary.total_tokens:>12,}"
            f"{summary.novel_tokens:>12,}"
            f"{summary.repeated_tokens:>12,}"
            f"{summary.duplication_rate:>9.1%}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Policy comparison benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_removing_repeated_source_reduces_indexed_delivery_cost(
    benchmark_db_path: str,
) -> None:
    """
    Compare the naive indexed trajectory with the no-repeat trajectory.

    The unique deliveries are identical between the two policies. Therefore,
    removing the repeated request must remove exactly the token cost of that
    repeated source response.
    """
    naive_result = _build_runner(benchmark_db_path).run(_build_naive_indexed_scenario())
    optimized_result = _build_runner(benchmark_db_path).run(_build_no_repeat_scenario())

    naive = summarize_result(naive_result)
    optimized = summarize_result(optimized_result)

    repeated_delivery = naive_result.deliveries[2]

    assert repeated_delivery.content_kind == "symbol_source"
    assert repeated_delivery.novel_tokens == 0
    assert repeated_delivery.repeated_tokens > 0

    assert naive.completed_steps == 5
    assert optimized.completed_steps == 4
    assert naive.successful_steps == 5
    assert optimized.successful_steps == 4

    assert optimized.total_tokens < naive.total_tokens
    assert naive.total_tokens - optimized.total_tokens == repeated_delivery.tokens

    assert optimized.novel_tokens == naive.novel_tokens
    assert optimized.repeated_tokens == 0
    assert optimized.duplication_rate == 0.0

    assert naive.total_tokens == optimized.total_tokens + repeated_delivery.tokens


@pytest.mark.benchmark
def test_raw_source_baseline_is_one_entirely_novel_delivery() -> None:
    """
    Verify the raw-source baseline's accounting semantics.
    """
    result = _build_raw_source_runner().run(_build_raw_source_scenario())
    summary = summarize_result(result)

    assert result.completed_steps == 1
    assert result.successful_steps == 1
    assert result.failed_steps == 0
    assert len(result.events) == 2
    assert len(result.deliveries) == 1

    delivery = result.deliveries[0]

    assert delivery.content_kind == "repository_source"
    assert delivery.tokens > 0
    assert delivery.novel_tokens == delivery.tokens
    assert delivery.repeated_tokens == 0

    assert summary.total_tokens == delivery.tokens
    assert summary.novel_tokens == delivery.tokens
    assert summary.repeated_tokens == 0
    assert summary.duplication_rate == 0.0

    tools = {tool.name: tool for tool in summary.tools}

    assert set(tools) == {"initial_context"}
    assert tools["initial_context"].calls == 1
    assert tools["initial_context"].delivered_results == 1
    assert tools["initial_context"].total_tokens == delivery.tokens


@pytest.mark.benchmark
def test_context_policy_comparison_records_all_measurements(
    benchmark_db_path: str,
    tmp_path: Path,
) -> None:
    """
    Compare raw source, naive indexed exploration, and no-repeat exploration.

    The test intentionally does not require either indexed policy to beat the
    raw baseline. It verifies that each policy is measured consistently and
    that the resulting differences are persisted accurately.
    """
    raw_summary = summarize_result(
        _build_raw_source_runner().run(_build_raw_source_scenario())
    )
    naive_summary = summarize_result(
        _build_runner(benchmark_db_path).run(_build_naive_indexed_scenario())
    )
    optimized_summary = summarize_result(
        _build_runner(benchmark_db_path).run(_build_no_repeat_scenario())
    )

    assert raw_summary.counter_name == "heuristic-chars-4"
    assert naive_summary.counter_name == "heuristic-chars-4"
    assert optimized_summary.counter_name == "heuristic-chars-4"

    assert raw_summary.total_tokens > 0
    assert naive_summary.total_tokens > 0
    assert optimized_summary.total_tokens > 0

    assert naive_summary.total_tokens > optimized_summary.total_tokens
    assert naive_summary.novel_tokens == optimized_summary.novel_tokens
    assert naive_summary.repeated_tokens > 0
    assert optimized_summary.repeated_tokens == 0

    payload = _policy_comparison_payload(
        raw_summary,
        naive_summary,
        optimized_summary,
    )

    report_path = tmp_path / "context-policy-comparison.json"
    report_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    restored = json.loads(report_path.read_text(encoding="utf-8"))

    policies = {item["policy"]: item for item in restored["policies"]}

    assert set(policies) == {
        "raw-source",
        "indexed-naive",
        "indexed-no-repeat",
    }
    assert policies["raw-source"]["total_tokens"] == raw_summary.total_tokens
    assert policies["indexed-naive"]["total_tokens"] == naive_summary.total_tokens
    assert (
        policies["indexed-no-repeat"]["total_tokens"] == optimized_summary.total_tokens
    )

    optimized_comparison = restored["comparisons"]["indexed_no_repeat_vs_naive"]
    expected_removed = naive_summary.total_tokens - optimized_summary.total_tokens

    assert optimized_comparison["removed_tokens"] == expected_removed
    assert optimized_comparison["total_token_difference"] == (-expected_removed)
    assert 0 < optimized_comparison["reduction_rate"] < 1

    print()
    print(
        _render_policy_comparison(
            [
                ("raw-source", raw_summary),
                ("indexed-naive", naive_summary),
                ("indexed-no-repeat", optimized_summary),
            ]
        )
    )
    print(f"\nbenchmark report: {report_path}")


@pytest.mark.benchmark
def test_policy_measurements_are_deterministic(
    benchmark_db_path: str,
) -> None:
    """
    Verify all three policy measurements are stable across repeated runs.

    Runtime fields are excluded because wall-clock performance is not expected
    to be deterministic.
    """

    def run_policies() -> dict[str, tuple[int, int, int, float]]:
        raw = summarize_result(
            _build_raw_source_runner().run(_build_raw_source_scenario())
        )
        naive = summarize_result(
            _build_runner(benchmark_db_path).run(_build_naive_indexed_scenario())
        )
        optimized = summarize_result(
            _build_runner(benchmark_db_path).run(_build_no_repeat_scenario())
        )

        return {
            "raw-source": (
                raw.total_tokens,
                raw.novel_tokens,
                raw.repeated_tokens,
                raw.duplication_rate,
            ),
            "indexed-naive": (
                naive.total_tokens,
                naive.novel_tokens,
                naive.repeated_tokens,
                naive.duplication_rate,
            ),
            "indexed-no-repeat": (
                optimized.total_tokens,
                optimized.novel_tokens,
                optimized.repeated_tokens,
                optimized.duplication_rate,
            ),
        }

    assert run_policies() == run_policies()
