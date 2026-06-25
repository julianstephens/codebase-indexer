from .runner import EvaluationScenario, ToolStep

EvaluationScenario(
    scenario_id="search-source-repeat-trace",
    description=(
        "Search for a symbol, inspect its source twice, trace its callers, "
        "and inspect one caller."
    ),
    steps=(
        ToolStep(
            name="search",
            arguments={"query": "charge", "limit": 10},
            purpose="Locate the target symbol.",
            content_kind="search_result",
        ),
        ToolStep(
            name="get_source",
            arguments={"qualified_name": "benchmark.service.charge"},
            purpose="Read the target implementation.",
            content_kind="symbol_source",
        ),
        ToolStep(
            name="get_source",
            arguments={"qualified_name": "benchmark.service.charge"},
            purpose="Exercise repeated-content accounting.",
            content_kind="symbol_source",
        ),
        ToolStep(
            name="trace_callers",
            arguments={
                "qualified_name": "benchmark.service.charge",
                "depth": 2,
            },
            purpose="Inspect the target blast radius.",
            content_kind="caller_trace",
        ),
        ToolStep(
            name="get_source",
            arguments={"qualified_name": "benchmark.views.checkout"},
            purpose="Inspect one direct caller.",
            content_kind="symbol_source",
        ),
    ),
)
