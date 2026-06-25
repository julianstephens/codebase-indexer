"""
tests/test_tools.py — Tests for indexer/tools.py

Coverage:
  - get_source(): successful lookup, not-found handling, project filtering,
                  source truncation, missing/corrupt databases, error recovery
  - search(): successful and empty searches, filtering, limit clamping,
              missing/corrupt databases, error recovery
  - trace_callers(): direct and multi-hop callers, empty traversal, depth
                     clamping, cycles, project filtering, error recovery
  - Adapter behavior: query functions receive normalized arguments and
                      rendered output is returned unchanged
  - Integration: results from one public tool can be used by another

Formatting-helper tests live with the renderer modules rather than here.
"""

from pathlib import Path

import pytest

from src.indexer import tools as tools_module
from src.indexer.store import open_path
from src.indexer.tools import get_source, search, trace_callers
from src.indexer.treesitter import NodeRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    name: str,
    qn: str,
    file_path: str = "src/utils.py",
    label: str = "Function",
    start_line: int = 1,
    end_line: int = 10,
    parent: str = "",
    signature: str = "",
    source: str = "",
    language: str = "python",
    properties: dict | None = None,
) -> NodeRecord:
    """
    Build a NodeRecord for tool integration tests.

    Args:
        name: short symbol name.
        qn: fully qualified symbol name.
        file_path: repository-relative source path.
        label: graph node label.
        start_line: 1-based start line.
        end_line: 1-based end line.
        parent: enclosing symbol name.
        signature: source signature.
        source: full source body.
        language: canonical language name.
        properties: optional language-specific properties.

    Returns:
        Populated NodeRecord.
    """
    resolved_signature = signature or f"def {name}():"
    resolved_source = source or f"{resolved_signature}\n    pass"
    return NodeRecord(
        label=label,
        name=name,
        qualified_name=qn,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=resolved_signature,
        source=resolved_source,
        language=language,
        parent=parent,
        properties=properties or {},
    )


def _build_db(tmp_path: Path) -> str:
    """
    Build a populated graph database for public tool tests.

    The graph contains two direct callers of charge and two callees:

        checkout ─┐
                  ├─> charge ─> save
        complete ─┘          └─> refund

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the populated SQLite database.
    """
    db_path = str(tmp_path / "test.db")
    store = open_path(db_path)
    store.upsert_project("test-app", "/repo", "python")

    records = [
        _make_record(
            "charge",
            "test_app.src.payments.service.charge",
            file_path="src/payments/service.py",
            start_line=1,
            end_line=20,
            signature=(
                "def charge(user: User, amount_cents: int, currency: str) -> Payment:"
            ),
            source=(
                "def charge(user: User, amount_cents: int, currency: str) -> Payment:\n"
                '    """Charge a user via Stripe."""\n'
                "    result = stripe.charge(user.token, amount_cents, currency)\n"
                "    payment = Payment.save(result)\n"
                "    return payment\n"
            ),
        ),
        _make_record(
            "refund",
            "test_app.src.payments.service.refund",
            file_path="src/payments/service.py",
            start_line=22,
            end_line=35,
            signature="def refund(payment: Payment) -> bool:",
            source=(
                "def refund(payment: Payment) -> bool:\n"
                "    return stripe.refund(payment.id)\n"
            ),
        ),
        _make_record(
            "checkout",
            "test_app.src.payments.views.checkout",
            file_path="src/payments/views.py",
            start_line=1,
            end_line=30,
            signature="def checkout(request: Request) -> Response:",
            source=(
                "def checkout(request: Request) -> Response:\n"
                "    charge(request.user, request.amount)\n"
            ),
        ),
        _make_record(
            "complete",
            "test_app.src.orders.processor.complete",
            file_path="src/orders/processor.py",
            start_line=5,
            end_line=25,
            signature="def complete(order: Order) -> None:",
            source=(
                "def complete(order: Order) -> None:\n"
                "    charge(order.user, order.total)\n"
            ),
        ),
        _make_record(
            "Payment",
            "test_app.src.payments.models.Payment",
            file_path="src/payments/models.py",
            label="Class",
            start_line=1,
            end_line=50,
            signature="class Payment(BaseModel):",
            source="class Payment(BaseModel):\n    id: str\n    amount: int\n",
        ),
        _make_record(
            "save",
            "test_app.src.payments.models.Payment.save",
            file_path="src/payments/models.py",
            label="Method",
            start_line=10,
            end_line=20,
            parent="Payment",
            signature="def save(self) -> None:",
            source="def save(self) -> None:\n    db.save(self)\n",
        ),
    ]

    store.begin()
    qn_to_id = store.insert_nodes(records, "test-app")
    store.insert_edges(
        [
            (
                "test_app.src.payments.views.checkout",
                "test_app.src.payments.service.charge",
                "CALLS",
                {"confidence": 0.95, "strategy": "import_map"},
            ),
            (
                "test_app.src.orders.processor.complete",
                "test_app.src.payments.service.charge",
                "CALLS",
                {"confidence": 0.85, "strategy": "fuzzy"},
            ),
            (
                "test_app.src.payments.service.charge",
                "test_app.src.payments.models.Payment.save",
                "CALLS",
                {"confidence": 0.90, "strategy": "same_module"},
            ),
            (
                "test_app.src.payments.service.charge",
                "test_app.src.payments.service.refund",
                "CALLS",
                {"confidence": 0.70, "strategy": "same_module"},
            ),
        ],
        qn_to_id,
        "test-app",
    )
    store.insert_files(
        {
            "src/payments/service.py": "def charge(): pass\ndef refund(): pass",
            "src/payments/views.py": "def checkout(): pass",
            "src/payments/models.py": "class Payment: pass",
            "src/orders/processor.py": "def complete(): pass",
        },
        "test-app",
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
def db_path(tmp_path: Path) -> str:
    """
    Return a populated database path.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to a populated SQLite database.
    """
    return _build_db(tmp_path)


@pytest.fixture
def missing_db_path(tmp_path: Path) -> str:
    """
    Return a path that does not contain a database.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Missing database path.
    """
    return str(tmp_path / "nonexistent.db")


# ---------------------------------------------------------------------------
# get_source
# ---------------------------------------------------------------------------


class TestGetSource:
    def test_found_returns_rendered_source(self, db_path: str) -> None:
        result = get_source(
            db_path,
            "test_app.src.payments.service.charge",
        )

        assert "def charge" in result
        assert "src/payments/service.py" in result
        assert "test_app.src.payments.service.charge" in result
        assert "Function" in result

    def test_found_includes_direct_relationships(self, db_path: str) -> None:
        result = get_source(
            db_path,
            "test_app.src.payments.service.charge",
        )

        assert "called by" in result.lower()
        assert "checkout" in result
        assert "complete" in result
        assert "calls" in result.lower()
        assert "save" in result
        assert "refund" in result

    def test_found_includes_token_estimate(self, db_path: str) -> None:
        result = get_source(
            db_path,
            "test_app.src.payments.service.charge",
        )

        assert "tokens" in result
        assert "─" in result

    def test_not_found_returns_renderer_message(self, db_path: str) -> None:
        result = get_source(db_path, "src.payments.missing")

        assert "not found" in result.lower()
        assert "search" in result.lower()
        assert "missing" in result

    def test_project_filter_found(self, db_path: str) -> None:
        result = get_source(
            db_path,
            "test_app.src.payments.service.charge",
            project="test-app",
        )

        assert "def charge" in result

    def test_project_filter_wrong_project_not_found(self, db_path: str) -> None:
        result = get_source(
            db_path,
            "test_app.src.payments.service.charge",
            project="wrong-project",
        )

        assert "not found" in result.lower()

    def test_class_node(self, db_path: str) -> None:
        result = get_source(
            db_path,
            "test_app.src.payments.models.Payment",
        )

        assert "class Payment" in result
        assert "Class" in result

    def test_method_node(self, db_path: str) -> None:
        result = get_source(
            db_path,
            "test_app.src.payments.models.Payment.save",
        )

        assert "def save" in result
        assert "Method" in result

    def test_node_without_callers_renders_empty_section(self, db_path: str) -> None:
        result = get_source(
            db_path,
            "test_app.src.payments.views.checkout",
        )

        assert "called by" in result.lower()
        assert "(0)" in result or "(none)" in result.lower()

    def test_node_without_callees_renders_empty_section(self, db_path: str) -> None:
        result = get_source(
            db_path,
            "test_app.src.payments.service.refund",
        )

        assert "calls" in result.lower()
        assert "(0)" in result or "(none)" in result.lower()

    def test_large_source_is_truncated(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "big.db")
        store = open_path(db_path)
        store.upsert_project("p", "/repo")
        record = _make_record(
            "huge_fn",
            "p.huge_fn",
            source="# line\n" * 7_000,
            start_line=1,
            end_line=7_000,
        )
        store.begin()
        store.insert_nodes([record], "p")
        store.commit()
        store.close()

        result = get_source(db_path, "p.huge_fn")

        assert "[source truncated" in result

    def test_missing_db_returns_error_string(self, missing_db_path: str) -> None:
        result = get_source(missing_db_path, "any.qn")

        assert isinstance(result, str)
        assert "database not found" in result.lower()

    def test_corrupt_db_returns_error_string(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.db"
        corrupt.write_bytes(b"this is not sqlite")

        result = get_source(str(corrupt), "any.qn")

        assert isinstance(result, str)
        assert "error" in result.lower()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_match_returns_rendered_result(self, db_path: str) -> None:
        result = search(db_path, "charge")

        assert "charge" in result
        assert "src/payments" in result
        assert "Function" in result
        assert "result" in result.lower()
        assert "─" in result

    def test_no_results_returns_empty_search_message(self, db_path: str) -> None:
        result = search(db_path, "zzz_no_such_symbol_xyz")

        assert "0 results" in result.lower()
        assert "no nodes matched" in result.lower()

    def test_label_filter_function_excludes_class(self, db_path: str) -> None:
        result = search(
            db_path,
            "Payment",
            label="Function",
        )

        assert "class Payment" not in result

    def test_label_filter_class_finds_class(self, db_path: str) -> None:
        result = search(
            db_path,
            "Payment",
            label="Class",
        )

        assert "test_app.src.payments.models.Payment" in result
        assert "Class" in result

    def test_project_filter(self, db_path: str) -> None:
        result = search(
            db_path,
            "charge",
            project="test-app",
        )

        assert "charge" in result

    def test_wrong_project_returns_empty_search(self, db_path: str) -> None:
        result = search(
            db_path,
            "charge",
            project="missing-project",
        )

        assert "0 results" in result.lower()

    def test_phrase_query_returns_string(self, db_path: str) -> None:
        result = search(db_path, '"def charge"')

        assert isinstance(result, str)

    def test_boolean_query_returns_string(self, db_path: str) -> None:
        result = search(db_path, "charge AND Payment")

        assert isinstance(result, str)

    def test_empty_query_never_raises(self, db_path: str) -> None:
        result = search(db_path, "")

        assert isinstance(result, str)

    def test_pagination_message_when_more_results(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "many.db")
        store = open_path(db_path)
        store.upsert_project("p", "/repo")
        records = [
            _make_record(
                f"process_{index}",
                f"p.mod.process_{index}",
                source=f"def process_{index}(): pass",
            )
            for index in range(30)
        ]
        store.begin()
        store.insert_nodes(records, "p")
        store.commit()
        store.close()

        result = search(db_path, "process", limit=5)

        assert "more result" in result.lower()

    def test_missing_db_returns_error_string(self, missing_db_path: str) -> None:
        result = search(missing_db_path, "anything")

        assert isinstance(result, str)
        assert "database not found" in result.lower()

    def test_corrupt_db_returns_error_string(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.db"
        corrupt.write_bytes(b"not a database")

        result = search(str(corrupt), "anything")

        assert isinstance(result, str)
        assert "error" in result.lower()


# ---------------------------------------------------------------------------
# trace_callers
# ---------------------------------------------------------------------------


class TestTraceCallers:
    def test_direct_callers_are_rendered(self, db_path: str) -> None:
        result = trace_callers(
            db_path,
            "test_app.src.payments.service.charge",
        )

        assert "test_app.src.payments.service.charge" in result
        assert "checkout" in result
        assert "complete" in result
        assert "hop 1" in result
        assert "blast radius" in result.lower()

    def test_trace_includes_token_estimate(self, db_path: str) -> None:
        result = trace_callers(
            db_path,
            "test_app.src.payments.service.charge",
        )

        assert "tokens" in result
        assert "─" in result

    def test_not_found_returns_renderer_message(self, db_path: str) -> None:
        result = trace_callers(db_path, "nonexistent.qn")

        assert "not found" in result.lower()
        assert "search" in result.lower()

    def test_no_callers_returns_empty_trace(self, db_path: str) -> None:
        result = trace_callers(
            db_path,
            "test_app.src.payments.views.checkout",
        )

        assert "0 callers" in result.lower()
        assert "no callers" in result.lower()

    def test_project_filter(self, db_path: str) -> None:
        result = trace_callers(
            db_path,
            "test_app.src.payments.service.charge",
            project="test-app",
        )

        assert "checkout" in result
        assert "complete" in result

    def test_wrong_project_returns_not_found(self, db_path: str) -> None:
        result = trace_callers(
            db_path,
            "test_app.src.payments.service.charge",
            project="wrong-project",
        )

        assert "not found" in result.lower()

    def test_multi_hop_trace(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "chain.db")
        store = open_path(db_path)
        store.upsert_project("p", "/repo")
        records = [
            _make_record("a", "p.a"),
            _make_record("b", "p.b"),
            _make_record("c", "p.c"),
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.insert_edges(
            [
                ("p.a", "p.b", "CALLS", {"confidence": 0.95}),
                ("p.b", "p.c", "CALLS", {"confidence": 0.85}),
            ],
            qn_to_id,
            "p",
        )
        store.commit()
        store.close()

        result = trace_callers(db_path, "p.c", depth=2)

        assert "p.b" in result
        assert "p.a" in result
        assert "hop 1" in result
        assert "hop 2" in result

    def test_depth_one_excludes_indirect_callers(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "depth-one.db")
        store = open_path(db_path)
        store.upsert_project("p", "/repo")
        records = [
            _make_record("a", "p.a"),
            _make_record("b", "p.b"),
            _make_record("c", "p.c"),
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.insert_edges(
            [
                ("p.a", "p.b", "CALLS", {}),
                ("p.b", "p.c", "CALLS", {}),
            ],
            qn_to_id,
            "p",
        )
        store.commit()
        store.close()

        result = trace_callers(db_path, "p.c", depth=1)

        assert "p.b" in result
        assert "p.a" not in result

    def test_cycle_terminates(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "cycle.db")
        store = open_path(db_path)
        store.upsert_project("p", "/repo")
        records = [
            _make_record("f0", "p.f0"),
            _make_record("f1", "p.f1"),
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.insert_edges(
            [
                ("p.f0", "p.f1", "CALLS", {}),
                ("p.f1", "p.f0", "CALLS", {}),
            ],
            qn_to_id,
            "p",
        )
        store.commit()
        store.close()

        result = trace_callers(db_path, "p.f0", depth=10)

        assert isinstance(result, str)
        assert "p.f1" in result

    def test_missing_db_returns_error_string(self, missing_db_path: str) -> None:
        result = trace_callers(missing_db_path, "any.qn")

        assert isinstance(result, str)
        assert "database not found" in result.lower()

    def test_corrupt_db_returns_error_string(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.db"
        corrupt.write_bytes(b"not a database")

        result = trace_callers(str(corrupt), "any.qn")

        assert isinstance(result, str)
        assert "error" in result.lower()


# ---------------------------------------------------------------------------
# Adapter behavior
# ---------------------------------------------------------------------------


class TestAdapterBehavior:
    def test_get_source_delegates_to_query_and_renderer(
        self,
        db_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinel = object()
        captured: dict[str, object] = {}

        def fake_query(_store, qualified_name, project):
            captured["qualified_name"] = qualified_name
            captured["project"] = project
            return sentinel

        monkeypatch.setattr(tools_module, "query_source", fake_query)
        monkeypatch.setattr(
            tools_module,
            "render_source_text",
            lambda result: "rendered source" if result is sentinel else "wrong",
        )

        result = get_source(db_path, "p.mod.fn", project="p")

        assert result == "rendered source"
        assert captured == {
            "qualified_name": "p.mod.fn",
            "project": "p",
        }

    def test_get_source_uses_not_found_renderer(
        self,
        db_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            tools_module,
            "query_source",
            lambda _store, _qualified_name, _project: None,
        )
        monkeypatch.setattr(
            tools_module,
            "render_node_not_found",
            lambda qualified_name: f"missing: {qualified_name}",
        )

        result = get_source(db_path, "p.missing")

        assert result == "missing: p.missing"

    def test_search_clamps_limit_to_fifty(
        self,
        db_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinel = object()
        captured: dict[str, object] = {}

        def fake_query(_store, query, file_pattern, project, label, limit):
            captured["query"] = query
            captured["file_pattern"] = file_pattern
            captured["project"] = project
            captured["label"] = label
            captured["limit"] = limit
            return sentinel

        monkeypatch.setattr(tools_module, "query_search", fake_query)
        monkeypatch.setattr(
            tools_module,
            "render_search_text",
            lambda result: "rendered search" if result is sentinel else "wrong",
        )

        result = search(
            db_path,
            "needle",
            project="p",
            label="Function",
            limit=200,
        )

        assert result == "rendered search"
        assert captured == {
            "query": "needle",
            "project": "p",
            "label": "Function",
            "file_pattern": None,
            "limit": 50,
        }

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            (-5, 1),
            (0, 1),
            (1, 1),
            (5, 5),
            (10, 10),
            (100, 10),
        ],
    )
    def test_trace_callers_clamps_depth(
        self,
        db_path: str,
        monkeypatch: pytest.MonkeyPatch,
        requested: int,
        expected: int,
    ) -> None:
        sentinel = object()
        captured: dict[str, int] = {}

        def fake_query(_store, _qualified_name, _project, depth):
            captured["depth"] = depth
            return sentinel

        monkeypatch.setattr(tools_module, "query_callers", fake_query)
        monkeypatch.setattr(
            tools_module,
            "render_trace_text",
            lambda result: "rendered trace" if result is sentinel else "wrong",
        )

        result = trace_callers(db_path, "p.fn", depth=requested)

        assert result == "rendered trace"
        assert captured["depth"] == expected

    def test_get_source_query_error_becomes_error_string(
        self,
        db_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail(_store, _qualified_name, _project):
            raise RuntimeError("boom")

        monkeypatch.setattr(tools_module, "query_source", fail)

        result = get_source(db_path, "p.fn")

        assert result == "Error: get_source failed: boom"

    def test_search_query_error_becomes_error_string(
        self,
        db_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail(_store, _query, _file_pattern, _project, _label, _limit):
            raise RuntimeError("boom")

        monkeypatch.setattr(tools_module, "query_search", fail)

        result = search(db_path, "needle")

        assert result == "Error: search failed: boom"

    def test_trace_query_error_becomes_error_string(
        self,
        db_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail(_store, _qualified_name, _project, _depth):
            raise RuntimeError("boom")

        monkeypatch.setattr(tools_module, "query_callers", fail)

        result = trace_callers(db_path, "p.fn")

        assert result == "Error: trace_callers failed: boom"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_search_result_qn_is_valid_get_source_input(
        self,
        db_path: str,
    ) -> None:
        search_result = search(db_path, "charge")

        for line in search_result.splitlines():
            candidate = line.strip()
            if candidate == "test_app.src.payments.service.charge":
                source_result = get_source(db_path, candidate)
                assert "def charge" in source_result
                return

        pytest.fail("search output did not contain the charge qualified name")

    def test_traced_caller_is_valid_get_source_input(
        self,
        db_path: str,
    ) -> None:
        trace_result = trace_callers(
            db_path,
            "test_app.src.payments.service.charge",
        )

        assert "checkout" in trace_result
        source_result = get_source(
            db_path,
            "test_app.src.payments.views.checkout",
        )
        assert "def checkout" in source_result

    def test_all_tools_return_non_empty_strings(self, db_path: str) -> None:
        qualified_name = "test_app.src.payments.service.charge"

        assert get_source(db_path, qualified_name)
        assert search(db_path, "charge")
        assert trace_callers(db_path, qualified_name)

    def test_all_tools_handle_missing_database(
        self,
        missing_db_path: str,
    ) -> None:
        assert isinstance(get_source(missing_db_path, "any.qn"), str)
        assert isinstance(search(missing_db_path, "anything"), str)
        assert isinstance(trace_callers(missing_db_path, "any.qn"), str)
