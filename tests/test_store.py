"""
tests/test_store.py — Tests for indexer/store.py

Coverage:
  - Factory functions: open_memory, open_path, open_path_readonly
  - Store lifecycle: close, check_integrity, checkpoint
  - Bulk write helpers: begin_bulk/end_bulk, drop_indexes/create_indexes
  - Transaction: begin, commit, rollback
  - Project CRUD: upsert_project, get_project, list_projects, delete_project
  - Node writes: insert_nodes, delete_nodes_for_file
  - Edge writes: insert_edges (resolved, unresolved, duplicate)
  - File writes: insert_files, insert_file_hashes
  - Node reads: get_node_by_qn, get_node_by_id, get_nodes_by_file,
                count_nodes, count_edges
  - Search: search_nodes with label, name_pattern, file_pattern, fts_query,
            pagination
  - Graph traversal: bfs_callers, bfs_callees, max_depth, max_nodes
  - File reads: get_file_source, get_file_hashes
  - Schema introspection: get_schema_summary
  - Skeleton: iter_skeleton ordering and label filtering
  - Dump / restore: dump_to_file, restore_from_file
  - Internal helpers: _batched, _node_record_to_row, _row_to_node
  - default_db_path
  - Error cases: missing QN, readonly writes, bad paths
"""

import json
import sqlite3
from pathlib import Path

import pytest

from src.indexer.errors import (
    FileNotFoundError as StoreFileNotFoundError,
)
from src.indexer.errors import (
    InvalidNodeRecordError,
    StoreOperationError,
)
from src.indexer.store import (
    BFSResult,
    EdgeRow,
    NodeRow,
    SearchParams,
    SearchResult,
    Store,
    _batched,
    _node_record_to_row,
    default_db_path,
    open_memory,
    open_path,
    open_path_readonly,
)
from src.indexer.treesitter import NodeRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> Store:
    """Fresh in-memory store for each test."""
    s = open_memory()
    yield s
    s.close()


@pytest.fixture
def populated_store(store: Store) -> Store:
    """
    In-memory store pre-populated with:
      - project "test-app"
      - 3 Function nodes: foo, bar, baz  (src/utils.py)
      - 1 Class node:     MyClass        (src/models.py)
      - 1 Method node:    method_a       (src/models.py, parent=MyClass)
      - edges: foo→bar (CALLS), bar→baz (CALLS), foo→MyClass (IMPORTS)
      - file content for src/utils.py and src/models.py
    """
    store.upsert_project("test-app", "/repo", "python")

    records = [
        _make_record("foo", "Function", "test_app.src.utils.foo", "src/utils.py", 1, 5),
        _make_record(
            "bar", "Function", "test_app.src.utils.bar", "src/utils.py", 7, 12
        ),
        _make_record(
            "baz", "Function", "test_app.src.utils.baz", "src/utils.py", 14, 18
        ),
        _make_record(
            "MyClass", "Class", "test_app.src.models.MyClass", "src/models.py", 1, 20
        ),
        _make_record(
            "method_a",
            "Method",
            "test_app.src.models.MyClass.method_a",
            "src/models.py",
            5,
            10,
            parent="MyClass",
        ),
    ]

    store.begin()
    qn_to_id = store.insert_nodes(records, "test-app")
    store.insert_edges(
        [
            (
                "test_app.src.utils.foo",
                "test_app.src.utils.bar",
                "CALLS",
                {"confidence": 0.95},
            ),
            (
                "test_app.src.utils.bar",
                "test_app.src.utils.baz",
                "CALLS",
                {"confidence": 0.85},
            ),
            (
                "test_app.src.utils.foo",
                "test_app.src.models.MyClass",
                "IMPORTS",
                {"confidence": 1.0},
            ),
        ],
        qn_to_id,
        "test-app",
    )
    store.insert_files(
        {
            "src/utils.py": "def foo(): pass\ndef bar(): pass\ndef baz(): pass",
            "src/models.py": "class MyClass:\n    def method_a(self): pass",
        },
        "test-app",
        {"src/utils.py": "python", "src/models.py": "python"},
    )
    store.insert_file_hashes(
        [
            ("src/utils.py", "aaa111", 1_000_000_000, 256),
            ("src/models.py", "bbb222", 2_000_000_000, 128),
        ],
        "test-app",
    )
    store.commit()

    return store


def _make_record(
    name: str,
    label: str,
    qn: str,
    file_path: str,
    start_line: int,
    end_line: int,
    parent: str = "",
    language: str = "python",
    properties: dict | None = None,
) -> NodeRecord:
    sig = f"def {name}():" if label == "Function" else f"class {name}:"
    src = f"{sig}\n    pass"
    return NodeRecord(
        label=label,
        name=name,
        qualified_name=qn,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=sig,
        source=src,
        language=language,
        parent=parent,
        properties=properties or {},
    )


# ---------------------------------------------------------------------------
# _batched
# ---------------------------------------------------------------------------


class TestBatched:
    def test_even_division(self):
        result = list(_batched([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_uneven_division(self):
        result = list(_batched([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_batch(self):
        result = list(_batched([1, 2, 3], 10))
        assert result == [[1, 2, 3]]

    def test_empty_list(self):
        result = list(_batched([], 5))
        assert result == []

    def test_batch_size_one(self):
        result = list(_batched([1, 2, 3], 1))
        assert result == [[1], [2], [3]]

    def test_batch_size_equals_length(self):
        result = list(_batched([1, 2, 3], 3))
        assert result == [[1, 2, 3]]

    def test_preserves_order(self):
        items = list(range(100))
        batches = list(_batched(items, 7))
        reconstructed = [x for batch in batches for x in batch]
        assert reconstructed == items


# ---------------------------------------------------------------------------
# _node_record_to_row
# ---------------------------------------------------------------------------


class TestNodeRecordToRow:
    def test_basic_fields(self):
        record = _make_record("foo", "Function", "proj.src.foo", "src/foo.py", 1, 5)
        row = _node_record_to_row(record, "proj")
        assert row[0] == "proj"  # project
        assert row[1] == "Function"  # label
        assert row[2] == "foo"  # name
        assert row[3] == "proj.src.foo"  # qualified_name
        assert row[4] == "src/foo.py"  # file_path
        assert row[5] == 1  # start_line
        assert row[6] == 5  # end_line

    def test_properties_serialised_to_json(self):
        record = _make_record(
            "foo",
            "Function",
            "proj.src.foo",
            "src/foo.py",
            1,
            5,
            properties={"async": True, "decorators": ["@login"]},
        )
        row = _node_record_to_row(record, "proj")
        props = json.loads(row[9])
        assert props["async"] is True
        assert props["decorators"] == ["@login"]

    def test_empty_properties_serialises_to_empty_object(self):
        record = _make_record("foo", "Function", "proj.src.foo", "src/foo.py", 1, 5)
        row = _node_record_to_row(record, "proj")
        assert json.loads(row[9]) == {}

    def test_raises_on_empty_qualified_name(self):
        record = _make_record("foo", "Function", "", "src/foo.py", 1, 5)
        with pytest.raises(InvalidNodeRecordError):
            _node_record_to_row(record, "proj")

    def test_returns_tuple_of_10(self):
        record = _make_record("foo", "Function", "proj.src.foo", "src/foo.py", 1, 5)
        row = _node_record_to_row(record, "proj")
        assert len(row) == 10


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    def test_open_memory_returns_store(self):
        s = open_memory()
        assert isinstance(s, Store)
        s.close()

    def test_open_memory_passes_integrity_check(self):
        s = open_memory()
        assert s.check_integrity() is True
        s.close()

    def test_open_path_creates_file(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s = open_path(db_path)
        assert Path(db_path).exists()
        s.close()

    def test_open_path_creates_parent_dirs(self, tmp_path):
        db_path = str(tmp_path / "deep" / "nested" / "test.db")
        s = open_path(db_path)
        assert Path(db_path).exists()
        s.close()

    def test_open_path_existing_file_preserves_data(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s = open_path(db_path)
        s.upsert_project("p", "/repo")
        s.close()

        s2 = open_path(db_path)
        assert s2.get_project("p") is not None
        s2.close()

    def test_open_path_readonly_returns_store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s = open_path(db_path)
        s.close()

        ro = open_path_readonly(db_path)
        assert isinstance(ro, Store)
        ro.close()

    def test_open_path_readonly_raises_on_missing_file(self, tmp_path):
        db_path = str(tmp_path / "nonexistent.db")
        with pytest.raises(StoreFileNotFoundError):
            open_path_readonly(db_path)

    def test_open_path_readonly_prevents_writes(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s = open_path(db_path)
        s.upsert_project("p", "/repo")
        s.close()

        ro = open_path_readonly(db_path)
        with pytest.raises(sqlite3.OperationalError):
            ro.upsert_project("new", "/new")
        ro.close()


# ---------------------------------------------------------------------------
# default_db_path
# ---------------------------------------------------------------------------


class TestDefaultDbPath:
    def test_returns_string(self):
        path = default_db_path("my-app")
        assert isinstance(path, str)

    def test_ends_with_project_db(self):
        path = default_db_path("my-app")
        assert path.endswith("my-app.db")

    def test_custom_cache_dir(self, tmp_path):
        path = default_db_path("my-app", str(tmp_path))
        assert str(tmp_path) in path
        assert path.endswith("my-app.db")

    def test_creates_cache_dir(self, tmp_path):
        cache = str(tmp_path / "new_cache")
        default_db_path("proj", cache)
        assert Path(cache).exists()


# ---------------------------------------------------------------------------
# Store lifecycle
# ---------------------------------------------------------------------------


class TestStoreLifecycle:
    def test_close_is_idempotent(self, store):
        store.close()
        store.close()  # should not raise

    def test_check_integrity_fresh_db(self, store):
        assert store.check_integrity() is True

    def test_checkpoint_does_not_raise(self, store):
        store.checkpoint()  # should be a no-op on in-memory db


# ---------------------------------------------------------------------------
# Bulk write helpers
# ---------------------------------------------------------------------------


class TestBulkWrite:
    def test_begin_end_bulk_cycle(self, store):
        store.begin_bulk()
        store.end_bulk()

    def test_drop_create_indexes_cycle(self, store):
        store.begin_bulk()
        store.drop_indexes()
        store.create_indexes()
        store.end_bulk()

    def test_begin_bulk_twice_raises(self, store):
        store.begin_bulk()
        with pytest.raises(StoreOperationError):
            store.begin_bulk()
        store.end_bulk()

    def test_end_bulk_without_begin_raises(self, store):
        with pytest.raises(StoreOperationError):
            store.end_bulk()

    def test_data_survives_bulk_cycle(self, store):
        store.upsert_project("p", "/repo")
        store.begin_bulk()
        store.drop_indexes()
        records = [_make_record("foo", "Function", "p.src.foo", "src/foo.py", 1, 5)]
        store.begin()
        store.insert_nodes(records, "p")
        store.commit()
        store.create_indexes()
        store.end_bulk()

        node = store.get_node_by_qn("p.src.foo")
        assert node is not None
        assert node.name == "foo"


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------


class TestTransaction:
    def test_commit_persists_data(self, store):
        store.upsert_project("p", "/repo")
        store.begin()
        records = [_make_record("foo", "Function", "p.src.foo", "src/foo.py", 1, 3)]
        store.insert_nodes(records, "p")
        store.commit()
        assert store.get_node_by_qn("p.src.foo") is not None

    def test_rollback_discards_data(self, store):
        store.upsert_project("p", "/repo")
        store.begin()
        records = [_make_record("bar", "Function", "p.src.bar", "src/bar.py", 1, 3)]
        store.insert_nodes(records, "p")
        store.rollback()
        assert store.get_node_by_qn("p.src.bar") is None

    def test_rollback_in_exception_handler(self, store):
        store.upsert_project("p", "/repo")
        try:
            store.begin()
            records = [_make_record("baz", "Function", "p.src.baz", "src/baz.py", 1, 3)]
            store.insert_nodes(records, "p")
            raise ValueError("simulated error")  # noqa: TRY003, TRY301
        except ValueError:
            store.rollback()
        assert store.get_node_by_qn("p.src.baz") is None


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------


class TestProjectCRUD:
    def test_upsert_and_get_project(self, store):
        store.upsert_project("my-app", "/repo/my-app", "python")
        p = store.get_project("my-app")
        assert p is not None
        assert p["name"] == "my-app"
        assert p["root_path"] == "/repo/my-app"
        assert p["language"] == "python"

    def test_get_nonexistent_project_returns_none(self, store):
        assert store.get_project("nope") is None

    def test_upsert_updates_existing(self, store):
        store.upsert_project("p", "/old")
        store.upsert_project("p", "/new", "go")
        p = store.get_project("p")
        assert p["root_path"] == "/new"
        assert p["language"] == "go"

    def test_upsert_updates_indexed_at(self, store):
        store.upsert_project("p", "/repo")
        t1 = store.get_project("p")["indexed_at"]
        import time

        time.sleep(0.01)
        store.upsert_project("p", "/repo")
        t2 = store.get_project("p")["indexed_at"]
        # indexed_at should be refreshed
        assert t2 >= t1

    def test_list_projects_empty(self, store):
        assert store.list_projects() == []

    def test_list_projects_sorted(self, store):
        store.upsert_project("z-app", "/z")
        store.upsert_project("a-app", "/a")
        store.upsert_project("m-app", "/m")
        names = [p["name"] for p in store.list_projects()]
        assert names == sorted(names)

    def test_delete_project_returns_1(self, store):
        store.upsert_project("p", "/repo")
        assert store.delete_project("p") == 1

    def test_delete_nonexistent_project_returns_0(self, store):
        assert store.delete_project("nope") == 0

    def test_delete_project_cascades_to_nodes(self, populated_store):
        populated_store.delete_project("test-app")
        assert populated_store.count_nodes("test-app") == 0

    def test_delete_project_cascades_to_edges(self, populated_store):
        populated_store.delete_project("test-app")
        assert populated_store.count_edges("test-app") == 0

    def test_project_language_nullable(self, store):
        store.upsert_project("p", "/repo")
        p = store.get_project("p")
        assert p["language"] is None


# ---------------------------------------------------------------------------
# Node writes
# ---------------------------------------------------------------------------


class TestNodeWrites:
    def test_insert_nodes_returns_qn_to_id(self, store):
        store.upsert_project("p", "/repo")
        records = [_make_record("foo", "Function", "p.src.foo", "src/foo.py", 1, 5)]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.commit()
        assert "p.src.foo" in qn_to_id
        assert isinstance(qn_to_id["p.src.foo"], int)
        assert qn_to_id["p.src.foo"] > 0

    def test_insert_multiple_nodes(self, store):
        store.upsert_project("p", "/repo")
        records = [
            _make_record("foo", "Function", "p.foo", "f.py", 1, 3),
            _make_record("bar", "Function", "p.bar", "f.py", 5, 7),
            _make_record("baz", "Function", "p.baz", "f.py", 9, 11),
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.commit()
        assert len(qn_to_id) == 3
        assert store.count_nodes("p") == 3

    def test_insert_nodes_raises_on_empty_qn(self, store):
        store.upsert_project("p", "/repo")
        records = [_make_record("foo", "Function", "", "f.py", 1, 3)]
        with pytest.raises(InvalidNodeRecordError):
            store.insert_nodes(records, "p")

    def test_insert_replaces_on_conflict(self, store):
        store.upsert_project("p", "/repo")
        r1 = _make_record("foo", "Function", "p.foo", "f.py", 1, 3)
        store.begin()
        store.insert_nodes([r1], "p")
        store.commit()

        # Update signature
        r2 = _make_record("foo", "Function", "p.foo", "f.py", 1, 5)
        r2.signature = "def foo(x: int) -> int:"
        store.begin()
        store.insert_nodes([r2], "p")
        store.commit()

        node = store.get_node_by_qn("p.foo")
        assert node.signature == "def foo(x: int) -> int:"
        assert store.count_nodes("p") == 1  # not doubled

    def test_insert_large_batch(self, store):
        store.upsert_project("p", "/repo")
        records = [
            _make_record(f"fn_{i}", "Function", f"p.fn_{i}", "f.py", i, i + 1)
            for i in range(600)  # exceeds _INSERT_BATCH_SIZE
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.commit()
        assert store.count_nodes("p") == 600
        assert len(qn_to_id) == 600

    def test_delete_nodes_for_file(self, populated_store):
        deleted = populated_store.delete_nodes_for_file("test-app", "src/utils.py")
        assert deleted == 3  # foo, bar, baz
        assert populated_store.count_nodes("test-app") == 2  # MyClass, method_a

    def test_delete_nodes_for_nonexistent_file(self, store):
        store.upsert_project("p", "/repo")
        result = store.delete_nodes_for_file("p", "nonexistent.py")
        assert result == 0

    def test_node_properties_round_trip(self, store):
        store.upsert_project("p", "/repo")
        props = {
            "async": True,
            "decorators": ["@login_required"],
            "visibility": "public",
        }
        r = _make_record("foo", "Function", "p.foo", "f.py", 1, 5, properties=props)
        store.begin()
        store.insert_nodes([r], "p")
        store.commit()
        node = store.get_node_by_qn("p.foo")
        assert node.properties["async"] is True
        assert node.properties["decorators"] == ["@login_required"]
        assert node.properties["visibility"] == "public"


# ---------------------------------------------------------------------------
# Edge writes
# ---------------------------------------------------------------------------


class TestEdgeWrites:
    def test_insert_edges_returns_count(self, populated_store):
        assert populated_store.count_edges("test-app") == 3

    def test_insert_edges_skips_unresolved(self, store):
        store.upsert_project("p", "/repo")
        records = [_make_record("foo", "Function", "p.foo", "f.py", 1, 3)]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        inserted = store.insert_edges(
            [("p.foo", "p.nonexistent", "CALLS", {})],
            qn_to_id,
            "p",
        )
        store.commit()
        assert inserted == 0

    def test_insert_edges_skips_duplicates(self, store):
        store.upsert_project("p", "/repo")
        records = [
            _make_record("foo", "Function", "p.foo", "f.py", 1, 3),
            _make_record("bar", "Function", "p.bar", "f.py", 5, 7),
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        edge = [("p.foo", "p.bar", "CALLS", {"confidence": 0.9})]
        store.insert_edges(edge, qn_to_id, "p")
        store.insert_edges(edge, qn_to_id, "p")  # duplicate
        store.commit()
        assert store.count_edges("p") == 1

    def test_edge_properties_round_trip(self, store):
        store.upsert_project("p", "/repo")
        records = [
            _make_record("foo", "Function", "p.foo", "f.py", 1, 3),
            _make_record("bar", "Function", "p.bar", "f.py", 5, 7),
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.insert_edges(
            [
                (
                    "p.foo",
                    "p.bar",
                    "CALLS",
                    {"confidence": 0.95, "strategy": "same_module"},
                )
            ],
            qn_to_id,
            "p",
        )
        store.commit()

        result = store.bfs_callees("p.foo", project="p", max_depth=1)
        assert result is not None
        edge = result.edges[0]
        assert edge.properties["confidence"] == 0.95
        assert edge.properties["strategy"] == "same_module"

    def test_different_edge_types(self, store):
        store.upsert_project("p", "/repo")
        records = [
            _make_record("foo", "Function", "p.foo", "f.py", 1, 3),
            _make_record("Bar", "Class", "p.Bar", "g.py", 1, 10),
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.insert_edges(
            [
                ("p.foo", "p.Bar", "CALLS", {}),
                ("p.foo", "p.Bar", "IMPORTS", {}),
            ],
            qn_to_id,
            "p",
        )
        store.commit()
        assert store.count_edges("p") == 2

    def test_insert_large_edge_batch(self, store):
        store.upsert_project("p", "/repo")
        n = 200
        records = [
            _make_record(f"f{i}", "Function", f"p.f{i}", "f.py", i, i + 1)
            for i in range(n)
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        edges = [(f"p.f{i}", f"p.f{i+1}", "CALLS", {}) for i in range(n - 1)]
        inserted = store.insert_edges(edges, qn_to_id, "p")
        store.commit()
        assert inserted == n - 1


# ---------------------------------------------------------------------------
# File writes
# ---------------------------------------------------------------------------


class TestFileWrites:
    def test_insert_and_get_file_source(self, store):
        store.upsert_project("p", "/repo")
        store.insert_files({"src/foo.py": "def foo(): pass"}, "p")
        src = store.get_file_source("p", "src/foo.py")
        assert src == "def foo(): pass"

    def test_get_file_source_missing_returns_none(self, store):
        store.upsert_project("p", "/repo")
        assert store.get_file_source("p", "nonexistent.py") is None

    def test_insert_files_with_language(self, store):
        store.upsert_project("p", "/repo")
        store.insert_files(
            {"src/foo.py": "x = 1"},
            "p",
            {"src/foo.py": "python"},
        )
        src = store.get_file_source("p", "src/foo.py")
        assert src == "x = 1"

    def test_insert_files_replace_on_conflict(self, store):
        store.upsert_project("p", "/repo")
        store.insert_files({"src/foo.py": "v1"}, "p")
        store.insert_files({"src/foo.py": "v2"}, "p")
        assert store.get_file_source("p", "src/foo.py") == "v2"

    def test_insert_file_hashes_and_get(self, store):
        store.upsert_project("p", "/repo")
        store.insert_file_hashes(
            [("src/foo.py", "abc123", 999, 512)],
            "p",
        )
        hashes = store.get_file_hashes("p")
        assert "src/foo.py" in hashes
        sha, mtime, size = hashes["src/foo.py"]
        assert sha == "abc123"
        assert mtime == 999
        assert size == 512

    def test_get_file_hashes_empty(self, store):
        store.upsert_project("p", "/repo")
        assert store.get_file_hashes("p") == {}

    def test_insert_file_hashes_replace_on_conflict(self, store):
        store.upsert_project("p", "/repo")
        store.insert_file_hashes([("src/foo.py", "v1", 1, 100)], "p")
        store.insert_file_hashes([("src/foo.py", "v2", 2, 200)], "p")
        hashes = store.get_file_hashes("p")
        sha, mtime, size = hashes["src/foo.py"]
        assert sha == "v2"
        assert mtime == 2


# ---------------------------------------------------------------------------
# Node reads
# ---------------------------------------------------------------------------


class TestNodeReads:
    def test_get_node_by_qn_found(self, populated_store):
        node = populated_store.get_node_by_qn("test_app.src.utils.foo")
        assert node is not None
        assert node.name == "foo"
        assert node.label == "Function"

    def test_get_node_by_qn_not_found(self, populated_store):
        assert populated_store.get_node_by_qn("nonexistent.qn") is None

    def test_get_node_by_qn_with_project_filter(self, populated_store):
        node = populated_store.get_node_by_qn(
            "test_app.src.utils.foo", project="test-app"
        )
        assert node is not None

    def test_get_node_by_qn_wrong_project_returns_none(self, populated_store):
        node = populated_store.get_node_by_qn(
            "test_app.src.utils.foo", project="other-project"
        )
        assert node is None

    def test_get_node_by_id(self, populated_store):
        node = populated_store.get_node_by_qn("test_app.src.utils.foo")
        assert node is not None
        by_id = populated_store.get_node_by_id(node.id)
        assert by_id is not None
        assert by_id.qualified_name == node.qualified_name

    def test_get_node_by_id_not_found(self, store):
        assert store.get_node_by_id(999999) is None

    def test_get_nodes_by_file(self, populated_store):
        nodes = populated_store.get_nodes_by_file("test-app", "src/utils.py")
        assert len(nodes) == 3
        names = {n.name for n in nodes}
        assert names == {"foo", "bar", "baz"}

    def test_get_nodes_by_file_ordered_by_start_line(self, populated_store):
        nodes = populated_store.get_nodes_by_file("test-app", "src/utils.py")
        lines = [n.start_line for n in nodes]
        assert lines == sorted(lines)

    def test_get_nodes_by_file_missing_returns_empty(self, store):
        store.upsert_project("p", "/repo")
        assert store.get_nodes_by_file("p", "nonexistent.py") == []

    def test_count_nodes(self, populated_store):
        assert populated_store.count_nodes("test-app") == 5

    def test_count_nodes_zero(self, store):
        store.upsert_project("p", "/repo")
        assert store.count_nodes("p") == 0

    def test_count_edges(self, populated_store):
        assert populated_store.count_edges("test-app") == 3

    def test_count_edges_zero(self, store):
        store.upsert_project("p", "/repo")
        assert store.count_edges("p") == 0

    def test_node_row_fields(self, populated_store):
        node = populated_store.get_node_by_qn("test_app.src.utils.foo")
        assert isinstance(node.id, int)
        assert node.project == "test-app"
        assert node.file_path == "src/utils.py"
        assert node.start_line == 1
        assert node.end_line == 5
        assert isinstance(node.signature, str)
        assert isinstance(node.source, str)
        assert isinstance(node.properties, dict)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearchNodes:
    def test_search_by_project(self, populated_store):
        result = populated_store.search_nodes(SearchParams(project="test-app"))
        assert result.total == 5

    def test_search_by_label(self, populated_store):
        result = populated_store.search_nodes(
            SearchParams(project="test-app", label="Function")
        )
        assert all(r.label == "Function" for r in result.rows)
        assert result.total == 3

    def test_search_by_label_class(self, populated_store):
        result = populated_store.search_nodes(
            SearchParams(project="test-app", label="Class")
        )
        assert result.total == 1
        assert result.rows[0].name == "MyClass"

    def test_search_by_name_pattern(self, populated_store):
        result = populated_store.search_nodes(
            SearchParams(project="test-app", name_pattern="%ba%")
        )
        names = {r.name for r in result.rows}
        assert "bar" in names
        assert "baz" in names
        assert "foo" not in names

    def test_search_by_file_pattern(self, populated_store):
        result = populated_store.search_nodes(
            SearchParams(project="test-app", file_pattern="src/utils%")
        )
        assert result.total == 3
        assert all(r.file_path == "src/utils.py" for r in result.rows)

    def test_search_fts_query(self, populated_store):
        result = populated_store.search_nodes(
            SearchParams(project="test-app", fts_query="MyClass")
        )
        assert result.total >= 1
        names = {r.name for r in result.rows}
        assert "MyClass" in names

    def test_search_pagination_limit(self, populated_store):
        result = populated_store.search_nodes(SearchParams(project="test-app", limit=2))
        assert len(result.rows) == 2
        assert result.total == 5

    def test_search_pagination_offset(self, populated_store):
        result_p1 = populated_store.search_nodes(
            SearchParams(project="test-app", limit=2, offset=0)
        )
        result_p2 = populated_store.search_nodes(
            SearchParams(project="test-app", limit=2, offset=2)
        )
        ids_p1 = {r.id for r in result_p1.rows}
        ids_p2 = {r.id for r in result_p2.rows}
        assert ids_p1.isdisjoint(ids_p2)

    def test_search_no_results(self, populated_store):
        result = populated_store.search_nodes(
            SearchParams(project="test-app", name_pattern="%zzznomatch%")
        )
        assert result.total == 0
        assert result.rows == []

    def test_search_returns_search_result_type(self, populated_store):
        result = populated_store.search_nodes(SearchParams(project="test-app"))
        assert isinstance(result, SearchResult)
        assert all(isinstance(r, NodeRow) for r in result.rows)

    def test_search_no_project_filter_returns_all(self, store):
        store.upsert_project("a", "/a")
        store.upsert_project("b", "/b")
        ra = _make_record("fn_a", "Function", "a.fn_a", "f.py", 1, 3)
        rb = _make_record("fn_b", "Function", "b.fn_b", "g.py", 1, 3)
        store.begin()
        store.insert_nodes([ra], "a")
        store.insert_nodes([rb], "b")
        store.commit()
        result = store.search_nodes(SearchParams())
        assert result.total == 2

    def test_search_default_limit_applied(self, store):
        store.upsert_project("p", "/repo")
        records = [
            _make_record(f"f{i}", "Function", f"p.f{i}", "f.py", i, i + 1)
            for i in range(50)
        ]
        store.begin()
        store.insert_nodes(records, "p")
        store.commit()
        result = store.search_nodes(SearchParams(project="p"))
        assert len(result.rows) == 20  # default limit
        assert result.total == 50


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------


class TestBFS:
    # ── bfs_callees ─────────────────────────────────────────────────────

    def test_bfs_callees_direct(self, populated_store):
        result = populated_store.bfs_callees(
            "test_app.src.utils.foo", project="test-app", max_depth=1
        )
        assert result is not None
        assert result.root.name == "foo"
        visited_names = {r.name for r, _ in result.visited}
        assert "bar" in visited_names

    def test_bfs_callees_depth_2(self, populated_store):
        result = populated_store.bfs_callees(
            "test_app.src.utils.foo", project="test-app", max_depth=2
        )
        visited_names = {r.name for r, _ in result.visited}
        assert "bar" in visited_names
        assert "baz" in visited_names

    def test_bfs_callees_hop_depths(self, populated_store):
        result = populated_store.bfs_callees(
            "test_app.src.utils.foo", project="test-app", max_depth=2
        )
        hop_map = {r.name: hop for r, hop in result.visited}
        assert hop_map.get("bar") == 1
        assert hop_map.get("baz") == 2

    def test_bfs_callees_max_depth_1_excludes_indirect(self, populated_store):
        result = populated_store.bfs_callees(
            "test_app.src.utils.foo", project="test-app", max_depth=1
        )
        visited_names = {r.name for r, _ in result.visited}
        assert "baz" not in visited_names

    def test_bfs_callees_not_found_returns_none(self, store):
        store.upsert_project("p", "/repo")
        assert store.bfs_callees("nonexistent.qn") is None

    def test_bfs_callees_returns_bfs_result(self, populated_store):
        result = populated_store.bfs_callees("test_app.src.utils.foo")
        assert isinstance(result, BFSResult)
        assert isinstance(result.root, NodeRow)
        assert all(isinstance(r, NodeRow) for r, _ in result.visited)
        assert all(isinstance(e, EdgeRow) for e in result.edges)

    def test_bfs_callees_max_nodes_respected(self, store):
        # Chain: f0 → f1 → f2 → ... → f9
        store.upsert_project("p", "/repo")
        n = 10
        records = [
            _make_record(f"f{i}", "Function", f"p.f{i}", "f.py", i, i + 1)
            for i in range(n)
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        edges = [(f"p.f{i}", f"p.f{i+1}", "CALLS", {}) for i in range(n - 1)]
        store.insert_edges(edges, qn_to_id, "p")
        store.commit()

        result = store.bfs_callees("p.f0", project="p", max_nodes=3)
        assert len(result.visited) <= 3

    # ── bfs_callers ─────────────────────────────────────────────────────

    def test_bfs_callers_direct(self, populated_store):
        # bar is called by foo
        result = populated_store.bfs_callers(
            "test_app.src.utils.bar", project="test-app", max_depth=1
        )
        assert result is not None
        visited_names = {r.name for r, _ in result.visited}
        assert "foo" in visited_names

    def test_bfs_callers_indirect(self, populated_store):
        # baz is called by bar (hop1), bar called by foo (hop2)
        result = populated_store.bfs_callers(
            "test_app.src.utils.baz", project="test-app", max_depth=2
        )
        visited_names = {r.name for r, _ in result.visited}
        assert "bar" in visited_names
        assert "foo" in visited_names

    def test_bfs_callers_hop_depths(self, populated_store):
        result = populated_store.bfs_callers(
            "test_app.src.utils.baz", project="test-app", max_depth=2
        )
        hop_map = {r.name: hop for r, hop in result.visited}
        assert hop_map.get("bar") == 1
        assert hop_map.get("foo") == 2

    def test_bfs_callers_no_callers(self, populated_store):
        # foo has no callers
        result = populated_store.bfs_callers(
            "test_app.src.utils.foo", project="test-app", max_depth=1
        )
        assert result is not None
        assert result.visited == []

    def test_bfs_callers_not_found_returns_none(self, store):
        store.upsert_project("p", "/repo")
        assert store.bfs_callers("nonexistent.qn") is None

    def test_bfs_callers_edge_types_filter(self, populated_store):
        # IMPORTS edges should not appear in default CALLS traversal
        result = populated_store.bfs_callers(
            "test_app.src.models.MyClass",
            project="test-app",
            max_depth=1,
            edge_types=["CALLS"],
        )
        assert result is not None
        visited_names = {r.name for r, _ in result.visited}
        assert "foo" not in visited_names  # foo→MyClass is IMPORTS, not CALLS

    def test_bfs_callers_with_imports_edge_type(self, populated_store):
        result = populated_store.bfs_callers(
            "test_app.src.models.MyClass",
            project="test-app",
            max_depth=1,
            edge_types=["IMPORTS"],
        )
        assert result is not None
        visited_names = {r.name for r, _ in result.visited}
        assert "foo" in visited_names

    def test_bfs_no_cycles(self, store):
        # Circular call: f0 → f1 → f0
        store.upsert_project("p", "/repo")
        records = [
            _make_record("f0", "Function", "p.f0", "f.py", 1, 3),
            _make_record("f1", "Function", "p.f1", "f.py", 5, 7),
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.insert_edges(
            [("p.f0", "p.f1", "CALLS", {}), ("p.f1", "p.f0", "CALLS", {})],
            qn_to_id,
            "p",
        )
        store.commit()

        # Should terminate without infinite loop
        result = store.bfs_callees("p.f0", project="p", max_depth=10)
        assert result is not None
        visited_names = {r.name for r, _ in result.visited}
        assert len(visited_names) == 1  # only f1, not f0 again


# ---------------------------------------------------------------------------
# Schema summary
# ---------------------------------------------------------------------------


class TestSchemaSummary:
    def test_summary_keys(self, populated_store):
        summary = populated_store.get_schema_summary("test-app")
        assert "node_labels" in summary
        assert "edge_types" in summary
        assert "total_nodes" in summary
        assert "total_edges" in summary
        assert "sample_qns" in summary

    def test_summary_counts(self, populated_store):
        summary = populated_store.get_schema_summary("test-app")
        assert summary["total_nodes"] == 5
        assert summary["total_edges"] == 3

    def test_summary_label_counts(self, populated_store):
        summary = populated_store.get_schema_summary("test-app")
        label_map = {entry["label"]: entry["count"] for entry in summary["node_labels"]}
        assert label_map.get("Function") == 3
        assert label_map.get("Class") == 1
        assert label_map.get("Method") == 1

    def test_summary_edge_types(self, populated_store):
        summary = populated_store.get_schema_summary("test-app")
        type_map = {t["type"]: t["count"] for t in summary["edge_types"]}
        assert type_map.get("CALLS") == 2
        assert type_map.get("IMPORTS") == 1

    def test_summary_sample_qns(self, populated_store):
        summary = populated_store.get_schema_summary("test-app")
        assert isinstance(summary["sample_qns"], list)
        assert len(summary["sample_qns"]) > 0

    def test_summary_empty_project(self, store):
        store.upsert_project("empty", "/repo")
        summary = store.get_schema_summary("empty")
        assert summary["total_nodes"] == 0
        assert summary["total_edges"] == 0
        assert summary["node_labels"] == []


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------


class TestIterSkeleton:
    def test_yields_tuples(self, populated_store):
        rows = list(populated_store.iter_skeleton("test-app"))
        assert len(rows) > 0
        for fp, sig, qn in rows:
            assert isinstance(fp, str)
            assert isinstance(sig, str)
            assert isinstance(qn, str)

    def test_ordered_by_file_path_then_start_line(self, populated_store):
        rows = list(populated_store.iter_skeleton("test-app"))
        file_paths = [fp for fp, _, _ in rows]
        # File paths should be in non-decreasing order
        assert file_paths == sorted(file_paths)

    def test_excludes_file_nodes_by_default(self, store):
        store.upsert_project("p", "/repo")
        records = [
            _make_record("Dockerfile", "File", "p.Dockerfile", "Dockerfile", 1, 10),
            _make_record("foo", "Function", "p.foo", "src/foo.py", 1, 5),
        ]
        store.begin()
        store.insert_nodes(records, "p")
        store.commit()
        rows = list(store.iter_skeleton("p"))
        qns = [qn for _, _, qn in rows]
        assert "p.foo" in qns
        assert "p.Dockerfile" not in qns

    def test_label_filter(self, populated_store):
        rows = list(populated_store.iter_skeleton("test-app", labels=["Function"]))
        for _fp, _sig, qn in rows:
            node = populated_store.get_node_by_qn(qn)
            assert node.label == "Function"

    def test_all_non_file_nodes_included(self, populated_store):
        rows = list(populated_store.iter_skeleton("test-app"))
        qns = {qn for _, _, qn in rows}
        assert "test_app.src.utils.foo" in qns
        assert "test_app.src.models.MyClass" in qns
        assert "test_app.src.models.MyClass.method_a" in qns

    def test_empty_project_yields_nothing(self, store):
        store.upsert_project("empty", "/repo")
        rows = list(store.iter_skeleton("empty"))
        assert rows == []


# ---------------------------------------------------------------------------
# Dump / restore
# ---------------------------------------------------------------------------


class TestDumpRestore:
    def test_dump_to_file_creates_file(self, populated_store, tmp_path):
        db_path = str(tmp_path / "dump.db")
        populated_store.dump_to_file(db_path)
        assert Path(db_path).exists()
        assert Path(db_path).stat().st_size > 0

    def test_dump_to_file_creates_parent_dirs(self, populated_store, tmp_path):
        db_path = str(tmp_path / "deep" / "nested" / "dump.db")
        populated_store.dump_to_file(db_path)
        assert Path(db_path).exists()

    def test_dump_preserves_data(self, populated_store, tmp_path):
        db_path = str(tmp_path / "dump.db")
        populated_store.dump_to_file(db_path)

        loaded = open_path(db_path)
        assert loaded.count_nodes("test-app") == 5
        assert loaded.count_edges("test-app") == 3
        node = loaded.get_node_by_qn("test_app.src.utils.foo")
        assert node is not None
        assert node.name == "foo"
        loaded.close()

    def test_restore_from_file(self, populated_store, tmp_path):
        db_path = str(tmp_path / "source.db")
        populated_store.dump_to_file(db_path)

        target = open_memory()
        target.restore_from_file(db_path)
        assert target.count_nodes("test-app") == 5
        target.close()

    def test_restore_raises_on_missing_file(self, store, tmp_path):
        with pytest.raises(StoreFileNotFoundError):
            store.restore_from_file(str(tmp_path / "nonexistent.db"))

    def test_round_trip_integrity(self, populated_store, tmp_path):
        db_path = str(tmp_path / "rt.db")
        populated_store.dump_to_file(db_path)
        loaded = open_path(db_path)
        assert loaded.check_integrity() is True
        loaded.close()


# ---------------------------------------------------------------------------
# _row_to_node and _row_to_edge
# ---------------------------------------------------------------------------


class TestRowConversions:
    def test_row_to_node_deserialises_properties(self, populated_store):
        node = populated_store.get_node_by_qn("test_app.src.utils.foo")
        assert isinstance(node.properties, dict)

    def test_row_to_node_handles_null_properties(self, store):
        # Insert a row with NULL properties directly to test robustness
        store.upsert_project("p", "/repo")
        store._conn.execute(
            """
            INSERT INTO nodes
                (project, label, name, qualified_name, file_path,
                 start_line, end_line, signature, source, properties)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """,
            ("p", "Function", "f", "p.f", "f.py", 1, 2, "def f():", "def f(): pass"),
        )
        store._conn.commit()
        node = store.get_node_by_qn("p.f")
        assert node is not None
        assert node.properties == {}

    def test_row_to_edge_deserialises_properties(self, populated_store):
        result = populated_store.bfs_callees(
            "test_app.src.utils.foo", project="test-app", max_depth=1
        )
        for edge in result.edges:
            assert isinstance(edge.properties, dict)

    def test_node_row_is_dataclass(self, populated_store):
        node = populated_store.get_node_by_qn("test_app.src.utils.foo")
        assert isinstance(node, NodeRow)

    def test_edge_row_is_dataclass(self, populated_store):
        result = populated_store.bfs_callees(
            "test_app.src.utils.foo", project="test-app", max_depth=1
        )
        for edge in result.edges:
            assert isinstance(edge, EdgeRow)
