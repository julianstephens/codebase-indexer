"""
schema.py — SQLite DDL for the repo knowledge graph.

Tables:
  projects     — indexed repos
  nodes        — every function/class/method/file with full source
  edges        — CALLS, IMPORTS, INHERITS, DEFINES relationships
  files        — raw file content (for fallback full-file retrieval)
  file_hashes  — mtime + sha256 per file (incremental re-index)
  nodes_fts    — FTS5 virtual table over name, signature, source
  adr          — architecture decision records (optional, per project)

Design notes:
  - qualified_name is the globally unique address for every node.
    Format: path.to.module.ClassName.method_name
    Used by get_source(), trace_callers(), and all edge resolution.
  - source stores the full body of each node. Nothing is stripped.
    The skeleton is derived at query time from the signature column.
  - properties is a JSON blob for language-specific extras:
    decorators, return types, visibility, async flag, etc.
  - Indexes are defined separately from the schema so the pipeline
    can drop them before bulk insert and recreate after (10-50x
    faster for large repos).
  - FTS5 content= + content_rowid= keeps the index in sync with
    the nodes table without duplicating storage.
"""

# ---------------------------------------------------------------------------
# Core schema
# ---------------------------------------------------------------------------

import sqlite3

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Projects ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS projects (
    name         TEXT PRIMARY KEY,
    root_path    TEXT NOT NULL,
    language     TEXT,                  -- dominant language, nullable
    indexed_at   TEXT NOT NULL DEFAULT (datetime('now')),
    schema_ver   INTEGER NOT NULL DEFAULT 1
);

-- ── Nodes ─────────────────────────────────────────────────────────────────
--
-- One row per extractable symbol: Function, Class, Method, Interface,
-- Type, Module, File, Resource.
--
-- qualified_name is the stable address used everywhere else in the system.
-- It is globally unique within a project.
--
-- signature is everything up to (but not including) the body opener:
--   Python:  "def charge(user: User, amount_cents: int) -> Payment:"
--   Go:      "func (s *PaymentService) Charge(user User, cents int) error"
--   TS:      "async function charge(user: User, amountCents: number): Promise<Payment>"
--
-- source is the full source text of this node including the body.
-- For File nodes (fallback extraction) this is the entire file.
--
-- properties is a JSON object for extras that vary by language:
--   {"async": true, "decorators": ["@login_required"], "visibility": "public",
--    "return_type": "Payment", "bases": ["BaseModel"], "hotspot_score": 0.82}

CREATE TABLE IF NOT EXISTS nodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project         TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    label           TEXT    NOT NULL,   -- Function | Class | Method | Interface |
                                        -- Type | Module | File | Resource
    name            TEXT    NOT NULL,   -- short name: "charge"
    qualified_name  TEXT    NOT NULL,   -- full address: "myapp.payments.service.charge"
    file_path       TEXT    NOT NULL,   -- repo-relative: "src/payments/service.py"
    start_line      INTEGER NOT NULL DEFAULT 0,
    end_line        INTEGER NOT NULL DEFAULT 0,
    signature       TEXT    NOT NULL DEFAULT '',
    source          TEXT    NOT NULL DEFAULT '',
    properties      TEXT             DEFAULT '{}',

    UNIQUE (project, qualified_name)
);

-- ── Edges ─────────────────────────────────────────────────────────────────
--
-- Typed relationships between nodes.
--
-- Edge types used by this system:
--   CALLS      — function/method A calls function/method B
--   IMPORTS    — file/module A imports file/module B
--   DEFINES    — file/class A defines function/method B
--   INHERITS   — class A inherits from class B
--   IMPLEMENTS — class A implements interface B
--   CONTAINS   — package/folder A contains file B
--
-- properties carries resolution metadata:
--   {"confidence": 0.95, "strategy": "same_module", "line": 42}
-- For HTTP_CALLS (future):
--   {"url_path": "/api/payments", "method": "POST", "confidence": 0.80}

CREATE TABLE IF NOT EXISTS edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    source_id   INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id   INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    type        TEXT    NOT NULL,
    properties  TEXT    NOT NULL DEFAULT '{}',

    -- Prevent duplicate edges of the same type between the same nodes
    UNIQUE (source_id, target_id, type)
);

-- ── Files ─────────────────────────────────────────────────────────────────
--
-- Raw file content for every indexed file.
-- Used by:
--   1. Fallback get_source() when a QN resolves to a File node.
--   2. Future diff/patch application during remediation.
--   3. Re-extraction without re-reading disk (incremental pipeline).

CREATE TABLE IF NOT EXISTS files (
    project     TEXT NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    path        TEXT NOT NULL,          -- repo-relative path
    language    TEXT,                   -- detected language, nullable
    source      TEXT NOT NULL,          -- full raw content
    line_count  INTEGER NOT NULL DEFAULT 0,
    size_bytes  INTEGER NOT NULL DEFAULT 0,

    PRIMARY KEY (project, path)
);

-- ── File hashes ────────────────────────────────────────────────────────────
--
-- sha256 + mtime_ns per file for incremental re-indexing.
-- A file is re-extracted only when sha256 differs from the stored value.
-- mtime_ns is a fast pre-check: if mtime hasn't changed, skip sha256.

CREATE TABLE IF NOT EXISTS file_hashes (
    project     TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    path        TEXT    NOT NULL,
    sha256      TEXT    NOT NULL,
    mtime_ns    INTEGER NOT NULL,
    size_bytes  INTEGER NOT NULL DEFAULT 0,

    PRIMARY KEY (project, path)
);

-- ── Architecture Decision Records ─────────────────────────────────────────
--
-- Optional per-project notes persisted across agent sessions.
-- Stored as structured sections in a JSON blob:
--   {"context": "...", "decision": "...", "consequences": "..."}

CREATE TABLE IF NOT EXISTS adr (
    project     TEXT PRIMARY KEY REFERENCES projects(name) ON DELETE CASCADE,
    content     TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# ---------------------------------------------------------------------------
# FTS5 virtual table
# ---------------------------------------------------------------------------
#
# Separate from SCHEMA because it must be created after the nodes table
# exists. content= makes this a "content table" FTS index — FTS5 reads
# from nodes directly and does not duplicate text storage.
#
# Searchable columns:
#   name           — short symbol name ("charge", "UserViewSet")
#   qualified_name — full dotted path for exact lookups
#   signature      — def/func line including type hints
#   source         — full body (enables "find code that does X")
#
# tokenize='unicode61' handles camelCase and snake_case reasonably.
# For camelCase splitting (search("getUserById") finds "getUser", "ById"),
# a custom tokenizer would be needed — out of scope for v1.

FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    name,
    qualified_name,
    signature,
    source,
    content     = 'nodes',
    content_rowid = 'id',
    tokenize    = 'unicode61 remove_diacritics 1'
);
"""

# Triggers keep the FTS index in sync with nodes without manual maintenance.
FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS nodes_fts_insert
    AFTER INSERT ON nodes BEGIN
        INSERT INTO nodes_fts (rowid, name, qualified_name, signature, source)
        VALUES (new.id, new.name, new.qualified_name, new.signature, new.source);
    END;

CREATE TRIGGER IF NOT EXISTS nodes_fts_delete
    AFTER DELETE ON nodes BEGIN
        INSERT INTO nodes_fts (nodes_fts, rowid, name, qualified_name, signature, source)
        VALUES ('delete', old.id, old.name, old.qualified_name, old.signature, old.source);
    END;

CREATE TRIGGER IF NOT EXISTS nodes_fts_update
    AFTER UPDATE ON nodes BEGIN
        INSERT INTO nodes_fts (nodes_fts, rowid, name, qualified_name, signature, source)
        VALUES ('delete', old.id, old.name, old.qualified_name, old.signature, old.source);
        INSERT INTO nodes_fts (rowid, name, qualified_name, signature, source)
        VALUES (new.id, new.name, new.qualified_name, new.signature, new.source);
    END;
"""

# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------
#
# Defined separately so the pipeline can:
#   1. DROP all indexes before bulk insert   (store.drop_indexes)
#   2. INSERT all nodes + edges in one tx
#   3. RECREATE indexes after               (store.create_indexes)
#
# This pattern gives 10-50x faster bulk inserts on large repos.
# The UNIQUE constraints on nodes and edges are enforced by the table
# definition itself, not by these indexes, so they survive the drop.

INDEXES = """
-- Node lookups
CREATE INDEX IF NOT EXISTS idx_nodes_project    ON nodes (project);
CREATE INDEX IF NOT EXISTS idx_nodes_label      ON nodes (label);
CREATE INDEX IF NOT EXISTS idx_nodes_file       ON nodes (file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_name       ON nodes (name);
CREATE INDEX IF NOT EXISTS idx_nodes_label_proj ON nodes (project, label);

-- Edge traversal (forward and reverse)
CREATE INDEX IF NOT EXISTS idx_edges_source     ON edges (source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target     ON edges (target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type       ON edges (type);
CREATE INDEX IF NOT EXISTS idx_edges_proj_type  ON edges (project, type);

-- Incremental re-index
CREATE INDEX IF NOT EXISTS idx_hashes_project   ON file_hashes (project);
"""

# Individual DROP statements matching each CREATE above.
# Used by store.drop_indexes() before bulk insert.
DROP_INDEXES = """
DROP INDEX IF EXISTS idx_nodes_project;
DROP INDEX IF EXISTS idx_nodes_label;
DROP INDEX IF EXISTS idx_nodes_file;
DROP INDEX IF EXISTS idx_nodes_name;
DROP INDEX IF EXISTS idx_nodes_label_proj;
DROP INDEX IF EXISTS idx_edges_source;
DROP INDEX IF EXISTS idx_edges_target;
DROP INDEX IF EXISTS idx_edges_type;
DROP INDEX IF EXISTS idx_edges_proj_type;
DROP INDEX IF EXISTS idx_hashes_project;
"""

# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


def initialize(conn: sqlite3.Connection) -> None:
    """
    Create all tables, FTS, triggers, and indexes on a fresh connection.
    Safe to call on an existing database — all statements use IF NOT EXISTS.

    Args:
        conn: an open sqlite3.Connection
    """
    conn.executescript(SCHEMA)
    conn.executescript(FTS)
    conn.executescript(FTS_TRIGGERS)
    conn.executescript(INDEXES)
    conn.commit()


def drop_indexes(conn: sqlite3.Connection) -> None:
    """
    Drop all non-unique indexes before a bulk insert.
    The UNIQUE constraints on nodes(project, qualified_name) and
    edges(source_id, target_id, type) are part of the table definition
    and are NOT dropped — they continue to enforce integrity.

    Args:
        conn: an open sqlite3.Connection
    """
    conn.executescript(DROP_INDEXES)
    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> None:
    """
    Recreate indexes after a bulk insert.

    Args:
        conn: an open sqlite3.Connection
    """
    conn.executescript(INDEXES)
    conn.execute("PRAGMA optimize")
    conn.commit()


def begin_bulk(conn: sqlite3.Connection) -> None:
    """
    Tune SQLite pragmas for maximum bulk write throughput.
    WAL mode is preserved (set in SCHEMA). Call drop_indexes() first.

    Args:
        conn: an open sqlite3.Connection
    """
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA cache_size = -65536")  # 64 MB page cache
    conn.execute("PRAGMA temp_store = MEMORY")


def end_bulk(conn: sqlite3.Connection) -> None:
    """
    Restore safe pragmas after bulk insert. Call create_indexes() first.

    Args:
        conn: an open sqlite3.Connection
    """
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -2000")  # default
    conn.execute("PRAGMA temp_store = DEFAULT")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.commit()


def check_integrity(conn: sqlite3.Connection) -> bool:
    """
    Basic sanity check. Returns True if the database looks healthy.
    A False result means the caller should delete and re-index.

    Args:
        conn: an open sqlite3.Connection
    """
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result[0] != "ok":
            return False
        # Schema version forward-compatibility check
        row = conn.execute("SELECT MAX(schema_ver) FROM projects").fetchone()
        return not (row and row[0] is not None and row[0] > SCHEMA_VERSION)
    except Exception:
        return False
