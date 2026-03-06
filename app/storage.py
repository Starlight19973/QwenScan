"""Persistent storage for QwenScan batches (SQLite + WAL)."""

import json
import sqlite3
import threading
from datetime import datetime

from app.config import DB_PATH

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA busy_timeout=5000")
    return _conn


def init_db():
    conn = _get_conn()
    with _lock:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS batches (
                id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'uploading',
                created_at TEXT,
                custom_prompt TEXT DEFAULT '',
                template_id TEXT DEFAULT '',
                base_url TEXT DEFAULT '',
                total_pages INTEGER DEFAULT 0,
                pages_done INTEGER DEFAULT 0,
                page_time_ms_total INTEGER DEFAULT 0,
                page_time_samples INTEGER DEFAULT 0,
                error TEXT,
                excel_filename TEXT
            );
            CREATE TABLE IF NOT EXISTS batch_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT REFERENCES batches(id),
                filename TEXT,
                status TEXT DEFAULT 'pending',
                pages INTEGER DEFAULT 0,
                pages_done INTEGER DEFAULT 0,
                page_time_ms_total INTEGER DEFAULT 0,
                page_time_samples INTEGER DEFAULT 0,
                error TEXT,
                started_at TEXT,
                finished_at TEXT,
                detected_template_id TEXT DEFAULT '',
                UNIQUE(batch_id, filename)
            );
            CREATE TABLE IF NOT EXISTS page_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT REFERENCES batches(id),
                filename TEXT,
                page INTEGER,
                result_json TEXT,
                duration_ms INTEGER,
                UNIQUE(batch_id, filename, page)
            );
            CREATE INDEX IF NOT EXISTS idx_bf_batch ON batch_files(batch_id);
            CREATE INDEX IF NOT EXISTS idx_pr_batch ON page_results(batch_id);
            CREATE INDEX IF NOT EXISTS idx_pr_file ON page_results(batch_id, filename);
        """)
        # Migration: add timing columns
        for col in (
            "started_at TEXT",
            "finished_at TEXT",
            "page_time_ms_total INTEGER DEFAULT 0",
            "page_time_samples INTEGER DEFAULT 0",
        ):
            try:
                conn.execute(f"ALTER TABLE batches ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass  # column already exists
        for col in (
            "started_at TEXT",
            "finished_at TEXT",
            "page_time_ms_total INTEGER DEFAULT 0",
            "page_time_samples INTEGER DEFAULT 0",
        ):
            try:
                conn.execute(f"ALTER TABLE batch_files ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass  # column already exists
        try:
            conn.execute("ALTER TABLE batch_files ADD COLUMN detected_template_id TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists
        try:
            conn.execute("ALTER TABLE page_results ADD COLUMN duration_ms INTEGER")
        except sqlite3.OperationalError:
            pass  # column already exists
        # Migration: upload_complete flag for streaming processing
        try:
            conn.execute("ALTER TABLE batches ADD COLUMN upload_complete INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # column already exists
        # Migration: allowed_templates for auto-mode filtering
        try:
            conn.execute("ALTER TABLE batches ADD COLUMN allowed_templates TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        # Migration: page_range for page selection
        try:
            conn.execute("ALTER TABLE batches ADD COLUMN page_range TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        # Migration: mode for OCR batches
        try:
            conn.execute("ALTER TABLE batches ADD COLUMN mode TEXT DEFAULT 'extract'")
        except sqlite3.OperationalError:
            pass
        # Migration: ocr_type for OCR mode selection
        try:
            conn.execute("ALTER TABLE batches ADD COLUMN ocr_type TEXT DEFAULT 'universal'")
        except sqlite3.OperationalError:
            pass
        # Table: ocr_tags
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ocr_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT REFERENCES batches(id),
                filename TEXT,
                tags_json TEXT,
                UNIQUE(batch_id, filename)
            )
        """)
        conn.commit()


def next_batch_id() -> str:
    """Generate sequential batch ID like '1', '2', '3', ..."""
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT MAX(CAST(id AS INTEGER)) FROM batches WHERE id GLOB '[0-9]*'").fetchone()
        last = row[0] if row[0] is not None else 0
        return str(last + 1)


def create_batch(batch_id, custom_prompt="", template_id="", base_url="",
                  allowed_templates="", page_range="", mode="extract"):
    conn = _get_conn()
    with _lock:
        conn.execute(
            "INSERT INTO batches (id, status, created_at, custom_prompt, template_id, base_url, "
            "allowed_templates, page_range, mode) "
            "VALUES (?, 'uploading', ?, ?, ?, ?, ?, ?, ?)",
            (batch_id, datetime.now().isoformat(), custom_prompt, template_id, base_url,
             allowed_templates, page_range, mode),
        )
        conn.commit()


def add_file(batch_id, filename):
    conn = _get_conn()
    with _lock:
        conn.execute(
            "INSERT OR IGNORE INTO batch_files (batch_id, filename, status) VALUES (?, ?, 'uploaded')",
            (batch_id, filename),
        )
        conn.commit()


def get_batch(batch_id):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    if not row:
        return None
    batch = dict(row)
    files = conn.execute(
        "SELECT filename, status, pages, pages_done, "
        "page_time_ms_total, page_time_samples, "
        "error, started_at, finished_at, detected_template_id "
        "FROM batch_files WHERE batch_id = ? ORDER BY id",
        (batch_id,),
    ).fetchall()
    batch["files"] = [dict(f) for f in files]
    return batch


def list_batches():
    conn = _get_conn()
    rows = conn.execute(
        "SELECT b.id, b.status, b.created_at, b.total_pages, b.pages_done, b.template_id, "
        "b.custom_prompt, b.allowed_templates, b.page_range, "
        "b.started_at, b.finished_at, "
        "(SELECT COUNT(*) FROM batch_files WHERE batch_id = b.id) as file_count "
        "FROM batches b ORDER BY b.created_at DESC LIMIT 50"
    ).fetchall()
    return [dict(r) for r in rows]


def update_batch(batch_id, **kwargs):
    if kwargs.get("status") == "interrupted":
        import traceback
        print(f"[INTERRUPTED] batch={batch_id} — caller trace:", flush=True)
        traceback.print_stack()
    conn = _get_conn()
    with _lock:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [batch_id]
        conn.execute(f"UPDATE batches SET {sets} WHERE id = ?", values)
        conn.commit()


def update_file(batch_id, filename, **kwargs):
    conn = _get_conn()
    with _lock:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [batch_id, filename]
        conn.execute(f"UPDATE batch_files SET {sets} WHERE batch_id = ? AND filename = ?", values)
        conn.commit()


def set_file_pages(batch_id, filename, pages):
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE batch_files SET pages = ? WHERE batch_id = ? AND filename = ?",
            (pages, batch_id, filename),
        )
        conn.commit()


def mark_file_started(batch_id, filename):
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE batch_files "
            "SET started_at = COALESCE(started_at, ?), finished_at = NULL "
            "WHERE batch_id = ? AND filename = ?",
            (datetime.now().isoformat(), batch_id, filename),
        )
        conn.commit()


def mark_file_finished(batch_id, filename):
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE batch_files SET finished_at = COALESCE(finished_at, ?) "
            "WHERE batch_id = ? AND filename = ?",
            (datetime.now().isoformat(), batch_id, filename),
        )
        conn.commit()


def get_file_duration_ms(batch_id: str, filename: str) -> int | None:
    """Get VLM processing duration for a file from page_results (page=0 for extraction mode)."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT duration_ms FROM page_results WHERE batch_id=? AND filename=? AND page=0",
        (batch_id, filename),
    ).fetchone()
    return row["duration_ms"] if row else None


def mark_processing_files_interrupted(batch_id):
    """Freeze timers for files that were active when batch was interrupted."""
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE batch_files "
            "SET status = 'interrupted', finished_at = COALESCE(finished_at, ?) "
            "WHERE batch_id = ? AND status IN ('converting', 'queued', 'processing')",
            (datetime.now().isoformat(), batch_id),
        )
        conn.commit()


def add_total_pages(batch_id, count):
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE batches SET total_pages = total_pages + ? WHERE id = ?",
            (count, batch_id),
        )
        conn.commit()


def _normalize_page_duration_ms(page_duration_ms: int | None) -> int | None:
    if page_duration_ms is None:
        return None
    try:
        value = int(page_duration_ms)
    except (TypeError, ValueError):
        return None
    return max(0, value)


def increment_pages_done(batch_id, filename, page_duration_ms: int | None = None):
    conn = _get_conn()
    duration = _normalize_page_duration_ms(page_duration_ms)
    with _lock:
        if duration is None:
            conn.execute(
                "UPDATE batch_files SET pages_done = pages_done + 1 WHERE batch_id = ? AND filename = ?",
                (batch_id, filename),
            )
            conn.execute(
                "UPDATE batches SET pages_done = pages_done + 1 WHERE id = ?",
                (batch_id,),
            )
        else:
            conn.execute(
                "UPDATE batch_files "
                "SET pages_done = pages_done + 1, "
                "page_time_ms_total = COALESCE(page_time_ms_total, 0) + ?, "
                "page_time_samples = COALESCE(page_time_samples, 0) + 1 "
                "WHERE batch_id = ? AND filename = ?",
                (duration, batch_id, filename),
            )
            conn.execute(
                "UPDATE batches "
                "SET pages_done = pages_done + 1, "
                "page_time_ms_total = COALESCE(page_time_ms_total, 0) + ?, "
                "page_time_samples = COALESCE(page_time_samples, 0) + 1 "
                "WHERE id = ?",
                (duration, batch_id),
            )
        conn.commit()


def save_page_result(batch_id, filename, page, result, page_duration_ms: int | None = None):
    conn = _get_conn()
    duration = _normalize_page_duration_ms(page_duration_ms)
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO page_results (batch_id, filename, page, result_json, duration_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (batch_id, filename, page, json.dumps(result, ensure_ascii=False), duration),
        )
        conn.commit()


def get_new_page_results(batch_id: str, after_id: int = 0) -> list[dict]:
    """Get page results added after the given id (for SSE streaming)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, filename, page, result_json, duration_ms "
        "FROM page_results WHERE batch_id = ? AND id > ? ORDER BY id",
        (batch_id, after_id),
    ).fetchall()
    return [dict(r) for r in rows]


def get_page_results(batch_id, filenames: list[str] | None = None):
    conn = _get_conn()
    if filenames:
        placeholders = ", ".join("?" for _ in filenames)
        rows = conn.execute(
            "SELECT filename, page, result_json FROM page_results "
            f"WHERE batch_id = ? AND filename IN ({placeholders}) ORDER BY filename, page",
            (batch_id, *filenames),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT filename, page, result_json FROM page_results WHERE batch_id = ? ORDER BY filename, page",
            (batch_id,),
        ).fetchall()
    results = []
    for r in rows:
        data = json.loads(r["result_json"])
        data["filename"] = r["filename"]
        data["page"] = r["page"]
        results.append(data)
    return results


def get_processed_pages(batch_id, filename):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT page FROM page_results WHERE batch_id = ? AND filename = ?",
        (batch_id, filename),
    ).fetchall()
    return {r["page"] for r in rows}


def sync_counters(batch_id):
    """Recalculate counters from actual DB state. Used on resume."""
    conn = _get_conn()
    with _lock:
        row = conn.execute(
            "SELECT COALESCE(SUM(pages), 0) as tp FROM batch_files WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        total_pages = row["tp"]

        row = conn.execute(
            "SELECT COUNT(*) as pd, "
            "COALESCE(SUM(duration_ms), 0) as ms_total, "
            "SUM(CASE WHEN duration_ms IS NOT NULL THEN 1 ELSE 0 END) as ms_samples "
            "FROM page_results WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        pages_done = row["pd"]
        page_time_ms_total = row["ms_total"]
        page_time_samples = row["ms_samples"] or 0

        conn.execute(
            "UPDATE batches SET total_pages = ?, pages_done = ?, "
            "page_time_ms_total = ?, page_time_samples = ? "
            "WHERE id = ?",
            (total_pages, pages_done, page_time_ms_total, page_time_samples, batch_id),
        )

        # Sync per-file counters too
        files = conn.execute(
            "SELECT filename, pages FROM batch_files WHERE batch_id = ?",
            (batch_id,),
        ).fetchall()
        for f in files:
            row = conn.execute(
                "SELECT COUNT(*) as pd, "
                "COALESCE(SUM(duration_ms), 0) as ms_total, "
                "SUM(CASE WHEN duration_ms IS NOT NULL THEN 1 ELSE 0 END) as ms_samples "
                "FROM page_results WHERE batch_id = ? AND filename = ?",
                (batch_id, f["filename"]),
            ).fetchone()
            conn.execute(
                "UPDATE batch_files SET pages_done = ?, page_time_ms_total = ?, page_time_samples = ? "
                "WHERE batch_id = ? AND filename = ?",
                (row["pd"], row["ms_total"], row["ms_samples"] or 0, batch_id, f["filename"]),
            )
            # Mark file as done if all pages processed
            if f["pages"] > 0 and row["pd"] >= f["pages"]:
                conn.execute(
                    "UPDATE batch_files SET status = 'done', finished_at = COALESCE(finished_at, ?) "
                    "WHERE batch_id = ? AND filename = ?",
                    (datetime.now().isoformat(), batch_id, f["filename"]),
                )

        conn.commit()


def delete_batch(batch_id):
    """Delete batch and all related records from DB. Returns list of filenames for cleanup."""
    conn = _get_conn()
    with _lock:
        files = conn.execute(
            "SELECT filename FROM batch_files WHERE batch_id = ?", (batch_id,)
        ).fetchall()
        filenames = [f["filename"] for f in files]
        row = conn.execute(
            "SELECT excel_filename FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
        excel = row["excel_filename"] if row and row["excel_filename"] else None
        conn.execute("DELETE FROM page_results WHERE batch_id = ?", (batch_id,))
        conn.execute("DELETE FROM batch_files WHERE batch_id = ?", (batch_id,))
        conn.execute("DELETE FROM batches WHERE id = ?", (batch_id,))
        conn.commit()
    return filenames, excel


def mark_upload_complete(batch_id):
    """Signal that all files have been uploaded for this batch."""
    conn = _get_conn()
    with _lock:
        conn.execute("UPDATE batches SET upload_complete = 1 WHERE id = ?", (batch_id,))
        conn.commit()


def is_upload_complete(batch_id) -> bool:
    """Check if all uploads for this batch are done."""
    conn = _get_conn()
    row = conn.execute("SELECT upload_complete FROM batches WHERE id = ?", (batch_id,)).fetchone()
    if not row:
        return True
    return bool(row["upload_complete"])


def get_unfinished_files(batch_id) -> list[dict]:
    """Get files that haven't been processed yet (status != done and status != error)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT filename, status, pages, pages_done, "
        "page_time_ms_total, page_time_samples, "
        "error, started_at, finished_at "
        "FROM batch_files WHERE batch_id = ? AND status NOT IN ('done', 'error') ORDER BY id",
        (batch_id,),
    ).fetchall()
    return [dict(r) for r in rows]



def mark_document_done(batch_id, filename, num_pages, result, duration_ms: int | None = None):
    """Save single result for entire document (multi-image mode) and update counters."""
    conn = _get_conn()
    duration = _normalize_page_duration_ms(duration_ms)
    with _lock:
        # Save result with page=0 meaning whole document
        conn.execute(
            "INSERT OR REPLACE INTO page_results (batch_id, filename, page, result_json, duration_ms) "
            "VALUES (?, ?, 0, ?, ?)",
            (batch_id, filename, json.dumps(result, ensure_ascii=False), duration),
        )
        # Update batch_files
        conn.execute(
            "UPDATE batch_files SET pages_done = ?, status = 'done', "
            "page_time_ms_total = COALESCE(page_time_ms_total, 0) + COALESCE(?, 0), "
            "page_time_samples = COALESCE(page_time_samples, 0) + 1, "
            "finished_at = COALESCE(finished_at, ?) "
            "WHERE batch_id = ? AND filename = ?",
            (num_pages, duration, datetime.now().isoformat(), batch_id, filename),
        )
        # Update batches
        if duration is not None:
            conn.execute(
                "UPDATE batches SET pages_done = pages_done + ?, "
                "page_time_ms_total = COALESCE(page_time_ms_total, 0) + ?, "
                "page_time_samples = COALESCE(page_time_samples, 0) + 1 "
                "WHERE id = ?",
                (num_pages, duration, batch_id),
            )
        else:
            conn.execute(
                "UPDATE batches SET pages_done = pages_done + ? WHERE id = ?",
                (num_pages, batch_id),
            )
        conn.commit()

def mark_interrupted_batches():
    """On startup: fix batches stuck in 'processing' status.

    If all files are done -> mark batch as 'done'.
    If started less than 10 minutes ago -> skip (may still be running).
    Otherwise -> mark as 'interrupted' (can be resumed).
    """
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT id, started_at FROM batches WHERE status = 'processing'"
        ).fetchall()
        if not rows:
            return
        now = datetime.now()
        for row in rows:
            bid = row["id"]
            started_at = row["started_at"]

            # Don't touch batches started less than 10 minutes ago —
            # the processing task may still be alive in another process
            if started_at:
                try:
                    t = datetime.fromisoformat(started_at)
                    if (now - t).total_seconds() < 600:
                        print(f"[STARTUP] batch={bid} started {started_at} — too recent (<10min), skipping", flush=True)
                        continue
                except Exception:
                    pass

            total = conn.execute(
                "SELECT COUNT(*) FROM batch_files WHERE batch_id = ?", (bid,)
            ).fetchone()[0]
            done = conn.execute(
                "SELECT COUNT(*) FROM batch_files WHERE batch_id = ? AND status = 'done'",
                (bid,),
            ).fetchone()[0]
            errored = conn.execute(
                "SELECT COUNT(*) FROM batch_files WHERE batch_id = ? AND status = 'error'",
                (bid,),
            ).fetchone()[0]
            if total > 0 and done + errored >= total:
                conn.execute(
                    "UPDATE batches SET status = 'done', "
                    "finished_at = COALESCE(finished_at, ?) WHERE id = ?",
                    (datetime.now().isoformat(), bid),
                )
                print(f"[STARTUP] batch={bid} all files done ({done}/{total}) -> status=done", flush=True)
            else:
                conn.execute(
                    "UPDATE batches SET status = 'interrupted' WHERE id = ?",
                    (bid,),
                )
                conn.execute(
                    "UPDATE batch_files "
                    "SET status = 'interrupted', finished_at = COALESCE(finished_at, ?) "
                    "WHERE batch_id = ? AND status IN ('converting', 'queued', 'processing')",
                    (datetime.now().isoformat(), bid),
                )
                print(
                    f"[STARTUP] batch={bid} files={done}done/{errored}err/{total}total -> status=interrupted",
                    flush=True,
                )
        conn.commit()


# ── OCR Tags ───────────────────────────────────────────────────────

def save_tags(batch_id: str, filename: str, tags: dict):
    conn = _get_conn()
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO ocr_tags (batch_id, filename, tags_json) VALUES (?, ?, ?)",
            (batch_id, filename, json.dumps(tags, ensure_ascii=False)),
        )
        conn.commit()


def get_tags(batch_id: str, filename: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT tags_json FROM ocr_tags WHERE batch_id = ? AND filename = ?",
        (batch_id, filename),
    ).fetchone()
    if not row:
        return None
    return json.loads(row["tags_json"])


def update_tags(batch_id: str, filename: str, tags: dict):
    save_tags(batch_id, filename, tags)


def update_page_ocr_text(batch_id: str, filename: str, page: int, new_text: str):
    """Update OCR text for a specific page (used for tag auto-replace)."""
    conn = _get_conn()
    with _lock:
        row = conn.execute(
            "SELECT result_json FROM page_results WHERE batch_id = ? AND filename = ? AND page = ?",
            (batch_id, filename, page),
        ).fetchone()
        if not row:
            return
        data = json.loads(row["result_json"])
        data["ocr_text"] = new_text
        # Update lines text too
        lines = data.get("lines", [])
        # Lines stay as-is (bbox coords), only the ocr_text field changes
        conn.execute(
            "UPDATE page_results SET result_json = ? WHERE batch_id = ? AND filename = ? AND page = ?",
            (json.dumps(data, ensure_ascii=False), batch_id, filename, page),
        )
        conn.commit()
