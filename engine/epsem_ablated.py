"""
Ablated D: EpSemMemory-Ablated — EP-SEM Without Tagging.

Inherits PersistentMemoryBase. Separates episodic (in-memory) and
semantic (SQLite) buffers with periodic compaction every N turns,
but WITHOUT structural tagging or conflict rejection.

Purpose: Ablation study — isolate the pure effect size of
selective tagging and knowledge refinement in EP-SEM.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import List

from .event_emitter import EventEmitter
from .base import PersistentMemoryBase, Message, WriteResult, ReadResult, load_config


class EpSemAblated(PersistentMemoryBase):
    """
    EP-SEM architecture without tagging/conflict detection.
    - Episodic buffer: in-memory list
    - Semantic store: SQLite (all entries stored without classification)
    - Compaction: every N turns, flush episodic → semantic (no filtering)
    """

    def __init__(self, db_path: str | None = None):
        super().__init__()
        self.event_emitter: EventEmitter | None = None
        config = load_config()
        self._compaction_interval: int = config.get("compaction_interval", 10)

        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "epsem_ablated.db")
        self._db_path = db_path
        self._conn = sqlite3.connect(self._db_path)
        self._setup_tables()

        self._episodic: List[Message] = []
        self._turn_count: int = 0

    def _setup_tables(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn INTEGER,
                role TEXT,
                content TEXT,
                is_constraint INTEGER DEFAULT 0,
                is_distractor INTEGER DEFAULT 0,
                created_at REAL
            )
        """)
        self._conn.commit()

    def write(self, message: Message) -> WriteResult:
        t0 = time.perf_counter()
        self._episodic.append(message)
        self._turn_count += 1

        tokens = len(message.content) // 3
        self._total_tokens += tokens

        compaction_latency = 0.0
        # Compact every N turns (blind flush — no tagging)
        if self._turn_count % self._compaction_interval == 0:
            t_compact = time.perf_counter()
            self._compact()
            compaction_latency = time.perf_counter() - t_compact

        elapsed = time.perf_counter() - t0
        return WriteResult(
            write_latency=elapsed,
            compaction_latency=compaction_latency,
            tokens_spent=tokens,
        )

    def read(self, query: str, max_tokens: int = 4096) -> ReadResult:
        t0 = time.perf_counter()
        parts: List[str] = []

        # 1. Semantic store (persisted entries)
        cur = self._conn.cursor()
        cur.execute("SELECT turn, role, content FROM semantic_store ORDER BY turn")
        for turn, role, content in cur.fetchall():
            parts.append(f"[{role}] (turn {turn}): {content}")

        # 2. Episodic buffer (not yet compacted)
        for msg in self._episodic:
            parts.append(f"[{msg.role}] (turn {msg.turn}): {msg.content}")

        context = "\n".join(parts)
        # Truncate if needed
        if len(context) // 3 > max_tokens:
            char_limit = max_tokens * 3
            context = context[-char_limit:]

        elapsed = time.perf_counter() - t0
        return ReadResult(context_text=context, read_latency=elapsed)

    def get_token_count(self) -> int:
        return self._total_tokens

    def reset(self) -> None:
        super().reset()
        self._episodic.clear()
        self._turn_count = 0
        cur = self._conn.cursor()
        cur.execute("DELETE FROM semantic_store")
        self._conn.commit()

    def test_cross_session_persistence(self) -> bool:
        """Clear episodic buffer, verify semantic store has content."""
        self._episodic.clear()
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM semantic_store")
        count = cur.fetchone()[0]
        if count == 0:
            return False
        # Verify we can reconstruct context from semantic store alone
        result = self.read("verify persistence")
        return len(result.context_text) > 0

    def _compact(self) -> None:
        """Flush all episodic entries to semantic store (no classification).
        NOTE: is_constraint/is_distractor are intentionally zeroed out.
        Ablated model must NOT have access to ground truth labels —
        this preserves ablation purity.
        """
        cur = self._conn.cursor()
        for msg in self._episodic:
            cur.execute(
                "INSERT INTO semantic_store (turn, role, content, is_constraint, is_distractor, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (msg.turn, msg.role, msg.content, 0, 0, time.time()),
            )
        self._conn.commit()
        self._episodic.clear()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
