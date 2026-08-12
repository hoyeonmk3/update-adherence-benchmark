"""
EpSemFiltered — EpSemMemory variant that returns CONSTRAINT-only from semantic store.
Episodic buffer is still included (to avoid compaction gap).
Purpose: Measure effect of EPHEMERAL filtering on Precision/F1.
"""
from __future__ import annotations
from .epsem_memory import EpSemMemory
from .base import ReadResult
import time
from typing import List


class EpSemFiltered(EpSemMemory):
    """Same write behavior as EpSemMemory, but read() skips EPHEMERAL entries."""

    def read(self, query: str, max_tokens: int = 4096) -> ReadResult:
        t0 = time.perf_counter()
        parts: List[str] = []

        # 1. CONSTRAINT only from semantic store (skip EPHEMERAL)
        cur = self._conn.cursor()
        cur.execute(
            "SELECT turn, role, content, tag FROM semantic_store "
            "WHERE tag = 'CONSTRAINT' ORDER BY turn"
        )
        for turn, role, content, tag in cur.fetchall():
            parts.append(f"[{tag}] [{role}] (turn {turn}): {content}")

        # 2. Episodic buffer (not yet compacted — include to avoid gap)
        for msg in self._episodic:
            parts.append(f"[{msg.role}] (turn {msg.turn}): {msg.content}")

        context = "\n".join(parts)
        # Truncate if needed
        if len(context) // 3 > max_tokens:
            char_limit = max_tokens * 3
            context = context[-char_limit:]

        elapsed = time.perf_counter() - t0
        return ReadResult(context_text=context, read_latency=elapsed)
