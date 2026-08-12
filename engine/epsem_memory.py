"""
Proposed E: EpSemMemory-Full — Complete EP-SEM Architecture.

Full implementation with:
- Structural tagging ([CONSTRAINT] / [EPHEMERAL] / [REJECTED])
- Synchronous block compaction every N turns
- LLM-based classification via Ollama
- Conflict rejection (semantic contradiction detection)
- Cross-session persistence via SQLite
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests

from .event_emitter import EventEmitter
from .base import PersistentMemoryBase, Message, WriteResult, ReadResult, load_config


class EpSemMemory(PersistentMemoryBase):
    """
    EP-SEM Full: Episodic-Semantic Memory with structural tagging,
    conflict rejection, and persistent semantic store.
    """

    def __init__(self, db_path: str | None = None):
        super().__init__()
        self.event_emitter: EventEmitter | None = None
        config = load_config()
        self._compaction_interval: int = config.get("compaction_interval", 10)
        self._ollama_url: str = config.get("ollama_endpoint", "http://localhost:11434")
        self._model: str = config.get("ollama_model", "qwen2.5:14b-instruct-q4_K_M")
        self._embed_model: str = config.get("embed_model", "nomic-embed-text")
        self._conflict_threshold: float = 0.85  # cosine similarity threshold for conflict check

        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "epsem_full.db")
        self._db_path = db_path
        self._conn = sqlite3.connect(self._db_path)
        self._setup_tables()

        self._episodic: List[Message] = []
        self._turn_count: int = 0

    def _setup_tables(self) -> None:
        cur = self._conn.cursor()
        # Semantic store — classified entries
        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn INTEGER,
                role TEXT,
                content TEXT,
                tag TEXT,
                is_constraint INTEGER DEFAULT 0,
                is_distractor INTEGER DEFAULT 0,
                created_at REAL,
                ttl_turns INTEGER DEFAULT -1
            )
        """)
        # Embedding cache (shared with RagMemory pattern)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)
        # Rejection log
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rejection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn INTEGER,
                content TEXT,
                conflicting_constraint_id INTEGER,
                reason TEXT,
                created_at REAL
            )
        """)
        self._conn.commit()


    def _emit_event(self, turn: int, event_type: str, data: dict | None = None):
        """Emit event if emitter is attached."""
        if self.event_emitter:
            self.event_emitter.emit(turn, "EpSemMemory", event_type, data)

    def write(self, message: Message) -> WriteResult:
        t0 = time.perf_counter()
        self._episodic.append(message)
        self._turn_count += 1

        tokens = len(message.content) // 3
        self._total_tokens += tokens

        compaction_latency = 0.0
        # Synchronous block compaction every N turns
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
        char_limit = max_tokens * 3
        cur = self._conn.cursor()

        # --- Priority 1: CONSTRAINT (never truncated) ---
        cur.execute(
            "SELECT turn, role, content, tag FROM semantic_store "
            "WHERE tag = 'CONSTRAINT' ORDER BY turn"
        )
        constraint_parts: List[str] = []
        for turn, role, content, tag in cur.fetchall():
            constraint_parts.append(f"[{tag}] [{role}] (turn {turn}): {content}")

        constraint_block = "\n".join(constraint_parts)
        remaining_chars = char_limit - len(constraint_block)

        # --- Priority 2: Episodic buffer (recent, not yet compacted) ---
        episodic_parts: List[str] = []
        for msg in self._episodic:
            episodic_parts.append(f"[{msg.role}] (turn {msg.turn}): {msg.content}")

        # --- Priority 3: Non-expired EPHEMERAL (lowest priority, fills remaining space) ---
        cur.execute(
            "SELECT turn, role, content, tag, ttl_turns FROM semantic_store "
            "WHERE tag = 'EPHEMERAL' ORDER BY turn DESC"
        )
        ephemeral_parts: List[str] = []
        for turn, role, content, tag, ttl in cur.fetchall():
            if ttl > 0 and (self._turn_count - turn) > ttl:
                continue  # TTL expired
            ephemeral_parts.append(f"[{tag}] [{role}] (turn {turn}): {content}")

        # Assemble: constraints first, then episodic, then ephemeral within budget
        secondary_block = "\n".join(episodic_parts + ephemeral_parts)
        if len(secondary_block) > remaining_chars and remaining_chars > 0:
            secondary_block = secondary_block[:remaining_chars]

        if constraint_block and secondary_block:
            context = constraint_block + "\n" + secondary_block
        else:
            context = constraint_block or secondary_block

        elapsed = time.perf_counter() - t0
        return ReadResult(context_text=context, read_latency=elapsed)

    def get_active_constraints(self):
        """All CONSTRAINT rows (EpSem has no archiving -> every constraint is active)."""
        cur = self._conn.cursor()
        cur.execute("SELECT content FROM semantic_store WHERE tag = 'CONSTRAINT'")
        return [r[0] for r in cur.fetchall()]

    def get_token_count(self) -> int:
        return self._total_tokens

    def reset(self) -> None:
        super().reset()
        self._episodic.clear()
        self._turn_count = 0
        cur = self._conn.cursor()
        cur.execute("DELETE FROM semantic_store")
        cur.execute("DELETE FROM rejection_log")
        # Preserve embedding_cache
        self._conn.commit()

    def test_cross_session_persistence(self) -> bool:
        """Clear episodic buffer, verify constraints survive in semantic store."""
        self._episodic.clear()
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM semantic_store WHERE tag = 'CONSTRAINT'")
        count = cur.fetchone()[0]
        if count == 0:
            return False
        result = self.read("verify persistence")
        return "[CONSTRAINT]" in result.context_text

    # -------------------------------------------------------------------
    # Compaction Process (§3-2 Control E)
    # -------------------------------------------------------------------
    def _compact(self) -> None:
        """
        Synchronous block compaction:
        Phase A: Classify each episodic entry via LLM
        Phase B: Check conflicts for CONSTRAINT candidates
        Phase C: Store classified entries in semantic store
        Phase D: Clear episodic buffer

        UNCERTAIN handling:
        UNCERTAIN entries are demoted to EPHEMERAL with TTL.
        This prevents classification retry loops and ensures
        deterministic behavior in repeat runs.
        """
        self._emit_event(self._turn_count, "compression", {"buffer_size": len(self._episodic)})
        cur = self._conn.cursor()
        for msg in self._episodic:
            # Phase A: Classify
            tag = self._classify(msg)
            self._emit_event(msg.turn, "classification", {"label": tag, "content": msg.content[:100]})

            # UNCERTAIN → demote to EPHEMERAL with shorter TTL
            if tag == "UNCERTAIN":
                tag = "EPHEMERAL"

            # Phase B: Conflict check for constraints
            if tag == "CONSTRAINT":
                conflict = self._check_conflict(msg)
                if conflict is not None:
                    cur.execute(
                        "INSERT INTO rejection_log (turn, content, conflicting_constraint_id, reason, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (msg.turn, msg.content, conflict["id"], conflict["reason"], time.time()),
                    )
                    tag = "REJECTED"
                    self._emit_event(msg.turn, "conflict_detected", {"content": msg.content[:100], "reason": conflict["reason"]})

            # Phase C: Store (skip REJECTED entries)
            if tag != "REJECTED":
                ttl = 20 if tag == "EPHEMERAL" else -1
                cur.execute(
                    "INSERT INTO semantic_store "
                    "(turn, role, content, tag, is_constraint, is_distractor, created_at, ttl_turns) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (msg.turn, msg.role, msg.content, tag,
                     int(msg.is_constraint), int(msg.is_distractor), time.time(), ttl),
                )
                self._total_tokens += 5

        self._conn.commit()
        # Phase D: Clear episodic
        self._episodic.clear()

    def _classify(self, message: Message) -> str:
        """Classify a message as CONSTRAINT, EPHEMERAL, or UNCERTAIN via LLM."""
        prompt = (
            "Classify the following message in an engineering conversation.\n"
            "Respond with EXACTLY one word: CONSTRAINT, EPHEMERAL, or UNCERTAIN.\n\n"
            "Rules:\n"
            "- CONSTRAINT: Contains specific engineering requirements, specifications, "
            "numerical limits, material properties, or design rules that MUST be preserved.\n"
            "- EPHEMERAL: Casual conversation, greetings, off-topic discussion, "
            "weather talk, or any non-engineering content.\n"
            "- UNCERTAIN: Engineering-related but not a firm constraint.\n\n"
            f"Message: {message.content}\n\n"
            "Classification:"
        )
        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False, "options": {"temperature": 0, "seed": 42}},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip().upper()

            # Parse — accept first valid tag found
            for tag in ["CONSTRAINT", "EPHEMERAL", "UNCERTAIN"]:
                if tag in result:
                    return tag
            return "UNCERTAIN"
        except Exception:
            # Fallback: default to UNCERTAIN (no ground truth access)
            # Removed ground truth label fallback
            # to prevent self-evaluation circularity
            return "UNCERTAIN"

    def _check_conflict(self, message: Message) -> Optional[dict]:
        """Check if a new constraint conflicts with existing ones."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, content FROM semantic_store WHERE tag = 'CONSTRAINT'"
        )
        existing = cur.fetchall()
        if not existing:
            return None

        new_emb = self._get_embedding(message.content)
        if new_emb is None:
            return None

        for cid, content in existing:
            existing_emb = self._get_embedding(content)
            if existing_emb is None:
                continue
            sim = self._cosine_sim(new_emb, existing_emb)
            # High similarity + different content → potential conflict
            if sim > self._conflict_threshold:
                # Ask LLM to verify contradiction
                is_conflict, reason = self._llm_conflict_check(message.content, content)
                if is_conflict:
                    return {"id": cid, "reason": reason}
        return None

    def _llm_conflict_check(self, new_content: str, existing_content: str) -> tuple[bool, str]:
        """Use LLM to determine if two constraints contradict each other."""
        prompt = (
            "Do the following two engineering constraints CONTRADICT each other?\n"
            "Respond with JSON: {\"contradicts\": true/false, \"reason\": \"...\"}\n\n"
            f"Existing constraint: {existing_content}\n"
            f"New statement: {new_content}\n\n"
            "JSON response:"
        )
        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False, "options": {"temperature": 0, "seed": 42}},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            # Parse JSON from response
            data = json.loads(result)
            return data.get("contradicts", False), data.get("reason", "")
        except Exception:
            return False, ""

    # -------------------------------------------------------------------
    # Embedding Helpers (shared pattern with RagMemory)
    # -------------------------------------------------------------------
    def _get_embedding(self, text: str) -> np.ndarray | None:
        content_hash = hashlib.md5(text.encode()).hexdigest()
        cur = self._conn.cursor()
        cur.execute("SELECT embedding FROM embedding_cache WHERE content_hash = ?", (content_hash,))
        row = cur.fetchone()
        if row:
            return np.frombuffer(row[0], dtype=np.float32)
        try:
            resp = requests.post(
                f"{self._ollama_url}/api/embeddings",
                json={"model": self._embed_model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            emb_list = resp.json().get("embedding", [])
            if not emb_list:
                return None
            emb = np.array(emb_list, dtype=np.float32)
            cur.execute(
                "INSERT OR REPLACE INTO embedding_cache (content_hash, embedding) VALUES (?, ?)",
                (content_hash, emb.tobytes()),
            )
            self._conn.commit()
            return emb
        except Exception:
            return None

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
