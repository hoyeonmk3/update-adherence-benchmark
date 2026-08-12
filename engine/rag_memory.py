"""
Control C: RagMemory — Hybrid Retrieval (BM25 + Dense + RRF).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests

from .base import MemoryBase, Message, WriteResult, ReadResult, load_config


class RagMemory(MemoryBase):
    """
    Hybrid RAG memory combining:
    1. SQLite FTS5 for BM25 keyword search
    2. nomic-embed-text via Ollama for dense vector search
    3. Reciprocal Rank Fusion (RRF) to merge results

    Purpose: Measure hybrid RAG's ability to handle multi-hop reasoning
    and detect false-positive retrieval from semantically similar but
    contextually different chunks.
    """

    def __init__(self, db_path: str | None = None):
        super().__init__()
        config = load_config()
        self._ollama_url: str = config.get("ollama_endpoint", "http://localhost:11434")
        self._embed_model: str = config.get("embed_model", "nomic-embed-text")
        self._top_k: int = config.get("rag_top_k", 5)
        self._rrf_k: int = 60  # RRF constant

        # SQLite setup — in-memory for benchmark isolation, or file for persistence
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "rag_memory.db")
        self._db_path = db_path
        self._conn = sqlite3.connect(self._db_path)
        self._setup_tables()

    def _setup_tables(self) -> None:
        """Create FTS5 table and embedding cache table."""
        cur = self._conn.cursor()
        # Chunks table (source of truth)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn INTEGER,
                role TEXT,
                content TEXT,
                is_constraint INTEGER DEFAULT 0,
                is_distractor INTEGER DEFAULT 0
            )
        """)
        # FTS5 virtual table for BM25
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(content, content_rowid='id')
        """)
        # Embedding cache table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)
        self._conn.commit()

    def write(self, message: Message) -> WriteResult:
        t0 = time.perf_counter()
        cur = self._conn.cursor()

        # Insert into chunks table
        cur.execute(
            "INSERT INTO chunks (turn, role, content, is_constraint, is_distractor) VALUES (?, ?, ?, ?, ?)",
            (message.turn, message.role, message.content,
             int(message.is_constraint), int(message.is_distractor)),
        )
        chunk_id = cur.lastrowid

        # Insert into FTS5 index
        cur.execute(
            "INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)",
            (chunk_id, message.content),
        )
        self._conn.commit()

        # Pre-compute and cache embedding
        self._get_embedding(message.content)

        tokens = len(message.content) // 3
        self._total_tokens += tokens
        elapsed = time.perf_counter() - t0
        return WriteResult(write_latency=elapsed, tokens_spent=tokens)

    def read(self, query: str, max_tokens: int = 4096) -> ReadResult:
        t0 = time.perf_counter()

        # 1. BM25 search via FTS5
        bm25_results = self._bm25_search(query)

        # 2. Dense vector search
        dense_results = self._dense_search(query)

        # 3. Reciprocal Rank Fusion
        fused = self._rrf_merge(bm25_results, dense_results)

        # 4. Build context from top-k chunks
        top_ids = [chunk_id for chunk_id, _ in fused[:self._top_k]]
        if top_ids:
            placeholders = ",".join("?" * len(top_ids))
            cur = self._conn.cursor()
            cur.execute(
                f"SELECT id, turn, role, content FROM chunks WHERE id IN ({placeholders}) ORDER BY turn",
                top_ids,
            )
            rows = cur.fetchall()
            parts = [f"[{role}] (turn {turn}): {content}" for _, turn, role, content in rows]
            context = "\n".join(parts)
        else:
            context = ""

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
        cur = self._conn.cursor()
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM chunks_fts")
        # NOTE: embedding_cache is intentionally preserved across resets.
        # Query embeddings and repeated chunk embeddings are reused across
        # scenarios and repeat runs, stabilizing latency measurements.
        self._conn.commit()

    # -----------------------------------------------------------------------
    # BM25 Search
    # -----------------------------------------------------------------------

    @staticmethod
    def _to_fts_query(text: str) -> str | None:
        """Convert natural language to FTS5-safe OR query."""
        import re as _re
        tokens = _re.findall(r'\w+', text.lower())
        tokens = [t for t in tokens if len(t) >= 2]
        if not tokens:
            return None
        return " OR ".join(f'"{t}"' for t in tokens)

    def _bm25_search(self, query: str, limit: int = 20) -> List[Tuple[int, float]]:
        """Search using SQLite FTS5 BM25 ranking."""
        cur = self._conn.cursor()
        fts_query = self._to_fts_query(query)
        if fts_query is None:
            return []
        try:
            cur.execute(
                """
                SELECT rowid, rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            )
            # FTS5 rank is negative (lower = better match)
            return [(row[0], -row[1]) for row in cur.fetchall()]
        except Exception:
            return []

    # -----------------------------------------------------------------------
    # Dense Vector Search
    # -----------------------------------------------------------------------
    def _dense_search(self, query: str, limit: int = 20) -> List[Tuple[int, float]]:
        """Search using cosine similarity of embeddings."""
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return []

        cur = self._conn.cursor()
        cur.execute("SELECT id, content FROM chunks")
        rows = cur.fetchall()

        scored = []
        for chunk_id, content in rows:
            chunk_emb = self._get_embedding(content)
            if chunk_emb is not None:
                sim = self._cosine_sim(query_emb, chunk_emb)
                scored.append((chunk_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # -----------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # -----------------------------------------------------------------------
    def _rrf_merge(
        self,
        bm25_results: List[Tuple[int, float]],
        dense_results: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        """Merge BM25 and dense results using RRF."""
        scores: dict[int, float] = {}
        for rank, (chunk_id, _) in enumerate(bm25_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (self._rrf_k + rank + 1)
        for rank, (chunk_id, _) in enumerate(dense_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (self._rrf_k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # -----------------------------------------------------------------------
    # Embedding Helpers
    # -----------------------------------------------------------------------
    def _get_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding from cache or compute via Ollama."""
        content_hash = hashlib.md5(text.encode()).hexdigest()

        # Check cache
        cur = self._conn.cursor()
        cur.execute("SELECT embedding FROM embedding_cache WHERE content_hash = ?", (content_hash,))
        row = cur.fetchone()
        if row:
            return np.frombuffer(row[0], dtype=np.float32)

        # Compute via Ollama
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

            # Cache
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
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    def __del__(self):
        """Close DB connection on cleanup."""
        try:
            self._conn.close()
        except Exception:
            pass
