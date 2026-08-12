"""
Control B: SummaryMemory — Recursive Summarization.
"""
from __future__ import annotations

import time
import json
from typing import List

import requests

from .base import MemoryBase, Message, WriteResult, ReadResult, load_config


class SummaryMemory(MemoryBase):
    """
    Accumulates messages until token count exceeds threshold (default 4096),
    then recursively summarizes [existing summary + recent N turns] via
    local LLM call.

    Purpose: Track semantic drift as detail (numbers, formulas) gets
    progressively lost through repeated summarization cycles.
    """

    def __init__(self):
        super().__init__()
        config = load_config()
        self._threshold: int = config.get("summary_token_threshold", 4096)
        self._ollama_url: str = config.get("ollama_endpoint", "http://localhost:11434")
        self._model: str = config.get("ollama_model", "qwen2.5:14b-instruct-q4_K_M")
        self._summary: str = ""
        self._recent: List[Message] = []
        self._recent_tokens: int = 0

    def write(self, message: Message) -> WriteResult:
        t0 = time.perf_counter()
        self._recent.append(message)
        msg_tokens = len(message.content) // 3
        self._recent_tokens += msg_tokens
        self._total_tokens += msg_tokens

        compaction_latency = 0.0
        # Trigger summarization when threshold exceeded
        if self._recent_tokens > self._threshold:
            t_compact = time.perf_counter()
            self._summary = self._summarize()
            compaction_latency = time.perf_counter() - t_compact
            self._recent.clear()
            self._recent_tokens = 0

        elapsed = time.perf_counter() - t0
        return WriteResult(
            write_latency=elapsed,
            compaction_latency=compaction_latency,
            tokens_spent=msg_tokens,
        )

    def read(self, query: str, max_tokens: int = 4096) -> ReadResult:
        t0 = time.perf_counter()
        parts = []
        if self._summary:
            parts.append(f"[Summary]\n{self._summary}")
        for msg in self._recent:
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
        self._summary = ""
        self._recent.clear()
        self._recent_tokens = 0

    def _summarize(self) -> str:
        """Call local LLM to produce a recursive summary."""
        recent_text = "\n".join(
            f"[{m.role}] (turn {m.turn}): {m.content}" for m in self._recent
        )
        prompt = (
            "You are a precise summarizer for engineering conversations. "
            "Preserve ALL numerical values, formulas, and constraint specifications exactly. "
            "Combine the existing summary with the new conversation turns into a single "
            "comprehensive summary.\n\n"
            f"=== Existing Summary ===\n{self._summary or '(none)'}\n\n"
            f"=== New Turns ===\n{recent_text}\n\n"
            "=== Updated Summary ==="
        )
        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            summary = data.get("response", "").strip()
            # Count tokens spent on summarization
            self._total_tokens += data.get("eval_count", 0)
            return summary
        except Exception as e:
            # Fallback: keep raw text if LLM fails
            return f"{self._summary}\n{recent_text}"
