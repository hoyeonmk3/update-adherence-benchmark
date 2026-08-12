"""
Control A: NaiveMemory — Full-Context Append with Truncation.
"""
from __future__ import annotations

import time
from typing import List

from .base import MemoryBase, Message, WriteResult, ReadResult


class NaiveMemory(MemoryBase):
    """
    Stores all messages in a flat list. On read(), concatenates in
    reverse-chronological order and truncates at max_tokens.

    Purpose: Measure the exact point where context truncation causes
    constraint loss (Recall → 0%) and track high inference cost.
    """

    def __init__(self):
        super().__init__()
        self._history: List[Message] = []

    def write(self, message: Message) -> WriteResult:
        t0 = time.perf_counter()
        self._history.append(message)
        # Rough token estimate: 1 token ≈ 4 chars (English) / 2 chars (Korean)
        tokens = len(message.content) // 3
        self._total_tokens += tokens
        elapsed = time.perf_counter() - t0
        return WriteResult(write_latency=elapsed, tokens_spent=tokens)

    def read(self, query: str, max_tokens: int = 4096) -> ReadResult:
        t0 = time.perf_counter()
        # Reverse-chronological concatenation
        parts: List[str] = []
        token_count = 0
        for msg in reversed(self._history):
            line = f"[{msg.role}] (turn {msg.turn}): {msg.content}"
            line_tokens = len(line) // 3
            if token_count + line_tokens > max_tokens:
                break
            parts.append(line)
            token_count += line_tokens
        # Reverse back to chronological order for context
        parts.reverse()
        context = "\n".join(parts)
        elapsed = time.perf_counter() - t0
        return ReadResult(context_text=context, read_latency=elapsed)

    def get_token_count(self) -> int:
        return self._total_tokens

    def reset(self) -> None:
        super().reset()
        self._history.clear()
