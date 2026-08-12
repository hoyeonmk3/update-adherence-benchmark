"""
MemoryBase ABC and data classes for the benchmark suite.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Configuration loader (uses Path(__file__) for absolute resolution)
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    """Load benchmark configuration from config.yaml."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data Classes (§3-1)
# ---------------------------------------------------------------------------
@dataclass
class Message:
    """A single conversation turn."""
    role: str               # "user" | "assistant" | "system"
    content: str
    turn: int
    is_constraint: bool = False
    is_distractor: bool = False


@dataclass
class WriteResult:
    """Return value of MemoryBase.write()."""
    write_latency: float = 0.0
    compaction_latency: float = 0.0
    tokens_spent: int = 0


@dataclass
class ReadResult:
    """Return value of MemoryBase.read()."""
    context_text: str = ""
    read_latency: float = 0.0


# ---------------------------------------------------------------------------
# Abstract Base Classes (§3-1)
# ---------------------------------------------------------------------------
class MemoryBase(ABC):
    """Base interface for all memory models."""

    def __init__(self):
        self._total_tokens: int = 0

    @abstractmethod
    def write(self, message: Message) -> WriteResult:
        """
        Ingest a conversation message into memory.
        Returns timing and token usage metrics.
        """

    @abstractmethod
    def read(self, query: str, max_tokens: int = 4096) -> ReadResult:
        """
        Retrieve context from memory for LLM consumption.
        Returns context text and read latency.
        """

    @abstractmethod
    def get_token_count(self) -> int:
        """Return cumulative tokens consumed so far."""

    def reset(self) -> None:
        """Reset memory to initial state (for between scenarios)."""
        self._total_tokens = 0


class PersistentMemoryBase(MemoryBase):
    """Extended ABC for memory models that persist across sessions (§3-1)."""

    @abstractmethod
    def test_cross_session_persistence(self) -> bool:
        """
        Verify that after clearing volatile buffers, the memory can
        reconstruct core constraints from persistent storage alone.
        """
