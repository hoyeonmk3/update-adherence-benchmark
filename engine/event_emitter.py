"""Lightweight event emitter for benchmark instrumentation."""
from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional

class EventEmitter:
    """Collects events during a benchmark run for later DB insertion."""

    def __init__(self):
        self._events: List[Dict[str, Any]] = []
        self._conflict_count = 0
        self._compression_count = 0
        self._constraint_count = 0

    def emit(self, turn: int, model: str, event_type: str, data: dict | None = None):
        self._events.append({
            "turn": turn,
            "model": model,
            "event_type": event_type,
            "data": json.dumps(data or {}),
            "ts": time.time(),
        })
        if event_type == "conflict_detected":
            self._conflict_count += 1
        elif event_type == "compression":
            self._compression_count += 1
        elif event_type == "classification" and data and data.get("label") == "CONSTRAINT":
            self._constraint_count += 1

    def get_events(self) -> List[Dict[str, Any]]:
        return self._events

    def get_snapshot_data(self) -> Dict[str, int]:
        return {
            "constraint_count": self._constraint_count,
            "conflict_count_since_last": self._conflict_count,
            "compression_count_since_last": self._compression_count,
        }

    def reset_counters(self):
        self._conflict_count = 0
        self._compression_count = 0

    def clear(self):
        self._events.clear()
        self._conflict_count = 0
        self._compression_count = 0
        self._constraint_count = 0
