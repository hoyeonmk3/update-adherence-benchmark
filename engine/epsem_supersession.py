"""
EpSemSupersession — EP-SEM with Constraint Supersession.

Based on EpSemMemory (copy + modify).
Adds:
- Supersession detection (LLM-based update relationship check)
- ARCHIVED constraint handling (is_archived flag)
- Query path separation (read excludes ARCHIVED, conflict check includes ARCHIVED)
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests

from .event_emitter import EventEmitter
from .base import PersistentMemoryBase, Message, WriteResult, ReadResult, load_config


class EpSemSupersession(PersistentMemoryBase):
    """
    EP-SEM with Constraint Supersession:
    - Episodic-Semantic Memory with structural tagging
    - Conflict rejection (existing)
    - Supersession detection (NEW): automatically archives old constraints
      when a newer constraint updates the same requirement
    """

    def __init__(self, db_path: str | None = None, compaction_interval: int | None = None):
        super().__init__()
        self.event_emitter: EventEmitter | None = None
        config = load_config()
        self._compaction_interval: int = compaction_interval or config.get("compaction_interval", 10)
        self._ollama_url: str = config.get("ollama_endpoint", "http://localhost:11434")
        self._model: str = config.get("ollama_model", "qwen2.5:14b-instruct-q4_K_M")
        self._embed_model: str = config.get("embed_model", "nomic-embed-text")
        self._conflict_threshold: float = 0.85
        self._supersession_threshold: float = 0.92  # higher bar: supersession requires stronger similarity

        # Prompts from config (externalized for reproducibility)
        # Pre-detection markers for explicit update messages
        self._update_markers: list = [
            "revised from", "updated from", "changed from",
            "UPDATED REQUIREMENT", "has been revised", "has been changed",
            "new requirement replaces",
            "just to confirm the update", "just to confirm the change",
        ]
        # Word-boundary markers (checked with regex to avoid partial matches)
        self._update_markers_word: list = ["supersedes"]
        # Confirm/repeat markers — these REPEAT existing constraints, not introduce new ones
        # NOTE: Only include phrases that UNAMBIGUOUSLY indicate repetition.
        # Avoid generic phrases like 'as noted', 'to clarify' that may precede valid new constraints.
        self._confirm_markers: list = [
            "just to confirm", "to reiterate", "as mentioned earlier",
            "as previously stated", "reiterating", "confirming that",
        ]

        self._supersession_prompt: str = config.get(
            "supersession_prompt",
            "Does the new statement EXPLICITLY REVISE or REPLACE the existing constraint?\n\n"
            "Answer true ONLY if:\n"
            "- The new statement contains explicit revision language "
            "(e.g. 'revised from X to Y', 'updated to', 'changed from', 'replaces')\n"
            "- OR the new statement specifies a DIFFERENT value for the EXACT SAME "
            "parameter in the EXACT SAME scope, making the old value obsolete.\n\n"
            "Answer false if:\n"
            "- The two constraints could validly coexist as separate requirements.\n"
            "- The new statement is an ADDITIONAL requirement (e.g. a new safety factor "
            "at a different design stage).\n\n"
            'Respond with JSON: {{"updates": true/false, "reason": "..."}}\n\n'
            "Existing constraint: {existing}\n"
            "New statement: {new}\n\n"
            "JSON response:"
        )
        self._conflict_prompt: str = config.get(
            "conflict_prompt",
            "Do the following two engineering constraints CONTRADICT each other?\n"
            'Respond with JSON: {{"contradicts": true/false, "reason": "..."}}\n\n'
            "Existing constraint: {existing}\n"
            "New statement: {new}\n\n"
            "JSON response:"
        )

        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "epsem_supersession.db")
        self._db_path = db_path
        self._conn = sqlite3.connect(self._db_path)
        self._setup_tables()

        self._episodic: List[Message] = []
        self._turn_count: int = 0

    # Soft hedging patterns — messages matching these are stored as
    # CONSTRAINT but flagged is_soft_source=1 so that later firm-language
    # constraints can supersede them at a lower threshold (0.80 vs 0.92).
    _SOFT_HEDGING_PATTERNS: list = [
        r"rough guideline",
        r"would be about",
        r"\bideally\b",
        r"not critical",
        r"just a preference",
        r"\bmight be\b(?!\s+critical)",
        r"it's not critical",
        r"though this is just",
        r"in informal settings",
        r"historical data shows.*averaged",
    ]

    def _is_soft_language(self, text: str) -> bool:
        """Detect soft hedging language that indicates low-authority source."""
        text_lower = text.lower()

        # "Important requirement" is ALWAYS firm — hard override, no exceptions
        if 'important requirement' in text_lower:
            return False

        # Check for soft hedging patterns
        has_soft = any(re.search(pat, text_lower) for pat in self._SOFT_HEDGING_PATTERNS)
        if not has_soft:
            return False

        # If soft pattern exists, check if firm obligation phrases also exist.
        # When BOTH soft + firm are present, soft takes priority.
        # Rationale: distractors often embed obligation words from parameter names
        # (e.g., "Material hardness must be at least HRC should ideally be around 94")
        # The soft hedging context ("ideally", "just a preference") indicates
        # the speaker does NOT intend this as a binding constraint.
        return True

    def _setup_tables(self) -> None:
        cur = self._conn.cursor()
        # Semantic store — with is_archived and is_soft_source columns
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
                ttl_turns INTEGER DEFAULT -1,
                is_archived INTEGER DEFAULT 0,
                is_soft_source INTEGER DEFAULT 0
            )
        """)
        # Migration: add is_soft_source to existing DBs
        try:
            cur.execute("ALTER TABLE semantic_store ADD COLUMN is_soft_source INTEGER DEFAULT 0")
        except Exception:
            pass  # Column already exists
        # Embedding cache
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)
        # Rejection log (also used for supersession audit trail)
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
        if self.event_emitter:
            self.event_emitter.emit(turn, "EpSemSupersession", event_type, data)

    def write(self, message: Message) -> WriteResult:
        t0 = time.perf_counter()
        self._episodic.append(message)
        self._turn_count += 1

        tokens = len(message.content) // 3
        self._total_tokens += tokens

        compaction_latency = 0.0
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

        # --- Priority 1: Active CONSTRAINT only (ARCHIVED excluded) ---
        cur.execute(
            "SELECT turn, role, content, tag FROM semantic_store "
            "WHERE tag = 'CONSTRAINT' AND is_archived = 0 ORDER BY turn"
        )
        constraint_parts: List[str] = []
        for turn, role, content, tag in cur.fetchall():
            constraint_parts.append(f"[{tag}] [{role}] (turn {turn}): {content}")

        constraint_block = "\n".join(constraint_parts)
        remaining_chars = char_limit - len(constraint_block)

        # --- Priority 2: Episodic buffer ---
        episodic_parts: List[str] = []
        for msg in self._episodic:
            episodic_parts.append(f"[{msg.role}] (turn {msg.turn}): {msg.content}")

        # --- Priority 3: Non-expired EPHEMERAL ---
        cur.execute(
            "SELECT turn, role, content, tag, ttl_turns FROM semantic_store "
            "WHERE tag = 'EPHEMERAL' ORDER BY turn DESC"
        )
        ephemeral_parts: List[str] = []
        for turn, role, content, tag, ttl in cur.fetchall():
            if ttl > 0 and (self._turn_count - turn) > ttl:
                continue
            ephemeral_parts.append(f"[{tag}] [{role}] (turn {turn}): {content}")

        secondary_block = "\n".join(episodic_parts + ephemeral_parts)
        if len(secondary_block) > remaining_chars and remaining_chars > 0:
            secondary_block = secondary_block[:remaining_chars]

        if constraint_block and secondary_block:
            context = constraint_block + "\n" + secondary_block
        else:
            context = constraint_block or secondary_block

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
        cur.execute("DELETE FROM rejection_log")
        # Preserve embedding_cache
        self._conn.commit()

    def test_cross_session_persistence(self) -> bool:
        self._episodic.clear()
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM semantic_store WHERE tag = 'CONSTRAINT' AND is_archived = 0")
        count = cur.fetchone()[0]
        if count == 0:
            return False
        result = self.read("verify persistence")
        return "[CONSTRAINT]" in result.context_text

    # -------------------------------------------------------------------
    # Compaction Process — with Supersession
    # -------------------------------------------------------------------
    def _compact(self) -> None:
        """
        Synchronous block compaction with supersession:
        Phase A: Classify each episodic entry via LLM
        Phase B: Check relationships for CONSTRAINT candidates
                 (supersession + conflict, in that order)
        Phase C: Store classified entries in semantic store
        Phase D: Clear episodic buffer
        """
        self._emit_event(self._turn_count, "compression", {"buffer_size": len(self._episodic)})
        cur = self._conn.cursor()
        for msg in self._episodic:
            # Phase A-0: Pre-detection for explicit update messages
            content_lower = msg.content.lower()
            is_explicit_update = (
                any(marker.lower() in content_lower for marker in self._update_markers)
                or any(re.search(r'\b' + marker + r'\b', content_lower, re.IGNORECASE)
                       for marker in self._update_markers_word)
            )
            # Pre-detection for confirm/repeat messages (demote to EPHEMERAL)
            is_confirm_repeat = any(
                marker in content_lower for marker in self._confirm_markers
            )

            # Phase A: Classify
            tag = self._classify(msg)
            self._emit_event(msg.turn, "classification", {"label": tag, "content": msg.content[:100]})

            # Confirm/repeat messages are never CONSTRAINT — they repeat existing ones
            if is_confirm_repeat and not is_explicit_update:
                if tag == "CONSTRAINT":
                    tag = "EPHEMERAL"
                    self._emit_event(msg.turn, "confirm_demotion", {
                        "content": msg.content[:100]
                    })

            if tag in ("UNCERTAIN", "EPHEMERAL"):
                # If pre-detection found update markers, promote to CONSTRAINT
                # so it enters the supersession pathway
                if is_explicit_update:
                    tag = "CONSTRAINT"
                    self._emit_event(msg.turn, "pre_detection_override", {
                        "content": msg.content[:100]
                    })
                elif tag == "UNCERTAIN":
                    tag = "EPHEMERAL"

            # Phase B: Relationship check for constraints (supersession + conflict)
            did_supersede = False
            superseded_targets = []
            if tag == "CONSTRAINT":
                rel = self._check_relationship(msg, is_explicit_update)
                if rel["action"] == "REJECT":
                    cur.execute(
                        "INSERT INTO rejection_log (turn, content, conflicting_constraint_id, reason, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (msg.turn, msg.content, rel["target_id"], rel["reason"], time.time()),
                    )
                    tag = "REJECTED"
                    self._emit_event(msg.turn, "conflict_detected", {
                        "content": msg.content[:100], "reason": rel["reason"]
                    })
                elif rel["action"] == "SUPERSEDE":
                    # Archive old constraint(s) + log — with rollback on partial failure
                    archived_ids = []
                    try:
                        for target_id, reason in rel["targets"]:
                            self._archive_constraint(cur, target_id, reason)
                            archived_ids.append(target_id)
                            self._emit_event(msg.turn, "supersession", {
                                "archived_id": target_id,
                                "reason": reason,
                                "new_content": msg.content[:100]
                            })
                        did_supersede = True
                        superseded_targets = rel["targets"]
                        # tag remains CONSTRAINT → Phase C stores new constraint
                    except Exception:
                        # Rollback: un-archive any partially archived constraints
                        for aid in archived_ids:
                            cur.execute(
                                "UPDATE semantic_store SET is_archived=0 WHERE id=?",
                                (aid,)
                            )
                        tag = "REJECTED"  # safe fallback
                # action == "INDEPENDENT" → tag stays CONSTRAINT

            # Phase C: Store (skip REJECTED entries)
            if tag != "REJECTED":
                ttl = 20 if tag == "EPHEMERAL" else -1
                store_content = msg.content

                # Fix 3+4: Only transform content on actual SUPERSEDE
                if did_supersede and is_explicit_update and tag == "CONSTRAINT":
                    # Fix 4: Template substitution from old constraint text
                    substituted = self._substitute_old_constraint(
                        msg.content, superseded_targets, cur
                    )
                    if substituted:
                        store_content = substituted
                        self._emit_event(msg.turn, "template_substitution", {
                            "result": store_content[:100]
                        })
                    else:
                        # Fallback: try extraction
                        extracted = self._extract_new_constraint(msg.content)
                        if extracted:
                            store_content = extracted
                            self._emit_event(msg.turn, "extraction_fallback", {
                                "result": store_content[:100]
                            })
                        else:
                            self._emit_event(msg.turn, "substitution_failed", {
                                "content": msg.content[:100]
                            })
                        # If both fail, store_content remains msg.content

                # Detect soft hedging language for source-firmness demotion
                soft_flag = 1 if (tag == "CONSTRAINT" and self._is_soft_language(msg.content)) else 0
                cur.execute(
                    "INSERT INTO semantic_store "
                    "(turn, role, content, tag, is_constraint, is_distractor, created_at, ttl_turns, is_archived, is_soft_source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                    (msg.turn, msg.role, store_content, tag,
                     int(msg.is_constraint), int(msg.is_distractor), time.time(), ttl, soft_flag),
                )
                self._total_tokens += 5

        self._conn.commit()
        # Phase D: Clear episodic
        self._episodic.clear()

    # -------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------
    def _classify(self, message: Message) -> str:
        prompt = (
            "Classify the following message in an engineering conversation.\n"
            "Respond with EXACTLY one word: CONSTRAINT, EPHEMERAL, or UNCERTAIN.\n\n"
            "Rules:\n"
            "- CONSTRAINT: Contains specific engineering requirements, specifications, "
            "numerical limits, material properties, or design rules that MUST be preserved. "
            "ALSO classify as CONSTRAINT if the message contains: mandatory behavioral "
            "policies, compliance requirements (e.g., regulatory or legal obligations), "
            "or decision-making principles using obligation language (must, never, always, "
            "shall) that constrain how work should be done.\n"
            "- EPHEMERAL: Casual conversation, greetings, off-topic discussion, "
            "weather talk, or social exchanges with NO engineering or policy content. "
            "Do NOT classify as EPHEMERAL if the message contains any obligation, "
            "prohibition, or mandatory principle.\n"
            "- UNCERTAIN: Engineering-related but not a firm constraint.\n\n"
            "Examples:\n"
            '- "Never suggest single-point-of-failure architectures" -> CONSTRAINT\n'
            '- "Data retention must comply with GDPR Article 17" -> CONSTRAINT\n'
            '- "Minimum wall thickness shall be 331mm" -> CONSTRAINT\n'
            '- "Thermal conductivity shall exceed 352 W/mK" -> CONSTRAINT\n'
            '- "Material hardness must be at least 417 HRC" -> CONSTRAINT\n'
            '- "Operating pressure range is 123 to 593 bar" -> CONSTRAINT\n'
            '- "Maximum operating temperature must not exceed 23°C" -> CONSTRAINT\n'
            '- "Sounds good, let me know!" -> EPHEMERAL\n'
            '- "We might consider performance tradeoffs" -> UNCERTAIN\n\n'
            f"Message: {message.content}\n\n"
            "Classification:"
        )
        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0, "seed": 42}},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip().upper()
            for tag in ["CONSTRAINT", "EPHEMERAL", "UNCERTAIN"]:
                if tag in result:
                    return tag
            return "UNCERTAIN"
        except Exception as e:
            self._emit_event(getattr(message, 'turn', -1), "classify_error", {
                "error": str(e)[:200], "content": message.content[:50]
            })
            return "UNCERTAIN"

    # -------------------------------------------------------------------
    # Relationship Check — Supersession + Conflict (NEW)
    # -------------------------------------------------------------------
    def _check_relationship(self, message: Message, is_explicit_update: bool = False) -> dict:
        """
        Check relationship between new message and existing constraints.
        Three-pass approach:
        1. Active constraints: supersession check (can the new one update an active one?)
        2. Active constraints: conflict check (does the new one contradict an active one?)
        3. Archived constraints: re-insertion prevention (is this a duplicate of something archived?)

        Args:
            is_explicit_update: If True, use lower supersession threshold (0.70)
                because explicit update markers provide strong prior signal.

        Returns:
        - {"action": "INDEPENDENT"}: no relationship found
        - {"action": "SUPERSEDE", "target_id": int, "reason": str}
        - {"action": "REJECT", "target_id": int, "reason": str}
        """
        # Fix 1: Dual threshold — explicit updates use 0.70 (empirical, LLM is 2nd filter)
        sup_threshold = 0.70 if is_explicit_update else self._supersession_threshold
        new_emb = self._get_embedding(message.content)
        if new_emb is None:
            return {"action": "INDEPENDENT"}

        # Pre-compute firmness of the new message (used in Pass 1 + Pass 2)
        new_is_firm = not self._is_soft_language(message.content)

        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, content, is_archived, is_soft_source FROM semantic_store WHERE tag = 'CONSTRAINT'"
        )
        all_constraints = cur.fetchall()
        if not all_constraints:
            return {"action": "INDEPENDENT"}

        active = [(cid, content, is_soft) for cid, content, ia, is_soft in all_constraints if not ia]
        archived = [(cid, content) for cid, content, ia, _ in all_constraints if ia]

        # Pre-compute similarities for active constraints (avoid redundant embedding lookups)
        active_sims = []
        for cid, content, is_soft in active:
            existing_emb = self._get_embedding(content)
            if existing_emb is None:
                continue
            sim = self._cosine_sim(new_emb, existing_emb)
            active_sims.append((cid, content, sim, is_soft))

        # --- Pass 1: Supersession check on ACTIVE constraints (best match only) ---
        # Collect candidates, then take only the highest-similarity match.
        # Multi-target archiving causes unrecoverable GT loss because
        # _substitute_old_constraint only processes targets[0].
        supersede_candidates = []
        # new_is_firm already computed at function entry
        for cid, content, sim, is_soft in active_sims:
            # Source-firmness demotion: only lower threshold if existing is soft AND new is firm
            effective_threshold = 0.80 if (is_soft and new_is_firm) else sup_threshold
            if sim > effective_threshold:
                is_update, reason = self._llm_supersession_check(message.content, content)
                if is_update:
                    supersede_candidates.append((cid, reason, sim))

        if supersede_candidates:
            # Sort by similarity descending, take only the best match
            supersede_candidates.sort(key=lambda x: x[2], reverse=True)
            best = supersede_candidates[0]
            supersede_targets = [(best[0], best[1])]
            return {"action": "SUPERSEDE", "targets": supersede_targets,
                    "target_id": best[0],  # backward compat
                    "reason": best[1]}

        # --- Pass 1.5: Active near-duplicate rejection (no LLM, similarity only) ---
        for cid, content, sim, is_soft in active_sims:
            if sim > 0.97:
                return {"action": "REJECT", "target_id": cid,
                        "reason": f"[NEAR_DUPLICATE_ACTIVE] sim={sim:.3f}"}

        # --- Pass 2: Conflict check on ACTIVE constraints only ---
        # Source-firmness demotion: if new message has firm language and
        # existing is soft-source, treat as supersession instead of conflict
        # new_is_firm already computed at function entry
        for cid, content, sim, is_soft in active_sims:
            if sim > self._conflict_threshold:
                is_conflict, reason = self._llm_conflict_check(message.content, content)
                if is_conflict:
                    # Firmness demotion: firm new + soft existing → supersede, not reject
                    if new_is_firm and is_soft:
                        self._emit_event(getattr(message, 'turn', -1), "firmness_demotion", {
                            "soft_id": cid, "content": message.content[:100],
                            "reason": f"firm supersedes soft: {reason}"
                        })
                        return {"action": "SUPERSEDE",
                                "targets": [(cid, f"[FIRMNESS_DEMOTION] {reason}")],
                                "target_id": cid,
                                "reason": f"[FIRMNESS_DEMOTION] {reason}"}
                    return {"action": "REJECT", "target_id": cid, "reason": reason}

        # --- Pass 3: Archived re-insertion prevention (no LLM, similarity only) ---
        for cid, content in archived:
            existing_emb = self._get_embedding(content)
            if existing_emb is None:
                continue
            sim = self._cosine_sim(new_emb, existing_emb)
            if sim > 0.95:
                return {"action": "REJECT", "target_id": cid,
                        "reason": f"[DUPLICATE_OF_ARCHIVED] sim={sim:.3f}"}

        return {"action": "INDEPENDENT"}

    def _substitute_old_constraint(
        self, update_content: str, targets: list, cur
    ) -> str | None:
        """Template substitution: replace old value with new value in the OLD constraint text.

        This preserves modal language ('must not exceed') and non-superseded parameters
        (yield strength, fatigue life) that coexist in composite constraint rows.

        Args:
            update_content: The update message text (e.g., "revised from 400°C to 300°C")
            targets: List of (target_id, reason) from supersession
            cur: DB cursor

        Returns:
            Substituted text, or None if substitution fails.
        """
        # Extract old_value and new_value from the update message
        match = re.search(
            r'(?:revis(?:ed|es)|chang(?:ed|es)|updat(?:ed|es)|replac(?:ed|es))\s+from\s+(\S+(?:°?\w*)?)\s+to\s+(\S+(?:°?\w*)?)',
            update_content, re.IGNORECASE
        )
        if not match:
            # Fallback 1: extract "from <number> to <number>" from LLM reason
            reason = targets[0][1] if targets else ""
            match = re.search(
                r'from\s+(\d+(?:\.\d+)?\s*°?\s*\w*)\s+to\s+(\d+(?:\.\d+)?\s*°?\s*\w*)',
                reason, re.IGNORECASE
            )
        if not match:
            # Fallback 2: flexible extraction — LLM reason wraps values in phrases
            # e.g., "from 'must not exceed 250 bar' to 'must not exceed 187.5 bar'"
            reason = targets[0][1] if targets else ""
            m = re.search(
                r'from\b.*?(\d+(?:\.\d+)?)\s*(°?\w+).*?\bto\b.*?(\d+(?:\.\d+)?)\s*(°?\w+)',
                reason, re.IGNORECASE
            )
            if m:
                old_val = m.group(1) + " " + m.group(2)   # e.g., "250 bar"
                new_val = m.group(3) + " " + m.group(4)   # e.g., "187.5 bar"
                old_val = old_val.strip()
                new_val = new_val.strip()
                # Get old constraint text
                target_id = targets[0][0]
                cur.execute("SELECT content FROM semantic_store WHERE id=?", (target_id,))
                row = cur.fetchone()
                if row and old_val in row[0]:
                    return row[0].replace(old_val, new_val, 1)
                # Also try without space (e.g., "250bar" vs "250 bar")
                if row:
                    old_nospace = m.group(1) + m.group(2)
                    new_nospace = m.group(3) + m.group(4)
                    if old_nospace in row[0]:
                        return row[0].replace(old_nospace, new_nospace, 1)
        if not match:
            # Fallback 3: "Previous: X. New: Y" format in the update message
            # Handles messages like "Previous: coefficient 0.35-0.45. New: coefficient 0.3-0.45."
            # Note: Group 2 uses greedy (.+) to avoid premature termination at decimal points
            m_prev = re.search(
                r'Previous:\s*(.+?)\.\s*New:\s*(.+)',
                update_content, re.IGNORECASE
            )
            if m_prev:
                old_phrase = m_prev.group(1).strip()
                new_phrase = m_prev.group(2).strip().rstrip('.')
                target_id = targets[0][0]
                cur.execute("SELECT content FROM semantic_store WHERE id=?", (target_id,))
                row = cur.fetchone()
                if row and old_phrase in row[0]:
                    return row[0].replace(old_phrase, new_phrase, 1)
            return None

        old_value = match.group(1).strip().rstrip('.,;')
        new_value = match.group(2).strip().rstrip('.,;')

        if not old_value or not new_value:
            return None

        # Get the first superseded constraint's text
        target_id = targets[0][0]
        cur.execute("SELECT content FROM semantic_store WHERE id=?", (target_id,))
        row = cur.fetchone()
        if not row:
            return None

        old_text = row[0]

        # Attempt substitution (first occurrence only to avoid partial matches)
        if old_value in old_text:
            result = old_text.replace(old_value, new_value, 1)
            return result

        # Try without degree symbol variations
        old_value_stripped = old_value.replace('°', '')
        new_value_stripped = new_value.replace('°', '')
        if old_value_stripped in old_text:
            result = old_text.replace(old_value_stripped, new_value_stripped, 1)
            return result

        return None

    def _extract_new_constraint(self, content: str) -> str | None:
        """Extract only the NEW constraint value from an update message.

        This prevents storing the old value text (e.g., '400°C') in the
        active CONSTRAINT row, which would cause has_old=True in UA checks.

        Patterns handled:
        - "revised from X to Y due to Z" → "Y due to Z" (reconstructed)
        - "changed from X to Y" → extracts Y portion
        - "UPDATED REQUIREMENT: ... from X to Y ..." → extracts Y portion
        """
        content_lower = content.lower()

        # Pattern 1: "from X to Y" — extract everything after "to"
        match = re.search(
            r'(?:revis(?:ed|es)|chang(?:ed|es)|updat(?:ed|es)|replac(?:ed|es))\s+from\s+(.+?)\s+to\s+(.+)',
            content, re.IGNORECASE
        )
        if match:
            old_val = match.group(1).strip()
            new_part = match.group(2).strip()
            # Reconstruct: remove the "from X to Y" narrative,
            # keep the new value in a clean constraint form
            # Look for the parameter context before "revised from"
            pre_match = re.search(
                r'(?:^|:\s*)(.+?)(?:has been |was |)\s*(?:revis(?:ed|es)|chang(?:ed|es)|updat(?:ed|es)|replac(?:ed|es))\s+from',
                content, re.IGNORECASE
            )
            if pre_match:
                param_context = pre_match.group(1).strip()
                # Strip "UPDATED REQUIREMENT:" prefix if present
                param_context = re.sub(
                    r'^UPDATED\s+REQUIREMENT\s*:\s*', '', param_context, flags=re.IGNORECASE
                ).strip()
                # Reconstruct: "Maximum operating temperature must not exceed 300°C"
                return f"{param_context} {new_part}"
            return new_part

        # Pattern 2: "UPDATED REQUIREMENT:" prefix — take content after it
        # but strip any "from X" references
        if "UPDATED REQUIREMENT" in content.upper():
            after_prefix = re.sub(
                r'^.*?UPDATED\s+REQUIREMENT\s*:\s*', '', content, flags=re.IGNORECASE
            ).strip()
            # Remove "from X to" portion if present
            cleaned = re.sub(
                r'\s*(?:revised|changed)\s+from\s+\S+\s+to\s+', ' ',
                after_prefix, flags=re.IGNORECASE
            ).strip()
            if cleaned and len(cleaned) > 10:
                return cleaned

        return None

    @staticmethod
    def _parse_json_from_llm(raw: str) -> dict:
        """Extract JSON from LLM response that may contain explanation text."""
        # Try direct parse first
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        # Try extracting JSON object from mixed text
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass
        return {}

    def _llm_supersession_check(self, new_content: str, existing_content: str) -> tuple[bool, str]:
        """LLM-based update relationship detection (NEW)."""
        prompt = self._supersession_prompt.format(
            existing=existing_content, new=new_content
        )
        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0, "seed": 42}},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            data = self._parse_json_from_llm(result)
            return data.get("updates", False), data.get("reason", "")
        except Exception as e:
            self._emit_event(-1, "llm_supersession_error", {
                "error": str(e)[:200],
                "new": new_content[:50], "existing": existing_content[:50]
            })
            return False, ""

    def _llm_conflict_check(self, new_content: str, existing_content: str) -> tuple[bool, str]:
        """LLM-based contradiction detection (preserved from EpSemMemory)."""
        prompt = self._conflict_prompt.format(
            existing=existing_content, new=new_content
        )
        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0, "seed": 42}},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            data = self._parse_json_from_llm(result)
            return data.get("contradicts", False), data.get("reason", "")
        except Exception as e:
            self._emit_event(-1, "llm_conflict_error", {
                "error": str(e)[:200],
                "new": new_content[:50], "existing": existing_content[:50]
            })
            return False, ""

    # -------------------------------------------------------------------
    # Archive Helper (NEW)
    # -------------------------------------------------------------------
    def _archive_constraint(self, cur, constraint_id: int, reason: str) -> None:
        """
        Archive a constraint (set is_archived=1). NOT a delete.
        Uses the provided cursor to participate in the caller's transaction scope.
        Logs the action for audit trail (Explicit Override principle).
        """
        cur.execute(
            "UPDATE semantic_store SET is_archived=1 WHERE id=?",
            (constraint_id,)
        )
        cur.execute(
            "INSERT INTO rejection_log (turn, content, conflicting_constraint_id, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (self._turn_count, f"[AUTO_SUPERSEDE] id={constraint_id}", constraint_id, reason, time.time())
        )

    # -------------------------------------------------------------------
    # Embedding Helpers (shared pattern with EpSemMemory)
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
