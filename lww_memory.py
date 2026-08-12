"""LWW (Last-Write-Wins) baseline arm.

Inherits the frozen EpSemSupersession (v9.8) and replaces only the binary LLM
judgment with an unconditional True: same-scope detection (0.92 embedding
screen) + unconditional overwrite. Retirement, storage, compaction, and
provenance are inherited unchanged from the frozen engine, so the difference
from SS is isolated to the presence or absence of the judge.
"""
from engine.epsem_supersession import EpSemSupersession


class LWWMemory(EpSemSupersession):
    """Last-write-wins: retire the top similarity-screen match without judgment."""

    def _llm_supersession_check(self, new_content, existing_content):
        return True, "lww_auto_overwrite"
