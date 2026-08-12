"""SS-noRetire (annotations-only arm) — component isolation.

Disables the frozen SS relationship judgment entirely (Pass 1 supersession +
Pass 2 conflict REJECT + Pass 3 reinsertion guard), leaving a pure append-only
engine. SS's storage structure is kept intact: per-message verbatim storage,
classification, and soft flags.

Verification target: if F/N recall stays at the SS level, the recall gain comes
from the storage structure; if UA drops to 0, UA comes from the retirement
operation — a full separation of the two effects.

Design note: disabling Pass 1 alone is not a clean isolation, because Pass 2
(conflict) would then REJECT incoming revised constraints as contradictions of
the old ones; disabling the whole judgment is the pure separation (verified
against the _compact/_check_relationship structure of epsem_supersession.py).
"""
from engine.epsem_supersession import EpSemSupersession


class SSNoRetire(EpSemSupersession):
    """Relationship judgment fully disabled — append-only SS (storage structure kept)."""

    def _check_relationship(self, message, is_explicit_update=False):
        return {"action": "INDEPENDENT"}
