"""Experiment D: build engine/epsem_supersession_alt.py from the frozen engine/epsem_supersession.py.
Only change: the five hard-coded similarity thresholds become instance attributes so a calibrated set can be injected.
Every replacement is asserted to match exactly once; remaining comparisons are listed for audit; output hash recorded."""
import hashlib, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent
src = ROOT / "engine" / "epsem_supersession.py"
dst = ROOT / "engine" / "epsem_supersession_alt.py"
T = src.read_text(encoding="utf-8")
print("source sha256:", hashlib.sha256(T.encode()).hexdigest()[:16])
ATTRS = ("self._supersession_threshold: float = 0.92  # higher bar: supersession requires stronger similarity\n"
         "        self._th_explicit: float = 0.70  # [alt] explicit-marker path (was literal)\n"
         "        self._th_soft: float = 0.80      # [alt] firm-supersedes-soft (was literal)\n"
         "        self._th_dup: float = 0.97       # [alt] near-duplicate rejection (was literal)\n"
         "        self._th_reins: float = 0.95     # [alt] archived re-insertion prevention (was literal)")
REPL = [
 ("class EpSemSupersession(", "class EpSemSupersessionAlt("),
 ("sup_threshold = 0.70 if is_explicit_update else self._supersession_threshold",
  "sup_threshold = self._th_explicit if is_explicit_update else self._supersession_threshold"),
 ("effective_threshold = 0.80 if (is_soft and new_is_firm) else sup_threshold",
  "effective_threshold = self._th_soft if (is_soft and new_is_firm) else sup_threshold"),
 ("if sim > 0.97:", "if sim > self._th_dup:"),
 ("if sim > 0.95:", "if sim > self._th_reins:"),
 ("self._supersession_threshold: float = 0.92  # higher bar: supersession requires stronger similarity", ATTRS),
]
for o, n in REPL:
    c = T.count(o); assert c == 1, (c, o); T = T.replace(o, n)
for i, line in enumerate(T.split("\n"), 1):
    if re.search(r"sim\s*[<>]=?\s*[0-9.]|sim\s*>\s*self\._|_threshold", line) and "def " not in line:
        print(f"  L{i}: {line.strip()[:110]}")
dst.write_text(T, encoding="utf-8")
print("written", dst, "sha256:", hashlib.sha256(T.encode()).hexdigest()[:16])
