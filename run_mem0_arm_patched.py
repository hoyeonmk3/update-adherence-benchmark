"""Mem0 arm, patched rerun of failed scenarios (library monkeypatch variant).

- Purpose: rerun only the scenarios that failed in the main run (run_mem0_arm.py)
  due to a fact-extraction parsing defect in mem0 2.0.17 (when "memory" comes back
  as a list of strings, m.get("text") raises AttributeError). No mem0 package file
  is modified; the json parser is replaced by a proxy only inside this process's
  mem0.memory.main namespace.
- Normalization rule: if the parsed result is a dict whose "memory" is a list,
  promote string items to {"text": string}. Dict items pass through unchanged;
  all other parsing is delegated to the original.
- Output: results/behavior_Mem0_patched.json (separate from the main-run file,
  same schema plus a patched=true field). A separate store directory is used, so
  the main-run outputs are untouched.
- Example: python run_mem0_arm_patched.py --only U007,U008
  (run only after the main chain finishes — avoids concurrent ollama load)
"""
import sys, json as _stdlib_json
from pathlib import Path
from datetime import datetime

import run_mem0_arm as base  # reuse the guard (ollama 0.24.0), config, and run_one

import mem0.memory.main as _mm


class _JsonProxy:
    """json replacement scoped to mem0.memory.main — normalizes loads only, delegates the rest."""

    def loads(self, s, **kw):
        obj = _stdlib_json.loads(s, **kw)
        if isinstance(obj, dict) and isinstance(obj.get("memory"), list):
            obj["memory"] = [m if isinstance(m, dict) else {"text": str(m)}
                             for m in obj["memory"]]
        return obj

    def __getattr__(self, name):
        return getattr(_stdlib_json, name)


_mm.json = _JsonProxy()

# isolated from the main-run store
base.STORE_DIR = "mem0_store_patched"
OUT = Path("results/behavior_Mem0_patched.json")


def main():
    import argparse, shutil, gc, traceback
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, required=True,
                    help="comma-separated scenario-id substrings (e.g. U007,U008)")
    args = ap.parse_args()
    wanted = [w.strip() for w in args.only.split(",") if w.strip()]

    shutil.rmtree(base.STORE_DIR, ignore_errors=True)
    dirs = [Path("scenarios/holdout_v4"),
            Path("scenarios/holdout_v5")]
    scenarios = []
    for d in dirs:
        for p in sorted(d.glob("*.json")):
            if any(w in p.name for w in wanted):
                scenarios.append(p)
    print(f"Mem0 patched arm: {len(scenarios)} scenarios {wanted} | {datetime.now().isoformat()}")

    results = []
    for i, p in enumerate(scenarios):
        with open(p) as f:
            sc = _stdlib_json.load(f)
        print(f"  [{i+1}/{len(scenarios)}] {sc['scenario_id']} ({sc['total_turns']}T) ...",
              end="", flush=True)
        try:
            r = base.run_one(sc)
            r["patched"] = True
            cu = r["context_ua"]
            ok = sum(1 for x in cu if x["has_new"] and not x["has_old"])
            bh = r["behavior"]
            bok = sum(1 for x in bh if x["new_tokens_hit"] and not x["old_tokens_hit"])
            print(f" R={r['recall']:.3f} ctxUA={ok}/{len(cu)} behUA*={bok}/{len(bh)} t={r['replay_s']}s")
            results.append(r)
        except Exception as e:
            print(f" ERROR {e}")
            traceback.print_exc()
            results.append({"scenario_id": sc["scenario_id"], "error": str(e), "patched": True})
        gc.collect()
        with open(OUT, "w") as f:
            _stdlib_json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"MEM0 PATCHED DONE {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
