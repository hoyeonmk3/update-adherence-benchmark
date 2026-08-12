"""SS(threshold=0.85) x 48 F/N scenarios — measuring the protective effect of the 0.92 conservatism.

Reference: canonical SS (0.92) F/N recall is 1.000/1.000. If 0.85 costs recall here,
the conservative threshold is justified; if it does not, the honest statement is that
0.85 dominates on this benchmark.
"""
import json, time, gc, sys, traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

import requests
from engine.base import Message
from engine.epsem_supersession import EpSemSupersession

OLLAMA = "http://localhost:11434"
_v = requests.get(f"{OLLAMA}/api/version", timeout=10).json().get("version", "?")
if _v != "0.24.0":
    print(f"FATAL: ollama {_v} != 0.24.0"); sys.exit(2)
print(f"[guard] ollama {_v} OK")

OUT = Path("results/ss085_fn.json")
scenarios = sorted(Path("scenarios/fn").glob("*.json"))
print(f"SS(0.85) × F/N: {len(scenarios)} scenarios | {datetime.now().isoformat()}")

results = []
for i, p in enumerate(scenarios):
    with open(p) as f:
        sc = json.load(f)
    turns, query, gt = sc["turns"], sc["query"], sc["ground_truth_constraints"]
    mts = set(sc["evaluation_protocol"]["measurement_turns"])
    raw_len = sum(len(t["content"]) for t in turns)
    print(f"  [{i+1}/{len(scenarios)}] {sc['scenario_id']} ({sc['total_turns']}T) ...", end="", flush=True)
    try:
        t0 = time.perf_counter()
        mem = EpSemSupersession(db_path=":memory:")
        mem._supersession_threshold = 0.85
        for t in turns:
            mem.write(Message(role=t["role"], content=t["content"], turn=t["turn"],
                              is_constraint=t.get("is_constraint", False),
                              is_distractor=t.get("is_distractor", False)))
            if t["turn"] in mts:
                mem.read(query)
        ctx = mem.read(query).context_text
        r = {"scenario_id": sc["scenario_id"], "scenario_type": sc["scenario_type"],
             "total_turns": sc["total_turns"],
             "recall": sum(1 for g in gt if g in ctx) / len(gt) if gt else 0.0,
             "found": sum(1 for g in gt if g in ctx), "total_gt": len(gt),
             "missed_gt": [g for g in gt if g not in ctx],
             "ccr": round(1 - len(ctx) / raw_len, 4),
             "wall_clock_s": round(time.perf_counter() - t0, 1)}
        print(f" R={r['recall']:.3f} CCR={r['ccr']:.3f} t={r['wall_clock_s']}s")
        results.append(r)
        del mem
        gc.collect()
    except Exception as e:
        print(f" ERROR {e}")
        traceback.print_exc()
        results.append({"scenario_id": sc["scenario_id"], "error": str(e)})
    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

print(f"SS085-FN DONE {datetime.now().isoformat()}")
