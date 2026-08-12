"""A3 runner: SS-noRetire x all 72 scenarios (store metrics only — recall/CCR/UA ledger)."""
import json, time, gc, sys, traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

import requests
from engine.base import Message
from noretire_memory import SSNoRetire

OLLAMA = "http://localhost:11434"
PINNED_VERSION = "0.24.0"
OUT = Path("results/a3_noretire.json")

v = requests.get(f"{OLLAMA}/api/version", timeout=10).json().get("version", "?")
if v != PINNED_VERSION:
    print(f"FATAL: ollama {v} != {PINNED_VERSION}"); sys.exit(2)
print(f"[guard] ollama {v} OK")

dirs = [Path("scenarios/holdout_v4"),
        Path("scenarios/holdout_v5")]
scenarios = []
for d in dirs:
    scenarios += sorted(d.glob("*.json"))
print(f"A3 SS-noRetire: {len(scenarios)} scenarios | {datetime.now().isoformat()}")

results = []
for i, p in enumerate(scenarios):
    with open(p) as f:
        sc = json.load(f)
    turns, query, gt = sc["turns"], sc["query"], sc["ground_truth_constraints"]
    mts = set(sc["evaluation_protocol"]["measurement_turns"])
    ua_targets = sc["evaluation_protocol"].get("ua_targets", [])
    raw_len = sum(len(t["content"]) for t in turns)
    print(f"  [{i+1}/{len(scenarios)}] {sc['scenario_id']} ({sc['total_turns']}T) ...", end="", flush=True)
    try:
        t0 = time.perf_counter()
        mem = SSNoRetire(db_path=":memory:")
        for t in turns:
            mem.write(Message(role=t["role"], content=t["content"], turn=t["turn"],
                              is_constraint=t.get("is_constraint", False),
                              is_distractor=t.get("is_distractor", False)))
            if t["turn"] in mts:
                mem.read(query)
        ctx = mem.read(query).context_text
        ua = [{"gt_id": tg["gt_id"], "has_new": tg["new_value"] in ctx,
               "has_old": tg["old_value"] in ctx} for tg in ua_targets]
        r = {"scenario_id": sc["scenario_id"], "scenario_type": sc["scenario_type"],
             "total_turns": sc["total_turns"],
             "recall": sum(1 for g in gt if g in ctx) / len(gt) if gt else 0.0,
             "ccr": round(1 - len(ctx) / raw_len, 4), "ua": ua,
             "wall_clock_s": round(time.perf_counter() - t0, 1)}
        ok = sum(1 for x in ua if x["has_new"] and not x["has_old"])
        print(f" R={r['recall']:.3f} CCR={r['ccr']:.3f} UA={ok}/{len(ua)} t={r['wall_clock_s']}s")
        results.append(r)
        del mem
        gc.collect()
    except Exception as e:
        print(f" ERROR {e}")
        traceback.print_exc()
        results.append({"scenario_id": sc["scenario_id"], "error": str(e)})
    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

print(f"A3 DONE {datetime.now().isoformat()}")
