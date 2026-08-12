"""Threshold sensitivity sweep — SS arm, 24 Update Conflict scenarios, supersession_threshold in {0.85, 0.90, 0.95}.

(0.92, the reported default, is already covered by the canonical data. No behavior
scoring — context metrics only, so this sweep is fast.)
"""
import json, time, gc, sys, traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

import requests
from engine.epsem_supersession import EpSemSupersession
from engine.base import Message

_v = requests.get("http://localhost:11434/api/version", timeout=10).json().get("version", "?")
if _v != "0.24.0":
    print(f"FATAL: ollama {_v} != 0.24.0 (pin violation) — aborting"); sys.exit(2)
print(f"[guard] ollama {_v} OK")

SC_DIRS = [Path("scenarios/holdout_v4"),
           Path("scenarios/holdout_v5")]
OUT_DIR = Path("results")
THRESHOLDS = [0.85, 0.90, 0.95]

def run_one(sc, threshold):
    turns, query, gt = sc["turns"], sc["query"], sc["ground_truth_constraints"]
    mts = set(sc["evaluation_protocol"]["measurement_turns"])
    ua_targets = sc["evaluation_protocol"].get("ua_targets", [])
    raw_len = sum(len(t["content"]) for t in turns)
    mem = EpSemSupersession(db_path=":memory:")
    mem._supersession_threshold = threshold
    for t in turns:
        mem.write(Message(role=t["role"], content=t["content"], turn=t["turn"],
                          is_constraint=t.get("is_constraint", False),
                          is_distractor=t.get("is_distractor", False)))
        if t["turn"] in mts:
            mem.read(query)
    ctx = mem.read(query).context_text
    ua = []
    for tg in ua_targets:
        hn, ho = tg["new_value"] in ctx, tg["old_value"] in ctx
        ua.append({"gt_id": tg["gt_id"], "has_new": hn, "has_old": ho})
    res = {
        "scenario_id": sc["scenario_id"], "total_turns": sc["total_turns"],
        "recall": sum(1 for g in gt if g in ctx) / len(gt) if gt else 0.0,
        "ccr": round(1 - len(ctx) / raw_len, 4), "ua": ua,
    }
    del mem
    gc.collect()
    return res

def main():
    scenarios = []
    for d in SC_DIRS:
        scenarios += sorted(d.glob("U0*.json"))
    print(f"sweep: {len(scenarios)} UC scenarios × {THRESHOLDS} | {datetime.now().isoformat()}")
    for th in THRESHOLDS:
        out = OUT_DIR / f"sweep_th{int(th*100)}.json"
        results = []
        print(f"\n===== threshold {th} | {datetime.now().isoformat()} =====")
        for i, p in enumerate(scenarios):
            with open(p) as f:
                sc = json.load(f)
            print(f"  [{i+1}/{len(scenarios)}] {sc['scenario_id']} ...", end="", flush=True)
            try:
                r = run_one(sc, th)
                ok = sum(1 for x in r["ua"] if x["has_new"] and not x["has_old"])
                print(f" R={r['recall']:.3f} UA={ok}/{len(r['ua'])}")
                results.append(r)
            except Exception as e:
                print(f" ERROR {e}")
                traceback.print_exc()
                results.append({"scenario_id": sc["scenario_id"], "error": str(e)})
            with open(out, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nSWEEP DONE {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
