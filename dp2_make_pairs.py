#!/usr/bin/env python3
"""Step 1 of the similarity analysis: extract (original constraint utterance, update utterance)
pairs for all 70 update events and attach condition and outcome labels.
Inputs are the released scenarios and canonical results. Output feeds dp2_embed.py."""
import json, glob, os
from collections import Counter

def canon_outcomes():
    out = {}
    for fn in ["v4_holdout_20260602_215257.json", "v5_holdout_merged.json"]:
        for r in json.load(open(os.path.join("results", "canonical", fn), encoding="utf-8"))["EpSemSupersession"]:
            if r.get("ua"):
                for d in r["ua"]["details"]:
                    n, o = d["has_new"], d["has_old"]
                    out[(r["scenario_id"], d["gt_id"])] = "C" if n and not o else "SR" if n else "SO" if o else "L"
    return out

def imp_outcomes():
    out = {}
    for r in json.load(open(os.path.join("results", "behavior_EpSemSupersession_imp.json"), encoding="utf-8")):
        if "error" in r:
            continue
        for e in r["context_ua"]:
            n, o = e["has_new"], e["has_old"]
            out[(r["scenario_id"], e["gt_id"])] = "C" if n and not o else "SR" if n else "SO" if o else "L"
    return out

pairs = []
canon = canon_outcomes()
imp = imp_outcomes()

def extract(sc, cond, out_map, sid_key):
    for tg in sc["evaluation_protocol"]["ua_targets"]:
        ov, nv = tg["old_value"], tg["new_value"]
        orig = next((t["content"] for t in sc["turns"] if ov in t["content"]), None)
        upd = next((t["content"] for t in sc["turns"] if nv in t["content"]), None)
        if not orig or not upd:
            print(f"  warning, incomplete pair: {sid_key} {tg['gt_id']}")
            continue
        pairs.append({"scenario": sid_key, "gt_id": tg["gt_id"], "cond": cond,
                      "update_turn": tg["update_turn"], "total_turns": sc["total_turns"],
                      "orig": orig, "upd": upd,
                      "outcome": out_map.get((sid_key, tg["gt_id"]), "?")})

for d in ["holdout_v4", "holdout_v5"]:
    for p in sorted(glob.glob(os.path.join("scenarios", d, "U0*.json"))):
        sc = json.load(open(p, encoding="utf-8"))
        extract(sc, "explicit", canon, sc["scenario_id"])

for p in sorted(glob.glob(os.path.join("scenarios", "implicit", "U0*.json"))):
    sc = json.load(open(p, encoding="utf-8"))
    tier = sc.get("implicit_tier", "?")
    extract(sc, tier, imp, sc["scenario_id"])

print(f"{len(pairs)} pairs: {Counter(x['cond'] for x in pairs)}")
print(f"outcome labels: {Counter(x['outcome'] for x in pairs)}")
json.dump(pairs, open("dp2_pairs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote dp2_pairs.json")
