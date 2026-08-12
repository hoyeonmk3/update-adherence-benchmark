"""TSRead arm — query-time newest-wins (timestamp-aware, no-deletion resolution).

Built on SS-noRetire (append-only verbatim store) with a post-processing step at
read time: CONSTRAINT lines in the assembled context are embedded, and within
each mutual-similarity >= 0.92 cluster only the most recent turn is kept. The
store itself is untouched (nothing is deleted); resolution happens at query
time. The threshold matches SS (0.92) for comparison parity.
"""
import json, time, gc, sys, re, math, traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

import requests
from engine.base import Message
from noretire_memory import SSNoRetire

OLLAMA = "http://localhost:11434"
_v = requests.get(f"{OLLAMA}/api/version", timeout=10).json().get("version", "?")
if _v != "0.24.0":
    print(f"FATAL: ollama {_v} != 0.24.0"); sys.exit(2)
print(f"[guard] ollama {_v} OK")

OUT = Path("results/behavior_TSRead.json")
GEN_MODEL = "qwen2.5:14b-instruct-q4_K_M"
LINE_RE = re.compile(r"^\[CONSTRAINT\] \[(\w+)\] \(turn (\d+)\): (.*)$")
THRESH = 0.92

_emb_cache = {}
def embed(text):
    if text in _emb_cache:
        return _emb_cache[text]
    r = requests.post(f"{OLLAMA}/api/embeddings",
                      json={"model": "nomic-embed-text", "prompt": text}, timeout=60)
    r.raise_for_status()
    v = r.json()["embedding"]
    _emb_cache[text] = v
    return v

def cos(a, b):
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a)); db = math.sqrt(sum(y*y for y in b))
    return num/(da*db) if da and db else 0.0

def ts_filter(context):
    """Cluster CONSTRAINT lines, keep only the most recent turn per cluster. Non-CONSTRAINT lines are preserved."""
    lines = context.split("\n")
    cons = []
    for idx, l in enumerate(lines):
        m = LINE_RE.match(l.strip())
        if m:
            cons.append({"idx": idx, "turn": int(m.group(2)), "content": m.group(3)})
    kept = []
    drop_idx = set()
    for c in sorted(cons, key=lambda x: -x["turn"]):  # newest first
        e = embed(c["content"])
        if any(cos(e, embed(k["content"])) >= THRESH for k in kept):
            drop_idx.add(c["idx"])  # similar to a newer item -> older version drops
        else:
            kept.append(c)
    return "\n".join(l for i, l in enumerate(lines) if i not in drop_idx)

def gen_answer(context, query):
    prompt = ("You are an assistant helping a user. Below is your memory from a long conversation with the user.\n"
              "=== MEMORY ===\n" + context + "\n=== END MEMORY ===\n\n"
              "Using ONLY the memory above, answer the user's question.\n"
              "Question: " + query + "\n"
              "List each applicable constraint verbatim, exactly as it appears in the memory.\nAnswer:")
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": GEN_MODEL, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0, "seed": 42}}, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "").strip()

TOKEN_RE = re.compile(r"[A-Za-z0-9°%/\.]+")
def behavior_score(answer, target):
    old_v, new_v = target["old_value"], target["new_value"]
    ot, nt = set(TOKEN_RE.findall(old_v)), set(TOKEN_RE.findall(new_v))
    old_only, new_only = sorted(ot - nt), sorted(nt - ot)
    def tok_in(tok): return re.search(re.escape(tok), answer) is not None
    return {"gt_id": target["gt_id"],
            "ans_new_verbatim": new_v in answer, "ans_old_verbatim": old_v in answer,
            "new_only_tokens": new_only, "old_only_tokens": old_only,
            "new_tokens_hit": [t for t in new_only if tok_in(t)],
            "old_tokens_hit": [t for t in old_only if tok_in(t)]}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=str, default=None)
    args = ap.parse_args()
    dirs = [Path("scenarios/holdout_v4"),
            Path("scenarios/holdout_v5")]
    scenarios = []
    for d in dirs:
        scenarios += sorted(d.glob("U0*.json"))
    if args.smoke:
        scenarios = [s for s in scenarios if args.smoke in s.name]
    print(f"TSRead arm: {len(scenarios)} scenarios | {datetime.now().isoformat()}")
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
                    ts_filter(mem.read(query).context_text)
            context = ts_filter(mem.read(query).context_text)
            replay_s = time.perf_counter() - t0
            answer = gen_answer(context, query)
            r = {"scenario_id": sc["scenario_id"], "scenario_type": sc.get("scenario_type", ""),
                 "total_turns": sc["total_turns"],
                 "recall": sum(1 for g in gt if g in context) / len(gt) if gt else 0.0,
                 "found": sum(1 for g in gt if g in context), "total_gt": len(gt),
                 "ccr": round(1 - len(context) / raw_len, 4) if raw_len else 0.0,
                 "context_ua": [{"gt_id": tg["gt_id"], "has_new": tg["new_value"] in context,
                                 "has_old": tg["old_value"] in context} for tg in ua_targets],
                 "behavior": [behavior_score(answer, tg) for tg in ua_targets],
                 "context_text": context, "answer_text": answer,
                 "event_counts": {}, "supersession_events": [],
                 "replay_s": round(replay_s, 1)}
            cu = r["context_ua"]
            ok = sum(1 for x in cu if x["has_new"] and not x["has_old"])
            bh = r["behavior"]
            bok = sum(1 for x in bh if x["new_tokens_hit"] and not x["old_tokens_hit"])
            print(f" R={r['recall']:.3f} ctxUA={ok}/{len(cu)} behUA*={bok}/{len(bh)} t={r['replay_s']}s")
            results.append(r)
            del mem
        except Exception as e:
            print(f" ERROR {e}")
            traceback.print_exc()
            results.append({"scenario_id": sc["scenario_id"], "error": str(e)})
        gc.collect()
        out = OUT if not args.smoke else OUT.with_name("behavior_TSRead_smoke.json")
        with open(out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"TSREAD ARM DONE {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
