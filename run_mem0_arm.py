"""Mem0 external-system arm — 24 Update Conflict scenarios, 10-turn batched add, store+behavior scoring.

- mem0ai 2.0.17 (pinned), LLM=qwen2.5:14b temp0, embedder=nomic (768), local qdrant.
- Protocol parity: 10-turn batched add (same cadence as the EpSem compaction interval),
  mid-run search at measurement_turns.
- Caveat: mem0 does not expose a generation seed, so this arm is a single run. Verbatim
  recall is disadvantaged by mem0's paraphrased storage; behavior-level UA is the primary
  comparison.
- Output schema matches behavior_*.json (analysis scripts are reused as-is).
"""
import json, time, gc, sys, re, shutil, traceback
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
import requests
from mem0 import Memory

OLLAMA = "http://localhost:11434"
PINNED = "0.24.0"
_v = requests.get(f"{OLLAMA}/api/version", timeout=10).json().get("version", "?")
if _v != PINNED:
    print(f"FATAL: ollama {_v} != {PINNED}"); sys.exit(2)
print(f"[guard] ollama {_v} OK | mem0ai 2.0.17")

BATCH = 10
SEARCH_K = 20
STORE_DIR = "mem0_store_run"
OUT = Path("results/behavior_Mem0.json")
GEN_MODEL = "qwen2.5:14b-instruct-q4_K_M"

def make_memory(collection):
    config = {
        "llm": {"provider": "ollama", "config": {"model": GEN_MODEL, "temperature": 0, "ollama_base_url": OLLAMA}},
        "embedder": {"provider": "ollama", "config": {"model": "nomic-embed-text", "ollama_base_url": OLLAMA, "embedding_dims": 768}},
        "vector_store": {"provider": "qdrant", "config": {"path": STORE_DIR, "collection_name": collection, "embedding_model_dims": 768, "on_disk": True}},
    }
    return Memory.from_config(config)

def gen_answer(context, query):
    prompt = (
        "You are an assistant helping a user. Below is your memory from a long conversation with the user.\n"
        "=== MEMORY ===\n" + context + "\n=== END MEMORY ===\n\n"
        "Using ONLY the memory above, answer the user's question.\n"
        "Question: " + query + "\n"
        "List each applicable constraint verbatim, exactly as it appears in the memory.\n"
        "Answer:"
    )
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

def assemble(mem, query, uid):
    s = mem.search(query, filters={"user_id": uid}, limit=SEARCH_K)
    items = s.get("results", s) if isinstance(s, dict) else s
    return "\n".join(f"- {it['memory']}" for it in items)

def run_one(sc):
    turns, query, gt = sc["turns"], sc["query"], sc["ground_truth_constraints"]
    mts = set(sc["evaluation_protocol"]["measurement_turns"])
    ua_targets = sc["evaluation_protocol"].get("ua_targets", [])
    raw_len = sum(len(t["content"]) for t in turns)
    uid = sc["scenario_id"]
    coll = re.sub(r"[^A-Za-z0-9]", "_", uid)
    mem = make_memory(coll)

    t0 = time.perf_counter()
    batch = []
    for t in turns:
        batch.append({"role": t["role"] if t["role"] in ("user", "assistant") else "user",
                      "content": t["content"]})
        if len(batch) >= BATCH:
            mem.add(messages=batch, user_id=uid)
            batch = []
        if t["turn"] in mts:
            assemble(mem, query, uid)
    if batch:
        mem.add(messages=batch, user_id=uid)
    context = assemble(mem, query, uid)
    replay_s = time.perf_counter() - t0
    answer = gen_answer(context, query)

    return {
        "scenario_id": uid, "scenario_type": sc.get("scenario_type", ""),
        "total_turns": sc["total_turns"],
        "recall": sum(1 for g in gt if g in context) / len(gt) if gt else 0.0,
        "found": sum(1 for g in gt if g in context), "total_gt": len(gt),
        "ccr": round(1 - len(context) / raw_len, 4) if raw_len else 0.0,
        "context_ua": [{"gt_id": tg["gt_id"], "has_new": tg["new_value"] in context,
                        "has_old": tg["old_value"] in context} for tg in ua_targets],
        "behavior": [behavior_score(answer, tg) for tg in ua_targets],
        "context_text": context, "answer_text": answer,
        "event_counts": {}, "supersession_events": [],
        "replay_s": round(replay_s, 1),
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=str, default=None)
    args = ap.parse_args()

    shutil.rmtree(STORE_DIR, ignore_errors=True)
    dirs = [Path("scenarios/holdout_v4"),
            Path("scenarios/holdout_v5")]
    scenarios = []
    for d in dirs:
        scenarios += sorted(d.glob("U0*.json"))
    if args.smoke:
        scenarios = [s for s in scenarios if args.smoke in s.name]

    print(f"Mem0 arm: {len(scenarios)} scenarios | {datetime.now().isoformat()}")
    results = []
    for i, p in enumerate(scenarios):
        with open(p) as f:
            sc = json.load(f)
        print(f"  [{i+1}/{len(scenarios)}] {sc['scenario_id']} ({sc['total_turns']}T) ...", end="", flush=True)
        try:
            r = run_one(sc)
            cu = r["context_ua"]
            ok = sum(1 for x in cu if x["has_new"] and not x["has_old"])
            bh = r["behavior"]
            bok = sum(1 for x in bh if x["new_tokens_hit"] and not x["old_tokens_hit"])
            print(f" R={r['recall']:.3f} ctxUA={ok}/{len(cu)} behUA*={bok}/{len(bh)} t={r['replay_s']}s")
            results.append(r)
        except Exception as e:
            print(f" ERROR {e}")
            traceback.print_exc()
            results.append({"scenario_id": sc["scenario_id"], "error": str(e)})
        gc.collect()
        out = OUT if not args.smoke else OUT.with_name("behavior_Mem0_smoke.json")
        with open(out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"MEM0 ARM DONE {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
