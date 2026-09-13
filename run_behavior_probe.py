"""Behavior-level UA runner (all configurations, one replay per arm).

Adds to the canonical protocol of run_holdout_v4.py (measurement_turns mid-run reads included):
- Ollama version guard (aborts on a stack mismatch).
- argparse: --models/--suffix/--llm/--scenario-dir/--rag-topk/--query-reform/--all-types/--no-answers.
- --answer-prompts p0,p1,p2: score several answer prompts from a single replay. answer_text/behavior hold the
  first style (p0 = the prompt of Fig. 3(c), byte-identical to the reported runs, schema-compatible with earlier
  outputs); answers_by_prompt/behavior_by_prompt hold every style. Replay and scoring logic are unchanged.
- Each record carries runner_sha256 and answer_prompts.
"""
import json, time, gc, sys, re, traceback, argparse, hashlib
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

import requests
from engine.naive_memory import NaiveMemory
from engine.rag_memory import RagMemory
from engine.epsem_memory import EpSemMemory
from engine.epsem_filtered import EpSemFiltered
from engine.epsem_supersession import EpSemSupersession
from engine.event_emitter import EventEmitter
from engine.base import Message
from lww_memory import LWWMemory

MODELS = [
    ("EpSemSupersession", EpSemSupersession),
    ("LWWMemory", LWWMemory),
    ("NaiveMemory", NaiveMemory),
    ("RagMemory", RagMemory),
    ("EpSemMemory", EpSemMemory),
    ("EpSemFiltered", EpSemFiltered),
]

OUT_DIR = Path(__file__).resolve().parent / "results"

RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# answer-prompt styles for the prompt ablation. p0 == the Fig. 3(c) prompt, byte-identical to the reported runs.
PROMPTS = {
    "p0": ("Using ONLY the memory above, answer the user's question.\n"
           "Question: {q}\n"
           "List each applicable constraint verbatim, exactly as it appears in the memory.\n"
           "Answer:"),
    "p1": ("Using ONLY the memory above, answer the user's question in your own words.\n"
           "Question: {q}\n"
           "Answer:"),
    "p2": ("Using ONLY the memory above, answer the user's question.\n"
           "Question: {q}\n"
           "Answer:"),
}
OLLAMA = "http://localhost:11434"
PINNED_VERSION = "0.24.0"

def version_guard():
    try:
        v = requests.get(f"{OLLAMA}/api/version", timeout=10).json().get("version", "?")
    except Exception as e:
        print(f"FATAL: cannot reach the ollama server ({e})"); sys.exit(2)
    if v != PINNED_VERSION:
        print(f"FATAL: ollama version {v} != pinned {PINNED_VERSION}"); sys.exit(2)
    print(f"[guard] ollama {v} OK")

def gen_answer(context, query, gen_model, style="p0"):
    prompt = (
        "You are an assistant helping a user. Below is your memory from a long conversation with the user.\n"
        "=== MEMORY ===\n" + context + "\n=== END MEMORY ===\n\n"
        + PROMPTS[style].format(q=query)
    )
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": gen_model, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0, "seed": 42}},
                      timeout=600)
    r.raise_for_status()
    return r.json().get("response", "").strip()

TOKEN_RE = re.compile(r"[A-Za-z0-9°%/\.]+")

def behavior_score(answer, target):
    old_v, new_v = target["old_value"], target["new_value"]
    ot, nt = set(TOKEN_RE.findall(old_v)), set(TOKEN_RE.findall(new_v))
    old_only, new_only = sorted(ot - nt), sorted(nt - ot)
    def tok_in(tok):
        return re.search(re.escape(tok), answer) is not None
    return {
        "gt_id": target["gt_id"],
        "ans_new_verbatim": new_v in answer,
        "ans_old_verbatim": old_v in answer,
        "new_only_tokens": new_only,
        "old_only_tokens": old_only,
        "new_tokens_hit": [t for t in new_only if tok_in(t)],
        "old_tokens_hit": [t for t in old_only if tok_in(t)],
    }

def compute_ua(context_text, ua_targets):
    return [{"gt_id": t["gt_id"], "has_new": t["new_value"] in context_text,
             "has_old": t["old_value"] in context_text} for t in ua_targets]

def reform_query(sc):
    dom = sc.get("domain", "") or "project"
    return f"Important requirement: current {dom} constraints and specifications"

def run_one(model_name, ModelClass, sc, args):
    turns, query, gt = sc["turns"], sc["query"], sc["ground_truth_constraints"]
    mts = set(sc["evaluation_protocol"]["measurement_turns"])
    ua_targets = sc["evaluation_protocol"].get("ua_targets", [])
    raw_len = sum(len(t["content"]) for t in turns)

    if model_name in ("EpSemMemory", "EpSemFiltered", "EpSemSupersession", "LWWMemory"):
        mem = ModelClass(db_path=":memory:")
    else:
        mem = ModelClass()
    if args.llm and hasattr(mem, "_model"):
        mem._model = args.llm
    if args.rag_topk and model_name == "RagMemory":
        mem._top_k = args.rag_topk
    emitter = None
    if hasattr(mem, "event_emitter"):
        emitter = EventEmitter()
        mem.event_emitter = emitter

    read_query = reform_query(sc) if args.query_reform else query

    t0 = time.perf_counter()
    for t in turns:
        mem.write(Message(role=t["role"], content=t["content"], turn=t["turn"],
                          is_constraint=t.get("is_constraint", False),
                          is_distractor=t.get("is_distractor", False)))
        if t["turn"] in mts:
            mem.read(read_query)

    context = mem.read(read_query).context_text
    replay_s = time.perf_counter() - t0

    gen_model = args.llm or "qwen2.5:14b-instruct-q4_K_M"
    styles = [s.strip() for s in args.answer_prompts.split(",") if s.strip()]
    answers = {}
    if not args.no_answers:
        for st in styles:
            answers[st] = gen_answer(context, query, gen_model, st)
    answer = answers.get(styles[0], "") if answers else ""
    found = [g for g in gt if g in context]

    ev_counts, ev_super = {}, []
    if emitter:
        for e in emitter.get_events():
            et = e["event_type"]
            ev_counts[et] = ev_counts.get(et, 0) + 1
            if any(k in et.lower() for k in ("supersede", "supersession", "retire",
                                             "confirm_demotion", "pre_detection", "template_sub",
                                             "extraction", "substitution", "conflict")):
                ev_super.append({"turn": e["turn"], "type": et, "data": e["data"][:400]})

    result = {
        "scenario_id": sc["scenario_id"],
        "version": sc.get("version", ""),
        "scenario_type": sc.get("scenario_type", ""),
        "total_turns": sc["total_turns"],
        "recall": len(found) / len(gt) if gt else 0.0,
        "found": len(found), "total_gt": len(gt),
        "ccr": round(1 - len(context) / raw_len, 4) if raw_len else 0.0,
        "context_ua": compute_ua(context, ua_targets),
        "behavior": [behavior_score(answer, t) for t in ua_targets] if answer else [],
        "context_text": context,
        "answer_text": answer,
        "answer_prompts": styles,
        "answers_by_prompt": answers,
        "behavior_by_prompt": {st: [behavior_score(a, t) for t in ua_targets] for st, a in answers.items()},
        "runner_sha256": RUNNER_SHA256,
        "event_counts": ev_counts,
        "supersession_events": ev_super,
        "replay_s": round(replay_s, 1),
    }
    del mem
    gc.collect()
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=None, help="comma-separated configuration filter")
    ap.add_argument("--suffix", type=str, default="", help="output file suffix (e.g. rerun, phi3, imp)")
    ap.add_argument("--llm", type=str, default=None, help="LLM override (e.g. phi3:medium)")
    ap.add_argument("--scenario-dir", type=str, default=None, help="scenario directory override")
    ap.add_argument("--rag-topk", type=int, default=None)
    ap.add_argument("--query-reform", action="store_true")
    ap.add_argument("--all-types", action="store_true", help="all scenario types, not only U0*")
    ap.add_argument("--no-answers", action="store_true", help="skip answer generation (store-level metrics only)")
    ap.add_argument("--answer-prompts", type=str, default="p0",
                    help="comma-separated answer-prompt styles (p0=reported, p1=own-words, p2=neutral)")
    args = ap.parse_args()
    for st in args.answer_prompts.split(","):
        assert st.strip() in PROMPTS, f"unknown prompt style {st}"
    print(f"[runner] sha256={RUNNER_SHA256[:16]} answer_prompts={args.answer_prompts}")

    version_guard()

    if args.scenario_dir:
        dirs = [Path(args.scenario_dir)]
    else:
        dirs = [Path(__file__).resolve().parent / "scenarios" / "holdout_v4",
                Path(__file__).resolve().parent / "scenarios" / "holdout_v5"]
    pattern = "*.json" if args.all_types else "U0*.json"
    scenarios = []
    for d in dirs:
        scenarios += sorted(d.glob(pattern))
    models = MODELS
    if args.models:
        names = [x.strip() for x in args.models.split(",")]
        models = [(n, c) for n, c in MODELS if n in names]

    print(f"scenarios: {len(scenarios)} | models: {[n for n,_ in models]} | suffix='{args.suffix}' "
          f"| llm={args.llm} | start {datetime.now().isoformat()}")

    for model_name, ModelClass in models:
        sfx = f"_{args.suffix}" if args.suffix else ""
        out_path = OUT_DIR / f"behavior_{model_name}{sfx}.json"
        results = []
        print(f"\n===== {model_name}{sfx} | {datetime.now().isoformat()} =====")
        for i, p in enumerate(scenarios):
            with open(p) as f:
                sc = json.load(f)
            print(f"  [{i+1}/{len(scenarios)}] {sc['scenario_id']} ({sc['total_turns']}T) ...",
                  end="", flush=True)
            try:
                r = run_one(model_name, ModelClass, sc, args)
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
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"saved {out_path}")

    print(f"\nSTAGE DONE {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
