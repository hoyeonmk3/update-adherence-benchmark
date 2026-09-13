"""Experiment D runner: EpSem-SS with an alternative embedder and calibrated (or fixed-transplanted) thresholds.
Store-level replay only (no answer generation). Protocol identical to run_behavior_probe (measurement_turns mid-run reads).
  --embedder bge|gte   --calib quantile|fixed   --sets uc,fn
Outputs results/altemb_{embedder}_{calib}_{set}.json with per-scenario recall/CCR/context_ua/events + provenance."""
import json, time, gc, sys, argparse, hashlib, traceback
from pathlib import Path
from datetime import datetime
ROOT = str(Path(__file__).resolve().parent); sys.path.insert(0, ROOT)
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, requests
from engine.epsem_supersession_alt import EpSemSupersessionAlt
from engine.event_emitter import EventEmitter
from engine.base import Message
from sentence_transformers import SentenceTransformer
RUNNER_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ENGINE_ALT_SHA = hashlib.sha256(Path(ROOT, "engine/epsem_supersession_alt.py").read_bytes()).hexdigest()
ST_MODELS = {"bge": "BAAI/bge-small-en-v1.5", "gte": "thenlper/gte-small"}
PINNED = "0.24.0"; OUT = Path(ROOT, "results")

class AltEmbedSS(EpSemSupersessionAlt):
    st_model = None; th = None; cache = {}
    def _get_embedding(self, text):
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in AltEmbedSS.cache:
            AltEmbedSS.cache[h] = np.asarray(AltEmbedSS.st_model.encode([text], normalize_embeddings=False, show_progress_bar=False)[0], dtype=np.float32)
        return AltEmbedSS.cache[h]
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        t = AltEmbedSS.th
        self._supersession_threshold = t["supersession"]; self._th_explicit = t["explicit"]
        self._th_soft = t["soft"]; self._conflict_threshold = t["conflict"]; self._th_dup = t["dup"]; self._th_reins = t["reins"]

def compute_ua(ctx, targets):
    return [{"gt_id": t["gt_id"], "has_new": t["new_value"] in ctx, "has_old": t["old_value"] in ctx} for t in targets]

def run_one(sc):
    turns, query, gt = sc["turns"], sc["query"], sc["ground_truth_constraints"]
    mts = set(sc["evaluation_protocol"]["measurement_turns"]); targets = sc["evaluation_protocol"].get("ua_targets", [])
    raw_len = sum(len(t["content"]) for t in turns)
    mem = AltEmbedSS(db_path=":memory:"); em = EventEmitter(); mem.event_emitter = em
    t0 = time.perf_counter()
    for t in turns:
        mem.write(Message(role=t["role"], content=t["content"], turn=t["turn"], is_constraint=t.get("is_constraint", False), is_distractor=t.get("is_distractor", False)))
        if t["turn"] in mts: mem.read(query)
    ctx = mem.read(query).context_text; el = time.perf_counter() - t0
    found = [g for g in gt if g in ctx]
    counts, sup = {}, []
    for e in em.get_events():
        counts[e["event_type"]] = counts.get(e["event_type"], 0) + 1
        if e["event_type"] in ("supersession", "firmness_demotion"): sup.append({"turn": e["turn"], "type": e["event_type"], "data": e["data"][:400]})
    del mem; gc.collect()
    return {"scenario_id": sc["scenario_id"], "scenario_type": sc.get("scenario_type", ""), "total_turns": sc["total_turns"],
            "recall": len(found) / len(gt) if gt else 0.0, "found": len(found), "total_gt": len(gt), "gt_missing": [g for g in gt if g not in ctx],
            "ccr": round(1 - len(ctx) / raw_len, 4) if raw_len else 0.0, "context_ua": compute_ua(ctx, targets), "context_text": ctx,
            "event_counts": counts, "supersession_events": sup, "replay_s": round(el, 1)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--embedder", required=True, choices=list(ST_MODELS)); ap.add_argument("--calib", required=True, choices=["quantile", "fixed"]); ap.add_argument("--sets", default="uc,fn")
    a = ap.parse_args()
    v = requests.get("http://localhost:11434/api/version", timeout=10).json().get("version"); assert v == PINNED, f"ollama {v} != {PINNED}"
    cal = json.load(open(OUT / "calib_map.json"))
    th = cal["map"][a.embedder] if a.calib == "quantile" else dict(cal["thresholds_nomic"])
    AltEmbedSS.th = th; AltEmbedSS.st_model = SentenceTransformer(ST_MODELS[a.embedder], device="cpu")
    print(f"[runner] sha={RUNNER_SHA[:16]} engine_alt={ENGINE_ALT_SHA[:16]} embedder={ST_MODELS[a.embedder]} calib={a.calib} thresholds={th} ollama={v} start={datetime.now().isoformat()}")
    base = Path(ROOT) / "scenarios"
    for s in [x.strip() for x in a.sets.split(",")]:
        pat = "U0*.json" if s == "uc" else "[FN]0*.json"
        files = sorted(base.glob(f"holdout_v4/{pat}")) + sorted(base.glob(f"holdout_v5/{pat}"))
        out = OUT / f"altemb_{a.embedder}_{a.calib}_{s}.json"; res = []
        print(f"===== {s} ({len(files)} scenarios) -> {out.name}")
        for i, f in enumerate(files):
            sc = json.load(open(f)); print(f"  [{i+1}/{len(files)}] {sc['scenario_id']} ({sc['total_turns']}T) ...", end="", flush=True)
            try:
                r = run_one(sc); r.update({"runner_sha256": RUNNER_SHA, "engine_alt_sha256": ENGINE_ALT_SHA, "embedder": ST_MODELS[a.embedder], "calib": a.calib, "thresholds": th})
                ok = sum(1 for x in r["context_ua"] if x["has_new"] and not x["has_old"])
                print(f" R={r['recall']:.3f} ctxUA={ok}/{len(r['context_ua'])} sup={r['event_counts'].get('supersession',0)}+{r['event_counts'].get('firmness_demotion',0)} t={r['replay_s']}s"); res.append(r)
            except Exception as e:
                print(" ERROR", e); traceback.print_exc(); res.append({"scenario_id": sc["scenario_id"], "error": str(e)})
            json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print("DONE", datetime.now().isoformat())
if __name__ == "__main__": main()
