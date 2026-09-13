"""Experiment D: quantile-map the engine's five similarity thresholds from the nomic space into alternative
embedding spaces, using a calibration corpus that contains NO update events: constraint-like user turns from the
Forgetting/Noise scenarios (holdout_v4 + holdout_v5, F0*/N0*), all within-scenario pairs.
Outputs calib_map.json (threshold map + distribution stats) and calib_sims.npz (raw pair similarities)."""
import json, glob, re, hashlib, itertools, sys, time
import numpy as np, requests
from pathlib import Path
ROOT = Path(__file__).resolve().parent; OUT = ROOT / "results"
OLLAMA = "http://localhost:11434"; NOMIC = "nomic-embed-text"
ST_MODELS = {"bge": "BAAI/bge-small-en-v1.5", "gte": "thenlper/gte-small"}
THRESH = {"supersession": 0.92, "explicit": 0.70, "soft": 0.80, "conflict": 0.85, "dup": 0.97, "reins": 0.95}
KEY = re.compile(r"requirement|guideline|must|shall|ideally|historical|preference|exceed|at least|within", re.I)
files = sorted(glob.glob(str(ROOT / "scenarios/holdout_v4/[FN]0*.json")) + glob.glob(str(ROOT / "scenarios/holdout_v5/[FN]0*.json")))
assert len(files) == 48, len(files)
scen = {}
for f in files:
    d = json.load(open(f))
    cand = [t["content"] for t in d["turns"] if t["role"] == "user" and (t.get("is_constraint") or (t.get("is_distractor") and re.search(r"\d", t["content"]) and KEY.search(t["content"])))]
    scen[d["scenario_id"]] = sorted(set(cand))
texts = sorted({t for v in scen.values() for t in v})
print(f"scenarios {len(scen)} | unique texts {len(texts)} | pairs {sum(len(v)*(len(v)-1)//2 for v in scen.values())}")
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
# nomic via ollama (same endpoint/model as the engine)
t0 = time.time(); nomic = {}
for i, t in enumerate(texts):
    r = requests.post(f"{OLLAMA}/api/embeddings", json={"model": NOMIC, "prompt": t}, timeout=60); r.raise_for_status()
    nomic[t] = np.array(r.json()["embedding"], dtype=np.float32)
    if i % 300 == 0: print(f"  nomic {i}/{len(texts)} {time.time()-t0:.0f}s")
from sentence_transformers import SentenceTransformer
alt = {}
for k, name in ST_MODELS.items():
    m = SentenceTransformer(name, device="cpu")
    E = m.encode(texts, normalize_embeddings=False, show_progress_bar=False, batch_size=64)
    alt[k] = {t: np.asarray(e, dtype=np.float32) for t, e in zip(texts, E)}; print("  encoded", k, E.shape)
sims = {"nomic": [], "bge": [], "gte": []}
for sid, v in scen.items():
    for a, b in itertools.combinations(v, 2):
        sims["nomic"].append(cos(nomic[a], nomic[b]))
        for k in ST_MODELS: sims[k].append(cos(alt[k][a], alt[k][b]))
S = {k: np.array(v) for k, v in sims.items()}
cal = {"corpus": {"scenarios": len(scen), "texts": len(texts), "pairs": int(len(S["nomic"]))}, "thresholds_nomic": THRESH, "map": {}, "frac_above": {}}
for k in ST_MODELS:
    cal["map"][k] = {}; cal["frac_above"][k] = {}
    for name, th in THRESH.items():
        q = float(np.mean(S["nomic"] < th))            # quantile of the nomic threshold in the calibration corpus
        th_alt = float(np.quantile(S[k], q))            # same quantile in the alternative space
        cal["map"][k][name] = round(th_alt, 4)
        cal["frac_above"][k][name] = {"nomic": round(float(np.mean(S['nomic'] >= th)), 5), "alt_at_mapped": round(float(np.mean(S[k] >= th_alt)), 5), "alt_at_fixed": round(float(np.mean(S[k] >= th)), 5)}
    rho = float(np.corrcoef(np.argsort(np.argsort(S["nomic"])), np.argsort(np.argsort(S[k])))[0, 1])
    cal["frac_above"][k]["spearman_vs_nomic"] = round(rho, 4)
cal["percentiles"] = {k: {str(p): round(float(np.percentile(S[k], p)), 4) for p in (50, 90, 95, 99, 99.9)} for k in S}
cal["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
json.dump(cal, open(OUT / "calib_map.json", "w"), indent=1); np.savez_compressed(OUT / "calib_sims.npz", **S)
print(json.dumps(cal["map"], indent=1)); print(json.dumps(cal["frac_above"], indent=1)); print("saved calib_map.json")
