"""Similarity analysis, stage 2: pairwise nomic-embed-text cosine measurement. Version guard built in."""
import json, sys, math
import requests

sys.stdout.reconfigure(line_buffering=True)
OLLAMA = "http://localhost:11434"
_v = requests.get(f"{OLLAMA}/api/version", timeout=10).json().get("version", "?")
if _v != "0.24.0":
    print(f"FATAL: ollama {_v} != 0.24.0"); sys.exit(2)
print(f"[guard] ollama {_v} OK")

def embed(text):
    r = requests.post(f"{OLLAMA}/api/embeddings",
                      json={"model": "nomic-embed-text", "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]

def cos(a, b):
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a)); db = math.sqrt(sum(y*y for y in b))
    return num / (da*db) if da and db else 0.0

pairs = json.load(open("dp2_pairs.json"))
cache = {}
for i, p in enumerate(pairs):
    for key in ("orig", "upd"):
        t = p[key]
        if t not in cache:
            cache[t] = embed(t)
    p["sim"] = round(cos(cache[p["orig"]], cache[p["upd"]]), 4)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(pairs)}")

json.dump(pairs, open("results/dp2_sims.json", "w"),
          ensure_ascii=False, indent=1)
print(f"DONE {len(pairs)} pairs, {len(cache)} unique texts embedded")
