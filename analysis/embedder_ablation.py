# -*- coding: utf-8 -*-
"""Embedder-replacement analysis: nomic-embed-text (frozen operating space) vs bge-small-en-v1.5 / gte-small.
   1) cross-check dp2_sims against the figure 8 CSV
   2) recompute pair cosines with alternative embedders (no prefixes, raw text, matching the engine)
   3) Spearman rank correlation / same-count recalibrated threshold / detected-set overlap / 70-event rule agreement
"""
import json, csv, math, sys
sys.stdout.reconfigure(encoding="utf-8")

SP = "."
FIG8 = "figures/data/figure8_similarity.csv"

pairs = json.load(open("results/dp2_sims.json", encoding="utf-8"))
assert len(pairs) == 70

# ---- 1) cross-check against figure 8 CSV ----
fig8 = list(csv.DictReader(open(FIG8, encoding="utf-8-sig")))
assert len(fig8) == 70
key = lambda s, g, c: (s, g, c)
fmap = {key(r["scenario"], r["gt_id"], r["condition"]): (float(r["similarity"]), r["outcome"]) for r in fig8}
mismatch = 0
for p in pairs:
    k = key(p["scenario"], p["gt_id"], p["cond"])
    assert k in fmap, k
    sim_f, out_f = fmap[k]
    if abs(sim_f - p["sim"]) > 1e-6 or out_f != p["outcome"]:
        mismatch += 1
print(f"[check] dp2_sims vs figure8 CSV: 70/70 keys matched, {mismatch} value mismatches")
assert mismatch == 0

explicit = [p for p in pairs if p["cond"] == "explicit"]
print(f"[check] {len(explicit)} explicit events, nomic >= 0.92 = {sum(1 for p in explicit if p['sim'] >= 0.92)} (expected 14)")

# ---- 2) alternative embedders ----
from sentence_transformers import SentenceTransformer
texts = sorted({t for p in pairs for t in (p["orig"], p["upd"])})
print(f"[embed] {len(texts)} unique texts")

def cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    return num / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k2 in range(i, j + 1):
                r[order[k2]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den

results = {}
for model_name in ["BAAI/bge-small-en-v1.5", "thenlper/gte-small"]:
    print(f"[model] loading and embedding with {model_name}...")
    m = SentenceTransformer(model_name, device="cpu")
    embs = {t: list(map(float, e)) for t, e in zip(texts, m.encode(texts, normalize_embeddings=False, show_progress_bar=False))}
    sims = [cos(embs[p["orig"]], embs[p["upd"]]) for p in pairs]

    rho_all = spearman([p["sim"] for p in pairs], sims)
    exp_idx = [i for i, p in enumerate(pairs) if p["cond"] == "explicit"]
    rho_exp = spearman([pairs[i]["sim"] for i in exp_idx], [sims[i] for i in exp_idx])

    # same-count (14) recalibrated operating point on the 45 explicit events
    exp_sims = sorted(((sims[i], i) for i in exp_idx), reverse=True)
    top14 = {i for _, i in exp_sims[:14]}
    th_lo, th_hi = exp_sims[14][0], exp_sims[13][0]  # rank-15 < th <= rank-14
    nomic14 = {i for i in exp_idx if pairs[i]["sim"] >= 0.92}
    overlap = len(top14 & nomic14)

    # 70-event rule agreement: above recalibrated threshold <-> outcome C (nomic baseline 69/70)
    th = (th_lo + th_hi) / 2
    agree = sum(1 for i, p in enumerate(pairs) if (sims[i] >= th) == (p["outcome"] == "C"))

    results[model_name] = dict(rho_all=rho_all, rho_exp=rho_exp, th=th, overlap=overlap, agree=agree,
                               sims={f"{p['scenario']}|{p['gt_id']}|{p['cond']}": round(sims[i], 4) for i, p in enumerate(pairs)})
    print(f"  Spearman(70)={rho_all:.4f}  Spearman(exp45)={rho_exp:.4f}")
    print(f"  recalibrated threshold (14 detections)={th:.4f} [{th_lo:.4f}, {th_hi:.4f}]  detected-set overlap={overlap}/14  rule agreement={agree}/70")

json.dump(results, open("analysis/embedder_ablation_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved: analysis/embedder_ablation_results.json")
