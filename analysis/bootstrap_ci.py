# -*- coding: utf-8 -*-
"""Scenario-level bootstrap 95% CIs for recall and CCR from the canonical per-scenario data."""
import csv, random, sys
sys.stdout.reconfigure(encoding="utf-8")

SRC = "figures/data/figure6_ccr_recall.csv"
rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
assert len(rows) == 360

data = {}  # model -> scenario_id -> (recall, ccr)
for r in rows:
    data.setdefault(r["model"], {})[r["scenario_id"]] = (float(r["recall"]), float(r["ccr"]))

scen = sorted(data["EpSem-SS"].keys())
assert len(scen) == 72
for m in data: assert sorted(data[m].keys()) == scen  # pairing integrity

B = 10000
rng = random.Random(20260812)
idx = list(range(72))
samples = [[rng.choice(idx) for _ in range(72)] for _ in range(B)]  # shared resamples (consistent with paired differences)

def ci(vals):
    means = sorted(sum(vals[i] for i in s) / 72 for s in samples)
    return means[249], means[9749]  # percentile 2.5 / 97.5

out = {}
for m in ["EpSem", "EpSem-F", "EpSem-SS"]:
    rec = [data[m][s][0] for s in scen]
    cc  = [data[m][s][1] for s in scen]
    out[m] = (sum(rec)/72, ci(rec), sum(cc)/72, ci(cc))

# paired differences: SS-F recall gain, F-SS CCR cost
gain = [data["EpSem-SS"][s][0] - data["EpSem-F"][s][0] for s in scen]
cost = [data["EpSem-F"][s][1] - data["EpSem-SS"][s][1] for s in scen]

for m, (r, rci, c, cci) in out.items():
    print(f"{m:9s} recall {r:.4f} CI [{rci[0]:.3f}, {rci[1]:.3f}]   CCR {c:.4f} CI [{cci[0]:.3f}, {cci[1]:.3f}]")
print(f"SS-F recall gain {sum(gain)/72:.4f} CI [{ci(gain)[0]:.3f}, {ci(gain)[1]:.3f}]")
print(f"F-SS CCR cost    {sum(cost)/72:.4f} CI [{ci(cost)[0]:.3f}, {ci(cost)[1]:.3f}]")
