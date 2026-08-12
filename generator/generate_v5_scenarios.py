#!/usr/bin/env python3
"""
Phase 2a: Generate holdout_v5 scenarios (+36 new)
=================================================
Imports generate_scenario() from scenario_generator.py and produces 36 additional
scenarios from a new SCENARIO_CONFIGS set.

- seed range disjoint from v4
- same domain/turn/type distribution
- writes generation_log.json
"""
import json, sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from scenario_generator import generate_scenario, DOMAINS

OUTPUT_DIR = Path("datasets/holdout_v5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# v5 Configs: +36 scenarios (seed 70000~)
# Same structure: F12 + N12 + U12, each 4x200T + 4x500T + 4x1000T
# Differentiated from v4 by distinct parameter combinations
# ─────────────────────────────────────────────

SCENARIO_CONFIGS_V5 = []

# FORGETTING (12) — seed 70000~
_f_raw_v5 = [
    ("F013", 200, 8,  "UNIFORM",    0.5),
    ("F014", 200, 5,  "CLUSTERED",  0.3),
    ("F015", 200, 13, "CLUSTERED",  0.5),
    ("F016", 200, 8,  "LATE_HEAVY", 0.5),
    ("F017", 500, 5,  "LATE_HEAVY", 0.3),
    ("F018", 500, 13, "CLUSTERED",  0.3),
    ("F019", 500, 20, "LATE_HEAVY", 0.5),
    ("F020", 500, 8,  "UNIFORM",    0.5),
    ("F021", 1000, 8,  "LATE_HEAVY", 0.3),
    ("F022", 1000, 20, "CLUSTERED",  0.5),
    ("F023", 1000, 5,  "UNIFORM",    0.5),
    ("F024", 1000, 13, "CLUSTERED",  0.3),
]
for i, (sid, turns, gtc, timing, dd) in enumerate(_f_raw_v5):
    SCENARIO_CONFIGS_V5.append({
        "scenario_id": f"hold-out-{sid}",
        "scenario_type": "FORGETTING",
        "total_turns": turns,
        "gt_count": gtc,
        "gt_timing": timing,
        "distractor_density": dd,
        "domain": DOMAINS[i % len(DOMAINS)],
        "seed": 70000 + i,
    })

# NOISE_TOLERANCE (12) — seed 80000~
# Same density distribution (30/50/70/90% x 3 turn counts)
_n_raw_v5 = [
    ("N013", 200, 8, 0.3), ("N014", 200, 8, 0.5),
    ("N015", 200, 8, 0.7), ("N016", 200, 8, 0.9),
    ("N017", 500, 8, 0.3), ("N018", 500, 8, 0.5),
    ("N019", 500, 8, 0.7), ("N020", 500, 8, 0.9),
    ("N021", 1000, 8, 0.3), ("N022", 1000, 8, 0.5),
    ("N023", 1000, 8, 0.7), ("N024", 1000, 8, 0.9),
]
for i, (sid, turns, gtc, dd) in enumerate(_n_raw_v5):
    SCENARIO_CONFIGS_V5.append({
        "scenario_id": f"hold-out-{sid}",
        "scenario_type": "NOISE_TOLERANCE",
        "total_turns": turns,
        "gt_count": gtc,
        "gt_timing": "UNIFORM",
        "distractor_density": dd,
        "domain": DOMAINS[i % len(DOMAINS)],
        "seed": 80000 + i,
        "noise_checkpoint": 30,
    })

# UPDATE_CONFLICT (12) — seed 90000~
_u_raw_v5 = [
    ("U013", 200, 5,  1, 0.5), ("U014", 200, 13, 1, 0.3),
    ("U015", 200, 5,  2, 0.5), ("U016", 200, 13, 3, 0.5),
    ("U017", 500, 5,  1, 0.5), ("U018", 500, 8,  3, 0.3),
    ("U019", 500, 13, 1, 0.5), ("U020", 500, 5,  2, 0.5),
    ("U021", 1000, 8,  2, 0.3), ("U022", 1000, 5,  1, 0.5),
    ("U023", 1000, 13, 2, 0.5), ("U024", 1000, 8,  1, 0.3),
]
for i, (sid, turns, gtc, updates, dd) in enumerate(_u_raw_v5):
    SCENARIO_CONFIGS_V5.append({
        "scenario_id": f"hold-out-{sid}",
        "scenario_type": "UPDATE_CONFLICT",
        "total_turns": turns,
        "gt_count": gtc,
        "gt_timing": "UNIFORM",
        "distractor_density": dd,
        "domain": DOMAINS[i % len(DOMAINS)],
        "seed": 90000 + i,
        "gt_update_count": updates,
    })

# ─────────────────────────────────────────────
# Generate
# ─────────────────────────────────────────────
gen_log = {
    "generator": "scenario_generator.py (v1.6)",
    "timestamp": datetime.now().isoformat(),
    "version": "v5",
    "n_scenarios": len(SCENARIO_CONFIGS_V5),
    "seed_range": "70000-90011",
    "scenarios": [],
}

stats = {"FORGETTING": 0, "NOISE_TOLERANCE": 0, "UPDATE_CONFLICT": 0}
errors = []

for config in SCENARIO_CONFIGS_V5:
    try:
        scenario = generate_scenario(config)
        sid = scenario["scenario_id"]
        stats[scenario["scenario_type"]] += 1

        out_file = OUTPUT_DIR / f"{sid.replace('hold-out-', '')}_v5.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(scenario, f, ensure_ascii=False, indent=2)

        mt = scenario["evaluation_protocol"]["measurement_turns"]
        gt = scenario["ground_truth_constraints"]
        ua = scenario["evaluation_protocol"].get("ua_targets", [])

        gen_log["scenarios"].append({
            "scenario_id": sid,
            "type": scenario["scenario_type"],
            "turns": scenario["total_turns"],
            "gt_count": len(gt),
            "measurement_turns": len(mt),
            "ua_targets": len(ua),
            "domain": scenario["domain"],
            "seed": config["seed"],
            "distractor_density": config.get("distractor_density", 0.3),
            "status": "OK",
        })

        print(f"  ✓ {sid:25s} | {scenario['scenario_type']:18s} | "
              f"turns={scenario['total_turns']:5d} | gt={len(gt):2d} | "
              f"cp={len(mt):2d} | ua={len(ua):2d} | domain={scenario['domain']}")

    except Exception as e:
        errors.append({"scenario_id": config["scenario_id"], "error": str(e)})
        print(f"  ✗ {config['scenario_id']}: {e}")

# Save generation log
gen_log["stats"] = stats
gen_log["errors"] = errors
with open(OUTPUT_DIR / "generation_log.json", "w") as f:
    json.dump(gen_log, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Generated: {sum(stats.values())} scenarios")
for stype, count in stats.items():
    print(f"  {stype}: {count}")
print(f"Errors: {len(errors)}")
print(f"Output: {OUTPUT_DIR.resolve()}")
print(f"Log: {(OUTPUT_DIR / 'generation_log.json').resolve()}")
