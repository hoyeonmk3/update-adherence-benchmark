"""
Benchmark v4 — Hold-out Runner (ScenarioSpec v1.6)
==================================================
- 36 hold-out scenarios (FORGETTING, NOISE_TOLERANCE, UPDATE_CONFLICT)
- mid-run reads at measurement_turns
- UA judgment from ua_targets
- 5 models x 36 scenarios

Usage:
    python3 run_holdout_v4.py [--smoke F001]
"""
import json, time, signal, faulthandler, sys, gc, resource, traceback
from pathlib import Path
from datetime import datetime

faulthandler.enable()
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, ".")

from engine.naive_memory import NaiveMemory
from engine.rag_memory import RagMemory
from engine.epsem_memory import EpSemMemory
from engine.epsem_filtered import EpSemFiltered
from engine.epsem_supersession import EpSemSupersession
from engine.base import Message

MODELS = {
    "NaiveMemory": NaiveMemory,
    "RagMemory": RagMemory,
    "EpSemMemory": EpSemMemory,
    "EpSemFiltered": EpSemFiltered,
    "EpSemSupersession": EpSemSupersession,
}

SCENARIO_DIR = Path("scenarios/holdout_v4")
RESULTS_FILE = Path(f"results/smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

def save_checkpoint(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def compute_ccr(context_length, raw_conversation_length):
    if raw_conversation_length == 0:
        return 0.0
    return round(1 - (context_length / raw_conversation_length), 3)

def compute_recall_at_turn(context_text, gt_list):
    """Exact match recall. Returns (recall, found, total, missed_list)."""
    if not gt_list:
        return 0.0, 0, 0, []
    found_list = [g for g in gt_list if g in context_text]
    missed_list = [g for g in gt_list if g not in context_text]
    found = len(found_list)
    return found / len(gt_list), found, len(gt_list), missed_list

def compute_ua(context_text, ua_targets):
    """UA judgment: per update, (old NOT IN context) AND (new IN context)."""
    if not ua_targets:
        return None
    results = []
    for target in ua_targets:
        has_new = target["new_value"] in context_text
        has_old = target["old_value"] in context_text
        accurate = has_new and not has_old
        results.append({
            "gt_id": target["gt_id"],
            "has_new": has_new,
            "has_old": has_old,
            "accurate": accurate,
            "category": "ok" if accurate else ("write_bug" if not has_new else "append_only"),
        })
    total = len(results)
    passed = sum(1 for r in results if r["accurate"])
    return {
        "total": total,
        "passed": passed,
        "ratio": passed / total if total > 0 else 0.0,
        "details": results,
    }

def gt_at_turn(gt_metadata, current_turn):
    """Return GT constraints visible at current_turn (inserted_at <= current_turn).
    For updated constraints, return the latest version."""
    visible = {}
    for gt_id, meta in gt_metadata.items():
        if meta["inserted_at"] <= current_turn:
            # If this is an update, it supersedes the original
            base_id = gt_id.replace("_upd", "")
            if "_upd" in gt_id:
                visible[base_id] = meta["text"]  # update supersedes
            elif base_id not in visible:
                visible[base_id] = meta["text"]
    return list(visible.values())


def run_scenario(model_name, ModelClass, scenario_data):
    """Run a single scenario with mid-run checkpoint reads."""
    sc_type = scenario_data["scenario_type"]
    turns = scenario_data["turns"]
    query = scenario_data["query"]
    gt = scenario_data["ground_truth_constraints"]
    gt_meta = scenario_data.get("gt_metadata", {})
    measurement_turns = scenario_data["evaluation_protocol"]["measurement_turns"]
    ua_targets = scenario_data["evaluation_protocol"].get("ua_targets", [])
    raw_conv_length = sum(len(t["content"]) for t in turns)

    # Initialize model
    if model_name in ("EpSemMemory", "EpSemFiltered", "EpSemSupersession"):
        mem = ModelClass(db_path=":memory:")
    else:
        mem = ModelClass()

    mt_set = set(measurement_turns)
    checkpoints = {}

    t0 = time.perf_counter()

    for t in turns:
        msg = Message(
            role=t["role"], content=t["content"], turn=t["turn"],
            is_constraint=t.get("is_constraint", False),
            is_distractor=t.get("is_distractor", False),
        )
        mem.write(msg)

        # Mid-run read at measurement turns
        if t["turn"] in mt_set:
            rr = mem.read(query)
            context = rr.context_text
            # GT visible at this turn
            visible_gt = gt_at_turn(gt_meta, t["turn"]) if gt_meta else gt
            recall, found, total, missed = compute_recall_at_turn(context, visible_gt)
            checkpoints[t["turn"]] = {
                "recall": recall,
                "found": found,
                "total_gt": total,
                "context_length": len(context),
                "missed_gt": missed,
            }

    # Final read (always)
    rr = mem.read(query)
    context = rr.context_text
    wall_clock = time.perf_counter() - t0

    # Final recall (against full ground_truth — latest version)
    final_recall, final_found, final_total, final_missed = compute_recall_at_turn(context, gt)
    ccr = compute_ccr(len(context), raw_conv_length)
    ua_result = compute_ua(context, ua_targets)

    result = {
        "scenario_id": scenario_data["scenario_id"],
        "scenario_type": sc_type,
        "domain": scenario_data.get("domain", ""),
        "total_turns": scenario_data["total_turns"],
        "recall": final_recall,
        "found": final_found,
        "total_gt": final_total,
        "missed_gt": final_missed,
        "ccr": ccr,
        "wall_clock_s": round(wall_clock, 2),
        "context_length": len(context),
        "raw_conv_length": raw_conv_length,
        "checkpoints": checkpoints,
    }
    if ua_result is not None:
        result["ua"] = ua_result

    # Cleanup
    del mem
    gc.collect()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=str, default=None,
                        help="Run single scenario (e.g., F001) for smoke test")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="Comma-separated scenario IDs (e.g., N011,N012)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (e.g., EpSemSupersession)")
    args = parser.parse_args()

    scenarios = sorted(SCENARIO_DIR.glob("*.json"))
    if args.smoke:
        scenarios = [s for s in scenarios if args.smoke in s.name]
    if args.scenarios:
        ids = [x.strip() for x in args.scenarios.split(",")]
        scenarios = [s for s in scenarios if any(sid in s.name for sid in ids)]
    if not scenarios:
        print(f"ERROR: No matching scenarios")
        return

    models_to_run = MODELS
    if args.models:
        model_names = [x.strip() for x in args.models.split(",")]
        models_to_run = {k: v for k, v in MODELS.items() if k in model_names}

    print(f"Found {len(scenarios)} scenarios in {SCENARIO_DIR}", flush=True)
    print(f"Models: {list(models_to_run.keys())}", flush=True)
    print(f"Results: {RESULTS_FILE}", flush=True)
    print(f"Start: {datetime.now().isoformat()}", flush=True)

    all_results = {}

    for model_name, ModelClass in models_to_run.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {model_name} | {datetime.now().isoformat()}", flush=True)
        print(f"{'='*60}", flush=True)

        model_results = []
        for sc_idx, sc_path in enumerate(scenarios):
            with open(sc_path) as f:
                sc = json.load(f)

            scenario_id = sc["scenario_id"]
            total_turns = sc["total_turns"]
            print(f"  [{sc_idx+1}/{len(scenarios)}] {scenario_id} "
                  f"({sc['scenario_type']}, {total_turns}T) ...", end="", flush=True)

            signal.alarm(3600)  # 1h timeout per scenario
            try:
                result = run_scenario(model_name, ModelClass, sc)
                signal.alarm(0)
                print(f" R={result['recall']:.3f} CCR={result['ccr']:.3f} "
                      f"t={result['wall_clock_s']:.1f}s", flush=True)
                if result.get("ua"):
                    print(f"    UA: {result['ua']['passed']}/{result['ua']['total']}", flush=True)
                model_results.append(result)

            except Exception as e:
                signal.alarm(0)
                print(f" ERROR: {e}", flush=True)
                traceback.print_exc()
                model_results.append({
                    "scenario_id": scenario_id,
                    "error": str(e),
                })

            # Checkpoint save after each scenario
            all_results[model_name] = model_results
            save_checkpoint(all_results, RESULTS_FILE)

        print(f"\n{model_name} complete: {len(model_results)} scenarios | "
              f"RSS={mem_mb():.0f}MB", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for model_name, results in all_results.items():
        valid = [r for r in results if "recall" in r]
        if not valid:
            print(f"{model_name}: all errors")
            continue
        avg_recall = sum(r["recall"] for r in valid) / len(valid)
        avg_ccr = sum(r["ccr"] for r in valid) / len(valid)
        ua_results = [r for r in valid if r.get("ua")]
        ua_str = ""
        if ua_results:
            ua_pass = sum(r["ua"]["passed"] for r in ua_results)
            ua_total = sum(r["ua"]["total"] for r in ua_results)
            ua_str = f" UA={ua_pass}/{ua_total}"
        print(f"  {model_name}: R={avg_recall:.3f} CCR={avg_ccr:.3f}{ua_str}", flush=True)

    print(f"\nResults saved: {RESULTS_FILE}", flush=True)
    print(f"End: {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()
