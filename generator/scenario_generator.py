#!/usr/bin/env python3
"""
Scenario Generator — ScenarioSpec v1.6 Contract Implementation
=============================================================
Rule-based skeleton + per-domain constraint/distractor template pools.
Seed-deterministic generation with no LLM calls.

Usage:
    python scenario_generator.py --output-dir ./datasets/holdout_v4/
"""
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# ─────────────────────────────────────────────
# 1. Domain-specific constraint/distractor pools
# ─────────────────────────────────────────────

DOMAINS = ["ENGINEERING", "MEDICAL", "LEGAL", "FINANCE", "GENERAL"]

CONSTRAINT_POOL = {
    "ENGINEERING": [
        "Maximum operating temperature must not exceed {v}°C",
        "Minimum wall thickness shall be {v}mm",
        "Tensile strength must be at least {v} MPa",
        "Operating pressure range is {v1} to {v2} bar",
        "Surface roughness Ra must not exceed {v} μm",
        "Vibration tolerance must be within {v}g RMS",
        "Material hardness must be at least {v} HRC",
        "Thermal conductivity shall exceed {v} W/mK",
        "Fatigue life must exceed {v} cycles",
        "Corrosion resistance rating must be at least {v}",
        "Maximum deflection under load is {v}mm",
        "Bolt torque specification is {v} Nm",
        "Clearance fit tolerance is H{v}/g{v2}",
        "Weld quality must meet ISO {v} Class B",
        "Assembly sequence requires step {v} before step {v2}",
        "Component weight must not exceed {v}kg",
        "Electrical insulation resistance minimum {v} MΩ",
        "Coolant flow rate must be at least {v} L/min",
        "Noise level must not exceed {v} dB at 1m distance",
        "Service interval is every {v} operating hours",
    ],
    "MEDICAL": [
        "Dosage must not exceed {v}mg per kg body weight",
        "Blood pressure target is below {v}/{v2} mmHg",
        "Heart rate alert threshold is above {v} bpm",
        "Medication interval must be at least {v} hours",
        "Patient BMI threshold for procedure is below {v}",
        "Surgical margin must be at least {v}mm",
        "Sterilization temperature must reach {v}°C for {v2} minutes",
        "Blood glucose target range is {v1} to {v2} mg/dL",
        "Ventilator tidal volume is {v}mL per kg ideal body weight",
        "Maximum radiation dose is {v} mSv per session",
        "Hemoglobin threshold for transfusion is below {v} g/dL",
        "Post-operative monitoring interval is every {v} minutes",
        "Drug concentration must remain below {v} ng/mL",
        "Wound closure tension must not exceed {v} N/cm",
        "Recovery room stay minimum is {v} hours",
        "Antibiotic prophylaxis must begin {v} minutes before incision",
        "Fluid resuscitation rate is {v} mL/kg/hour",
        "Pain score reassessment interval is {v} minutes",
        "Infection marker CRP threshold is {v} mg/L",
        "Renal function GFR threshold is above {v} mL/min",
    ],
    "LEGAL": [
        "Contract termination notice period is {v} business days",
        "Liability cap must not exceed {v}% of contract value",
        "Confidentiality obligation extends {v} years after termination",
        "Force majeure notification deadline is {v} hours",
        "Arbitration venue must be in {v} jurisdiction",
        "Indemnification limit is {v} times annual fee",
        "Data retention period is {v} years from collection date",
        "Response time for legal hold notice is {v} business days",
        "Minimum insurance coverage is {v} million USD",
        "Compliance audit frequency is every {v} months",
        "Subcontracting requires approval within {v} days",
        "Intellectual property assignment covers {v} year period",
        "Warranty period extends {v} months from delivery",
        "Dispute resolution must begin within {v} days of notice",
        "Payment terms are net {v} days from invoice",
        "Penalty rate is {v}% per day of delay",
        "Document execution requires {v} authorized signatories",
        "Amendment requires {v} days written notice",
        "Non-compete radius is {v} kilometers",
        "Escrow release conditions require {v} approvals",
    ],
    "FINANCE": [
        "Maximum portfolio exposure to single sector is {v}%",
        "Minimum reserve ratio must be maintained at {v}%",
        "Transaction approval threshold is {v} USD",
        "Credit risk score minimum is {v} points",
        "Dividend payout ratio must not exceed {v}%",
        "Debt-to-equity ratio ceiling is {v}",
        "Quarterly reporting deadline is {v} business days after quarter end",
        "Interest rate cap on variable loans is {v}%",
        "Minimum collateral coverage ratio is {v}x",
        "Foreign exchange exposure limit is {v}% of assets",
        "Trading volume daily limit is {v} million USD",
        "Capital adequacy ratio must exceed {v}%",
        "Loan-to-value ratio maximum is {v}%",
        "Risk-adjusted return threshold is {v}%",
        "Stress test scenario loss limit is {v}%",
        "Customer concentration limit is {v}% of revenue",
        "Operational risk capital allocation is {v}% of gross income",
        "Liquidity coverage ratio must exceed {v}%",
        "Net stable funding ratio minimum is {v}%",
        "Market risk VaR limit is {v} million USD",
    ],
    "GENERAL": [
        "Maximum response time must not exceed {v} seconds",
        "Availability SLA target is {v}%",
        "Data backup frequency is every {v} hours",
        "User session timeout is {v} minutes",
        "Maximum concurrent users supported is {v}",
        "API rate limit is {v} requests per minute",
        "Log retention period is {v} days",
        "Password minimum length is {v} characters",
        "System restart window is between {v}:00 and {v2}:00 UTC",
        "Cache invalidation interval is {v} seconds",
        "Maximum file upload size is {v}MB",
        "Notification delivery timeout is {v} seconds",
        "Database connection pool size is {v} connections",
        "Health check interval is every {v} seconds",
        "Batch processing window is {v} hours",
        "Error rate threshold for alert is {v}%",
        "Content moderation review time is {v} minutes",
        "Search results page size is {v} items",
        "Auto-save interval is every {v} seconds",
        "Report generation timeout is {v} minutes",
    ],
}

DISTRACTOR_TEMPLATES = {
    "topic_drift": [
        "By the way, I was thinking about {topic} recently.",
        "This reminds me of something about {topic}.",
        "Speaking of which, have you considered {topic}?",
        "On a different note, {topic} is quite interesting.",
        "I read an article about {topic} yesterday.",
    ],
    "semantic_near_miss": [
        "The {param} should ideally be around {v}, though this is just a preference.",
        "Some people suggest {param} of {v}, but it's not critical.",
        "In informal settings, {param} might be {v}.",
        "A rough guideline for {param} would be about {v}.",
        "Historical data shows {param} averaged {v} in past projects.",
    ],
    "temporal_confusion": [
        "Previously we discussed setting {param} to {v}, but that was for the old project.",
        "The old standard required {param} of {v}, which has since been updated.",
        "Before the revision, {param} was {v}.",
        "In version 1, {param} was {v}, but we changed it.",
        "The original proposal had {param} at {v}, which was later revised.",
    ],
}

DRIFT_TOPICS = [
    "weekend plans", "a new restaurant", "the weather forecast",
    "a documentary about space", "office renovation plans",
    "a recent conference", "team building activities",
    "supply chain challenges", "remote work policies",
    "a new software tool", "industry trends", "sustainability goals",
    "recent news in technology", "a book recommendation",
    "upcoming holiday plans", "training opportunities",
]

ASSISTANT_TEMPLATES = [
    "Understood, I've noted that. {detail}",
    "Got it. I'll keep that in mind. {detail}",
    "Thank you for specifying. {detail}",
    "Noted. {detail}",
    "I'll make sure to account for that. {detail}",
    "Acknowledged. {detail}",
    "That's clear. {detail}",
    "I've recorded that requirement. {detail}",
]

ASSISTANT_DETAILS = [
    "Let me know if there are other specifications.",
    "Is there anything else to consider?",
    "I'll update the documentation accordingly.",
    "Should I proceed with the next item?",
    "I'll factor this into the analysis.",
    "Would you like to discuss related parameters?",
    "",
]


# ─────────────────────────────────────────────
# 2. Scenario config definitions (§6)
# ─────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    scenario_id: str
    scenario_type: str  # FORGETTING, NOISE_TOLERANCE, UPDATE_CONFLICT
    total_turns: int
    gt_count: int
    gt_timing: str  # UNIFORM, LATE_HEAVY, CLUSTERED
    distractor_density: float
    domain: str
    seed: int
    gt_update_count: int = 0
    noise_checkpoint: int = 30
    update_delay: int = 20


# Allocation table — 36 scenarios + domain round-robin
SCENARIO_CONFIGS: List[Dict[str, Any]] = []

# FORGETTING (12)
_f_raw = [
    ("F001", 200, 5,  "UNIFORM",    0.3),
    ("F002", 200, 13, "UNIFORM",    0.3),
    ("F003", 200, 5,  "LATE_HEAVY", 0.3),
    ("F004", 200, 13, "LATE_HEAVY", 0.3),
    ("F005", 500, 8,  "UNIFORM",    0.3),
    ("F006", 500, 20, "UNIFORM",    0.3),
    ("F007", 500, 8,  "LATE_HEAVY", 0.3),
    ("F008", 500, 13, "CLUSTERED",  0.5),
    ("F009", 1000, 13, "UNIFORM",   0.3),
    ("F010", 1000, 13, "LATE_HEAVY", 0.5),
    ("F011", 1000, 20, "UNIFORM",   0.3),
    ("F012", 1000, 8,  "CLUSTERED", 0.3),
]
for i, (sid, turns, gtc, timing, dd) in enumerate(_f_raw):
    SCENARIO_CONFIGS.append({
        "scenario_id": f"hold-out-{sid}",
        "scenario_type": "FORGETTING",
        "total_turns": turns,
        "gt_count": gtc,
        "gt_timing": timing,
        "distractor_density": dd,
        "domain": DOMAINS[i % len(DOMAINS)],
        "seed": 40000 + i,
    })

# NOISE_TOLERANCE (12)
_n_raw = [
    ("N001", 200, 8, 0.3), ("N002", 200, 8, 0.5),
    ("N003", 200, 8, 0.7), ("N004", 200, 8, 0.9),
    ("N005", 500, 8, 0.3), ("N006", 500, 8, 0.5),
    ("N007", 500, 8, 0.7), ("N008", 500, 8, 0.9),
    ("N009", 1000, 8, 0.3), ("N010", 1000, 8, 0.5),
    ("N011", 1000, 8, 0.7), ("N012", 1000, 8, 0.9),
]
for i, (sid, turns, gtc, dd) in enumerate(_n_raw):
    SCENARIO_CONFIGS.append({
        "scenario_id": f"hold-out-{sid}",
        "scenario_type": "NOISE_TOLERANCE",
        "total_turns": turns,
        "gt_count": gtc,
        "gt_timing": "UNIFORM",
        "distractor_density": dd,
        "domain": DOMAINS[i % len(DOMAINS)],
        "seed": 50000 + i,
        "noise_checkpoint": 30,
    })

# UPDATE_CONFLICT (12)
_u_raw = [
    ("U001", 200, 8,  1, 0.3), ("U002", 200, 8,  2, 0.3),
    ("U003", 200, 8,  3, 0.3), ("U004", 200, 13, 2, 0.3),
    ("U005", 500, 8,  1, 0.3), ("U006", 500, 13, 2, 0.3),
    ("U007", 500, 13, 3, 0.3), ("U008", 500, 8,  2, 0.5),
    ("U009", 1000, 13, 1, 0.3), ("U010", 1000, 13, 2, 0.3),
    ("U011", 1000, 13, 3, 0.3), ("U012", 1000, 8,  3, 0.5),
]
for i, (sid, turns, gtc, updates, dd) in enumerate(_u_raw):
    SCENARIO_CONFIGS.append({
        "scenario_id": f"hold-out-{sid}",
        "scenario_type": "UPDATE_CONFLICT",
        "total_turns": turns,
        "gt_count": gtc,
        "gt_timing": "UNIFORM",
        "distractor_density": dd,
        "domain": DOMAINS[i % len(DOMAINS)],
        "seed": 60000 + i,
        "gt_update_count": updates,
    })


# ─────────────────────────────────────────────
# 3. Measurement turns computation (§3-0, §3-1, §3-2, §3-3)
# ─────────────────────────────────────────────

def compute_measurement_turns(scenario_type: str, total_turns: int,
                               gt_insertions: List[int] = None,
                               noise_checkpoint: int = 30,
                               update_schedule: List[Dict] = None,
                               update_delay: int = 20) -> List[int]:
    """Compute absolute measurement turn numbers per ScenarioSpec v1.6."""
    if scenario_type == "FORGETTING":
        # dense early, then every 50 turns
        base = [10, 25, 50]
        rest = list(range(100, total_turns, 50))
        turns = base + rest
        # Cap: max <= total_turns - 1; the last one is total_turns - 1
        turns = [t for t in turns if t < total_turns]
        if total_turns - 1 not in turns:
            turns.append(total_turns - 1)
        # 1000-turn scenarios cap at 499 per spec v1.3
        if total_turns >= 1000:
            turns = [t for t in turns if t <= 499]
            if 499 not in turns:
                turns.append(499)
        return sorted(set(turns))

    elif scenario_type == "NOISE_TOLERANCE":
        # §3-2: inserted_at + noise_checkpoint per GT
        if gt_insertions is None:
            return [total_turns - 1]
        mturns = set()
        for ins_turn in gt_insertions:
            mt = ins_turn + noise_checkpoint
            if mt < total_turns:
                mturns.add(mt)
        if not mturns:
            mturns.add(min(total_turns - 1, gt_insertions[0] + noise_checkpoint))
        return sorted(mturns)

    elif scenario_type == "UPDATE_CONFLICT":
        # §3-3: update_turn + update_delay per update + final
        if update_schedule is None:
            return [total_turns - 1]
        mturns = set()
        for upd in update_schedule:
            mt = upd["update_turn"] + update_delay
            if mt < total_turns:
                mturns.add(mt)
        mturns.add(total_turns - 1)
        return sorted(mturns)

    return [total_turns - 1]


# ─────────────────────────────────────────────
# 4. GT insertion placement
# ─────────────────────────────────────────────

def compute_gt_positions(gt_count: int, total_turns: int, timing: str,
                          rng: random.Random,
                          earliest: int = 0, latest: int = None) -> List[int]:
    """Compute turn numbers for GT constraint insertion."""
    if latest is None:
        latest = total_turns - 20

    available = list(range(earliest, latest + 1, 2))  # user turns only (even)
    if not available:
        available = list(range(0, total_turns, 2))

    if timing == "UNIFORM":
        if gt_count >= len(available):
            return sorted(available[:gt_count])
        step = len(available) / gt_count
        positions = [available[int(i * step)] for i in range(gt_count)]

    elif timing == "LATE_HEAVY":
        # 70% in latter half
        mid = len(available) // 2
        early_pool = available[:mid]
        late_pool = available[mid:]
        early_count = max(1, int(gt_count * 0.3))
        late_count = gt_count - early_count
        early_picks = sorted(rng.sample(early_pool, min(early_count, len(early_pool))))
        late_picks = sorted(rng.sample(late_pool, min(late_count, len(late_pool))))
        positions = early_picks + late_picks

    elif timing == "CLUSTERED":
        # 2-3 clusters
        n_clusters = min(3, gt_count)
        cluster_size = gt_count // n_clusters
        remainder = gt_count % n_clusters
        cluster_starts = [available[int(i * len(available) / (n_clusters + 1))]
                          for i in range(1, n_clusters + 1)]
        positions = []
        for ci, start in enumerate(cluster_starts):
            size = cluster_size + (1 if ci < remainder else 0)
            idx = available.index(start) if start in available else 0
            cluster_positions = available[idx:idx + size]
            positions.extend(cluster_positions)
        positions = positions[:gt_count]

    else:  # FRONT_HEAVY (legacy compat)
        early_pool = available[:len(available) // 3]
        if gt_count <= len(early_pool):
            positions = sorted(rng.sample(early_pool, gt_count))
        else:
            positions = sorted(rng.sample(available, min(gt_count, len(available))))

    return sorted(set(positions))[:gt_count]


# ─────────────────────────────────────────────
# 5. Constraint text generation
# ─────────────────────────────────────────────

def generate_constraint_text(domain: str, index: int, rng: random.Random) -> str:
    """Generate a constraint string from the domain pool."""
    pool = CONSTRAINT_POOL[domain]
    template = pool[index % len(pool)]
    v = rng.randint(10, 500)
    v2 = rng.randint(v + 1, v + 200)
    v1 = rng.randint(1, v - 1) if v > 1 else 1
    text = template.replace("{v}", str(v))
    text = text.replace("{v1}", str(v1))
    text = text.replace("{v2}", str(v2))
    return text


def generate_updated_constraint(old_text: str, rng: random.Random) -> str:
    """Generate an updated version of a constraint (new value)."""
    import re
    numbers = re.findall(r'\d+', old_text)
    if not numbers:
        return old_text + " (revised)"
    # Change the first significant number
    target = numbers[0]
    new_val = int(target) + rng.choice([-1, 1]) * rng.randint(10, 100)
    if new_val < 1:
        new_val = int(target) + rng.randint(10, 50)
    return old_text.replace(target, str(new_val), 1)


# ─────────────────────────────────────────────
# 6. Distractor generation
# ─────────────────────────────────────────────

def generate_distractor(dtype: str, domain: str, rng: random.Random,
                         constraint_text: str = "") -> str:
    """Generate a distractor message."""
    if dtype == "topic_drift":
        template = rng.choice(DISTRACTOR_TEMPLATES["topic_drift"])
        topic = rng.choice(DRIFT_TOPICS)
        return template.replace("{topic}", topic)
    elif dtype == "semantic_near_miss":
        template = rng.choice(DISTRACTOR_TEMPLATES["semantic_near_miss"])
        pool = CONSTRAINT_POOL[domain]
        ref = rng.choice(pool)
        # MIN-1 fix: strip all placeholders before using as param
        import re as _re
        param = _re.sub(r'\{[^}]+\}', '', ref).strip().rstrip("is").rstrip("must be").strip()
        if len(param) > 50:
            param = param[:50]
        v = str(rng.randint(1, 999))
        return template.replace("{param}", param).replace("{v}", v)
    elif dtype == "temporal_confusion":
        template = rng.choice(DISTRACTOR_TEMPLATES["temporal_confusion"])
        param = "the specification"
        v = str(rng.randint(1, 999))
        return template.replace("{param}", param).replace("{v}", v)
    return "Let me think about that for a moment."


def generate_assistant_response(rng: random.Random) -> str:
    """Generate a generic assistant response."""
    template = rng.choice(ASSISTANT_TEMPLATES)
    detail = rng.choice(ASSISTANT_DETAILS)
    return template.replace("{detail}", detail).strip()


# ─────────────────────────────────────────────
# 7. Main scenario generation
# ─────────────────────────────────────────────

def generate_scenario(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a complete scenario JSON from config."""
    rng = random.Random(config["seed"])
    sc_type = config["scenario_type"]
    total = config["total_turns"]
    gt_count = config["gt_count"]
    domain = config["domain"]
    dd = config["distractor_density"]
    timing = config.get("gt_timing", "UNIFORM")
    gt_update_count = config.get("gt_update_count", 0)
    noise_checkpoint = config.get("noise_checkpoint", 30)
    update_delay = 20

    # Distractor type ratios (§2 default)
    dtype_ratios = {"topic_drift": 0.5, "semantic_near_miss": 0.3, "temporal_confusion": 0.2}

    # ── Compute GT positions ──
    # §3-1 (FORGETTING): latest = min(total - 20, max(measurement_turns))
    # §3-2 (NOISE): latest = total - noise_checkpoint
    # §3-3 (UPDATE): latest = total - 20
    if sc_type == "FORGETTING":
        # Pre-compute measurement_turns for latest constraint
        pre_mturns = compute_measurement_turns("FORGETTING", total)
        latest = min(total - 20, max(pre_mturns)) if pre_mturns else total - 20
    elif sc_type == "NOISE_TOLERANCE":
        latest = total - noise_checkpoint
    else:
        latest = total - 20

    earliest = 0
    gt_positions = compute_gt_positions(gt_count, total, timing, rng,
                                         earliest=earliest, latest=latest)

    # ── Generate GT constraints ──
    gt_constraints = []
    gt_metadata = {}
    for i, pos in enumerate(gt_positions):
        gt_id = f"gt_{i+1:03d}"
        text = generate_constraint_text(domain, i, rng)
        gt_constraints.append(text)
        gt_metadata[gt_id] = {
            "text": text,
            "inserted_at": pos,
            "visibility": "EXPLICIT",
            "updated_at": None,
            "superseded_by": None,
        }

    # ── Handle updates (UPDATE_CONFLICT) ──
    update_schedule = []
    ua_targets = []
    ground_truth = list(gt_constraints)  # start with all originals

    if sc_type == "UPDATE_CONFLICT" and gt_update_count > 0:
        updatable = list(range(len(gt_constraints)))
        rng.shuffle(updatable)
        used_update_turns = set(gt_positions)  # BLK-2: track all occupied turns
        for ui in range(min(gt_update_count, len(updatable))):
            gt_idx = updatable[ui]
            gt_id = f"gt_{gt_idx+1:03d}"
            old_text = gt_constraints[gt_idx]
            new_text = generate_updated_constraint(old_text, rng)
            insert_turn = gt_positions[gt_idx]
            update_turn = insert_turn + max(20, rng.randint(20, total // 3))
            update_turn = min(update_turn, total - update_delay - 5)
            if update_turn <= insert_turn + 20:
                update_turn = insert_turn + 20
            # BLK-1 fix: update_turn must be even (user turn)
            if update_turn % 2 != 0:
                update_turn += 1
            # BLK-2 fix: avoid collision with existing GT or other update turns
            # Search forward then backward for a free even user turn
            original_ut = update_turn
            found_free = False
            # Forward search
            candidate = original_ut
            while candidate < total - update_delay - 2:
                if candidate % 2 == 0 and candidate not in used_update_turns:
                    update_turn = candidate
                    found_free = True
                    break
                candidate += 2
            # Backward search if forward failed
            if not found_free:
                candidate = original_ut - 2
                while candidate > insert_turn + 10:
                    if candidate % 2 == 0 and candidate not in used_update_turns:
                        update_turn = candidate
                        found_free = True
                        break
                    candidate -= 2
            assert found_free, f"No free even turn for update (gt={gt_id}, insert={insert_turn})"
            assert update_turn % 2 == 0, f"update_turn {update_turn} must be even"
            used_update_turns.add(update_turn)

            update_schedule.append({
                "gt_id": gt_id,
                "update_turn": update_turn,
                "old_value": old_text,
                "new_value": new_text,
            })
            ua_targets.append({
                "gt_id": gt_id,
                "old_value": old_text,
                "new_value": new_text,
                "update_turn": update_turn,
            })

            # Update ground_truth: remove old, add new
            if old_text in ground_truth:
                ground_truth.remove(old_text)
            ground_truth.append(new_text)

            # Update metadata
            new_gt_id = f"gt_{gt_idx+1:03d}_upd"
            gt_metadata[gt_id]["updated_at"] = update_turn
            gt_metadata[gt_id]["superseded_by"] = new_gt_id
            gt_metadata[new_gt_id] = {
                "text": new_text,
                "inserted_at": update_turn,
                "visibility": "EXPLICIT",
                "updated_at": None,
                "superseded_by": None,
            }

    # ── Compute measurement turns ──
    measurement_turns = compute_measurement_turns(
        sc_type, total,
        gt_insertions=gt_positions,
        noise_checkpoint=noise_checkpoint,
        update_schedule=update_schedule,
        update_delay=update_delay,
    )

    # ── Build turn list ──
    turns_data = []
    gt_turn_set = set(gt_positions)
    update_turn_map = {u["update_turn"]: u for u in update_schedule}

    # Compute distractor positions
    non_gt_user_turns = [t for t in range(0, total, 2) if t not in gt_turn_set
                         and t not in update_turn_map]
    n_distractors = int(len(non_gt_user_turns) * dd)
    rng.shuffle(non_gt_user_turns)
    distractor_turns = set(non_gt_user_turns[:n_distractors])

    for turn_num in range(total):
        if turn_num % 2 == 0:  # user turn
            is_constraint = False
            is_distractor = False
            distractor_type = None
            gt_id_ref = None

            if turn_num in gt_turn_set:
                # GT constraint turn
                gt_idx = gt_positions.index(turn_num)
                gt_id_ref = f"gt_{gt_idx+1:03d}"
                content = f"Important requirement: {gt_constraints[gt_idx]}"
                is_constraint = True

            elif turn_num in update_turn_map:
                # Update turn
                upd = update_turn_map[turn_num]
                content = f"Update to previous requirement: {upd['new_value']} (replaces the earlier specification)"
                is_constraint = True
                gt_id_ref = upd["gt_id"] + "_upd"

            elif turn_num in distractor_turns:
                # Distractor turn
                is_distractor = True
                r = rng.random()
                if r < dtype_ratios["topic_drift"]:
                    distractor_type = "topic_drift"
                elif r < dtype_ratios["topic_drift"] + dtype_ratios["semantic_near_miss"]:
                    distractor_type = "semantic_near_miss"
                else:
                    distractor_type = "temporal_confusion"
                content = generate_distractor(distractor_type, domain, rng)

            else:
                # Normal user turn
                content = rng.choice([
                    f"Let's continue with the {domain.lower()} specifications.",
                    f"What about the next requirement for this {domain.lower()} project?",
                    f"Can you summarize what we've covered so far?",
                    f"I have another consideration for this project.",
                    f"Let me check the current status of our {domain.lower()} requirements.",
                    f"Are there any dependencies between the specifications we discussed?",
                    f"How does this compare to industry standards?",
                    f"Let's review the constraints we've established.",
                ])

            turn_entry = {
                "turn": turn_num,
                "role": "user",
                "content": content,
                "is_constraint": is_constraint,
                "is_distractor": is_distractor,
            }
            if distractor_type:
                turn_entry["distractor_type"] = distractor_type
            if gt_id_ref:
                turn_entry["gt_id"] = gt_id_ref

        else:  # assistant turn
            content = generate_assistant_response(rng)
            turn_entry = {
                "turn": turn_num,
                "role": "assistant",
                "content": content,
                "is_constraint": False,
                "is_distractor": False,
            }

        turns_data.append(turn_entry)

    # ── Build query ──
    query = f"What are all the current {domain.lower()} constraints and requirements?"

    # ── Assemble output ──
    output = {
        "scenario_id": config["scenario_id"],
        "version": "v4",
        "scenario_type": sc_type,
        "params": {
            "total_turns": total,
            "gt_count": gt_count,
            "gt_timing": {
                "distribution": timing,
                "earliest": earliest,
                "latest": latest,
            },
            "gt_visibility": "EXPLICIT",
            "distractor_density": dd,
            "distractor_types": dtype_ratios,
            "domain": domain,
            "seed": config["seed"],
        },
        "seed": config["seed"],
        "total_turns": total,
        "domain": domain,
        "query": query,
        "turns": turns_data,
        "ground_truth_constraints": ground_truth,
        "gt_metadata": gt_metadata,
        "evaluation_protocol": {
            "metric": "exact_match_recall",
            "measurement_turns": measurement_turns,
            "ua_targets": ua_targets,
        },
    }

    if sc_type == "UPDATE_CONFLICT":
        output["params"]["gt_update_count"] = gt_update_count
        output["params"]["update_delay"] = update_delay
        output["update_schedule"] = update_schedule
    if sc_type == "NOISE_TOLERANCE":
        output["params"]["noise_checkpoint"] = noise_checkpoint

    return output


# ─────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ScenarioSpec v1.6 Scenario Generator")
    parser.add_argument("--output-dir", type=str, default="./datasets/holdout_v4/",
                        help="Output directory for generated scenarios")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"FORGETTING": 0, "NOISE_TOLERANCE": 0, "UPDATE_CONFLICT": 0}
    total_turns_sum = 0

    for config in SCENARIO_CONFIGS:
        scenario = generate_scenario(config)
        sid = scenario["scenario_id"]
        stats[scenario["scenario_type"]] += 1
        total_turns_sum += scenario["total_turns"]

        if args.dry_run:
            mt = scenario["evaluation_protocol"]["measurement_turns"]
            gt = scenario["ground_truth_constraints"]
            print(f"{sid:25s} | {scenario['scenario_type']:18s} | "
                  f"turns={scenario['total_turns']:5d} | "
                  f"gt={len(gt):2d} | "
                  f"checkpoints={len(mt):2d} | "
                  f"domain={scenario['domain']}")
        else:
            out_file = out_dir / f"{sid.replace('hold-out-', '')}_v4.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(scenario, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated: {sum(stats.values())} scenarios")
    for stype, count in stats.items():
        print(f"  {stype}: {count}")
    print(f"Total turns: {total_turns_sum:,}")
    if not args.dry_run:
        print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
