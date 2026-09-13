# -*- coding: utf-8 -*-
"""Prompt-ablation aggregation (behavior-level UA under three answer prompts).
Strict behavior scoring re-implemented from paper §III-C: a distinguishing value token counts only if it
appears on the same answer line as the parameter phrase (common wording of old/new constraint).
Validation: p0 (canonical prompt) strict counts must reproduce Fig. 9 canonical numbers
(Naive 14, RAG 0, EpSem 21, EpSem-F 24, EpSem-SS 34, LWW 35 of 45)."""
import json, re, os, glob, collections, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(REPO, "results")
ARMS = [("EpSemSupersession", "EpSem-SS"), ("LWWMemory", "LWW"), ("NaiveMemory", "Naive"),
        ("RagMemory", "RAG"), ("EpSemMemory", "EpSem"), ("EpSemFiltered", "EpSem-F")]
CANON_STRICT = {"Naive": 14, "RAG": 0, "EpSem": 21, "EpSem-F": 24, "EpSem-SS": 34, "LWW": 35}
TOKEN_RE = re.compile(r"[A-Za-z0-9°%/\.]+")

S = {}
for f in glob.glob(os.path.join(REPO, "scenarios", "**", "*.json"), recursive=True):
    try:
        d = json.load(open(f, encoding="utf-8"))
        if isinstance(d, dict) and "scenario_id" in d: S[d["scenario_id"]] = d
    except Exception: pass

def norm(s):
    return re.sub(r"\s+", " ", s.replace("\u00b0", "°")).strip().lower()

def param_phrase(old, new):
    ow, nw = old.split(), new.split()
    i = 0
    while i < min(len(ow), len(nw)) and ow[i] == nw[i]: i += 1
    return " ".join(ow[:i])

def strict_score(answer, tgt):
    old_v, new_v = tgt["old_value"], tgt["new_value"]
    ot, nt = set(TOKEN_RE.findall(old_v)), set(TOKEN_RE.findall(new_v))
    new_only, old_only = sorted(nt - ot), sorted(ot - nt)
    phrase = norm(param_phrase(old_v, new_v))
    # fall back to a shorter phrase (first 3 words) if the model paraphrases the tail
    short = " ".join(phrase.split()[:3])
    lines = [norm(l) for l in answer.split("\n") if l.strip()]
    def hit(toks):
        for l in lines:
            if (phrase and phrase in l) or (short and short in l):
                if any(re.search(re.escape(norm(t)), l) for t in toks): return True
        return False
    n, o = hit(new_only), hit(old_only)
    return "NewOnly" if n and not o else ("Both" if n and o else ("OldOnly" if o else "Neither"))

def loose_score(b):
    n, o = bool(b["new_tokens_hit"]), bool(b["old_tokens_hit"])
    return "NewOnly" if n and not o else ("Both" if n and o else ("OldOnly" if o else "Neither"))

def load(path): return {r["scenario_id"]: r for r in json.load(open(path, encoding="utf-8"))}

out = []
summary = {}   # (arm, prompt) -> Counter
per_len = {}   # (arm, prompt) -> {len: Counter}
integrity = []
for key, name in ARMS:
    C = load(os.path.join(REPO, "results", f"behavior_{key}.json"))
    A = load(os.path.join(SP, f"behavior_{key}_prompt_ablation.json"))
    ctx_same = sum(1 for s in A if A[s].get("context_text") == C[s]["context_text"])
    p0_same = sum(1 for s in A if A[s].get("answers_by_prompt", {}).get("p0") == C[s]["answer_text"])
    errs = sum(1 for r in A.values() if "error" in r)
    integrity.append((name, len(A), ctx_same, p0_same, errs, A[next(iter(A))].get("runner_sha256", "")[:16]))
    # canonical strict (validation)
    cc = collections.Counter()
    for s, r in C.items():
        tg = {t["gt_id"]: t for t in S[s]["evaluation_protocol"]["ua_targets"]}
        for b in r["behavior"]: cc[strict_score(r["answer_text"], tg[b["gt_id"]])] += 1
    summary[(name, "canon")] = cc
    for p in ("p0", "p1", "p2"):
        c = collections.Counter(); pl = collections.defaultdict(collections.Counter); lo = collections.Counter()
        for s, r in A.items():
            if "error" in r: continue
            tg = {t["gt_id"]: t for t in S[s]["evaluation_protocol"]["ua_targets"]}
            ans = r["answers_by_prompt"][p]
            for b in r["behavior_by_prompt"][p]:
                cat = strict_score(ans, tg[b["gt_id"]]); c[cat] += 1; pl[r["total_turns"]][cat] += 1
                lo[loose_score(b)] += 1
        summary[(name, p)] = c; per_len[(name, p)] = pl; summary[(name, p + "_loose")] = lo

print("## Integrity vs canonical (context / p0 answer identity)")
print("| arm | n | ctx same | p0 same | errors | runner sha |"); print("|---|---|---|---|---|---|")
for row in integrity: print("| " + " | ".join(str(x) for x in row) + " |")

print("\n## Strict re-scoring validation (canonical answer_text -> Fig. 9 numbers)")
print("| arm | Fig.9 NewOnly | re-scored NewOnly | Both | OldOnly | Neither |"); print("|---|---|---|---|---|---|")
ok = True
for key, name in ARMS:
    c = summary[(name, "canon")]; m = c["NewOnly"] == CANON_STRICT[name]; ok &= m
    print(f"| {name} | {CANON_STRICT[name]} | {c['NewOnly']}{'' if m else ' ✗'} | {c['Both']} | {c['OldOnly']} | {c['Neither']} |")
print("VALIDATION:", "PASS" if ok else "FAIL — strict re-implementation does not reproduce canonical; numbers below are provisional")

print("\n## Behavior-level UA (strict) by prompt — 45 update events, UC-24")
print("| arm | p0 NewOnly | p0 % | p1 NewOnly | p1 % | p2 NewOnly | p2 % | p0 Both/Old/Neither | p1 Both/Old/Neither | p2 Both/Old/Neither |")
print("|---|---|---|---|---|---|---|---|---|---|")
for key, name in ARMS:
    cells = []
    for p in ("p0", "p1", "p2"):
        c = summary[(name, p)]; n = sum(c.values()) or 1
        cells += [str(c["NewOnly"]), f"{100*c['NewOnly']/n:.1f}"]
    tail = [f"{summary[(name,p)]['Both']}/{summary[(name,p)]['OldOnly']}/{summary[(name,p)]['Neither']}" for p in ("p0", "p1", "p2")]
    print(f"| {name} | " + " | ".join(cells + tail) + " |")

print("\n## Ranking by NewOnly per prompt")
for p in ("p0", "p1", "p2"):
    order = sorted(ARMS, key=lambda a: -summary[(a[1], p)]["NewOnly"])
    print(f"- {p}: " + " > ".join(f"{n}({summary[(n,p)]['NewOnly']})" for _, n in order))

print("\n## Loose (token anywhere) NewOnly for reference")
print("| arm | p0 | p1 | p2 |"); print("|---|---|---|---|")
for key, name in ARMS:
    print(f"| {name} | " + " | ".join(str(summary[(name, p + '_loose')]['NewOnly']) for p in ("p0", "p1", "p2")) + " |")

print("\n## EpSem-SS strict by conversation length")
for p in ("p0", "p1", "p2"):
    pl = per_len[("EpSem-SS", p)]
    print(f"- {p}: " + ", ".join(f"{L}T {pl[L]['NewOnly']}/{sum(pl[L].values())}" for L in sorted(pl)))

print("\n## Answer length (chars, mean) by prompt — EpSem-SS")
A = load(os.path.join(SP, "behavior_EpSemSupersession_prompt_ablation.json"))
for p in ("p0", "p1", "p2"):
    L = [len(r["answers_by_prompt"][p]) for r in A.values()]
    print(f"- {p}: {sum(L)/len(L):.0f}")
