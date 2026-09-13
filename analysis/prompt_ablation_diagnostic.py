# -*- coding: utf-8 -*-
r"""Experiment A — free-form-tolerant deterministic scoring variant (diagnostic, NOT a replacement of the paper rule).
Motivation: the §III-C strict rule was co-designed with the verbatim-listing prompt (p0). Under own-words/neutral prompts
the same rule misses adherent answers for three mechanical reasons observed in the traces:
  (a) unit spacing/hyphenation: "412 mm", "173-hour" vs tokens "412mm", "173"
  (b) parameter phrase paraphrased: "Arbitration Venue: Must be in jurisdiction number 100" vs phrase "Arbitration venue must be in"
  (c) provenance mention: "412 mm (updated from an earlier requirement of 320 mm)" names the old value as superseded -> scored Both
Variant rules (all deterministic):
  T  tolerant tokens: numbers matched as bare numerals with optional unit glue (\b412\s*-?\s*mm\b or bare \b412\b)
  P  phrase = first two content words of the shared prefix (e.g. "surgical margin"), case-insensitive, same line
  C  Both + supersession cue on that line ("updated from", "previously", "replaces", "earlier", "instead of", "formerly", "changed from", "revised from", "rather than", "not") -> NewOnly+prov
Reported: strict (paper) vs T+P vs T+P+C, all arms x p0/p1/p2, plus canonical p0 sanity."""
import json, re, os, glob, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(REPO, "results")
exec(open(os.path.join(SP, "aggregate_expA.py"), encoding="utf-8").read().split("out = []")[0])
STOP = {"must", "is", "are", "be", "the", "a", "an", "of", "to", "at", "in", "for", "per", "shall", "not", "exceed", "least", "least,", "within", "range", "should"}
CUE = re.compile(r"updated from|previously|replac|earlier|instead of|formerly|changed from|revised from|rather than|\bnot\b|superseded|was\b|old\b|original", re.I)

def num_tokens(v):
    return [t for t in TOKEN_RE.findall(v) if any(ch.isdigit() for ch in t)]

def tok_regex(tok):
    m = re.match(r"^(\d+(?:\.\d+)?)(.*)$", tok)
    if not m: return re.compile(re.escape(tok), re.I)
    num, unit = m.groups()
    if unit: return re.compile(rf"(?<![\d.]){re.escape(num)}\s*-?\s*{re.escape(unit)}(?![A-Za-z])|(?<![\d.]){re.escape(num)}(?![\d.])", re.I)
    return re.compile(rf"(?<![\d.]){re.escape(num)}(?![\d.])")

def content_words(phrase):
    return [w for w in re.findall(r"[a-z]+", phrase.lower()) if w not in STOP][:2]

def score_variant(answer, tgt, use_cue):
    old_v, new_v = tgt["old_value"], tgt["new_value"]
    ot, nt = set(num_tokens(old_v)), set(num_tokens(new_v))
    new_only, old_only = sorted(nt - ot), sorted(ot - nt)
    if not new_only or not old_only:  # fall back to all distinguishing tokens
        ot2, nt2 = set(TOKEN_RE.findall(old_v)), set(TOKEN_RE.findall(new_v)); new_only, old_only = sorted(nt2 - ot2), sorted(ot2 - nt2)
    cw = content_words(param_phrase(old_v, new_v))
    lines = [l.replace("°", "°") for l in answer.split("\n") if l.strip()]
    n = o = cue = False
    for l in lines:
        ll = l.lower()
        if not all(w in ll for w in cw): continue
        ln = any(tok_regex(t).search(l) for t in new_only); lo = any(tok_regex(t).search(l) for t in old_only)
        n |= ln; o |= lo
        if ln and lo and CUE.search(l): cue = True
    if n and not o: return "NewOnly"
    if n and o: return "NewOnly+prov" if (use_cue and cue) else "Both"
    if o: return "OldOnly"
    return "Neither"

def table(title, fn):
    print(f"\n## {title}")
    print("| arm | p0 | p1 | p2 | p0 cats | p1 cats | p2 cats |"); print("|---|---|---|---|---|---|---|")
    for key, name in ARMS:
        A = load(os.path.join(SP, f"behavior_{key}_prompt_ablation.json")); cells = []; cats = []
        for p in ("p0", "p1", "p2"):
            c = collections.Counter()
            for s, r in A.items():
                tg = {t["gt_id"]: t for t in S[s]["evaluation_protocol"]["ua_targets"]}
                for b in r["behavior_by_prompt"][p]: c[fn(r["answers_by_prompt"][p], tg[b["gt_id"]])] += 1
            adh = c["NewOnly"] + c["NewOnly+prov"]
            cells.append(f"{adh} ({100*adh/45:.1f}%)")
            cats.append(f"N{c['NewOnly']}+P{c['NewOnly+prov']}/B{c['Both']}/O{c['OldOnly']}/X{c['Neither']}")
        print(f"| {name} | " + " | ".join(cells + cats) + " |")

table("Paper strict rule (reference)", lambda a, t: strict_score(a, t))
table("Variant T+P (tolerant tokens + 2-word parameter phrase)", lambda a, t: score_variant(a, t, False))
table("Variant T+P+C (+ provenance cue -> adherent)", lambda a, t: score_variant(a, t, True))

print("\n## Canonical p0 sanity under variants (EpSem-SS, should stay ~34)")
C = load(os.path.join(REPO, "results", "behavior_EpSemSupersession.json"))
for nm, fn in (("strict", lambda a, t: strict_score(a, t)), ("T+P", lambda a, t: score_variant(a, t, False)), ("T+P+C", lambda a, t: score_variant(a, t, True))):
    c = collections.Counter()
    for s, r in C.items():
        tg = {t["gt_id"]: t for t in S[s]["evaluation_protocol"]["ua_targets"]}
        for b in r["behavior"]: c[fn(r["answer_text"], tg[b["gt_id"]])] += 1
    print(f"- {nm}: {dict(c)}")

print("\n## Residual Neither under T+P+C — EpSem-SS p1/p2 (for manual audit)")
A = load(os.path.join(SP, "behavior_EpSemSupersession_prompt_ablation.json"))
for p in ("p1", "p2"):
    for s, r in A.items():
        tg = {t["gt_id"]: t for t in S[s]["evaluation_protocol"]["ua_targets"]}
        for b in r["behavior_by_prompt"][p]:
            t = tg[b["gt_id"]]
            if score_variant(r["answers_by_prompt"][p], t, True) == "Neither":
                cw = content_words(param_phrase(t["old_value"], t["new_value"]))
                hits = [l[:160] for l in r["answers_by_prompt"][p].split("\n") if cw and cw[0] in l.lower()]
                print(f"[{p} {s} {b['gt_id']}] new={t['new_value']} | lines: {hits[:2]}")
