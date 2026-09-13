"""Call-count probe: count language-model / embedding calls and prompt+response sizes per configuration on one scenario.
Monkeypatches requests.post inside the frozen engine; engine code untouched. Store-level replay only (no answers)."""
import json, sys, time, collections
sys.path.insert(0, str(Path(__file__).resolve().parent))
import requests
from engine.epsem_filtered import EpSemFiltered
from engine.epsem_supersession import EpSemSupersession
from engine.base import Message
SC = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "scenarios" / "holdout_v4" / "U001_v4.json")
sc = json.load(open(SC)); mts = set(sc["evaluation_protocol"]["measurement_turns"])
_post = requests.post
def run(cls, name):
    stats = collections.defaultdict(lambda: [0, 0, 0, 0.0])  # calls, prompt chars, response chars, seconds
    def patched(url, json=None, **kw):
        t0 = time.perf_counter(); r = _post(url, json=json, **kw); dt = time.perf_counter() - t0
        key = url.split("/api/")[-1]; s = stats[key]; s[0] += 1
        if json: s[1] += len(json.get("prompt", ""))
        try: s[2] += len(r.json().get("response", "")) if key == "generate" else 0
        except Exception: pass
        s[3] += dt; return r
    requests.post = patched
    mem = cls(db_path=":memory:"); t0 = time.perf_counter()
    for t in sc["turns"]:
        mem.write(Message(role=t["role"], content=t["content"], turn=t["turn"], is_constraint=t.get("is_constraint", False), is_distractor=t.get("is_distractor", False)))
        if t["turn"] in mts: mem.read(sc["query"])
    ctx = mem.read(sc["query"]).context_text; el = time.perf_counter() - t0
    requests.post = _post
    print(f"== {name}: replay {el:.0f}s | ctx {len(ctx)} chars")
    for k, (c, pc, rc, s) in stats.items(): print(f"   {k:12s} calls {c:5d} | prompt chars {pc:8d} | response chars {rc:7d} | {s:.0f}s")
    return stats
run(EpSemFiltered, "EpSem-F"); run(EpSemSupersession, "EpSem-SS")
