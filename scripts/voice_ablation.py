"""PREREG_voice_ablation.md: does the voice note itself raise recognition kappa vs Sonnet, holding the model fixed?

Training items only. Missing readout = noise, labels lower-cased, off-list skipped, off-schema dropped.
Paired item bootstrap (10,000 resamples of the training items, seed 0) for the note-minus-official difference.

    python scripts/voice_ablation.py
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ("recognition", "echo", "topic", "noise")


def labs(r):
    vs = r.get("verdicts") if isinstance(r, dict) else None
    if not isinstance(vs, list) or not vs or not all(isinstance(x, dict) and "label" in x for x in vs):
        return None
    return {int(x.get("index", 0)): str(x["label"]).strip().lower() for x in vs}


def from_verdicts(path, runs, key):
    out = {}
    p = ROOT / path
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        v = json.loads(line)
        if v.get("run") in runs and v.get("result") is not None and labs(v["result"]) is not None:
            out[v[key] or v["h"]] = labs(v["result"])
    return out


def kappa(pairs, binary):
    f = (lambda x: x == "recognition") if binary else (lambda x: x)
    a = [(f(s), f(g)) for s, g in pairs]
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in a) / n
    cs, cg = Counter(x for x, _ in a), Counter(y for _, y in a)
    pe = sum(cs[k] * cg[k] for k in set(cs) | set(cg)) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def compare(son, item, off, note, name):
    rows = defaultdict(list)  # item -> [(sonnet, official, note)]
    for h in sorted(set(son) & set(off) & set(note)):
        for i, s in son[h].items():
            a, b = off[h].get(i, "noise"), note[h].get(i, "noise")
            if s in C and a in C and b in C:
                rows[item[h]].append((s, a, b))
    allr = [r for v in rows.values() for r in v]
    if not allr:
        return {"name": name, "readouts": 0}
    items = sorted(rows)
    rng = random.Random(0)
    d = []
    for _ in range(10_000):
        rs = [r for it in (rng.choice(items) for _ in items) for r in rows[it]]
        d.append(kappa([(s, b) for s, _, b in rs], True) - kappa([(s, a) for s, a, _ in rs], True))
    d = sorted(x for x in d if x == x)
    ci = [round(d[int(0.025 * len(d))], 3), round(d[int(0.975 * len(d)) - 1], 3)]
    missed = [r for r in allr if r[0] == "recognition" and r[1] != "recognition"]
    false = [r for r in allr if r[0] != "recognition" and r[1] == "recognition"]
    return {"name": name, "readouts": len(allr), "items": len(items), "sonnet_recognitions": sum(r[0] == "recognition" for r in allr),
            "kappa_rec_official": round(kappa([(s, a) for s, a, _ in allr], True), 3),
            "kappa_rec_note": round(kappa([(s, b) for s, _, b in allr], True), 3),
            "diff_note_minus_official": round(kappa([(s, b) for s, _, b in allr], True) - kappa([(s, a) for s, a, _ in allr], True), 3),
            "diff_ci95_item_bootstrap": ci, "note_works": ci[0] > 0,
            "kappa_4way_official": round(kappa([(s, a) for s, a, _ in allr], False), 3),
            "kappa_4way_note": round(kappa([(s, b) for s, _, b in allr], False), 3),
            "sonnet_recognitions_missed_by_official": len(missed),
            "of_those_recovered_by_note": sum(r[2] == "recognition" for r in missed),
            "official_false_recognitions": len(false),
            "of_those_removed_by_note": sum(r[2] != "recognition" for r in false),
            "recognitions_official_vs_note": [sum(r[1] == "recognition" for r in allr), sum(r[2] == "recognition" for r in allr)],
            "echo_official_vs_note": [sum(r[1] == "echo" for r in allr), sum(r[2] == "echo" for r in allr)]}


def main() -> None:
    son, item = {}, {}
    for line in (ROOT / "runs/sonnet_teacher/verdicts.jsonl").read_text(encoding="utf-8").splitlines():
        v = json.loads(line)
        if v.get("run") == "teacher" and v.get("result") and labs(v["result"]) is not None:
            son[v["h"]], item[v["h"]] = labs(v["result"]), v["item"]
    q38_off = from_verdicts("runs/open_teacher/qwen__qwen3.8-27b/verdicts.jsonl", ("official",), "h")
    q38_note = from_verdicts("runs/open_teacher/qwen__qwen3.8-27b/verdicts.jsonl", ("smoke", "teacher"), "h_orig")
    q36_note = from_verdicts("runs/open_teacher/qwen__qwen3.6-27b/verdicts.jsonl", ("note_alibaba",), "h_orig")
    q36_off = {}
    for line in (ROOT / "runs/vllm-cache.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["k"].startswith("qwen3.6-27b|readout_recognition|") and labs(r["r"]) is not None:
            q36_off[r["k"].rsplit("|", 1)[1]] = labs(r["r"])
    res = {"primary_qwen3.8": compare(son, item, q38_off, q38_note, "Qwen3.8-27B: official vs + voice note (same serving)"),
           "secondary_qwen3.6": compare(son, item, q36_off, q36_note, "Qwen3.6-27B: official (vLLM) vs + voice note (Alibaba)")}
    (ROOT / "runs/open_teacher/voice_ablation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
