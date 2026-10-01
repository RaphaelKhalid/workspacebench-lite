"""Reproduce the README numbers from data/.

uv run python scripts/reproduce.py gold                   # Sonnet 5 labels on the 768 test prompts (~$4.40)
uv run python scripts/reproduce.py table                  # the README results table
uv run python scripts/reproduce.py gold --split train     # Sonnet 5 labels on 1,927 training prompts (~$10.90)
uv run python scripts/reproduce.py train                  # the teacher gate and the voice-note ablation
uv run python scripts/reproduce.py judge --model qwen/qwen3.8-27b --voice-note --name qwen38_note   # re-run a judge
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "runs" / "repro"
sys.path.insert(0, str(ROOT / "src"))
import sonnet_b1 as b1  # scripts/ is on sys.path when run as a script

C = ("recognition", "echo", "topic", "noise")


def r3(x: float) -> Decimal:
    """Round half up to 3 places (0.6475 -> 0.648), as the published tables do."""
    return Decimal(str(x)).quantize(Decimal("0.001"), ROUND_HALF_UP)


N_TRAIN_GOLD = 1927  # the training prompts Sonnet labelled, in the released order


def prompts(split: str, limit: int | None = None) -> list[dict]:
    from wsbench.evals.jailbreak_recognition.prompts import READOUT_SYSTEM

    with gzip.open(DATA / f"{split}_prompts.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    for r in rows:  # every prompt is the official one, byte for byte
        r["system"] = READOUT_SYSTEM
        assert hashlib.sha256((r["system"] + "\x00" + r["user"]).encode()).hexdigest() == r["h"], r[
            "h"
        ]
    return rows[:limit]


def first_label(result: dict | None, rule: str) -> str | None:
    """test: the family's own postprocess (missing/off-list -> noise); train: the gate's rule (off-schema dropped)."""
    if result is None:
        return None
    if rule == "test":
        from wsbench.evals.jailbreak_recognition.judge import postprocess_readout

        return postprocess_readout(result, [""])["labels"][0]
    vs = result.get("verdicts")
    if (
        not isinstance(vs, list)
        or not vs
        or not all(isinstance(x, dict) and "label" in x for x in vs)
    ):
        return None
    return {int(x.get("index", 0)): str(x["label"]).strip().lower() for x in vs}.get(1, "noise")


def label(
    rows,
    out: Path,
    *,
    model,
    rule,
    note="",
    reasoning=None,
    temperature=None,
    provider=None,
    max_usd=5.0,
):
    """Label prompts in batches with the upstream client; resumable; stops before passing max_usd."""
    from wsbench import llm

    from wsbjev import alt_routes, keys

    os.environ.setdefault("OPENROUTER_API_KEY", keys.get("OPENROUTER_API_KEY") or "")
    alt_routes._pin_providers(llm)
    alt_routes._PROVIDER.set(provider)
    schema = json.loads((DATA / "readout_schema.json").read_text(encoding="utf-8"))
    key = "cell" if rule == "test" else "h"
    done = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    todo, spent = [r for r in rows if r[key] not in done], 0.0
    for i in range(0, len(todo), 64):
        if spent >= max_usd:
            print(f"stopped at ${spent:.2f} (max ${max_usd:.2f})")
            break
        batch, got = todo[i : i + 64], {}
        spend = asyncio.run(
            llm.stream_json_async(
                [(r["system"] + note, r["user"]) for r in batch],
                schema=schema,
                model=model,
                on_result=lambda j, res, got=got: got.__setitem__(j, res),
                reasoning=reasoning,
                temperature=temperature,
                concurrency=16,
                max_tokens=1500 if note or provider else None,
            )
        )
        spent += spend.usd
        for j, r in enumerate(batch):
            lab = first_label(got.get(j), rule)
            if lab is not None:
                done[r[key]] = lab
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(done, sort_keys=True), encoding="utf-8")
        print(f"  {len(done)}/{len(rows)} labelled, ${spent:.2f}")
    return done


def cmd_gold(a) -> None:
    rows = prompts(a.split, a.limit or (N_TRAIN_GOLD if a.split == "train" else None))
    label(
        rows,
        OUT / f"gold_{a.split}.json",
        model="anthropic/claude-sonnet-5",
        rule=a.split,
        max_usd=a.max_usd,
    )


def cmd_judge(a) -> None:
    rows = prompts("test", a.limit or None)
    note = (
        (ROOT / "prompts" / "jb_v1_voice_note.txt").read_text(encoding="utf-8")
        if a.voice_note
        else ""
    )
    label(
        rows,
        OUT / f"judge_{a.name}.json",
        model=a.model,
        rule="test",
        note=note,
        reasoning={"enabled": False},
        temperature=0.0,
        provider=a.provider,
        max_usd=a.max_usd,
    )


def cmd_table(a) -> None:
    cells = json.loads((DATA / "cells.json").read_text(encoding="utf-8"))
    gold_all = json.loads(Path(a.gold).read_text(encoding="utf-8"))
    judges = json.loads((DATA / "test_labels.json").read_text(encoding="utf-8"))
    judges.update(
        {p.stem[6:]: json.loads(p.read_text(encoding="utf-8")) for p in OUT.glob("judge_*.json")}
    )
    items = cells["splits"]["test_items"]
    for name, sl in (
        ("primary (568 cells)", slice(200, 768)),
        ("secondary (768 cells)", slice(0, 768)),
    ):
        gold = {c: gold_all[c] for c in cells["send_order"][sl] if c in gold_all}
        n_h = Counter(cells["cell_stratum"][c] for c in gold)
        w = {
            c: cells["N_h"][cells["cell_stratum"][c]] / n_h[cells["cell_stratum"][c]] for c in gold
        }
        print(
            f"\n{name}: {len(gold)} gold cells\n| judge | κ_rec | LB95 | joint cells | κ 4-way |\n|---|---|---|---|---|"
        )
        for j, lab in judges.items():
            r = b1.compare(gold, w, lab, items)
            print(
                f"| {j} | {r3(r['kappa_recognition'])} | {r3(r['lb95_recognition'])} | {r['joint_cells']} | {r3(r['kappa_4way'])} |"
            )


def kappa(pairs) -> float:
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    cx, cy = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def paired(gold, item, a, b, name_a, name_b) -> None:
    """kappa_rec of two judges on the readouts all three share; item-bootstrap CI of the difference (b - a)."""
    rows = defaultdict(list)
    for h in sorted(set(gold) & set(a) & set(b)):
        if gold[h] in C and a[h] in C and b[h] in C:
            rows[item[h]].append(
                (gold[h] == "recognition", a[h] == "recognition", b[h] == "recognition")
            )
    flat = [r for v in rows.values() for r in v]
    ka, kb = kappa([(g, x) for g, x, _ in flat]), kappa([(g, y) for g, _, y in flat])
    its, rng, d = sorted(rows), random.Random(0), []
    for _ in range(10_000):
        s = [r for it in (rng.choice(its) for _ in its) for r in rows[it]]
        d.append(kappa([(g, y) for g, _, y in s]) - kappa([(g, x) for g, x, _ in s]))
    d = sorted(x for x in d if not math.isnan(x))
    print(
        f"{len(flat)} readouts, {len(its)} items, {sum(g for g, _, _ in flat)} Sonnet recognitions: "
        f"{name_a} {ka:.3f} | {name_b} {kb:.3f} | difference 95% CI [{d[int(0.025 * len(d))]:.3f}, {d[int(0.975 * len(d)) - 1]:.3f}]"
    )


def cmd_train(a) -> None:
    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))
    item = {r["h"]: r["item"] for r in prompts("train")}
    t = json.loads((DATA / "train_labels.json").read_text(encoding="utf-8"))
    note, q38, q36 = (
        t["Qwen3.8-27B + voice note"],
        t["Qwen3.8-27B, official prompt"],
        t["Qwen3.6-27B, official prompt"],
    )
    print("teacher gate:     ", end="")
    paired(gold, item, q36, note, "Qwen3.6-27B, official", "Qwen3.8-27B + note")
    print("voice-note ablation: ", end="")
    paired(gold, item, q38, note, "Qwen3.8-27B, official", "Qwen3.8-27B + note")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gold")
    g.add_argument("--split", choices=["test", "train"], default="test")
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--max-usd", type=float, default=15.0)
    j = sub.add_parser("judge")
    j.add_argument("--model", required=True)
    j.add_argument("--name", required=True)
    j.add_argument("--voice-note", action="store_true")
    j.add_argument("--provider", default="DeepInfra")
    j.add_argument("--limit", type=int, default=0)
    j.add_argument("--max-usd", type=float, default=2.0)
    for name in ("table", "train"):
        s = sub.add_parser(name)
        s.add_argument(
            "--gold", default=str(OUT / f"gold_{'test' if name == 'table' else 'train'}.json")
        )
    a = ap.parse_args()
    {"gold": cmd_gold, "judge": cmd_judge, "table": cmd_table, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
