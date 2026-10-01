# Reproduce

Every number in the README can be recomputed from this repo. We checked that the commands below reproduce the published numbers exactly, confidence intervals included, from our own Sonnet labels.

## What ships, what doesn't
- **Ships, in [`data/`](data/):** open data only.
  - The official judge prompts for the 768 test cells and the 3,529 training prompts. They're checked byte for byte against WorkspaceBench's own prompt via their sha256.
  - Each open judge's per-cell labels.
  - The Kev-4B training records.
  - The J-lens readouts.
  - The item splits and sampling strata.
- **Doesn't ship: Sonnet 5's labels.** WorkspaceBench doesn't redistribute judge verdicts either. You regenerate them with the script below. Sonnet's labels can vary slightly from run to run, so regenerated numbers may move a little.

## Setup
```bash
git config --global core.longpaths true   # Windows only: WorkspaceBench has deeply nested files
git clone --recursive https://github.com/RaphaelKhalid/workspacebench-lite
cd workspacebench-lite && uv sync
export OPENROUTER_API_KEY=sk-or-...
```

## Steps
| what | command | cost |
|---|---|---|
| Sonnet 5 labels on the 768 test cells | `uv run python scripts/reproduce.py gold` | ≈ $4.40 |
| The results table (primary 568 + all 768 cells) | `uv run python scripts/reproduce.py table` | free |
| Sonnet 5 labels on 1,927 training prompts | `uv run python scripts/reproduce.py gold --split train` | ≈ $10.90 |
| The teacher gate and the voice-note ablation | `uv run python scripts/reproduce.py train` | free |
| Re-run an open judge (it joins the table) | `uv run python scripts/reproduce.py judge --model qwen/qwen3.8-27b --voice-note --name mine` | ≈ $0.40 |

- **Resumable:** the paid commands can be stopped and restarted, and they stop before passing `--max-usd`.
- **Kev-4B students:**
  - **Records:** `data/kev_train/<teacher>.jsonl.gz`.
  - **Recipe:** `kev.train --init_from jaredpalmer/kev-4b`, 2 epochs, lr 5e-5, batch 1 × accum 8, bf16, `--max_state 2560`, augmentations off.
  - **Cost:** about $2 on one A100. Our pod scripts are in [`pod/`](pod/).
- **The full pipeline:** run the `open-jb` route (see the README) on `data/jlens_13site.jsonl.gz`, after gunzipping it.
