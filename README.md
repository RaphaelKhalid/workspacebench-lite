# WorkspaceBench-lite: an open-weights judge for `jailbreak_recognition`

**An Apache-2.0 judge agrees with WorkspaceBench's Sonnet 5 judge at κ = 0.714, past the benchmark's κ ≥ 0.70 judge-swap bar. It costs about 4× less through an API and about 15× less self-hosted.**

| | Official judge | This repo |
|---|---|---|
| Models | Sonnet 5 summarizer + Sonnet 5 judge | Qwen3.6-27B summarizer + **Qwen3.8-27B** judge (open weights, Apache-2.0) |
| Cost per J-lens arm (~144k cells) | **≈ $907** | **≈ $220** via OpenRouter · **≈ $60** self-hosted on one A100 (estimate) |
| Recognition κ vs Sonnet 5 (568 held-out cells) | n/a (it is the reference) | **0.714** |
| Unjudged cells | n/a | 1.2% (the benchmark's limit is 2%) |

## What this is
- **The family:** `jailbreak_recognition` asks whether a model's internal readouts show it *recognising* a jailbreak. It has 86 WildChat conversations.
- **Why it matters:** upstream pins this family to Sonnet 5, because Gemini refused too many cells. It is about 80% of the benchmark's judging bill: $907 of $1,136 per J-lens arm.
- **What this repo adds:** a drop-in open-weights judge for it, checked against Sonnet with the benchmark's own judge-swap metric (Cohen's κ on the binary cell label, [plans/0007](https://github.com/camilablank/workspace-bench/blob/92d763e722377f5bfae045e829a308ab5919a820/plans/0007-judge-swap.md)).

## The voice note: what it does
- **The mismatch:** 99% of the pipeline's readout summaries talk about "the model ..." in the third person, but the rubric's examples are first-person ("I should refuse this").
- **The note:** [`prompts/jb_v1_voice_note.txt`](prompts/jb_v1_voice_note.txt) restates the rubric's own examples in the summary's voice. Nothing else in the prompt changes.
- **Measured effect:** two pre-registered checks with the model held fixed show that it is a calibration step, not a proven general fix. It consistently makes the judge more conservative about recognition, and its effect on κ differs by item set:

  | Qwen3.8-27B | official prompt | + voice note |
  |---|---|---|
  | Held-out test (568 cells), κ_rec | 0.648 | **0.714** |
  | Held-out test, recognitions (judge vs Sonnet's 30.0) | 45.8 (over-calls) | 30.0 |
  | Training items (1,898 readouts), κ_rec | 0.753 | 0.711 (n.s.) |
  | Training items, recognitions | 48 | 28 |

- **Why it's allowed:** prompt review is what the benchmark prescribes when a swapped judge scores below 0.7.

## Quickstart
```bash
git clone --recursive https://github.com/RaphaelKhalid/workspacebench-lite
cd workspacebench-lite
uv sync        # WorkspaceBench (submodule pinned at 92d763e) + this package
export OPENROUTER_API_KEY=sk-or-...
uv run python -m wsbjev judge family=jailbreak_recognition   readouts=path/to/jlens.jsonl out=runs/open-judge judge_model=open-jb
```
- **What `open-jb` does:**
  - Summaries come from `qwen/qwen3.6-27b`.
  - Verdicts come from `qwen/qwen3.8-27b`, with the voice note appended.
  - Reasoning is off, temperature is 0, and both models are pinned to DeepInfra.
- **Readouts** come from WorkspaceBench's pipeline ([producing readouts](https://github.com/camilablank/workspace-bench/blob/92d763e722377f5bfae045e829a308ab5919a820/docs/producing_readouts.md)).
- **Scoring** uses the family's own pass@any rule.

## Results
On 568 held-out cells from 17 conversations, which were never used for training or prompt design.

| judge | recognition κ vs Sonnet 5 | one-sided 95% LB |
|---|---|---|
| **Qwen3.8-27B + voice note** | **0.714** | 0.552 |
| Qwen3.6-27B, official prompt | 0.693 | 0.527 |
| Qwen3.8-27B, official prompt | 0.648 | 0.448 |
| free Jev (zero-shot) | 0.654 | 0.459 |
| Kev-4B distilled from Qwen3.6-27B | 0.639 | 0.471 |
| Kev-4B distilled from Qwen3.8-27B + voice note | 0.582 | 0.419 |
| Kev-4B distilled from GLM-5.3 Flash | 0.580 | 0.354 |

## How we got here
1. **Port.** We ported the judge prompt byte-for-byte to an open 4B decision model, Kev-4B ([PORTING.md](PORTING.md)).
2. **Distil.** We distilled Kev-4B from open teachers. The best reached 0.639: a student copies its teacher's mistakes.
3. **Diagnose.** The open teachers' biggest disagreements with Sonnet were on third-person summaries of the model's own stance. That led to the voice note. This was diagnosed on training conversations only.
4. **Upgrade.** On training conversations, agreement jumped from 0.511 (Qwen3.6-27B) to 0.693 (Qwen3.8-27B + note). A same-model ablation shows the model upgrade drove that jump, not the note.
5. **Calibrate.** On the held-out test, Qwen3.8-27B over-calls recognition with the official prompt (0.648). With the note it matches Sonnet's recognition rate and reaches **0.714**.
6. **Open problem.** A 4B student of that teacher drops back to 0.582. Distilling the gain into a 4B judge is still unsolved.

## Caveats
- **The bar is passed on the point estimate only.** The one-sided 95% lower bound is 0.552, and over all 768 Sonnet-labelled test cells the score is 0.675.
- **The passing configuration needs the note.** Plain Qwen3.8-27B does not pass (0.648), and the note's effect on κ was the opposite sign on training items. Treat the pass as specific to this configuration.
- **The held-out cells were read by several judges.** That includes two Qwen3.8 configurations; the note version was pre-registered and read first. Each configuration was pre-registered (sha256 in [`prereg/`](prereg/)) and read once, and every read is reported in [`results/`](results/).
- **Our summaries come from Qwen3.6-27B.** The official pipeline's summaries come from Sonnet. The gold labels are Sonnet 5 judging these same summaries.

## Cost details
| | measured |
|---|---|
| Sonnet 5, judge stage, via OpenRouter | $0.0056 / cell |
| Qwen3.8-27B judge, DeepInfra | $0.00049 / cell (768 cells) |
| Open pipeline end to end, summarizer + judge | ≈ $0.0015 / cell (one 65-cell item) |
| Official judging, one J-lens arm (dry-run costing at list prices) | $907 |

The self-hosted figure assumes the measured 27B throughput on one A100 at $1.39/h (≈ 1.9 cells/s per stage).

## Layout
| folder | contents |
|---|---|
| `src/wsbjev/` | judge routes (`open-jb`, `vllm:`, `replay:`, `capture`) and the Kev-4B port |
| `prompts/` | the voice note |
| `scripts/` | labelling, scoring and the κ estimator with item bootstrap |
| `pod/` | Kev-4B setup, training and test scripts for one A100 |
| `prereg/` | pre-registrations, each frozen with sha256 before its data existed |
| `results/` | every result, including the failed attempts |
| `ported_questions/` | the 86 jailbreak items as Kev-4B questions |
| `dashboard/` | the live training monitor |

## Cite
This work builds on WorkspaceBench. Following its guidelines, please cite **the repo and the write-up**:

```bibtex
@misc{blank2026workspacebench,
  author       = {Blank, Camila and Bhatia, Agam and Ong, Euan and Nanda, Neel},
  title        = {WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace},
  year         = {2026},
  howpublished = {\url{https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the}}
}
```
- **WorkspaceBench repo:** https://github.com/camilablank/workspace-bench
- **The items are WildChat conversations.** Zhao, Ren, Hessel, Cardie, Choi, Deng, *WildChat: 1M ChatGPT Interaction Logs in the Wild*, ICLR 2024 (arXiv:2405.01470), ODC-BY.
- **Citing this repo:** use [`CITATION.cff`](CITATION.cff), which feeds GitHub's "Cite this repository" button.

**Credits:** [WorkspaceBench](https://github.com/camilablank/workspace-bench) by Blank, Bhatia, Ong and Nanda (MIT; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)) · WildChat (ODC-BY) · Qwen (Apache-2.0) · Kev-4B by jaredpalmer (Apache-2.0).

**Licence:** MIT ([LICENSE](LICENSE)).
