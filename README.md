# WorkspaceBench-lite: an open-weights judge for `jailbreak_recognition`

**Qwen3.8-27B (open weights, Apache-2.0) with a one-paragraph prompt clarification reaches recognition κ = 0.714 against WorkspaceBench's Sonnet 5 judge (one-sided 95% lower bound 0.552). That clears the benchmark's κ ≥ 0.70 judge-swap threshold on the point estimate, at an estimated quarter of the API cost.**

Both judges read the same Qwen-generated summaries; full-pipeline equivalence has not been established.

## Why
- **The family:** `jailbreak_recognition` asks whether a model's internal readouts show it recognising a jailbreak (86 WildChat conversations).
- **Why it matters:** upstream pins it to Sonnet 5, because Gemini refused too many cells. It is about 80% of WorkspaceBench's judging bill.
- **Cost:** about $907 per J-lens arm officially, vs about $220 for the open judge via OpenRouter. An estimate of about $60 self-hosted.
- **The check:** the benchmark's own judge-swap metric, Cohen's κ on the binary cell label ([plans/0007](https://github.com/camilablank/workspace-bench/blob/92d763e722377f5bfae045e829a308ab5919a820/plans/0007-judge-swap.md)).

## Use it
```bash
git clone --recursive https://github.com/RaphaelKhalid/workspacebench-lite
cd workspacebench-lite && uv sync
export OPENROUTER_API_KEY=sk-or-...

# free: print the judge prompt for a sample cell from WorkspaceBench's example readouts
uv run python -m wsbjev judge family=jailbreak_recognition judge_model=open-jb dry_run=True \
  readouts=third_party/workspace-bench/examples/readouts/jailbreak_recognition.jsonl out=runs/dry

# your own J-lens readouts
uv run python -m wsbjev judge family=jailbreak_recognition judge_model=open-jb \
  readouts=path/to/jlens.jsonl out=runs/open-judge
```
- **`open-jb`:** summaries by `qwen/qwen3.6-27b`; verdicts by `qwen/qwen3.8-27b` with the [voice note](prompts/jb_v1_voice_note.txt) appended. Reasoning off, temperature 0, both pinned to DeepInfra.
- **Numbers of record:** like any judge override, upstream doesn't count its runs as numbers of record.

## Results
568 cells from 17 conversations held out from training and from writing the voice note; 200 other cells from them were used to audition teachers. κ is computed on the cells both judges judged.

| judge | κ_rec vs Sonnet 5 | LB95 | joint cells |
|---|---|---|---|
| **Qwen3.8-27B + voice note** | **0.714** | 0.552 | 561 |
| Qwen3.6-27B, official prompt | 0.693 | 0.527 | 530 |
| Qwen3.8-27B, official prompt | 0.648 | 0.448 | 564 |
| Kev-4B (4B) distilled from Qwen3.6-27B | 0.639 | 0.471 | 568 |
| Kev-4B distilled from Qwen3.8-27B + voice note | 0.582 | 0.419 | 568 |
| Kev-4B distilled from GLM-5.3 Flash | 0.580 | 0.354 | 568 |

- **The voice note is a calibration step, not a general fix.** The pipeline's summaries are third-person ("the model is…"), while the rubric's examples are first-person. The note restates those examples in the summaries' voice.
  - **On the test:** it makes Qwen3.8 more conservative. The judge stops over-calling recognition and matches Sonnet's rate: 0.648 → 0.714.
  - **On training conversations:** no significant difference (0.753 vs 0.711).
- **Students copy their teachers.** 4B students stay at 0.58–0.64.

## Caveats
- **Point estimate only:** it passes on the point estimate (LB 0.552), and scores 0.675 over all 768 Sonnet-labelled test cells.
- **The note is required:** the pass depends on it; plain Qwen3.8 doesn't pass.
- **Several reads of the same cells:** the test cells were read by several judges. Each configuration was pre-registered ([`prereg/`](prereg/)) and read once, and every read of the configurations here is in [`results/`](results/).
- **Grid:** agreement was measured on the 13-site read grid, not the ~144k-cell grid the cost figures assume.

## Reproduce
- **Everything above can be recomputed:** see [REPRODUCE.md](REPRODUCE.md).
- **What ships:** open-model data, in [`data/`](data/).
- **What doesn't:** Sonnet 5's labels. They aren't redistributed (WorkspaceBench doesn't redistribute them either); `scripts/reproduce.py gold` regenerates them for about $4.40.

## Upstream contributions
- [#66](https://github.com/camilablank/workspace-bench/pull/66): a judge override also changes the summarizer (the README said otherwise).
- [#67](https://github.com/camilablank/workspace-bench/pull/67): documents which route a judge uses and which reasoning setting it requests.
- [#68](https://github.com/camilablank/workspace-bench/pull/68): one moderation-flagged prompt no longer aborts a whole eval family.

## Open work
- **Fresh conversations:** evaluate the frozen configuration on conversations outside the current test pool.
- **Full pipelines:** compare complete pipelines end to end, including item-level scores, on the full read grid.
- **Sonnet's ceiling:** measure how well Sonnet agrees with itself, to know the achievable agreement.
- **Human audit:** check disagreements against human judgment. Agreeing with Sonnet shows imitation of an instrument, not correctness.
- **Engineering:** a self-hosted (`vllm:`) route with the voice note, and a 4B judge that keeps the gain.

## Cite
This builds on WorkspaceBench. Following its guidelines, cite the repo and the write-up:
```bibtex
@misc{blank2026workspacebench,
  author       = {Blank, Camila and Bhatia, Agam and Ong, Euan and Nanda, Neel},
  title        = {WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace},
  year         = {2026},
  howpublished = {\url{https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the}}
}
```
- **Data:** the conversations come from WildChat (Zhao et al., ICLR 2024, ODC-BY).
- **Citing this repo:** use [`CITATION.cff`](CITATION.cff).
- **Credits and licences:** [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This repo is MIT ([LICENSE](LICENSE)).
