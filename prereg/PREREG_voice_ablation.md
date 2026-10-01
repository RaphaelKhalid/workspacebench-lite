# Does the voice note itself raise agreement with Sonnet? (pre-registration, 2026-10-01, before any ablation call)

**Question.** Holding the model fixed, does appending `prompts/jb_v1_voice_note.txt` (sha256 7f744e7a…) to the official jb-v1 system prompt raise recognition κ vs Sonnet 5?

**Cells.** The 1,927 Sonnet-labelled TRAINING prompts: the first 1,927 of `runs/sonnet_teacher/prompts_train.jsonl` in the pre-registered order. There are no test cells.

**Arms.** All use the same settings as PREREG_qwen_max.md: OpenRouter, DeepInfra pinned, reasoning disabled, temperature 0, max_tokens 1,500, 3 attempts.

| | official prompt | official + voice note |
|---|---|---|
| Qwen3.8-27B | **new run** | existing (PREREG_qwen_max.md teacher labels) |
| Qwen3.6-27B | existing vLLM labels (different serving; caveat) | **new run** |

**Primary.** Within Qwen3.8-27B, κ_rec(note) − κ_rec(official) on the readouts all arms share.
- The CI is a paired item bootstrap: 10,000 resamples of the 57 training items, seed 0.
- The note "works" if the 95% CI excludes 0.

**Secondary.**
- The same paired difference within Qwen3.6-27B, confounded by serving (vLLM vs OpenRouter/DeepInfra).
- κ 4-way.
- For the Sonnet-recognition readouts that the official arm missed, the share the note arm recovers, and the reverse.
- Echo usage.

**Rules.** The same as the gate: missing readout = noise, labels lower-cased, off-list labels skipped, off-schema responses dropped.

**Spend.** Expected ≈ $2.70: Qwen3.8 ≈ $1.10, Qwen3.6 ≈ $1.60. Hard stops: Qwen3.8 +$1.35 on its ledger, Qwen3.6 $1.90, so at most $3.25. The per-model script cap is raised to allow it.

## Amendment 1 (before any κ was computed for the Qwen3.6 arm)
- **What happened:** on DeepInfra, Qwen3.6-27B's structured output degenerates. It writes the verdict JSON, then blank lines until max_tokens: 94 of 149 calls, with 0 reasoning tokens.
- **Stop:** the run stopped itself on its 8-consecutive-failures guard after 79 prompts, at $0.55.
- **Change:** the Qwen3.6 + note arm is relabelled from scratch on a different pinned provider (Alibaba, the model's maker).
  - The run is tagged `note_alibaba`, and the DeepInfra rows are not used.
  - Settings are otherwise identical.
- **Budget:** the arm's stop is set so the experiment total stays under the pre-registered $3.25.
  - This may leave the arm short of 1,927 prompts. In that case the paired comparison uses the readouts the arms share, in the pre-registered order.
