# Result: the voice note does not raise recognition κ (PREREG_voice_ablation.md)

**Primary (pre-registered).** Qwen3.8-27B, same serving: DeepInfra pinned, reasoning off, temperature 0. 1,898 Sonnet-labelled training readouts over 57 items, with 39 Sonnet recognitions.

| Qwen3.8-27B | κ recognition vs Sonnet 5 | κ 4-way | recognitions said | echo said |
|---|---|---|---|---|
| official jb-v1 prompt | **0.753** | 0.635 | 48 | 11 |
| + voice note | 0.711 | 0.653 | 28 | 27 |

- **Note minus official:** −0.042, paired item-bootstrap 95% CI [−0.195, +0.106]. **The note does not work by the pre-registered test.**
- **What the note does instead:** it makes the judge more conservative.
  - It removes 11 of the official prompt's 15 false recognitions.
  - It recovers none of the 6 Sonnet recognitions the official prompt missed.
  - It roughly doubles echo use.
- **What the earlier gain was:** the 0.511 → 0.693 teacher gain in PREREG_qwen_max.md came from the model change (Qwen3.6 → Qwen3.8), not from the note.

**Secondary (Qwen3.6-27B ± note): no usable data.**
- **DeepInfra:** structured output degenerated into whitespace until max_tokens (94 of 149 calls).
- **Alibaba (Amendment 1):** it ignored the JSON schema (`readouts`/`class` keys) or returned no JSON (461 of 1,078). Both are dropped by the pre-registered rules.
- **A clean Qwen3.6 comparison** would need schema-enforced vLLM serving, the same as the baseline.

**What still holds.**
- **Summary voice:** 99% of the pipeline's summaries describe "the model" in the third person, while the rubric's examples are first-person. That is a fact about the instrument, but this ablation does not show that it lowers a strong judge's agreement.
- **The test result:** the held-out 0.714 (PREREG_qwen38_judge.md) stands as a measurement of the Qwen3.8 + note configuration. The note is not why it passes.

**Spend.** OpenRouter $3.21: Qwen3.8 official arm $1.02, Qwen3.6 on DeepInfra $0.55, Qwen3.6 on Alibaba $1.64. That is within the $3.25 hard stop.
