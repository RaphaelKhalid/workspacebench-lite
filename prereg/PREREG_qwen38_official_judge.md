# Qwen3.8-27B with the UNMODIFIED official prompt as the judge (pre-registration, 2026-10-01, before any call)

**Why.** PREREG_voice_ablation.md found that the voice note does not raise κ_rec on training items. There Qwen3.8-27B scored official 0.753 vs note 0.711. So the cleanest judge swap is the model alone, with WorkspaceBench's jb-v1 prompt byte-for-byte.

**Configuration.** `qwen/qwen3.8-27b`, DeepInfra pinned, reasoning disabled, temperature 0, max_tokens 1,500, 3 attempts. The official prompts are unchanged (`runs/sonnet/prompts_test_ordered.jsonl`), with Qwen3.6-27B summaries as in every run.

**Cells.** The 768 Sonnet-labelled test cells (positions 0–767).

**Metric.**
- **Primary:** κ_rec vs API Sonnet 5 on the 568 clean cells (positions 200–767) with sonnet_b1's weighted estimator. Bar: κ_rec ≥ 0.70, the benchmark's rule.
- **Also reported:** the LB95, κ 4-way, the 768-cell secondary, unjudged-cell rate (must be ≤ 2%), and the Qwen3.8 + voice note read (0.714) side by side.

**Disclosure.** This is an additional read of these test cells, after three Kev-4B student reads, the zero-shot reads, and the Qwen3.8 + note read. It is one configuration, read once, and all reads are reported.

**Spend.** OpenRouter ≤ $0.60.
