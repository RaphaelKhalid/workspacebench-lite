# Result: Qwen3.8-27B with the unmodified official prompt does not pass (PREREG_qwen38_official_judge.md)

**Primary: 568 clean test cells.** Qwen3.8-27B with the official jb-v1 prompt, byte-for-byte, scores κ_rec **0.648** (LB95 0.448) and κ 4-way 0.630, on 564 joint cells (4 unjudged).
- That is below the 0.70 bar.
- **It over-calls recognition:** 45.8 weighted recognitions vs Sonnet's 30.0. Item pass rate is 0.765 vs 0.588.
- **Secondary (all 768 cells):** κ_rec 0.594.

| judge vs Sonnet 5 (568 clean cells) | κ_rec | LB95 | weighted recognitions (judge / Sonnet) |
|---|---|---|---|
| Qwen3.8-27B + voice note | **0.714** | 0.552 | 30.0 / 30.0 |
| Qwen3.8-27B, official prompt | 0.648 | 0.448 | 45.8 / 30.0 |

**What the voice note does, across both reads.** It consistently makes the judge more conservative about recognition.
- **Training items:** 48 → 28 recognitions, and κ_rec 0.753 → 0.711 (n.s., PREREG_voice_ablation.md).
- **Test items:** 45.8 → 30.0 weighted recognitions, and κ_rec 0.648 → 0.714.

Its effect on agreement depends on whether the plain model over-calls on those items. So it acts as a calibration change, and a general gain in κ is not established.

**Spend.** OpenRouter $0.38.
