# Third-party notices

## WorkspaceBench
`ported_questions/OFFICIAL_SYSTEM_PROMPT.txt` and the judge prompts inside `ported_questions/jailbreak_recognition_ported.jsonl` are copied byte-for-byte from WorkspaceBench (https://github.com/camilablank/workspace-bench @ 92d763e). WorkspaceBench is installed as a dependency and is not vendored here.

```
MIT License

Copyright (c) 2026 Camila Blank

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## WildChat (Zhao et al. 2024)
- **Source:** Zhao, Ren, Hessel, Cardie, Choi, Deng. *WildChat: 1M ChatGPT Interaction Logs in the Wild.* ICLR 2024. arXiv:2405.01470. https://huggingface.co/datasets/allenai/WildChat-1M
- **Included:** the conversations inside `ported_questions/` and `data/` (the judge prompts and the Kev training records) are verbatim WildChat conversations, via WorkspaceBench.
- **Licence:** ODC-BY (Open Data Commons Attribution License v1.0). Attribution is required when redistributing or building on these items; this file provides it.

## CHIVE (Karvonen et al.)
- **Included:** the `[role]: content` transcript render used inside the ported judge prompts. It is WorkspaceBench's `jailbreak_recognition` render contract. https://github.com/adamkarvonen/chive
- No CHIVE code is included.

## Qwen (Alibaba)
- The open pipeline calls `Qwen/Qwen3.6-27B` (summarizer) and `Qwen/Qwen3.8-27B` (judge). Both are Apache-2.0 weights, served here through OpenRouter (DeepInfra).
- `data/` contains outputs of these models, of GLM-5.3 Flash (MIT) and of Kev-4B (Apache-2.0, jaredpalmer). No weights are redistributed.
