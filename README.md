# 🦦 Gemma 4 31B — Layer Importance & Quantization Study

Empirical study of Google's **Gemma 4 31B Instruct**: which transformer layers are redundant, which quantization schemes preserve quality, and what the Pareto frontier looks like below 3 bits-per-weight.

- **Interactive site**: https://kikocisbot.github.io/gemma4-31b-study/
- **Raw data + scripts**: https://huggingface.co/datasets/KikoCis/gemma4-31b-layer-study

## Featured result — custom IQ2_M ⭐

**[KikoCis/gemma-4-31b-it-IQ2_M-GGUF](https://huggingface.co/KikoCis/gemma-4-31b-it-IQ2_M-GGUF)** — 10.17 GB, 2.84 BPW

| Metric   | This IQ2_M | f16 baseline |
|----------|-----------:|-------------:|
| Size     | **10.17 GB** | 61.4 GB |
| Char-F1  | **84.71%** | 84.76% |
| BLEU-4   | **22.39**  | 21.02   |
| Exact    | **12%**    | 12%     |

Matches f16 on Char-F1 and **beats f16 on BLEU-4** at **6× smaller**. Beats Unsloth UD-IQ2_M on every metric at 0.6 GB smaller — CLI-focused imatrix calibration pushes the sub-3-BPW Pareto frontier further than Unsloth's generic Dynamic 2.0.

## Code &amp; Agentic benchmarks — IQ2_M (first public evaluation of Gemma 4 31B on these)

Google officially reports only **LiveCodeBench v6 = 80%** and **Tau2 = 76.9%** for Gemma 4 31B IT. HumanEval+/MBPP+/BFCL scores are not published. This appears to be the first public evaluation of Gemma 4 31B on these benchmarks — done on the custom IQ2_M quantization (not f16).

| Benchmark | Score | Passed / Total | Top failure classes |
|-----------|------:|---------------:|--------------------|
| HumanEval+ | **88.41%** pass@1 | 145 / 164 | wrong_output (15), missing_import (3) |
| MBPP+ | **82.01%** pass@1 | 310 / 378 | wrong_output (35), missing_import (28) |
| BFCL v3 simple | **92.25%** accuracy | 369 / 400 | args_mismatch (30), wrong_name (1) |

118 failure cases saved in [`benchmarks/failures_iq2m.jsonl`](lab/gemma-e2b-cli/benchmarks/failures_iq2m.jsonl) — seed for domain-mixed imatrix calibration and failure-driven LoRA training (next phase).

## Full benchmark (NL2Bash, 50 examples, Stanford/Tellina)

| Rank | Model | Size | BPW | Char-F1 | BLEU-4 |
|-----:|-------|------|-----|--------:|-------:|
| 1 | [Unsloth UD-IQ3_XXS](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) | 11.84 GB | ~2.98 | 85.06% | 20.26 |
| 2 | Base f16 (reference) | 61.4 GB | 16.0 | 84.76% | 21.02 |
| 3 | [Base IQ3_XS (this project)](https://huggingface.co/KikoCis/gemma-4-31b-it-IQ3_XS-GGUF) | 13.1 GB | 3.40 | 84.76% | 18.95 |
| **4** | **[IQ2_M CLI imatrix (this project)](https://huggingface.co/KikoCis/gemma-4-31b-it-IQ2_M-GGUF)** | **10.17 GB** | **2.84** | **84.71%** | **22.39** |
| 5 | [Unsloth UD-IQ2_M](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) | 10.75 GB | ~2.70 | 84.02% | 18.64 |
| 6 | [Otter v3 Q3_K_M (58-layer pruned)](https://huggingface.co/KikoCis/gemma-4-31b-otter-v3-GGUF) | 14.1 GB | 3.98 | 79.89% | 12.80 |
| 7 | [Otter v3 IQ3_XS (58-layer pruned)](https://huggingface.co/KikoCis/gemma-4-31b-otter-v3-GGUF) | 12.1 GB | 3.41 | 78.72% | 15.14 |
| 8 | [Unsloth UD-IQ2_XXS](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) | 8.53 GB | ~2.06 | 76.82% | 11.49 |
| 9 | Base Q2_K (3rd party) | 11.0 GB | 2.70 | 58.60% | 6.92 |

## Key findings

- **Custom IQ2_M is the new sub-3-BPW Pareto point** — matches f16 on Char-F1 and beats it on BLEU-4, at 6× smaller than f16 and 0.6 GB smaller than Unsloth UD-IQ2_M.
- **Q2_K collapses on Gemma 4** (F1 58.6%). The wide logit distribution + RMSNorm scaling is incompatible with scalar blockwise 2-bit.
- **Layer pruning without recovery fine-tuning is dominated** by dynamic quantization: the 58-layer Otter-v3 IQ3_XS (12.1 GB) loses 6 F1 points to the 60-layer IQ2_M (10.17 GB).
- **Layers 27 and 29 are net-negative** on perplexity: single-layer ablation REDUCES PPL by 74% and 71% respectively. Block ablation {27, 29}: PPL drops from 496 to 117 (-76%).

## Cross-model benchmark comparison (first public eval of Qwen3.6 + Gemma 4 on HumanEval+/MBPP+/BFCL)

| Model | Size | Active | HumanEval+ | MBPP+ | BFCL v3 | NL2Bash F1 |
|-------|------|--------|------------|-------|---------|------------|
| **Gemma 4 31B IQ2_M (ours)** | **10.4 GB** | 31B | **88.41%** | **82.01%** | 92.25% | **84.71%** |
| Qwen3.6 Q8_0 (≈f16) | 35.2 GB | 3B | 81.10% | 82.80% | **95.25%** | — |
| Qwen3.6 Unsloth UD-IQ2_M | 11.0 GB | 3B | 82.32% | 79.37% | 94.25% | — |
| **Qwen3.6 IQ2_M (ours)** | **11.1 GB** | 3B | 80.49% | 78.31% | **94.75%** | 81.63% |
| **Gemma 4 E4B Q8_0** | **7.8 GB** | 4.5B | 73.78% | 73.28% | 93.75% | 79.75% |
| Gemma 4 31B IQ1_M | 7.4 GB | 31B | 21.34% | 40.50% | 86.75% | 53.89% |

### Recommendation by use case

| Use case | Best model | Size |
|----------|-----------|------|
| Code generation | Gemma 4 31B IQ2_M | 10.4 GB |
| Agentic / tool-calling | Qwen3.6 IQ2_M | 11.1 GB |
| Sub-8 GB all-rounder | Gemma 4 E4B Q8_0 | 7.8 GB |

## Real-World Agent Test — The Benchmark Trap

**Critical finding (April 2026): benchmark scores do not predict agent capability.**

We tested models on an autonomous Docker-based bioinformatics challenge (download P53_HUMAN from UniProt, parse JSON, extract secondary structure + cancer mutations, generate HTML report). Same container, same tools, same prompt.

| Model | Size | BFCL | **Agent Score** | Failure Mode |
|-------|------|------|:---:|------|
| Claude Sonnet 4.6 | Cloud | — | **10/10** | None |
| E4B Base Q3_K_M | 3.8 GB | 80.25% | **6/10** | Completed, data incomplete |
| 31B IQ2_M (ours) | 10.4 GB | 92.25% | **2/10** | Empty "N/A" results |
| **E4B FT Q4_K_M** | **5.1 GB** | **95.50%** | **0/10** | "Hello World" + infinite loop |
| **E4B FT Q3_K_M** | **4.5 GB** | **93.75%** | **0/10** | `apt-get install python3` ×14 |

**Key lesson:** The fine-tuned E4B models (93-95% BFCL) are catastrophically worse than the base model (80% BFCL, 6/10 agent) on real tasks. Fine-tuning for BFCL taught the model to format JSON perfectly while destroying error recovery, strategy adaptation, and anti-repetition capabilities.

**The fine-tuned E4B models have been withdrawn. Use base Gemma 4 E4B instead.**

## Published models

- ⭐ 🤗 **[gemma-4-31b-it-IQ2_M-GGUF](https://huggingface.co/KikoCis/gemma-4-31b-it-IQ2_M-GGUF)** — best for code, 10.17 GB, HE+ 88.41%
- ⭐ 🤗 **[Qwen3.6-35B-A3B-IQ2_M-GGUF](https://huggingface.co/KikoCis/Qwen3.6-35B-A3B-IQ2_M-GGUF)** — best for agentic, 11.1 GB, BFCL 94.75%
- 🤗 [gemma-4-31b-it-IQ3_XS-GGUF](https://huggingface.co/KikoCis/gemma-4-31b-it-IQ3_XS-GGUF) — 13.1 GB, F1 84.76%, ties f16
- 🤗 [gemma-4-31b-otter-v3-GGUF](https://huggingface.co/KikoCis/gemma-4-31b-otter-v3-GGUF) — research artifact (58-layer pruned, no fine-tuning)
- 📊 [gemma4-31b-layer-study dataset](https://huggingface.co/datasets/KikoCis/gemma4-31b-layer-study) — raw data + scripts + visualization

## Methodology

3-phase layer analysis:
- **Phase A** — Block Influence (BI): per-layer residual-stream modification, `1 − cos(h_in, h_out)`.
- **Phase B** — Logit Lens: projecting each hidden state through the final norm + `lm_head`.
- **Phase C** — Single-layer ablation: measure perplexity delta when a layer is zeroed.

Benchmark: NL2Bash 50-example subset from the official Stanford/Tellina test split. Each question runs as an isolated `llama-cli` subprocess with `--reasoning off`, `--max-tokens 200`, `--temp 0.1`, `--ctx-size 4096`. Scoring via NLTK BLEU + character-level F1 + exact match.

All measurements run on consumer hardware with GPU acceleration.

## License

Apache 2.0 (matching base Gemma 4 license).

## SWE-bench Lite (30 problems, simplified patch matching)

| Model | Accuracy | Time | Cost | Solved |
|-------|:--------:|------|------|--------|
| Claude Opus 4.6 | 10% (3/30) | 43 min | ~$15 | django (3) |
| **Gemma4 31B v6** | **10% (3/30)** | 105 min | **$0** | astropy (1) + django (2) |

Neither model was trained on coding tasks — included as reference baseline.

## Complete Model Comparison (Scorer v2 — deep validation)

> **v2 scorer verifies actual data correctness, not just keyword presence.**
> v1 inflated scores by +10-12 points per model (e.g. giving 10/10 for "Helix: 0.00%").

| Model | Size | RAB (v2) | SWE-bench | BFCL | Cost |
|-------|------|:--------:|:---------:|:----:|------|
| Claude Opus 4.6 | Cloud | **16/40** | 10% | — | $4.72/run |
| **Gemma4 31B v6** | **16 GB** | **17/40** | **10%** | TBD | **$0** |
| Gemma4 31B Base q4 | 16 GB | 3/10\* | — | 92% | $0 |
| Gemma4 31B Base Q8 | 33 GB | 1/10\* | TBD | — | $0 |
| Gemma4 E4B Base | 3.8 GB | 1/10\* | — | 80% | $0 |
| Gemma4 E4B v5 | 4 GB | 0/10\* | — | TBD | $0 |
| Qwen3.5-0.8B | 0.8 GB | 0/10\* | — | — | $0 |
| **Gemma4 E4B v3** | **4.5 GB** | **0/10\*** | — | **95%** | $0 |

\* CH1 only — not comparable to full 4-challenge totals

**Key insight**: CH1 (Bioinformatics) exposed that NO model correctly parsed UniProt JSON (all output "Helix: 0%"). The real differentiator is CH3 (Data) and CH4 (DevOps) where models handle well-documented APIs.
