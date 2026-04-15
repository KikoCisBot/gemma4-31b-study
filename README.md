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

## Published models

- ⭐ 🤗 **[gemma-4-31b-it-IQ2_M-GGUF](https://huggingface.co/KikoCis/gemma-4-31b-it-IQ2_M-GGUF)** — recommended, 10.17 GB, F1 84.71%, BLEU-4 22.39
- 🤗 [gemma-4-31b-it-IQ3_XS-GGUF](https://huggingface.co/KikoCis/gemma-4-31b-it-IQ3_XS-GGUF) — 13.1 GB, F1 84.76%, ties f16
- 🤗 [gemma-4-31b-otter-v3-GGUF](https://huggingface.co/KikoCis/gemma-4-31b-otter-v3-GGUF) — research artifact (58-layer pruned, no fine-tuning)
- 📊 [gemma4-31b-layer-study dataset](https://huggingface.co/datasets/KikoCis/gemma4-31b-layer-study) — raw data + scripts + visualization

## Methodology

3-phase layer analysis:
- **Phase A** — Block Influence (BI): per-layer residual-stream modification, `1 − cos(h_in, h_out)`.
- **Phase B** — Logit Lens: projecting each hidden state through the final norm + `lm_head`.
- **Phase C** — Single-layer ablation: measure perplexity delta when a layer is zeroed.

Benchmark: NL2Bash 50-example subset from the official Stanford/Tellina test split. Each question runs as an isolated `llama-cli` subprocess with `--reasoning off`, `--max-tokens 200`, `--temp 0.1`, `--ctx-size 4096`. Scoring via NLTK BLEU + character-level F1 + exact match.

All measurements run on Apple M4 Max 128GB with full Metal GPU offload.

## License

Apache 2.0 (matching base Gemma 4 license).
