# Models and revisions
Revisions below were recorded by `analysis/step4_method_hardening/baselines/step4_model_info.json` (Step 4 run). Models used in earlier steps were loaded by name without a pinned revision; pin them yourself for exact replication.

| Role | Model | Revision (git SHA) |
|---|---|---|
| Similarity S | pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb | not recorded |
| Contradiction C (main) | roberta-large-mnli | 2a8f12d27941090092df78e4ba6f0928eb5eac98 |
| Alternative verifier | MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7 |
| Alternative verifier | cross-encoder/nli-deberta-v3-large | bab4bc7178836f731dcfd18c06ca9def0a137712 |
| Biomedical verifier | pritamdeka/PubMedBERT-MNLI-MedNLI | f1b6ce2e0d49f295b4cbcdc56c01b5fab6d068ab |
| Relevance baseline | ncbi/MedCPT-Cross-Encoder | 71caf65d4927987813984f54c284405a13fcca49 |
| Relevance baseline | cross-encoder/ms-marco-MiniLM-L-6-v2 | 233902d25c440f23af6f7d6e94d2946bac0bee0a |
| Downstream generator 1 | Qwen2.5-7B-Instruct (4-bit NF4, greedy) | not recorded |
| Downstream generator 2 | mistralai/Mistral-7B-Instruct-v0.3 (4-bit NF4, greedy) | not recorded |
| Zero-shot LLM judge (Step 6) | Qwen2.5-7B-Instruct (4-bit NF4), P(Yes) from logits | not recorded |

Both generators and the judge were run on a free Colab T4 with bitsandbytes 4-bit NF4 quantisation (`LOAD_4BIT = True`).

microsoft/deberta-large-mnli failed to run in half precision (overflow) and was excluded.
