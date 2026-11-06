# Step 6 notebooks (completed)

All notebooks were run on a free Colab T4. Each has a `DRY_RUN` flag in its config cell that swaps in fake models to test the code path; **numbers printed in dry-run mode are meaningless and no dry-run output is used anywhere in the paper or in `data/`.**

| Notebook | Question | Outputs in |
|---|---|---|
| `step6a_llm_judge_baselines` | ERS vs. zero-shot Qwen2.5-7B-Instruct judge prompts (PubMedQA, HealthContradict, SciFact) | `data/judge_baselines/` |
| `step6b_second_generator` | Does the downstream result hold for Mistral-7B-Instruct-v0.3? | `data/second_generator/` |
| `step6c_claim_free_consensus` | Can passages be checked against each other when no claim exists? | `data/claim_free/` |
| `step6d_scifact_external` | ERS on SciFact | `data/scifact/` |
| `step6e_cost_timing` | Per-passage cost of ERS vs. judge | `data/cost/` |
| `step6g_scifact_rejudge_paired` | SciFact judge re-run for the paired analysis. **Supersedes the withdrawn 6f**, whose output is not used. | `data/scifact/s6g_*` |

Notes
- Both 7B generators and the judge run in 4-bit NF4 (so precision is matched across generators). The judge is from the same family as generator 1 (Qwen); this is disclosed in the manuscript.
- Judge prompts were fixed before running; the judge cache key is md5(model name, system prompt, user prompt).
- Mistral may require a Hugging Face login and licence acceptance.
- SciFact is CC BY-NC 2.0: the notebooks save scores and IDs only, not abstracts.
- The paired analyses and Step 8 hybrid/cascade analyses are in `analysis/step6/` and `analysis/step8/`.
