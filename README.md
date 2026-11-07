# RAV-Med: contradiction-aware evidence reliability scoring for medical RAG

Code, derived scores and annotation records for the manuscript *"RAV-Med: Beyond Semantic Similarity: Contradiction-Aware Evidence Reliability Scoring for Pre-Generation Verification in Medical Retrieval-Augmented Generation"* (Huma Iqrar).

Archived version (DOI): to be added by the author after the first Zenodo release (this line is replaced with the DOI at release time).

## The score
`ERS = alpha*S - beta*C` with alpha = beta = 1. S is the cosine similarity between claim and passage (biomedical SBERT encoder); C is the contradiction probability of roberta-large-mnli (premise = passage, hypothesis = claim). A passage is SAFE if ERS >= T (T = 0.2 in the paper), RISKY if T-0.2 <= ERS < T, UNSAFE otherwise. See `rav_med/ers.py` and `MODELS.md`.

```python
from rav_med import ERSScorer
sc = ERSScorer()
print(sc.score(["Aspirin lowers reinfarction risk."], ["Aspirin did not reduce reinfarction (RR 1.01)."]))
```
`python tests_selftest.py` checks the score arithmetic and bands without downloading models.

## Quick check of the headline numbers
`python reproduce_key_numbers.py` recomputes 20 numbers quoted in the manuscript (AUROCs, generator accuracies, cost ratio, S14 counts) from the shipped files and prints PASS/FAIL for each. It needs only numpy, pandas and scikit-learn, no GPU and no downloads.

## Contents
| Path | What |
|---|---|
| `rav_med/` | ERS implementation |
| `data/pubmedqa/` | 1,000 queries x 4 conditions: claim, passage, S, C, ERS; length-matched control; re-parsed Qwen2.5-7B downstream predictions and tables. Derived from PubMedQA (MIT). |
| `data/healthcontradict/` | Scores only (S, C, ERS per document and claim type), the short claim file and result tables. **No document text is included.** |
| `data/judge_baselines/` | Step 6a/6g: Qwen2.5-7B-Instruct zero-shot judge scores (four prompts), the judge cache (md5 key to P(Yes), no text), AUROC tables, paired ERS-minus-comparator differences. |
| `data/second_generator/` | Step 6b: Mistral-7B-Instruct-v0.3 predictions and tables. |
| `data/claim_free/` | Step 6c: claim-free consensus scores and pools. |
| `data/scifact/` | Steps 6d and 6g: SciFact item scores (IDs and scores only, **no abstract text**), AUROC and paired tables. |
| `data/cost/` | Step 6e: per-passage timing of ERS and of the judge, settings. |
| `data/combination_cascade/` | Step 8: hybrid (S, C, judge) models and ERS-then-judge cascades. |
| `data/error_analysis/` | Author's classification of the 53 low-C passages (Supplementary S14). |
| `notebooks/` | Colab notebooks that produced each result; see `notebooks/README_step6.md`. |
| `analysis/` | Step 1 re-parse, Step 4 baselines, `step6/` paired analysis, `step7/` threshold, transfer and contradiction-feature analyses, `step8/` hybrid and cascade analyses (each with its fixed plan and results file). |
| `annotation/step3/` | Human-validation protocol, scripts, adjudicated gold labels, answer keys, reports, deviations log. Annotation workbooks are not included because they contain HealthContradict text. |
| `annotation/step5_clinician/` | Protocol, scripts, analysis report and deviations log for the blinded clinician review. The physician's item-level labels are deliberately not released; only the aggregate analysis report is. |

## Reproducing
1. **Headline numbers:** `python reproduce_key_numbers.py`.
2. **Step 6 paired analysis and Step 8:** `analysis/step6/paired_analysis_6a.py`, `analysis/step8/step8_analysis.py` and `step8_scifact.py` are repo-relative. They need the original HealthContradict file: clone its repository and set `HC_JSONL=/path/to/dataset/dataset_ready.jsonl` (the scripts exit with a message if it is missing). Outputs go to `reproduced_outputs/` (or `RAV_OUT`). On the released data the regenerated CSVs matched the shipped ones exactly (maximum difference 0.0; bootstrap seeds fixed).
3. **Step 7:** `analysis/step7/step7_analyses.py` still has hard-coded paths and needs the private clinician key and returned workbook, which are not released; the shipped CSVs and results file are its outputs.
4. **Recomputing scores:** run the notebooks on a GPU (free Colab T4 suffices; both 7B models run in 4-bit NF4; sessions can disconnect, so notebooks cache intermediates). Install `requirements.txt`.
5. Seeds: 42 for data construction and bootstrap unless stated; Step 8 folds use seed 2026.

## Known limitations of this release
- Pinned model revisions were recorded only for the Step 4 models (`MODELS.md`); the Mistral revision was not recorded. The 4-bit results may differ slightly on other hardware or library versions.
- Annotators were non-clinicians who had seen the results (annotator 2 is the author); the clinician review covers a small sample (see manuscript).
- `step7_analyses.py` is not runnable from this repository alone (see above).
- Notebook runs with `DRY_RUN = True` use fake models; their printed numbers are not results and none are in this repository.
- HealthContradict and SciFact text is not redistributed.

## Licence
- Code: MIT (`LICENSE`).
- Derived scores, tables and annotation records: CC BY 4.0 (see `LICENSE_DATA.md`), except that files derived from SciFact are limited to non-commercial research use (see below).
- Upstream data: PubMedQA is MIT. SciFact is CC BY-NC 2.0 per its dataset card; only IDs and scores derived from it are included, for non-commercial research use. The HealthContradict repository states no licence that we could find; check its terms before reusing its text.
