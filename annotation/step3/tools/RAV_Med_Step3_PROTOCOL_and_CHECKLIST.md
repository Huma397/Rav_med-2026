# RAV-Med Step 3 — Human Validation: Protocol & Checklist
(Analysis plan fixed BEFORE any labels are seen. Any later change must be logged under "Deviations" and reported in the paper.)

## 1. Design
- Two independent annotators (A1 = medical expert, A2 = author). Each sees a DIFFERENT shuffled order; joined on `item_id`.
- Part A: 130 claim–passage items (+12 repeats) → labels CONTRADICTS / SUPPORTS / NEITHER (confidence 1–3).
- Part B: 60 HealthContradict windows (+6 repeats) → YES / NO / MIXED / NOT_RELEVANT.
- Repeats give test–retest (intra-annotator) agreement.
- Answer keys, conditions and scores are never shown to annotators (blinding).

## 2. Primary outcomes (pre-specified)
1. Inter-annotator Cohen's κ with 95% bootstrap CI (2,000 resamples, seed 42); raw agreement.
2. Supplements for skewed labels: Gwet's AC1 and PABAK (κ alone can mislead when one label dominates).
3. Intra-annotator agreement on repeats.
4. Gold labels = adjudicated labels for disagreements; agreed labels otherwise.
5. Part A: construction validity per condition (Wilson 95% CI); AUROC of NLI-C, −ERS and −similarity against human CONTRADICTS (random stratum only; the targeted low-C stratum is reported separately).
6. Part B: stance fidelity of the top-1 window; ERS pairwise accuracy overall and on pairs where humans confirm both stances.

## 3. Sensitivity analyses (pre-specified)
- Expert-only gold (A1's labels) vs adjudicated gold; conclusions must hold under both.
- Agreement excluding confidence = 1 items.
- Interpretation bands (Landis & Koch) are descriptive only; the paper reports numbers, not adjectives alone.

## 4. Known limitations to disclose
- A2 is the author (risk of bias); mitigated by blinding, independent labelling and an expert annotator.
- Two annotators only; n=130/60 gives κ CI about ±0.08–0.10 at κ≈0.8.
- Annotator qualifications as per profile sheets.

## 5. Procedure
1. Send each annotator ONLY their own workbook (never the key, manifest, scripts, or the other annotator's file).
2. Do NOT edit item text, reorder, add or delete rows. Do not discuss items between annotators until both files are returned.
3. Fill Practice first (check vs Practice_key), then the main sheet, then the Annotator_profile sheet.
4. Every row needs a label from the dropdown and confidence 1/2/3. Comment is optional.
5. Return the .xlsx unchanged in name/structure.

## 6. Checks before analysis (author runs)
```
python3 make_manifest.py                       # already done; manifest.csv is the integrity reference
python3 validate_returned_sheet.py A 1 <file>  # also A 2, B 1, B 2   -> must print PASS
```
Fails on: blank/invalid label or confidence, missing/duplicate item_id, altered claim/passage text, blank profile. Warns on reordering or >90% one label.
If FAIL: return the file to the annotator to fix; keep the failed copy.

## 7. Adjudication
1. `python3 make_adjudication_sheet.py A a1.xlsx a2.xlsx adj_A.xlsx` (same for B).
2. A1 and A2 discuss ONLY listed disagreements, fill `final_label`; if still undecided, A1 (expert) decides and the item is flagged.
3. Save as CSV (item_id, final_label); keep the original files unmodified (audit trail).
4. Run analysis twice: with and without the adjudication CSV; report both.

## 8. Analysis commands
```
python3 analyze_partA.py a1.xlsx a2.xlsx PartA_ANSWER_KEY_do_not_share.csv [adj_A.csv] outA
python3 analyze_partB.py b1.xlsx b2.xlsx PartB_ANSWER_KEY_do_not_share.csv [adj_B.csv] outB
```

## 9. Deviations log
| Date | What changed | Why |
|---|---|---|
| | | |

## 10. Ethics note
Annotators label publicly available text and are not subjects; obtain verbal/written consent for naming in the Acknowledgements (profile sheet question).
