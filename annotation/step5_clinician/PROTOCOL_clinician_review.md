# RAV-Med Step 5: Blinded clinician review (protocol, fixed before any clinician label is seen)

## Purpose
The two earlier annotators were not clinicians and had seen the results. This step adds one independent, blinded clinician as a third rater on a subset of items the others already labelled, and adds one clinical judgement ("could this passage mislead?") that non-clinicians cannot make reliably.

## Design
- Part A: 60 claim-passage items sampled from the earlier random stratum by construction condition (12 CLEAN, 14 DRIFT, 14 NOISE, 20 CONTRADICTION; seed 2026) + 4 repeats = 64 rows. Labels: CONTRADICTS / SUPPORTS / NEITHER, confidence 1-3, and could_mislead YES/NO.
- Part B: 40 HealthContradict windows (20 from correct, 20 from incorrect documents) + 3 repeats = 43 rows. Labels: YES / NO / MIXED / NOT_RELEVANT, confidence 1-3.
- Item IDs are new (C###, D###) and shuffled; the clinician never sees conditions, scores, earlier labels or the key.
- Send ONLY `for_clinician/RAV_Med_clinician_review.xlsx`. Never send `private_do_not_share/` or `tools/`.

## Eligibility of the reviewer
Licensed or finally qualified clinician (physician, pharmacist, nurse practitioner or comparable). Must not have read the paper or seen results (profile sheet asks). Record qualification, years, specialty, consent to be named.

## Pre-specified outcomes (analyze_clinician.py)
1. Clinician vs adjudicated non-clinician gold, and vs each annotator: raw agreement, Cohen's kappa (2,000-resample bootstrap CI, seed 42), Gwet's AC1.
2. Clinician test-retest on the repeated rows.
3. Part A construction validity by condition (Wilson 95% CI), now by clinician label.
4. AUROC of NLI-C, -ERS and -similarity against clinician CONTRADICTS.
5. Clinical risk: share of could_mislead = YES per condition; AUROC of -ERS (and comparators) for could_mislead; precision and recall of the ERS < 0 and ERS < 0.2 flags.
6. Part B: stance fidelity of the top-1 window per document role; AUROC of top-1 ERS for correct vs incorrect documents.
Sensitivity: exclude clinician confidence = 1.

## Reporting rules
- Report all outcomes whatever the result. A low clinician-vs-annotator kappa is reported as such, and the earlier gold labels are then described as non-clinician labels with that caveat.
- Sample sizes are small (n = 60 and 40); report CIs, make no claims beyond them.
- Any change to this plan is logged below before the analysis is run.

## Procedure
1. Send the workbook; the clinician works alone, no web or AI tools.
2. On return: `python3 tools/validate_clinician_sheet.py returned.xlsx` must print PASS (if FAIL, return the file; keep the failed copy).
3. `python3 tools/analyze_clinician.py returned.xlsx private_do_not_share outdir`
4. Keep the returned file unmodified in the repository.

## Deviations log
| Date | What changed | Why |
|---|---|---|
| | | |

## Message you can send to the clinician
> Hello, I would be grateful for a few hours of your clinical judgement for a research study on medical retrieval-augmented generation. You will read about 100 short medical texts and say how each relates to a claim or question; no patient data is involved and the texts are public. Please work alone and do not use the web or AI tools. Complete the Practice sheets first. If you are happy to be named in the Acknowledgements, tell me in the profile sheet; this is optional. Thank you.
