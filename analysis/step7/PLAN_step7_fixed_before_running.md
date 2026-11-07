# Step 7: analyses of existing real data (plan fixed before any of these analyses was run)
Written 2026-10-27. Data: PubMedQA N=1000 scores (4,000 passages), SciFact scores from Step 6d (80-word-window unit as primary, whole abstract as secondary), HealthContradict document scores (reference claim as primary), clinician Part A labels. No new model runs, no invented values.

## 1. Threshold trade-off curves
Operating rule (as in the downstream test): a passage is kept if ERS >= T, blocked otherwise.
Tasks (good = should be kept; bad = should be blocked):
- PubMedQA: good CLEAN; bad CONTRADICTION (primary); bad DRIFT+NOISE (secondary). Cluster = query.
- SciFact (window): good SUPPORT; bad CONTRADICT (primary); bad IRRELEVANT and NEI_ONTOPIC (secondary). Cluster = claim.
- HealthContradict (reference claim, top-1 window): good = correct document, bad = incorrect document, unique (question, document) units. Cluster = question.
- Clinician (Part A, 60 base items): good = could_mislead NO, bad = could_mislead YES. No clustering (items independent).
Report for T in {0, 0.2, 0.4}: share of bad blocked and share of good kept with 95% cluster-bootstrap CIs (2,000 resamples, seed 42). Curves over T in [-1, 1] step 0.05 for ERS, similarity-only (S >= t) and contradiction-only (C < t) filters drawn in the same trade-off plane. T = 0.2 is NOT re-chosen from these curves; the curves only describe it.

## 2. Does alpha = beta = 1 transfer?
Score families (as in the earlier sweep): ratio S - r*C with r in {0, .25, .5, .75, 1, 1.5, 2, 3, 5, 1e6}; hard relevance gate with tau in {.2,.3,.4,.5,.6,.7} (score = 2 + (1 - C) if S >= tau else S).
Metrics (one number per dataset):
- PubMedQA: mean of AUROC(CLEAN vs CONTRADICTION) and AUROC(CLEAN vs DRIFT+NOISE) ("balanced").
- SciFact (window): mean of AUROC(SUPPORT vs CONTRADICT) and AUROC(SUPPORT vs IRRELEVANT).
- HealthContradict: pairwise accuracy with the reference claim.
Leave-one-dataset-out: for each target dataset, pick the parameter that maximises the mean metric over the OTHER two datasets, separately per family, then report the target metric at (a) that parameter, (b) alpha = beta = 1 (r = 1), (c) the target's own best parameter (oracle, descriptive only). CI for (a) - (b) by cluster bootstrap with the parameter fixed. Interpretation rule fixed in advance: a family "transfers" if (a) - (b) has a CI that excludes 0 in the positive direction on at least two of three targets and is not negative on the third; otherwise the paper keeps alpha = beta = 1 and reports the gate as a sensitivity result.

## 3. Error analysis
- (a) PubMedQA CONTRADICTION passages with C < 0.1 (n = 53): read each claim-passage pair and assign one failure type. Types are decided after reading and written down with counts. This classification is by the author-assisted reading and must be verified by the author or the clinician before it goes into the paper.
- (b) Clinician Part A: CLEAN items the clinician did not label SUPPORTS (4 of 12): reading and description.
- (c) SciFact failures cannot be read here (abstract text is not available in the outputs, by design); quantitative description only (S, C of contradicting abstracts that are SAFE).
