# Step 8 plan: hybrid score and ERS-then-judge cascade (written BEFORE any result was computed)
Date written: 2026-11-02. Data: existing real outputs only (no new model runs). No result of this analysis has been seen when this file is saved.

## Question
Can the cheap score (ERS) and the 7B judge be combined so that (A) the combination ranks evidence better than either alone, and (B) most of the judge's accuracy is kept while the judge is called on only a fraction of passages?

## Data (all existing, real)
- PubMedQA: 4,000 items (s6a_pubmedqa_judge_scores.csv). Good = CLEAN (1,000); bad = DRIFT, NOISE, CONTRADICTION (3,000). Cluster = pubid.
- HealthContradict: 3,552 scored windows (ref and free claims); pairwise task, cluster = topic. Primary = reference claim; free claim reported as secondary.
- SciFact: ERS exists; judge scores per item exist only after Step 6g. SciFact analyses are run only when 6g passes its reproduction check, with the same rules.
- Judge features: J_rel, J_con, J_sup, J_trust (probability of "Yes"), transformed to logit with probabilities clipped to [1e-4, 1-1e-4].

## Analysis A: hybrid score (fixed in advance)
- Models: H1 = L2 logistic regression (C = 1.0, features standardised on the training data) on [S, C, logit J_trust]; H2 = same on [S, C, logit J_rel, logit J_con, logit J_sup, logit J_trust]. No other model, no tuning of C.
- Comparators: ERS = S - C; J_trust; J_rel minus J_con (the judge analogue of ERS).
- A1 within-dataset: grouped 5-fold cross-validation (clusters never split; fold assignment by a random permutation of clusters, seed 2026), out-of-fold AUROC (PubMedQA, label good vs bad) and out-of-fold pairwise accuracy (HealthContradict, label = evidence-correct document).
- A2 cross-dataset transfer (primary external test): fit H1 and H2 on ALL of PubMedQA, apply unchanged to HealthContradict (pairwise accuracy, reference claim) and later SciFact (AUROC SUPPORT vs others).
- Uncertainty: cluster bootstrap (2,000), paired differences hybrid minus ERS and hybrid minus J_trust.
- Criterion for claiming a hybrid benefit: on at least one natural benchmark (HealthContradict or SciFact) the hybrid exceeds BOTH ERS and J_trust with a paired 95% CI excluding 0, and is not worse than either on the others (CI lower bound > -0.02). Otherwise the paper reports "no benefit of combination found".

## Analysis B: cascade (fixed in advance)
- Rule: let e = ERS, j = J_trust. If e >= hi keep without the judge; if e < lo block without the judge; otherwise keep iff j >= tau. Judge fraction f = share of items with lo <= e < hi.
- Grid: lo in {-0.4, -0.2, 0, 0.1, 0.2}, hi in {0.2, 0.3, 0.4, 0.5, 0.6} with lo < hi, tau in {0.3, 0.5, 0.7}.
- Selection (on training folds only): maximise balanced accuracy = mean(bad blocked, good kept) subject to f <= budget, for budgets 10%, 25%, 50%.
- Evaluation: grouped 5-fold CV (same folds); out-of-fold balanced accuracy, bad blocked, good kept, f. Cost per passage = ERS time + f x judge time, with ERS = 51.1 ms and judge = 868.2 ms (Step 6e, T4 GPU).
- References: ERS alone at the paper's T = 0.2 (fixed), ERS alone with T chosen by CV, judge alone with tau chosen by CV (f = 100%).
- Non-inferiority margin (fixed): a cascade supports the cost claim if the lower 95% bound of (cascade minus judge alone) in balanced accuracy is above -0.02, and it beats ERS alone if the lower bound of (cascade minus ERS at T = 0.2) is above 0.
- Tasks: PubMedQA pooled (primary) and CLEAN vs CONTRADICTION (secondary); SciFact SUPPORT vs others pooled when 6g passes.

## Limits stated in advance
- PubMedQA conditions are constructed and the judge is near ceiling there, so PubMedQA demonstrates the cost trade-off, not a natural-data benefit. The natural-data question is answered only by HealthContradict (pairwise) and SciFact.
- One judge model, 4-bit, same family as the first generator. One fold partition (seed 2026); sensitivity to 5 other partitions is reported for the main metrics.
- All results are reported whatever they show; deviations will be logged in DEVIATIONS_LOG.md.
