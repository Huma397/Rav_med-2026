# Step 8 results (final; plan: PLAN_fixed_before_running.md)
Files: A_hybrid_results.csv, A_hybrid_scifact.csv, B_cascade_results.csv, B_cascade_scifact.csv; scripts step8_analysis.py, step8_scifact.py.

## Deviation D1 (disclosed): comparator ties
First run used the clipped logit of the judge probabilities (needed for the logistic features) also as the J_trust and J_rel-J_con COMPARATOR scores. Clipping at 1e-4 creates ties and lowered the comparators (e.g. SciFact J_trust 0.924 instead of 0.945). Comparators were corrected to the raw probabilities, as in Step 6a; features are unchanged. The first-run conclusions (hybrid beats J_trust on PubMedQA and HealthContradict) were artefacts of this and are withdrawn. All numbers below are from the corrected run. Corrected values reproduce 6a (PubMedQA J_trust 0.991, SciFact 0.945).

## A. Hybrid scores (H1 = S, C, J_trust; H2 = all six; logistic, 5-fold grouped CV or transferred from PubMedQA)
Difference from J_trust (95% cluster-bootstrap CI):
- PubMedQA pooled: H1 -0.022 (-0.028,-0.015); H2 -0.005 (-0.010,-0.001). Both far above ERS (0.822 -> 0.969 / 0.986), but below the judge (0.991).
- PubMedQA CLEAN vs CONTRADICTION: H1 -0.040; H2 0.000 (-0.004, +0.005).
- HealthContradict reference claim (pairwise accuracy; ERS 0.804, J_trust 0.818): H1 transferred 0.828 (vs J_trust +0.010, -0.010 to +0.031; vs ERS +0.024, 0.006 to 0.046); H2 transferred 0.787 (-0.031 vs J_trust); within-HC CV H1 0.811, H2 0.830 (vs J_trust +0.012, -0.011 to +0.034).
- HealthContradict label-free: all 0.39-0.57, no clear benefit.
- SciFact pooled AUROC (ERS 0.824, J_trust 0.945): H1 transferred 0.941 (-0.004, -0.010 to +0.001); H2 0.925 (-0.020). SUPPORT vs CONTRADICT: H1 0.885 vs J_trust 0.876 (+0.009, 0.000 to 0.018); H2 0.845 (-0.032).
Criterion (hybrid exceeds both ERS and J_trust with CI excluding 0 on a natural benchmark): NOT met. Report: "no benefit of combination over the judge found"; the hybrid recovers most of the judge's accuracy relative to ERS but never exceeds it.

## B. Cascade (ERS first, judge only on the borderline band; balanced accuracy of keep/block; cost per passage on T4)
PubMedQA pooled: ERS T=0.2 0.742; ERS CV-tuned 0.827; cascade 25% judge 0.869 (256 ms); 50% 0.890 (441 ms); judge alone 0.938 (868 ms). 10% budget collapsed to ERS T=0.2.
SciFact: ERS T=0.2 0.739; ERS CV-tuned 0.770; cascade 10% 0.771 (109 ms); 25% 0.816 (220 ms); 50% 0.846 (476 ms); judge 0.872.
Non-inferiority to the judge (margin 0.02) fails at every budget on both datasets (closest: SciFact 50%, -0.026, CI -0.042 to -0.010). The cascade is an intermediate cost/accuracy option, not a replacement for the judge.
