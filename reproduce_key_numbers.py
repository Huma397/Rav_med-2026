"""Recompute the headline numbers of the manuscript from the files shipped in this repository.
No model download, no GPU and no HealthContradict text needed. Prints PASS/FAIL per check; exit code 1 if any fails.
Run from the repository root:  python reproduce_key_numbers.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = Path(__file__).resolve().parent
fails = []
def check(name, got, want, tol=0.0006):
    ok = abs(got - want) <= tol
    print(("PASS " if ok else "FAIL ") + f"{name}: recomputed {got:.4f}, manuscript {want}")
    if not ok: fails.append(name)

# 1. PubMedQA controlled corruption (Table 1/2): 1,000 queries x 4 conditions
d = pd.read_csv(R / "data/pubmedqa/rav_med_full_results_N1000.csv")
y = (d.condition == "CLEAN").astype(int)
check("PubMedQA pooled AUROC, ERS", roc_auc_score(y, d.ERS), 0.822)
x = d[d.condition.isin(["CLEAN", "CONTRADICTION"])]; yc = (x.condition == "CLEAN").astype(int)
check("PubMedQA CLEAN vs CONTRADICTION AUROC, similarity only", roc_auc_score(yc, x.similarity), 0.101)
check("PubMedQA CLEAN vs CONTRADICTION AUROC, ERS", roc_auc_score(yc, x.ERS), 0.811)

# 2. Prompted judge baselines (Step 6a), same 4,000 rows with judge scores
j = pd.read_csv(R / "data/judge_baselines/s6a_pubmedqa_judge_scores.csv"); yj = (j.condition == "CLEAN").astype(int)
check("PubMedQA pooled AUROC, reliability prompt (J_trust)", roc_auc_score(yj, j.J_trust), 0.991)
check("PubMedQA pooled AUROC, ERS (judge file)", roc_auc_score(yj, j.ERS), 0.822)

# 3. SciFact, second run of the judge (Step 6g)
s = pd.read_csv(R / "data/scifact/s6g_scifact_item_scores_nowindowtext.csv"); ys = (s.cond == "SUPPORT").astype(int)
check("SciFact pooled AUROC, ERS", roc_auc_score(ys, s.ERS), 0.824)
check("SciFact pooled AUROC, reliability prompt", roc_auc_score(ys, s.J_trust), 0.945)
z = s[s.cond.isin(["SUPPORT", "CONTRADICT"])]; yz = (z.cond == "SUPPORT").astype(int)
check("SciFact SUPPORT vs CONTRADICT AUROC, ERS", roc_auc_score(yz, z.ERS), 0.828)
check("SciFact SUPPORT vs CONTRADICT AUROC, reliability prompt (Table 6 is from the first run; the second run differs by < 0.001)", roc_auc_score(yz, z.J_trust), 0.877, 0.0015)
check("SciFact items", len(s), 1909, 0)

# 4. Downstream accuracy (tables shipped for both generators)
a = pd.read_csv(R / "data/pubmedqa/step1_7b_table_accuracy_reparsed.csv").set_index("setting").accuracy
check("Generator 1 accuracy, CLEAN", a["CLEAN"], 0.640, 0.0005); check("Generator 1 accuracy, no retrieval", a["NO_RETRIEVAL"], 0.496, 0.0005)
check("Generator 1 drop under CONTRADICTION", a["INJ_CONTRADICTION"] - a["CLEAN"], -0.164, 0.0005)
b = pd.read_csv(R / "data/second_generator/step1_table_accuracy.csv").set_index("setting").accuracy
check("Generator 2 accuracy, CLEAN", b["CLEAN"], 0.692, 0.0005); check("Generator 2 drop under CONTRADICTION", b["INJ_CONTRADICTION"] - b["CLEAN"], -0.062, 0.0005)

# 5. HealthContradict pairwise accuracy (reference claim, full ERS)
h = pd.read_csv(R / "data/healthcontradict/hc_table_pairwise_accuracy.csv")
check("HealthContradict pairwise accuracy, ERS, reference claim", float(h[(h.claim == "reference claim") & h.scorer.str.startswith("Full ERS")].pooled.iloc[0]), 0.804)

# 6. Cost (Step 6e)
c = pd.read_csv(R / "data/cost/s6e_cost_timing.csv")
check("Cost ratio judge/ERS (T4)", float(c.cost_relative_to_ERS_gpu.iloc[2]), 17, 0.05)

# 7. Error analysis of missed contradictions (S14)
e = pd.read_csv(R / "data/error_analysis/S14_author_classification.csv").type.value_counts()
check("S14 type A", float(e["A"]), 35, 0); check("S14 type B", float(e["B"]), 17, 0); check("S14 mixed", float(e["C"]), 1, 0)

print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}")
sys.exit(1 if fails else 0)
