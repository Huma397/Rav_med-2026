# 6a real results (Qwen2.5-7B-Instruct 4-bit zero-shot judge; 36,176 cached judgements)
Pre-specified endpoints (see notebook header). No single judge prompt was declared primary; all variants are reported.
HealthContradict (81 questions, 920 pairs), pairwise accuracy, reference claim: ERS 0.804 [0.771,0.836]; S 0.691; -C 0.801; J_rel 0.799; J_trust 0.818 [0.781,0.853].
 Paired ERS minus J_trust -0.014 [-0.043,+0.013]; minus J_rel +0.005 [-0.021,+0.032] -> indistinguishable.
Label-free claim: every scorer 0.39-0.51 (chance); ERS 0.427. Claim dependence affects all methods.
PubMedQA (synthetic): judges beat ERS in every task (paired differences -0.03 to -0.18, CIs exclude 0).
SciFact (natural claims, full abstract; unpaired CIs only, per-item judge scores were not saved):
 SUPPORT vs CONTRADICT ERS 0.828 [0.795,0.857] vs J_trust 0.877 [0.849,0.903]; SUPPORT vs IRRELEVANT ERS 0.879 vs J_rel 1.000; SUPPORT vs NEI_ONTOPIC ERS 0.735 vs J_rel 0.925; pooled ERS 0.824 vs J_trust 0.945.
