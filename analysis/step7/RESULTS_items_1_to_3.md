# Step 7 results (items 1 to 3). All numbers computed from real data; plan in PLAN_fixed_before_running.md

## Item 1: threshold trade-offs (T = ERS cut-off; blocked = ERS < T). 95% cluster-bootstrap CIs in tables/item1_threshold_operating_points.csv
At the paper's T = 0.2:
| Task | Bad blocked | Good kept |
|---|---|---|
| PubMedQA: CLEAN kept, CONTRADICTION blocked (1,000 vs 1,000) | 83.1% (80.8 to 85.4) | 79.8% (77.3 to 82.2) |
| PubMedQA: DRIFT+NOISE blocked (2,000) | 61.2% (58.8 to 63.6) | same 79.8% |
| SciFact window: CONTRADICT blocked (265) | 64.9% (59.0 to 71.2) | SUPPORT kept 88.4% (85.4 to 91.1) |
| SciFact window: IRRELEVANT blocked (693) | 79.7% (76.6 to 82.7) | same |
| SciFact window: NEI_ONTOPIC blocked (443) | 31.6% (27.0 to 36.2) | same |
| HealthContradict (ref claim): incorrect documents blocked (887) | 61.0% (52.7 to 68.2) | correct kept 82.5% (78.5 to 85.8) |
| Clinician: could-mislead blocked (19) | 84.2% (68.4 to 100) | harmless kept 46.3% (31.7 to 61.0) |

Reading the clinician row: the 41 "harmless" passages are 11 CLEAN, 14 DRIFT, 14 NOISE and 2 CONTRADICTION items. ERS keeps 82% of the harmless CLEAN items but only 36% of the harmless irrelevant (DRIFT/NOISE) ones, because blocking irrelevant text is intended. So the low "good kept" is partly the filter doing its job, not a cost to useful evidence. State this when quoting the row.
Other thresholds: T = 0 blocks 68.7% of PubMedQA contradictions (keeps 85.7% of clean) and 55.8% of SciFact contradictions; T = 0.4 blocks 88.8% and 76.6% but keeps only 67.8% and 77.2% of good passages. T = 0.2 sits at a visible elbow for PubMedQA and SciFact; it was not re-chosen from these curves.
Figure: figures/Fig_threshold_tradeoff.png. Contradiction-only matches or beats ERS on the conflict tasks (clearest on the clinician panel and on PubMedQA), and similarity-only is below chance on conflict tasks because contradicting text is more similar to the claim.

## Item 2: does alpha = beta = 1 transfer? (leave-one-dataset-out; tables/item2_*.csv)
Metrics: PubMedQA balanced AUROC, SciFact mean of (SUPPORT vs CONTRADICT, SUPPORT vs IRRELEVANT), HealthContradict pairwise accuracy (reference claim).
- Linear weights (r = beta/alpha): the weight picked on the other two datasets is never better than r = 1. Differences vs r = 1: PubMedQA -0.025 (CI -0.030 to -0.019), SciFact -0.014 (-0.018 to -0.010), HealthContradict -0.001 (-0.008 to +0.007). Each dataset's own best r gives at most +0.0002 (PubMedQA), +0.0054 (HealthContradict) and +0.0245 (SciFact, best r = 0.25) over r = 1. Conclusion: equal weighting is robust; no weight transfers better.
- Hard relevance gate (score = 2 + (1 - C) if S >= tau, else S): tau = 0.4 was chosen on the other two datasets for every target. Gain over r = 1: PubMedQA +0.105 (+0.091 to +0.121), SciFact +0.022 (+0.010 to +0.035), HealthContradict -0.010 (-0.034 to +0.013). By the rule fixed in advance (positive with CI above 0 on two of three targets, not negative on the third) the gate passes, but note the third target's point estimate is slightly negative (CI includes 0). Own-best tau: 0.4, 0.5, 0.2.
- What this supports saying: alpha = beta = 1 is a reasonable default; a relevance gate at about 0.4 is a stable improvement on the two conflict-plus-irrelevance benchmarks and neutral on HealthContradict (where all passages are on-topic). The gate was NOT tested downstream, and its scores are ranks, not the SAFE/RISKY/UNSAFE bands, so present it as a sensitivity result or future extension, not as the headline method.

## Item 3: error analysis
(a) PubMedQA CONTRADICTION passages with contradiction score < 0.1 (n = 53 of 1,000). I read all 53 and found two types:
- Type A, 35 passages: a long, often multi-clause claim prefixed with "It is not true that ...". The NLI model misses the scope of the negation. These claims contain a contrast or concession word (although, while, however, but) in 46% of cases, versus 8% across all CONTRADICTION passages; median claim length 27 words vs 20.
- Type B, 18 passages: a one-word edit by the rule-based generator that is ungrammatical or not a contradiction (for example "whether or not" turned into "whether or", "may or may not" into "may or may", "not only" into "only", "lack of" into "presence of" in an otherwise unchanged sentence, "unusual but not rare" into "unusual but rare").
- Check against humans: the 10 earlier human-labelled "targeted" items fall into these types. All 7 type A items were labelled CONTRADICTS by both annotators; of 3 type B items, 1 was CONTRADICTS and 2 were SUPPORTS (broken edits). Tiny sample, so treat the split as indicative: roughly two thirds (35) of the 53 are real NLI misses (3.5% of all contradicting passages) and up to one third (18) are generation artifacts (1.8%). The classification is from my reading plus that check; the author should verify it before it enters the paper.
(b) Clinician: 4 of the 12 CLEAN items were not labelled SUPPORTS. Three are on-topic passages that do not establish the specific claim (for example a tacrolimus assay-comparison passage vs a claim about precision and no pretreatment); ERS kept them (0.55 to 0.67) because they are on-topic and not contradictory. One (C048) the clinician labelled CONTRADICTS and could-mislead YES: the claim says "clear non-linear trends", the passage reports weak or absent associations; NLI gave C = 0.04, ERS = 0.68. This is a genuine false "safe" and a good example for the Limitations: ERS cannot tell "on-topic but not established" from "supported".
(c) SciFact (scores only; abstract text not available here): 93 of 265 contradicting windows (35%) passed T = 0.2. Their contradiction score is low (median 0.09; 55% below 0.1), and their similarity is high (mean 0.61), i.e. the NLI model returned "not contradiction" for them. 19% of all contradicting windows have C < 0.1 and 48% have S >= 0.6. To classify these failures one would have to read the abstracts; send me the claim ids if you want that done on your machine.

## What this changes for the paper
1. Report the operating-point table and the figure; justify T = 0.2 as an elbow, with the clinician-row caveat.
2. Keep alpha = beta = 1; add the leave-one-dataset-out result as evidence of robustness; present the gate as a stable sensitivity improvement.
3. Add a short error-analysis subsection with the two PubMedQA types, the clinician examples and the SciFact false-safe rate.
4. Limitations: ERS misses long negated claims and over-specific claims; construction artifacts affect about 1.8% of CONTRADICTION passages.
