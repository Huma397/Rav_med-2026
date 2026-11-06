# Step 8: hybrid score and cascade, exactly as in PLAN_fixed_before_running.md
import json, hashlib, numpy as np, pandas as pd
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
HC_JSONL = os.environ.get("HC_JSONL", "")
if not HC_JSONL or not os.path.exists(HC_JSONL):
    sys.exit("Set HC_JSONL to the HealthContradict file dataset/dataset_ready.jsonl (clone the original repository; its text is not redistributed here).")
OUTDIR = Path(os.environ.get("RAV_OUT", REPO / "reproduced_outputs")); OUTDIR.mkdir(parents=True, exist_ok=True)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
R6A = str(REPO / "data" / "judge_baselines") + "/"; OUT = str(OUTDIR) + "/"
ERS_MS, JUDGE_MS = 51.13, 868.15   # Step 6e (T4)
SEED, NB = 2026, 2000
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def folds(groups, k=5, seed=SEED):
    u = np.unique(groups); r = np.random.default_rng(seed); perm = r.permutation(len(u)); fid = {g: i % k for g, i in zip(u[perm], range(len(u)))}
    return np.array([fid[g] for g in groups])
def feats(df, which):
    cols = {"H1": ["S", "C", "Jt"], "H2": ["S", "C", "Jr", "Jc", "Js", "Jt"]}[which]; return df[cols].values
def fit_pred(tr, te, which):
    sc = StandardScaler().fit(feats(tr, which)); m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(feats(tr, which)), tr.y)
    return m.decision_function(sc.transform(feats(te, which)))
def prep(df, S="similarity", C="contradiction"):
    d = df.copy(); d["S"] = d[S]; d["C"] = d[C]; d["E"] = d.S - d.C
    for a, b in [("Jr", "J_rel"), ("Jc", "J_con"), ("Js", "J_sup"), ("Jt", "J_trust")]: d[a] = lg(d[b].values)
    d["Jrc"] = d.J_rel - d.J_con; d["Jtr"] = d.J_trust; return d
def boot_w(groups, B, seed):
    u, g = np.unique(groups, return_inverse=True); W = np.random.default_rng(seed).multinomial(len(u), np.ones(len(u)) / len(u), size=B); return g, W
def auc_boot(y, scores, g, W):
    out = []
    for b in range(W.shape[0]):
        w = W[b][g]; m = w > 0
        out.append(roc_auc_score(y[m], scores[m], sample_weight=w[m]))
    return np.array(out)
rows = []
# ---------------- PubMedQA ----------------
pq = prep(pd.read_csv(R6A + "s6a_pubmedqa_judge_scores.csv")); pq["y"] = (pq.condition == "CLEAN").astype(int); fo = folds(pq.pubid.values)
oof = {k: np.zeros(len(pq)) for k in ["H1", "H2"]}
for f in range(5):
    tr, te = pq[fo != f], pq[fo == f]
    for k in oof: oof[k][fo == f] = fit_pred(tr, te, k)
SC = {"ERS": pq.E.values, "J_trust": pq.Jtr.values, "J_rel_minus_con": pq.Jrc.values, "H1 (S,C,J_trust)": oof["H1"], "H2 (all six)": oof["H2"]}
def task_rows(name, mask):
    x = pq[mask]; y = x.y.values; g, W = boot_w(x.pubid.values, NB, SEED)
    est = {k: roc_auc_score(y, v[mask]) for k, v in SC.items()}; bs = {k: auc_boot(y, v[mask], g, W) for k, v in SC.items()}
    for k in SC:
        r = dict(dataset="PubMedQA", task=name, scorer=k, AUROC=est[k], lo=np.percentile(bs[k], 2.5), hi=np.percentile(bs[k], 97.5))
        for ref in ["ERS", "J_trust"]:
            if k != ref and "H" in k[:2]: d = bs[k] - bs[ref]; r[f"diff_vs_{ref}"] = est[k] - est[ref]; r[f"diff_vs_{ref}_lo"] = np.percentile(d, 2.5); r[f"diff_vs_{ref}_hi"] = np.percentile(d, 97.5)
        rows.append(r)
task_rows("CLEAN vs all corrupted (pooled), out-of-fold", np.ones(len(pq), bool))
task_rows("CLEAN vs CONTRADICTION, out-of-fold", pq.condition.isin(["CLEAN", "CONTRADICTION"]).values)
# sensitivity to the fold partition (pooled AUROC of H1 / H2 over 5 other partitions)
sens = []
for sd in range(5):
    f2 = folds(pq.pubid.values, seed=100 + sd); o = {k: np.zeros(len(pq)) for k in ["H1", "H2"]}
    for f in range(5):
        for k in o: o[k][f2 == f] = fit_pred(pq[f2 != f], pq[f2 == f], k)
    sens.append({k: roc_auc_score(pq.y, o[k]) for k in o})
S_ = pd.DataFrame(sens); print("partition sensitivity (pooled AUROC):\n", S_.round(4).describe().loc[["min", "max"]])
# ---------------- HealthContradict ----------------
raw = pd.DataFrame([json.loads(l) for l in open(HC_JSONL)]); raw["topic_id"] = raw.topic_id.astype(int)
ct = np.where(raw.query_stance == "yes", raw.doc_a, raw.doc_b); it = np.where(raw.query_stance == "yes", raw.doc_b, raw.doc_a)
did = lambda t: hashlib.md5(t.encode()).hexdigest()[:12]
P = pd.DataFrame(dict(topic_id=raw.topic_id, c=[did(t) for t in ct], i=[did(t) for t in it]))
hc = pd.read_csv(R6A + "s6a_healthcontradict_judge_scores.csv")
hc = hc.rename(columns={"J_rel": "J_rel", "J_con": "J_con", "J_sup": "J_sup", "J_trust": "J_trust"}); hc["similarity"] = hc.S; hc["contradiction"] = hc.C; hc = prep(hc)
def pair_w(score_by_key, P):
    a = np.array([score_by_key[(t, d)] for t, d in zip(P.topic_id, P.c)]); b = np.array([score_by_key[(t, d)] for t, d in zip(P.topic_id, P.i)]); return (a > b) + 0.5 * (a == b)
def pair_boot(W_by_name, topics, B=NB, seed=SEED):
    u, g = np.unique(topics, return_inverse=True); Wm = np.random.default_rng(seed).multinomial(len(u), np.ones(len(u)) / len(u), size=B)
    cnt = np.bincount(g); res = {}
    for k, w in W_by_name.items():
        s = np.bincount(g, weights=w); res[k] = (Wm @ s) / (Wm @ cnt)
    return res
# transfer models fitted on ALL PubMedQA
tr_all = pq; trf = {}
for k in ["H1", "H2"]:
    sc = StandardScaler().fit(feats(tr_all, k)); m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(feats(tr_all, k)), tr_all.y); trf[k] = (sc, m)
for ctype, label in [("ref", "reference claim"), ("free", "label-free claim")]:
    d = hc[hc.claim_type == ctype].copy(); d["key"] = list(zip(d.topic_id, d.doc_id))
    scores = {"ERS": d.E.values, "J_trust": d.Jtr.values, "J_rel_minus_con": d.Jrc.values}
    for k in ["H1", "H2"]: sc, m = trf[k]; scores[f"{k} transferred from PubMedQA"] = m.decision_function(sc.transform(feats(d, k)))
    # within-HC grouped CV (reference claim and free claim each fitted with the label 'document is evidence-correct')
    lab = {}
    for t, c_, i_ in zip(P.topic_id, P.c, P.i): lab[(t, c_)] = 1; lab[(t, i_)] = 0
    d["y"] = [lab[k] for k in d.key]; fo2 = folds(d.topic_id.values)
    for k in ["H1", "H2"]:
        o = np.zeros(len(d))
        for f in range(5): o[fo2 == f] = fit_pred(d[fo2 != f], d[fo2 == f], k)
        scores[f"{k} within-HC cross-validated"] = o
    W = {}; est = {}
    for name, sv in scores.items():
        m_ = dict(zip(d.key, sv)); w = pair_w(m_, P); W[name] = w; est[name] = w.mean()
    bs = pair_boot(W, P.topic_id.values)
    for name in scores:
        r = dict(dataset="HealthContradict", task=f"pairwise accuracy, {label}", scorer=name, AUROC=est[name], lo=np.percentile(bs[name], 2.5), hi=np.percentile(bs[name], 97.5))
        if name.startswith("H"):
            for ref in ["ERS", "J_trust"]: dd = bs[name] - bs[ref]; r[f"diff_vs_{ref}"] = est[name] - est[ref]; r[f"diff_vs_{ref}_lo"] = np.percentile(dd, 2.5); r[f"diff_vs_{ref}_hi"] = np.percentile(dd, 97.5)
        rows.append(r)
A = pd.DataFrame(rows).rename(columns={"AUROC": "AUROC_or_pairwise_acc"}); A.to_csv(OUT + "A_hybrid_results.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 60); print(A.round(3).to_string(index=False))
# ---------------- Analysis B: cascade on PubMedQA ----------------
LO = [-0.4, -0.2, 0.0, 0.1, 0.2]; HI = [0.2, 0.3, 0.4, 0.5, 0.6]; TAU = [0.3, 0.5, 0.7]; BUDGETS = [0.10, 0.25, 0.50]
def decide(e, j, lo, hi, tau): keep = np.where(e >= hi, True, np.where(e < lo, False, j >= tau)); return keep, ((e >= lo) & (e < hi))
def metrics(keep, y): bad = y == 0; good = y == 1; bb = (~keep[bad]).mean(); gk = keep[good].mean(); return bb, gk, (bb + gk) / 2
def run_cascade(d, task_mask, label):
    x = d[task_mask].reset_index(drop=True); y = x.y.values; e = x.E.values; j = x.Jtr.values; fo = folds(x.pubid.values if "pubid" in x else x.claim_id.values)
    res = {}
    def pick(tr_idx, budget):
        best = None
        for lo in LO:
            for hi in HI:
                if lo >= hi: continue
                for tau in TAU:
                    keep, ing = decide(e[tr_idx], j[tr_idx], lo, hi, tau)
                    if ing.mean() > budget: continue
                    bb, gk, ba = metrics(keep, y[tr_idx])
                    if best is None or ba > best[0]: best = (ba, lo, hi, tau)
        return best
    keep_oof = {}; frac_oof = {}
    for bud in BUDGETS: keep_oof[bud] = np.zeros(len(x), bool); frac_oof[bud] = np.zeros(len(x), bool)
    keep_ers_cv = np.zeros(len(x), bool); keep_j_cv = np.zeros(len(x), bool)
    for f in range(5):
        tr = np.where(fo != f)[0]; te = np.where(fo == f)[0]
        for bud in BUDGETS:
            b_ = pick(tr, bud)
            if b_ is None: keep_oof[bud][te] = e[te] >= 0.2; continue
            _, lo, hi, tau = b_; k_, ing = decide(e[te], j[te], lo, hi, tau); keep_oof[bud][te] = k_; frac_oof[bud][te] = ing
        Ts = np.arange(-0.4, 0.81, 0.05); bT = max(Ts, key=lambda T: metrics(e[tr] >= T, y[tr])[2]); keep_ers_cv[te] = e[te] >= bT
        tj = np.arange(0.05, 0.96, 0.05); bt = max(tj, key=lambda t: metrics(j[tr] >= t, y[tr])[2]); keep_j_cv[te] = j[te] >= bt
    cl = x.pubid.values if "pubid" in x else x.claim_id.values; g, W = boot_w(cl, NB, SEED)
    sets = {"ERS alone, T = 0.2 (fixed)": (e >= 0.2, np.zeros(len(x), bool), 0.0), "ERS alone, T chosen by CV": (keep_ers_cv, np.zeros(len(x), bool), 0.0), "Judge alone (J_trust), threshold chosen by CV": (keep_j_cv, np.ones(len(x), bool), 1.0)}
    for bud in BUDGETS: sets[f"Cascade, judge budget {int(bud*100)}%"] = (keep_oof[bud], frac_oof[bud], None)
    def wbal(keep):
        bad = (y == 0); good = (y == 1); out = []
        for b in range(NB):
            w = W[b][g].astype(float); bb = ((~keep) * w * bad).sum() / (w * bad).sum(); gk = (keep * w * good).sum() / (w * good).sum(); out.append((bb + gk) / 2)
        return np.array(out)
    B_ = {k: wbal(v[0]) for k, v in sets.items()}
    for k, (keep, ing, fr) in sets.items():
        bb, gk, ba = metrics(keep, y); f_ = ing.mean() if fr is None or fr == 0.0 else 1.0
        if k.startswith("ERS"): f_ = 0.0
        r = dict(task=label, method=k, bad_blocked=bb, good_kept=gk, balanced_acc=ba, bal_lo=np.percentile(B_[k], 2.5), bal_hi=np.percentile(B_[k], 97.5), judge_fraction=f_, cost_ms_per_passage=ERS_MS + f_ * JUDGE_MS if not k.startswith("Judge") else JUDGE_MS)
        for ref in ["Judge alone (J_trust), threshold chosen by CV", "ERS alone, T = 0.2 (fixed)"]:
            if k.startswith("Cascade"): dd = B_[k] - B_[ref]; r["diff_vs_" + ("judge" if ref.startswith("Judge") else "ERS_T0.2")] = ba - metrics(sets[ref][0], y)[2]; r["diff_vs_" + ("judge" if ref.startswith("Judge") else "ERS_T0.2") + "_lo"] = np.percentile(dd, 2.5); r["diff_vs_" + ("judge" if ref.startswith("Judge") else "ERS_T0.2") + "_hi"] = np.percentile(dd, 97.5)
        res[k] = r
    return pd.DataFrame(res.values())
CB = pd.concat([run_cascade(pq, np.ones(len(pq), bool), "PubMedQA pooled (CLEAN good; DRIFT, NOISE, CONTRADICTION bad)"), run_cascade(pq, pq.condition.isin(["CLEAN", "CONTRADICTION"]).values, "PubMedQA CLEAN vs CONTRADICTION")], ignore_index=True)
CB.to_csv(OUT + "B_cascade_results.csv", index=False); print(CB.round(3).to_string(index=False))
