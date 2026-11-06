# Step 8, SciFact part, as pre-specified (transfer from all PubMedQA; cascade with grouped 5-fold CV, grid and budgets unchanged)
import numpy as np, pandas as pd, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
OUTDIR = Path(os.environ.get("RAV_OUT", REPO / "reproduced_outputs")); OUTDIR.mkdir(parents=True, exist_ok=True)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
R6A=str(REPO/"data")+"/"; OUT=str(OUTDIR)+"/"
ERS_MS,JUDGE_MS,SEED,NB=51.13,868.15,2026,2000
lg=lambda p:np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
def prep(d):
    d=d.copy(); d["E"]=d.S-d.C
    for a,b in [("Jr","J_rel"),("Jc","J_con"),("Js","J_sup"),("Jt","J_trust")]: d[a]=lg(d[b].values)
    d["Jrc"]=d.J_rel-d.J_con; d["Jtr"]=d.J_trust; return d
cols={"H1":["S","C","Jt"],"H2":["S","C","Jr","Jc","Js","Jt"]}
pq=pd.read_csv(R6A+"judge_baselines/s6a_pubmedqa_judge_scores.csv"); pq["S"]=pq["similarity"]; pq["C"]=pq["contradiction"]; pq=prep(pq); pq["y"]=(pq.condition=="CLEAN").astype(int)
sf=prep(pd.read_csv(R6A+"scifact/s6g_scifact_item_scores_nowindowtext.csv")); sf["y"]=(sf.cond=="SUPPORT").astype(int)
mods={}
for k,c in cols.items(): sc=StandardScaler().fit(pq[c].values); mods[k]=(sc,LogisticRegression(C=1.0,max_iter=1000).fit(sc.transform(pq[c].values),pq.y))
def folds(g,k=5,seed=SEED):
    u=np.unique(g); p=np.random.default_rng(seed).permutation(len(u)); m={x:i%k for i,x in enumerate(u[p])}; return np.array([m[x] for x in g])
def boot(groups,B=NB,seed=SEED):
    u,g=np.unique(groups,return_inverse=True); return g,np.random.default_rng(seed).multinomial(len(u),np.ones(len(u))/len(u),size=B)
def waucs(y,s,g,W): return np.array([roc_auc_score(y[W[b][g]>0],s[W[b][g]>0],sample_weight=W[b][g][W[b][g]>0]) for b in range(W.shape[0])])
rows=[]
for tname,negs in [("SUPPORT vs all others (pooled)",["CONTRADICT","IRRELEVANT","NEI_ONTOPIC"]),("SUPPORT vs CONTRADICT",["CONTRADICT"])]:
    x=sf[sf.cond.isin(["SUPPORT"]+negs)].reset_index(drop=True); y=x.y.values; g,W=boot(x.claim_id.values)
    S={"ERS":x.E.values,"J_trust":x.Jtr.values,"J_rel_minus_con":x.Jrc.values}
    for k in cols: sc,m=mods[k]; S[f"{k} transferred from PubMedQA"]=m.decision_function(sc.transform(x[cols[k]].values))
    bs={k:waucs(y,v,g,W) for k,v in S.items()}; est={k:roc_auc_score(y,v) for k,v in S.items()}
    for k in S:
        r=dict(task=tname,scorer=k,AUROC=est[k],lo=np.percentile(bs[k],2.5),hi=np.percentile(bs[k],97.5))
        if k.startswith("H"):
            for ref in ["ERS","J_trust"]: d=bs[k]-bs[ref]; r[f"diff_vs_{ref}"]=est[k]-est[ref]; r[f"lo_vs_{ref}"]=np.percentile(d,2.5); r[f"hi_vs_{ref}"]=np.percentile(d,97.5)
        rows.append(r)
A=pd.DataFrame(rows); A.to_csv(OUT+"A_hybrid_scifact.csv",index=False)
# cascade
LO=[-0.4,-0.2,0,0.1,0.2]; HI=[0.2,0.3,0.4,0.5,0.6]; TAU=[0.3,0.5,0.7]; BUD=[0.10,0.25,0.50]
def dec(e,j,lo,hi,tau): return np.where(e>=hi,True,np.where(e<lo,False,j>=tau)),((e>=lo)&(e<hi))
def met(k,y): bb=(~k[y==0]).mean(); gk=k[y==1].mean(); return bb,gk,(bb+gk)/2
e=sf.E.values; j=sf.Jtr.values; y=sf.y.values; cl=sf.claim_id.values; fo=folds(cl)
ko={b:np.zeros(len(sf),bool) for b in BUD}; fr={b:np.zeros(len(sf),bool) for b in BUD}; kE=np.zeros(len(sf),bool); kJ=np.zeros(len(sf),bool)
for f in range(5):
    tr=np.where(fo!=f)[0]; te=np.where(fo==f)[0]
    for b in BUD:
        best=None
        for lo in LO:
            for hi in HI:
                if lo>=hi: continue
                for tau in TAU:
                    k_,ing=dec(e[tr],j[tr],lo,hi,tau)
                    if ing.mean()>b: continue
                    ba=met(k_,y[tr])[2]
                    if best is None or ba>best[0]: best=(ba,lo,hi,tau)
        if best is None: ko[b][te]=e[te]>=0.2
        else: _,lo,hi,tau=best; ko[b][te],fr[b][te]=dec(e[te],j[te],lo,hi,tau)
    Ts=np.arange(-0.4,0.81,0.05); T=max(Ts,key=lambda t:met(e[tr]>=t,y[tr])[2]); kE[te]=e[te]>=T
    ts=np.arange(0.05,0.96,0.05); t=max(ts,key=lambda t:met(j[tr]>=t,y[tr])[2]); kJ[te]=j[te]>=t
g,W=boot(cl)
def wb(k):
    o=[]
    for b in range(NB):
        w=W[b][g].astype(float); bad=(y==0); good=(y==1); o.append((((~k)*w*bad).sum()/(w*bad).sum()+(k*w*good).sum()/(w*good).sum())/2)
    return np.array(o)
sets={"ERS alone, T = 0.2 (fixed)":(e>=0.2,0.0),"ERS alone, T chosen by CV":(kE,0.0),"Judge alone (J_trust), threshold chosen by CV":(kJ,1.0)}
for b in BUD: sets[f"Cascade, judge budget {int(b*100)}%"]=(ko[b],fr[b].mean())
Bs={k:wb(v[0]) for k,v in sets.items()}; ref="Judge alone (J_trust), threshold chosen by CV"; rows=[]
for k,(kp,f_) in sets.items():
    bb,gk,ba=met(kp,y); r=dict(method=k,bad_blocked=bb,good_kept=gk,balanced_acc=ba,lo=np.percentile(Bs[k],2.5),hi=np.percentile(Bs[k],97.5),judge_fraction=f_,cost_ms=JUDGE_MS if k.startswith("Judge") else ERS_MS+f_*JUDGE_MS)
    if k.startswith("Cascade"): d=Bs[k]-Bs[ref]; r.update(diff_vs_judge=ba-met(sets[ref][0],y)[2],diff_lo=np.percentile(d,2.5),diff_hi=np.percentile(d,97.5))
    rows.append(r)
C=pd.DataFrame(rows); C.to_csv(OUT+"B_cascade_scifact.csv",index=False)
pd.set_option("display.width",250); print(A.round(3).to_string(index=False)); print(C.round(3).to_string(index=False))
