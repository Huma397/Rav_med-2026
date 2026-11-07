import json,hashlib,numpy as np,pandas as pd
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
HC_JSONL = os.environ.get("HC_JSONL", "")
if not HC_JSONL or not os.path.exists(HC_JSONL):
    sys.exit("Set HC_JSONL to the HealthContradict file dataset/dataset_ready.jsonl (clone the original repository; its text is not redistributed here).")
OUTDIR = Path(os.environ.get("RAV_OUT", REPO / "reproduced_outputs")); OUTDIR.mkdir(parents=True, exist_ok=True)
from sklearn.metrics import roc_auc_score
Z=str(REPO/"data"/"judge_baselines")+"/"; rng=np.random.default_rng(2026); B=4000
raw=pd.DataFrame([json.loads(l) for l in open(HC_JSONL,encoding="utf-8")]); raw["topic_id"]=raw.topic_id.astype(int)
ct=np.where(raw.query_stance=="yes",raw.doc_a,raw.doc_b); it=np.where(raw.query_stance=="yes",raw.doc_b,raw.doc_a)
did=lambda t:hashlib.md5(t.encode()).hexdigest()[:12]
P=pd.DataFrame(dict(topic_id=raw.topic_id,c=[did(t) for t in ct],i=[did(t) for t in it]))
d=pd.read_csv(Z+"s6a_healthcontradict_judge_scores.csv")
SC={"S":("S",1),"-C":("C",-1),"ERS":("ERS",1),"J_rel":("J_rel",1),"J_trust":("J_trust",1),"-J_con":("J_con",-1),"J_rel-J_con":("J_rel_minus_con",1),"J_sup-J_con":("J_sup_minus_con",1)}
topics=np.array(sorted(P.topic_id.unique())); res=[]
for ctp in ["ref","free"]:
    k=d[d.claim_type==ctp].set_index(["topic_id","doc_id"])
    W={}
    for n,(col,sg) in SC.items():
        a=sg*k.loc[list(zip(P.topic_id,P.c)),col].values; b=sg*k.loc[list(zip(P.topic_id,P.i)),col].values
        W[n]=(a>b)+0.5*(a==b)
    # cluster weights
    idx={t:np.where(P.topic_id.values==t)[0] for t in topics}
    cnt=np.array([len(idx[t]) for t in topics])
    draws=rng.multinomial(len(topics),np.ones(len(topics))/len(topics),size=B)  # cluster resample counts
    def acc(w):
        s=np.array([w[idx[t]].sum() for t in topics]); return (draws@s)/(draws@cnt)
    A={n:acc(W[n]) for n in W}
    for n in W:
        if n=="ERS": continue
        diff=A["ERS"]-A[n]; res.append(dict(claim=ctp,comparator=n,acc_ERS=W["ERS"].mean(),acc_comp=W[n].mean(),diff=W["ERS"].mean()-W[n].mean(),lo=np.percentile(diff,2.5),hi=np.percentile(diff,97.5)))
H=pd.DataFrame(res); print("HC paired (ERS minus comparator), cluster bootstrap over 81 questions"); print(H.round(3).to_string(index=False)); H.to_csv(OUTDIR/"paired_healthcontradict_ERS_minus_comparator.csv",index=False)
# PubMedQA paired AUROC differences
q=pd.read_csv(Z+"s6a_pubmedqa_judge_scores.csv"); pids=q.pubid.unique(); n=len(pids); pos={p:i for i,p in enumerate(pids)}
q["g"]=q.pubid.map(pos)
SQ={"ERS":("ERS",1),"J_rel":("J_rel",1),"J_trust":("J_trust",1),"J_rel-J_con":("J_rel_minus_con",1),"J_sup-J_con":("J_sup_minus_con",1)}
Wd=rng.multinomial(n,np.ones(n)/n,size=1500); out=[]
for tn,negs in {"CLEAN vs CONTRADICTION":["CONTRADICTION"],"CLEAN vs DRIFT+NOISE":["DRIFT","NOISE"],"pooled":["CONTRADICTION","DRIFT","NOISE"]}.items():
    x=q[q.condition.isin(["CLEAN"]+negs)]; y=(x.condition=="CLEAN").astype(int).values; g=x.g.values
    sw=[Wd[:,g[j]] for j in range(1)]  # placeholder
    sc={nm:sg*x[c].values for nm,(c,sg) in SQ.items()}
    est={nm:roc_auc_score(y,s) for nm,s in sc.items()}
    bs={nm:[] for nm in sc}
    for b in range(1500):
        w=Wd[b][g]; m=w>0
        for nm,s in sc.items(): bs[nm].append(roc_auc_score(y[m],s[m],sample_weight=w[m]))
    for nm in sc:
        if nm=="ERS": continue
        df=np.array(bs["ERS"])-np.array(bs[nm]); out.append(dict(task=tn,comparator=nm,AUROC_ERS=est["ERS"],AUROC_comp=est[nm],diff=est["ERS"]-est[nm],lo=np.percentile(df,2.5),hi=np.percentile(df,97.5)))
Q=pd.DataFrame(out); print("\nPubMedQA paired AUROC (ERS minus comparator), cluster bootstrap over queries"); print(Q.round(3).to_string(index=False)); Q.to_csv(OUTDIR/"paired_pubmedqa_ERS_minus_comparator.csv",index=False)
# reproduce pairwise accuracies
for ctp in ["ref","free"]:
    print(ctp,"reproduced ERS pairwise:",round(float(0),3) if False else "")
