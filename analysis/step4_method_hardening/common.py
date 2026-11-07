import json, hashlib, os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'..','..','data')
# HC_JSONL: path to dataset/dataset_ready.jsonl from the original HealthContradict repository (not redistributed here)
HC_JSONL=os.environ.get('HC_JSONL','dataset_ready.jsonl')
from sklearn.metrics import roc_auc_score
RS=[0,0.25,0.5,0.75,1,1.5,2,3,5,1e6]           # ratio beta/alpha; 1e6 ~ contradiction-only
TAUS=[0.2,0.3,0.4,0.5,0.6,0.7]
def load_pq(path=os.path.join(DATA,'pubmedqa','rav_med_full_results_N1000.csv')):
    d=pd.read_csv(path); return d
def load_hc(ct='ref',col=('S_top1','C_top1')):
    raw=pd.DataFrame([json.loads(l) for l in open(HC_JSONL,encoding='utf-8')]); raw['topic_id']=raw.topic_id.astype(int)
    did=lambda t:hashlib.md5(t.encode('utf-8')).hexdigest()[:12]
    cor=np.where(raw.query_stance=='yes',raw.doc_a,raw.doc_b); inc=np.where(raw.query_stance=='yes',raw.doc_b,raw.doc_a)
    pr=pd.DataFrame(dict(topic_id=raw.topic_id,ans=raw.query_stance,cid=[did(t) for t in cor],iid=[did(t) for t in inc]))
    s=pd.read_csv(os.path.join(DATA,'healthcontradict','hc_doc_scores.csv')); s=s[s.claim_type==ct].drop_duplicates(['topic_id','doc_id']).set_index(['topic_id','doc_id'])
    S,C=col
    pr['Sc']=[s.loc[(t,i),S] for t,i in zip(pr.topic_id,pr.cid)]; pr['Cc']=[s.loc[(t,i),C] for t,i in zip(pr.topic_id,pr.cid)]
    pr['Si']=[s.loc[(t,i),S] for t,i in zip(pr.topic_id,pr.iid)]; pr['Ci']=[s.loc[(t,i),C] for t,i in zip(pr.topic_id,pr.iid)]
    return pr
# score families: each returns a score (higher = more reliable)
def sc_ratio(S,C,r): return S-r*C
def sc_gate(S,C,tau): return np.where(S>=tau,2.0+(1-C),S)         # irrelevant (S<tau) always ranked below relevant
def sc_prod(S,C,_=None): return S*(1-C)
def sc_min(S,C,_=None): return np.minimum(S,1-C)
FAM={'ratio':(RS,sc_ratio),'gate':(TAUS,sc_gate),'prod':([None],sc_prod),'min':([None],sc_min)}
def pq_tasks(d,ids,fn,p):
    """returns dict of AUROCs (positive = CLEAN) for tasks on the given pubids"""
    x=d[d.pubid.isin(ids)].copy(); x['s']=fn(x.similarity.values,x.contradiction.values,p)
    cl=x[x.condition=='CLEAN']
    def a(conds):
        o=x[x.condition.isin(conds)]; y=np.r_[np.ones(len(cl)),np.zeros(len(o))]; return roc_auc_score(y,np.r_[cl.s.values,o.s.values])
    t1=a(['CONTRADICTION']); t2=a(['DRIFT','NOISE']); t3=a(['CONTRADICTION','DRIFT','NOISE'])
    return dict(T1_conflict=t1,T2_irrelevance=t2,T3_pooled=t3,balanced=(t1+t2)/2)
def hc_acc(pr,fn,p,idx=None):
    q=pr if idx is None else pr.loc[idx]
    a=fn(q.Sc.values,q.Cc.values,p); b=fn(q.Si.values,q.Ci.values,p); return float(np.mean((a>b)+0.5*(a==b)))
