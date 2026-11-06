import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
rng=np.random.default_rng(42)
d=load_pq(); L=pd.read_csv(os.path.join(DATA,'pubmedqa','rav_med_lenmatched_results.csv')); pr=load_hc(); prF=load_hc('free')
ids=np.array(sorted(d.pubid.unique())); topics=np.array(sorted(pr.topic_id.unique()))
out=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.append(s)
# ---- 1. full-data curves (descriptive) ----
rows=[]
for fam,(grid,fn) in FAM.items():
    for p in grid:
        r=dict(family=fam,param=p); 
        for nm,dd in [('PQ',d),('LM',L)]:
            t=pq_tasks(dd,ids,fn,p); r.update({f'{nm}_{k}':round(v,4) for k,v in t.items()})
        r['HC_ref']=round(hc_acc(pr,fn,p),4); r['HC_free']=round(hc_acc(prF,fn,p),4); rows.append(r)
cur=pd.DataFrame(rows); cur.to_csv('s4_curves_fulldata.csv',index=False)
P('FULL-DATA CURVES (descriptive, not for model selection)'); P(cur.to_string(index=False))
# ---- 2. in-dataset split: tune on half, test on other half; 200 random 2-fold splits ----
def best(fam,score_fn,idx_tune):
    grid,fn=FAM[fam]; v=[score_fn(fn,p,idx_tune) for p in grid]; k=int(np.nanargmax(v)); return grid[k]
def pq_split(R=200):
    res=[]
    for rep in range(R):
        perm=rng.permutation(ids); A,B=perm[:500],perm[500:]
        for tune,test in [(A,B),(B,A)]:
            for fam in ['ratio','gate']:
                p=best(fam,lambda fn,p,i:pq_tasks(d,i,fn,p)['balanced'],tune); fn=FAM[fam][1]
                t=pq_tasks(d,test,fn,p); base=pq_tasks(d,test,sc_ratio,1)
                res.append(dict(rep=rep,family=fam,param=p,**{f'tuned_{k}':v for k,v in t.items()},**{f'default_{k}':v for k,v in base.items()}))
    return pd.DataFrame(res)
R1=pq_split(); R1.to_csv('s4_pq_split_results.csv',index=False)
P('\nPUBMEDQA 2-FOLD TUNE/TEST (200 random splits x 2 folds; tuned on balanced AUROC of conflict and irrelevance tasks)')
for fam,g in R1.groupby('family'):
    P(f' family={fam}: chosen param distribution {g.param.value_counts().to_dict()}')
    for k in ['T1_conflict','T2_irrelevance','T3_pooled','balanced']:
        dlt=g[f'tuned_{k}']-g[f'default_{k}']; P(f'   {k}: tuned {g["tuned_"+k].mean():.4f} vs default(alpha=beta=1) {g["default_"+k].mean():.4f}; diff {dlt.mean():+.4f} (2.5-97.5 pct across splits {np.percentile(dlt,2.5):+.4f}, {np.percentile(dlt,97.5):+.4f})')
# ---- 3. HC split by question ----
def hc_split(R=200):
    res=[]
    for rep in range(R):
        perm=rng.permutation(topics); A,B=perm[:40],perm[40:]
        for tune,test in [(A,B),(B,A)]:
            it=pr.index[pr.topic_id.isin(tune)]; ie=pr.index[pr.topic_id.isin(test)]
            for fam in ['ratio','gate']:
                p=best(fam,lambda fn,p,i:hc_acc(pr,fn,p,i),it); fn=FAM[fam][1]
                res.append(dict(rep=rep,family=fam,param=p,tuned=hc_acc(pr,fn,p,ie),default=hc_acc(pr,sc_ratio,1,ie),conly=hc_acc(pr,sc_ratio,1e6,ie)))
    return pd.DataFrame(res)
R2=hc_split(); R2.to_csv('s4_hc_split_results.csv',index=False)
P('\nHEALTHCONTRADICT 2-FOLD BY QUESTION (200 random splits x 2 folds; reference claim)')
for fam,g in R2.groupby('family'):
    dl=g.tuned-g['default']; P(f' family={fam}: chosen {g.param.value_counts().to_dict()}; test acc tuned {g.tuned.mean():.4f} vs default {g["default"].mean():.4f} vs C-only {g.conly.mean():.4f}; diff {dl.mean():+.4f} (2.5-97.5 pct {np.percentile(dl,2.5):+.4f}, {np.percentile(dl,97.5):+.4f})')
# ---- 4. transfer ----
P('\nTRANSFER (parameter chosen on one dataset, evaluated on the other)')
for fam in ['ratio','gate']:
    pq_p=best(fam,lambda fn,p,i:pq_tasks(d,i,fn,p)['balanced'],ids); hc_p=best(fam,lambda fn,p,i:hc_acc(pr,fn,p,i),pr.index)
    fn=FAM[fam][1]
    P(f' {fam}: tuned on PubMedQA -> param {pq_p}; HC accuracy {hc_acc(pr,fn,pq_p):.4f} (default {hc_acc(pr,sc_ratio,1):.4f})')
    P(f' {fam}: tuned on HC -> param {hc_p}; PubMedQA balanced AUROC {pq_tasks(d,ids,fn,hc_p)["balanced"]:.4f} (default {pq_tasks(d,ids,sc_ratio,1)["balanced"]:.4f})')
# ---- 5. band calibration at default scoring ----
P('\nRISK-BAND CALIBRATION (ERS, alpha=beta=1). Share of passages that are CLEAN (PubMedQA) / evidence-correct (HC ref) within each band')
d['ERS1']=d.similarity-d.contradiction
for T in [0.0,0.2,0.4]:
    band=np.where(d.ERS1>=T,'SAFE',np.where(d.ERS1>=T-0.2,'RISKY','UNSAFE')); g=pd.crosstab(band,d.condition)
    hcs=pd.concat([pd.DataFrame(dict(e=pr.Sc-pr.Cc,c=1)),pd.DataFrame(dict(e=pr.Si-pr.Ci,c=0))]); hb=np.where(hcs.e>=T,'SAFE',np.where(hcs.e>=T-0.2,'RISKY','UNSAFE'))
    for b in ['SAFE','RISKY','UNSAFE']:
        n=g.loc[b].sum() if b in g.index else 0; cl=g.loc[b,'CLEAN'] if b in g.index else 0; hn=(hb==b).sum(); hcor=hcs.c[hb==b].sum()
        P(f' T={T}: {b:6s} PubMedQA n={n:4d} clean share={cl/max(n,1):.3f} | HC n={hn:4d} correct-doc share={hcor/max(hn,1):.3f}')
open('s4_sweep_report.txt','w').write('\n'.join(out))
