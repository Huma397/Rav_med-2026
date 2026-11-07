"""Step 7, items 1 and 2 (plan: PLAN_fixed_before_running.md). Real data only."""
import json, hashlib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R='/mnt/user-data/outputs/'; OUT=R+'step7_existing_data/'; SEED=42; NB=2000
# ---------------- data
pq=pd.read_csv(R+'rav_med_full_results_N1000.csv').rename(columns={'similarity':'S','contradiction':'C'})
sf=pd.read_csv(R+'step6_new_experiments/results_real/6d/s6d_scifact_scores_win_notext.csv')
sfa=pd.read_csv(R+'step6_new_experiments/results_real/6d/s6d_scifact_scores_full_notext.csv')
raw=pd.DataFrame([json.loads(l) for l in open('/home/claude/hc/repo/dataset/dataset_ready.jsonl',encoding='utf-8')]); raw['topic_id']=raw.topic_id.astype(int)
did=lambda t: hashlib.md5(t.encode('utf-8')).hexdigest()[:12]
cor=np.where(raw.query_stance=='yes',raw.doc_a,raw.doc_b); inc=np.where(raw.query_stance=='yes',raw.doc_b,raw.doc_a)
pairs=pd.DataFrame(dict(topic_id=raw.topic_id,cid=[did(t) for t in cor],iid=[did(t) for t in inc]))
hs=pd.read_csv('/home/claude/s2/hc_doc_scores.csv'); hs=hs[hs.claim_type=='ref'].drop_duplicates(['topic_id','doc_id']).set_index(['topic_id','doc_id'])
pairs['Sc']=[hs.loc[(t,d),'S_top1'] for t,d in zip(pairs.topic_id,pairs.cid)]; pairs['Cc']=[hs.loc[(t,d),'C_top1'] for t,d in zip(pairs.topic_id,pairs.cid)]
pairs['Si']=[hs.loc[(t,d),'S_top1'] for t,d in zip(pairs.topic_id,pairs.iid)]; pairs['Ci']=[hs.loc[(t,d),'C_top1'] for t,d in zip(pairs.topic_id,pairs.iid)]
units=pd.concat([pairs[['topic_id','cid','Sc','Cc']].rename(columns={'cid':'doc','Sc':'S','Cc':'C'}).assign(good=1),
                 pairs[['topic_id','iid','Si','Ci']].rename(columns={'iid':'doc','Si':'S','Ci':'C'}).assign(good=0)]).drop_duplicates(['topic_id','doc'])
# the same doc cannot be both good and bad in one question
dup=units.duplicated(['topic_id','doc'],keep=False).sum()
key=pd.read_csv(R+'step5_clinician/private_do_not_share/ClinPartA_KEY.csv'); lab=pd.read_excel(R+'step5_clinician/returned/RAV_Med_clinician_review_FINAL_clarified.xlsx',sheet_name='PartA',dtype=str)
cl=key.merge(lab.rename(columns={'item_id':'cid'})[['cid','could_mislead']],on='cid'); cl=cl[cl.repeat_of.isna()].rename(columns={'similarity':'S','contradiction':'C'})
cl['good']=(cl.could_mislead.str.upper()=='NO').astype(int); cl['ERS']=cl.S-cl.C; units['ERS']=units.S-units.C; assert len(cl)==60
# ---------------- tasks for item 1: (name, good df, bad df, cluster col)
def T(df,col,vals): return df[df[col].isin(vals)]
tasks=[('PubMedQA: CLEAN vs CONTRADICTION',T(pq,'condition',['CLEAN']),T(pq,'condition',['CONTRADICTION']),'pubid'),
       ('PubMedQA: CLEAN vs DRIFT+NOISE',T(pq,'condition',['CLEAN']),T(pq,'condition',['DRIFT','NOISE']),'pubid'),
       ('SciFact (window): SUPPORT vs CONTRADICT',T(sf,'cond',['SUPPORT']),T(sf,'cond',['CONTRADICT']),'claim_id'),
       ('SciFact (window): SUPPORT vs IRRELEVANT',T(sf,'cond',['SUPPORT']),T(sf,'cond',['IRRELEVANT']),'claim_id'),
       ('SciFact (window): SUPPORT vs NEI_ONTOPIC',T(sf,'cond',['SUPPORT']),T(sf,'cond',['NEI_ONTOPIC']),'claim_id'),
       ('HealthContradict (ref claim): correct vs incorrect doc',units[units.good==1],units[units.good==0],'topic_id'),
       ('Clinician Part A: could_mislead NO vs YES',cl[cl.good==1],cl[cl.good==0],None)]
rng=np.random.default_rng(SEED)
def rates(g,b,thr):
    return (b.ERS<thr).mean(), (g.ERS>=thr).mean()
rows=[]
for name,g,b,cc in tasks:
    g=g.copy(); b=b.copy()
    for d in (g,b):
        if 'ERS' not in d: d['ERS']=d.S-d.C
    for thr in (0.0,0.2,0.4):
        pb,pg=rates(g,b,thr)
        if cc:
            cg={c:ix for c,ix in g.groupby(cc).indices.items()}; cb={c:ix for c,ix in b.groupby(cc).indices.items()}
            ids=np.array(sorted(set(cg)|set(cb))); bs_b=[];bs_g=[]
            for _ in range(NB):
                pick=rng.choice(ids,len(ids),replace=True)
                gi=np.concatenate([cg[c] for c in pick if c in cg]); bi=np.concatenate([cb[c] for c in pick if c in cb])
                bs_b.append((b.ERS.values[bi]<thr).mean()); bs_g.append((g.ERS.values[gi]>=thr).mean())
        else:
            bs_b=[(b.ERS.values[rng.integers(0,len(b),len(b))]<thr).mean() for _ in range(NB)]; bs_g=[(g.ERS.values[rng.integers(0,len(g),len(g))]>=thr).mean() for _ in range(NB)]
        rows.append(dict(task=name,n_good=len(g),n_bad=len(b),T=thr,bad_blocked=pb,bad_lo=np.percentile(bs_b,2.5),bad_hi=np.percentile(bs_b,97.5),good_kept=pg,good_lo=np.percentile(bs_g,2.5),good_hi=np.percentile(bs_g,97.5)))
T1=pd.DataFrame(rows); T1.to_csv(OUT+'tables/item1_threshold_operating_points.csv',index=False); pd.set_option('display.width',250); print(T1.round(3).to_string(index=False)); print('dup units',dup)
# full curves for plotting: ERS, S-only, C-only
curves=[]
for name,g,b,cc in tasks:
    for sn,fg,fb in (('ERS',g.ERS if 'ERS' in g else g.S-g.C,b.ERS if 'ERS' in b else b.S-b.C),('Similarity only',g.S,b.S),('Contradiction only',-g.C,-b.C)):
        grid=np.unique(np.r_[np.quantile(np.r_[fg,fb],np.linspace(0,1,401))])
        for t in grid: curves.append(dict(task=name,scorer=sn,thr=t,good_lost=1-(fg>=t).mean(),bad_blocked=(fb<t).mean()))
pd.DataFrame(curves).to_csv(OUT+'tables/item1_curves.csv',index=False)
# ---------------- item 2: leave-one-dataset-out
RS=[0,.25,.5,.75,1,1.5,2,3,5,1e6]; TAUS=[.2,.3,.4,.5,.6,.7]
fam={'ratio':[(r,(lambda S,C,r=r:S-r*C)) for r in RS],'gate':[(t,(lambda S,C,t=t:np.where(S>=t,2.0+(1-C),S))) for t in TAUS]}
def auc(y,x,w=None): return roc_auc_score(y,x,sample_weight=w)
def _wauc(df,pos,negs,s,w):
    m=df.cond_.isin([pos]+negs).values; y=(df.cond_.values[m]==pos).astype(int); return auc(y,s[m],None if w is None else w[m])
def m_pq(fn,w=None):
    s=fn(pq.S.values,pq.C.values); return (_wauc(pq.assign(cond_=pq.condition),'CLEAN',['CONTRADICTION'],s,w)+_wauc(pq.assign(cond_=pq.condition),'CLEAN',['DRIFT','NOISE'],s,w))/2
def m_sf(fn,w=None):
    s=fn(sf.S.values,sf.C.values); d=sf.assign(cond_=sf.cond); return (_wauc(d,'SUPPORT',['CONTRADICT'],s,w)+_wauc(d,'SUPPORT',['IRRELEVANT'],s,w))/2
def m_hc(fn,w=None):
    a=fn(pairs.Sc.values,pairs.Cc.values); b=fn(pairs.Si.values,pairs.Ci.values); x=(a>b)+0.5*(a==b)
    return float(np.average(x,weights=w))
pq['code']=pq.pubid.astype('category').cat.codes; sf['code']=sf.claim_id.astype('category').cat.codes; pairs['code']=pairs.topic_id.astype('category').cat.codes
DS={'PubMedQA':(m_pq,pq.code.values),'SciFact':(m_sf,sf.code.values),'HealthContradict':(m_hc,pairs.code.values)}
tab={}
for fname,lst in fam.items():
    for p,fn in lst:
        for d,(m,_) in DS.items(): tab[(fname,p,d)]=m(fn)
G=pd.DataFrame([dict(family=f,param=p,dataset=d,metric=v) for (f,p,d),v in tab.items()]); G.to_csv(OUT+'tables/item2_metric_grid.csv',index=False)
print(G.pivot_table(index=['family','param'],columns='dataset',values='metric').round(4))
rows=[]
rb=np.random.default_rng(SEED)
for fname,lst in fam.items():
    params=[p for p,_ in lst]; fns=dict(lst)
    for tgt,(m,ids) in DS.items():
        others=[d for d in DS if d!=tgt]
        sel=max(params,key=lambda p:(np.mean([tab[(fname,p,d)] for d in others]),-params.index(p)))
        own=max(params,key=lambda p:(tab[(fname,p,tgt)],-params.index(p)))
        ref_fn=dict(fam['ratio'])[1]; base=tab[('ratio',1,tgt)]
        diffs=[]; codes=ids; nc=codes.max()+1
        for _ in range(NB):
            cnt=rb.multinomial(nc,np.ones(nc)/nc); w=cnt[codes].astype(float)
            diffs.append(m(fns[sel],w)-m(ref_fn,w))
        rows.append(dict(target=tgt,family=fname,param_chosen_on_other_two=sel,metric_chosen=tab[(fname,sel,tgt)],metric_alpha_beta_1=base,diff=tab[(fname,sel,tgt)]-base,diff_lo=np.percentile(diffs,2.5),diff_hi=np.percentile(diffs,97.5),own_best_param=own,metric_own_best=tab[(fname,own,tgt)]))
T2=pd.DataFrame(rows); T2.to_csv(OUT+'tables/item2_lodo_transfer.csv',index=False); print(T2.round(4).to_string(index=False))
