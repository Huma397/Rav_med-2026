"""RAV-Med Step 3, Part B analysis.
Usage: python3 analyze_partB.py annotator1.xlsx annotator2.xlsx PartB_ANSWER_KEY_do_not_share.csv [adjudicated.csv|""] [outdir]
adjudicated.csv (optional): item_id, final_label (after the two annotators discuss disagreements)."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from scipy.stats import beta
f1,f2,fk=sys.argv[1:4]; adj=sys.argv[4] if len(sys.argv)>4 and sys.argv[4].endswith('.csv') else None
out=sys.argv[5] if len(sys.argv)>5 else '.'; LAB=['YES','NO','MIXED','NOT_RELEVANT']; rng=np.random.default_rng(42)
key=pd.read_csv(fk)
def load(p,t):
    d=pd.read_excel(p,sheet_name='PartB')[['item_id','label','confidence']].rename(columns={'label':f'label_{t}','confidence':f'conf_{t}'})
    if d[f'label_{t}'].isna().any(): print(f'WARNING: annotator {t} left {int(d[f"label_{t}"].isna().sum())} items unlabeled')
    bad=set(d[f'label_{t}'].dropna())-set(LAB); assert not bad,f'invalid labels from annotator {t}: {bad}'; return d
d=key.merge(load(f1,'1'),on='item_id').merge(load(f2,'2'),on='item_id'); assert len(d)==len(key),'item_id mismatch'
isrep=d.repeat_of.notna()&(d.repeat_of!=''); base=d[~isrep].copy(); rep=d[isrep]
R=[]
def say(s=''): print(s); R.append(s)
def wil(k,n):
    return (np.nan,np.nan) if n==0 else ((beta.ppf(.025,k,n-k+1) if k>0 else 0),(beta.ppf(.975,k+1,n-k) if k<n else 1))
def boot(fn,df,n=2000):
    v=[]
    for _ in range(n):
        try: v.append(fn(df.iloc[rng.integers(0,len(df),len(df))]))
        except Exception: pass
    return np.percentile(v,[2.5,97.5])
b=base.dropna(subset=['label_1','label_2'])
k=cohen_kappa_score(b.label_1,b.label_2,labels=LAB); lo,hi=boot(lambda s:cohen_kappa_score(s.label_1,s.label_2,labels=LAB),b)
say(f'INTER-ANNOTATOR (n={len(b)} windows): Cohen kappa = {k:.3f} (95% CI {lo:.3f}-{hi:.3f}); raw agreement = {(b.label_1==b.label_2).mean():.3f}')
say(pd.crosstab(b.label_1,b.label_2).reindex(index=LAB,columns=LAB,fill_value=0).to_string())
col3=lambda x: x.where(x.isin(['YES','NO']),'OTHER')
k3=cohen_kappa_score(col3(b.label_1),col3(b.label_2),labels=['YES','NO','OTHER']); kc=cohen_kappa_score(b.label_1.isin(['YES','NO']),b.label_2.isin(['YES','NO']))
say(f'  collapsed 3-way (YES / NO / OTHER = MIXED or NOT_RELEVANT): kappa = {k3:.3f}; clear stance vs no clear stance: kappa = {kc:.3f}')
for t in ['1','2']:
    m=rep.merge(base[['item_id','label_'+t]].rename(columns={'item_id':'repeat_of','label_'+t:'orig'}),on='repeat_of').dropna(subset=['label_'+t,'orig'])
    if len(m): say(f'INTRA-ANNOTATOR annotator {t} (n={len(m)}): agreement = {(m["label_"+t]==m.orig).mean():.3f}')
if adj:
    a=pd.read_csv(adj); b=b.merge(a,on='item_id',how='left'); b['gold']=b.final_label; say(f'\nGold = adjudicated labels (n={b.gold.notna().sum()})')
else:
    b['gold']=np.where(b.label_1==b.label_2,b.label_1,np.nan); say(f'\nGold = windows where both annotators agree (n={b.gold.notna().sum()} of {len(b)}); disagreements need adjudication')
g=b.dropna(subset=['gold']).copy(); g['truth']=g.doc_stance.str.upper(); g['opp']=np.where(g.truth=='YES','NO','YES')
say('\nDOES THE TOP-1 WINDOW KEEP THE DOCUMENT STANCE? (gold human label vs dataset document stance)')
for role in ['correct','incorrect']:
    x=g[g.doc_role==role]; n=len(x); kk=(x.gold==x.truth).sum(); lo,hi=wil(kk,n)
    say(f'  {role} documents: window stance = document stance in {kk}/{n} = {kk/max(n,1):.3f} (Wilson {lo:.3f}-{hi:.3f}); opposite stance {(x.gold==x.opp).sum()}; MIXED {(x.gold=="MIXED").sum()}; NOT_RELEVANT {(x.gold=="NOT_RELEVANT").sum()}')
say('\n  Full table, gold human label (columns) by dataset document stance (rows; dataset has only YES/NO):')
say(pd.crosstab([g.doc_role,g.truth],g.gold).reindex(columns=LAB,fill_value=0).to_string())
cl=g[g.gold.isin(['YES','NO'])]
if len(cl):
    kk=(cl.gold==cl.truth).sum(); lo,hi=wil(kk,len(cl)); say(f'  Among windows where humans see a clear stance (YES or NO; n={len(cl)}), the stance matches the dataset in {kk}/{len(cl)} = {kk/len(cl):.3f} (Wilson {lo:.3f}-{hi:.3f})')
say('  Note: MIXED or NOT_RELEVANT on a window is not an annotator error. It means the top-scoring excerpt does not carry the document stance, which is a property of the excerpt, not of the dataset label.')
g['rel']=np.where(g.gold==g.evidence_answer.str.upper(),'SUPPORTS',np.where(g.gold==np.where(g.evidence_answer=='yes','NO','YES'),'CONTRADICTS','NEITHER'))
say('\nHUMAN RELATION TO THE REFERENCE CLAIM (SUPPORTS = window gives the evidence-supported answer; CONTRADICTS = opposite answer)')
say(g.groupby('rel')[['S_top1','C_top1','ERS_top1']].agg(['count','mean']).round(3).to_string())
sc=g[g.rel.isin(['SUPPORTS','CONTRADICTS'])].copy(); sc['y']=(sc.rel=='SUPPORTS').astype(int)
if sc.y.nunique()==2:
    for nm,col,sg in [('ERS','ERS_top1',1),('similarity','S_top1',1),('-contradiction','C_top1',-1)]:
        a=roc_auc_score(sc.y,sg*sc[col]); lo,hi=boot(lambda s:roc_auc_score(s.y,sg*s[col]),sc); say(f'  AUROC separating human SUPPORTS from CONTRADICTS windows, {nm}: {a:.3f} (95% CI {lo:.3f}-{hi:.3f}, n={len(sc)})')
# pairwise accuracy of ERS on the sampled pairs
piv=b.pivot_table(index='topic_id',columns='doc_role',values='ERS_top1'); win=((piv.correct>piv.incorrect)+0.5*(piv.correct==piv.incorrect)).astype(float)
say(f'\nERS pairwise accuracy on the {len(win)} sampled pairs (all): {win.mean():.3f}')
conf=g.pivot_table(index='topic_id',columns='doc_role',values='gold',aggfunc='first').dropna()
ok=[t for t in conf.index if conf.loc[t,'correct']==g[g.topic_id==t].truth.iloc[0] and conf.loc[t,'incorrect']==g[(g.topic_id==t)&(g.doc_role=='incorrect')].truth.iloc[0]]
if ok: say(f'ERS pairwise accuracy on pairs where humans confirm BOTH window stances ({len(ok)} pairs): {win.loc[ok].mean():.3f}')
g.to_csv(f'{out}/partB_gold_with_scores.csv',index=False); open(f'{out}/partB_analysis_report.txt','w').write('\n'.join(R)); print(f'\nSaved {out}/partB_analysis_report.txt')
