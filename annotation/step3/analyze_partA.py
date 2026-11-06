"""RAV-Med Step 3, Part A analysis.
Usage: python3 analyze_partA.py annotator1.xlsx annotator2.xlsx PartA_ANSWER_KEY_do_not_share.csv [adjudicated.csv] [outdir]
adjudicated.csv (optional): columns item_id, final_label  (final label after the two annotators discuss disagreements)."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix
from scipy.stats import beta
f1,f2,fk=sys.argv[1:4]; adj=sys.argv[4] if len(sys.argv)>4 and sys.argv[4].endswith('.csv') else None
out=sys.argv[5] if len(sys.argv)>5 else '.'; LAB=['CONTRADICTS','SUPPORTS','NEITHER']; rng=np.random.default_rng(42)
key=pd.read_csv(fk)
def load(p,tag):
    d=pd.read_excel(p,sheet_name='PartA')[['item_id','label','confidence']].rename(columns={'label':f'label_{tag}','confidence':f'conf_{tag}'})
    miss=d[f'label_{tag}'].isna().sum()
    if miss: print(f'WARNING: annotator {tag} left {miss} items unlabeled'); 
    bad=set(d[f'label_{tag}'].dropna())-set(LAB)
    assert not bad, f'invalid labels from annotator {tag}: {bad}'
    return d
d=key.merge(load(f1,'1'),on='item_id').merge(load(f2,'2'),on='item_id')
assert len(d)==len(key), 'item_id mismatch between files and key'
base=d[d.repeat_of.isna()|(d.repeat_of=='')].copy(); rep=d[~(d.repeat_of.isna()|(d.repeat_of==''))]
def boot(fn,df,n=2000):
    v=[]
    for _ in range(n):
        s=df.iloc[rng.integers(0,len(df),len(df))]
        try: v.append(fn(s))
        except Exception: pass
    return np.percentile(v,[2.5,97.5])
def wilson(k,n):
    if n==0: return (np.nan,np.nan)
    lo=beta.ppf(.025,k,n-k+1) if k>0 else 0; hi=beta.ppf(.975,k+1,n-k) if k<n else 1; return lo,hi
R=[]
def say(s=''): print(s); R.append(s)
b=base.dropna(subset=['label_1','label_2'])
k=cohen_kappa_score(b.label_1,b.label_2,labels=LAB); lo,hi=boot(lambda s:cohen_kappa_score(s.label_1,s.label_2,labels=LAB),b)
say(f'INTER-ANNOTATOR (n={len(b)} base items): Cohen kappa = {k:.3f} (95% CI {lo:.3f}-{hi:.3f}); raw agreement = {(b.label_1==b.label_2).mean():.3f}')
say(pd.crosstab(b.label_1,b.label_2).reindex(index=LAB,columns=LAB,fill_value=0).to_string())
for c in ['CLEAN','DRIFT','NOISE','CONTRADICTION']:
    g=b[b.condition==c]; say(f'  agreement within {c}: {(g.label_1==g.label_2).mean():.3f} (n={len(g)})')
for t in ['1','2']:
    m=rep.merge(base[['item_id','label_'+t]].rename(columns={'item_id':'repeat_of','label_'+t:'orig'}),on='repeat_of').dropna(subset=['label_'+t,'orig'])
    if len(m): say(f'INTRA-ANNOTATOR annotator {t} (n={len(m)} repeated items): agreement = {(m["label_"+t]==m.orig).mean():.3f}, kappa = {cohen_kappa_score(m["label_"+t],m.orig):.3f}')
# gold labels
if adj:
    a=pd.read_csv(adj); b=b.merge(a,on='item_id',how='left'); b['gold']=b.final_label; say(f'\nGold = adjudicated labels (n={b.gold.notna().sum()})')
else:
    b['gold']=np.where(b.label_1==b.label_2,b.label_1,np.nan); say(f'\nGold = items where both annotators agree (n={b.gold.notna().sum()} of {len(b)}); disagreements need adjudication')
g=b.dropna(subset=['gold']); r=g[g.stratum=='random']
say('\nCONSTRUCTION VALIDITY (random stratum, gold labels)')
tgt={'CLEAN':'SUPPORTS','DRIFT':'NEITHER','NOISE':'NEITHER','CONTRADICTION':'CONTRADICTS'}
for c,l in tgt.items():
    x=r[r.condition==c]; n=len(x); kk=(x.gold==l).sum(); lo,hi=wilson(kk,n); dist=x.gold.value_counts().to_dict()
    say(f'  {c}: labeled {l} in {kk}/{n} = {kk/max(n,1):.3f} (Wilson 95% CI {lo:.3f}-{hi:.3f}); all labels {dist}')
say('  Full table, condition (rows) by gold human label (columns), random stratum:'); say(pd.crosstab(r.condition,r.gold).reindex(columns=LAB,fill_value=0).to_string())
t=g[g.stratum=='targeted_lowC']; say(f'  TARGETED low-contradiction-score CONTRADICTION items (n={len(t)}): {t.gold.value_counts().to_dict()}  -> labeled CONTRADICTS = true NLI misses; other = construction noise')
say('\nSYSTEM vs HUMAN (random stratum, gold labels; human positive = CONTRADICTS)')
r=r.copy(); r['h']=(r.gold=='CONTRADICTS').astype(int)
def auc(col,sign=1): return lambda s: roc_auc_score(s.h,sign*s[col])
for nm,col,sg in [('NLI contradiction probability','contradiction',1),('-ERS','ERS',-1),('-similarity (similarity-only)','similarity',-1)]:
    a=auc(col,sg)(r); lo,hi=boot(auc(col,sg),r); say(f'  AUROC for human CONTRADICTS, {nm}: {a:.3f} (95% CI {lo:.3f}-{hi:.3f})')
for T in [0.5]:
    p=(r.contradiction>=T).astype(int); tn,fp,fn,tp=confusion_matrix(r.h,p,labels=[0,1]).ravel()
    say(f'  NLI at C>={T}: sensitivity {tp/max(tp+fn,1):.3f}, specificity {tn/max(tn+fp,1):.3f}, kappa vs human {cohen_kappa_score(r.h,p):.3f} (TP {tp}, FP {fp}, FN {fn}, TN {tn})')
r['hs']=(r.gold=='SUPPORTS').astype(int)
if r.hs.nunique()==2:
    for nm,col,sg in [('ERS','ERS',1),('similarity','similarity',1),('-contradiction','contradiction',-1)]:
        say(f'  AUROC for human SUPPORTS, {nm}: {roc_auc_score(r.hs,sg*r[col]):.3f}')
r.to_csv(f'{out}/partA_gold_with_scores.csv',index=False); open(f'{out}/partA_analysis_report.txt','w').write('\n'.join(R)); print(f'\nSaved {out}/partA_analysis_report.txt')
