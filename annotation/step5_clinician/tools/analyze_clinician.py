"""Usage: analyze_clinician.py <returned.xlsx> <private_dir> [outdir]
Pre-specified analysis (see PROTOCOL). Clinician is a blinded third rater on items already labelled by two non-clinicians."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from scipy.stats import beta
f,priv=sys.argv[1],sys.argv[2]; out=sys.argv[3] if len(sys.argv)>3 else '.'; rng=np.random.default_rng(42)
R=[]; 
def say(s=''): print(s); R.append(s)
def ac1(a,b):
    a=np.asarray(a); b=np.asarray(b); cats=sorted(set(a)|set(b)); q=len(cats)
    pa=(a==b).mean(); pi=np.array([((a==c).mean()+(b==c).mean())/2 for c in cats]); pe=(pi*(1-pi)).sum()/(q-1) if q>1 else 0
    return (pa-pe)/(1-pe) if pe<1 else np.nan
def boot(fn,df,n=2000):
    v=[]
    for _ in range(n):
        try: v.append(fn(df.iloc[rng.integers(0,len(df),len(df))]))
        except Exception: pass
    return np.percentile(v,[2.5,97.5])
def wil(k,n):
    return (beta.ppf(.025,k,n-k+1) if k else 0, beta.ppf(.975,k+1,n-k) if k<n else 1)
def agree(d,x,y,tag):
    k=cohen_kappa_score(d[x],d[y]); lo,hi=boot(lambda s:cohen_kappa_score(s[x],s[y]),d)
    say(f'  {tag}: n={len(d)}, raw agreement {(d[x]==d[y]).mean():.3f}, Cohen kappa {k:.3f} (95% CI {lo:.3f}-{hi:.3f}), Gwet AC1 {ac1(d[x],d[y]):.3f}')
for part,sheet,keyf,labs in (('A','PartA','ClinPartA_KEY.csv',['CONTRADICTS','SUPPORTS','NEITHER']),('B','PartB','ClinPartB_KEY.csv',['YES','NO','MIXED','NOT_RELEVANT'])):
    key=pd.read_csv(f'{priv}/{keyf}'); d=pd.read_excel(f,sheet_name=sheet,dtype=str).rename(columns={'item_id':'cid'})
    d['clin']=d.label.str.strip().str.upper().str.replace(' ','_'); d['clin_conf']=d.confidence.astype(float)
    cols=['cid','clin','clin_conf']+(['could_mislead'] if part=='A' else [])
    d=key.merge(d[cols],on='cid'); say(f'\n===== PART {part} =====')
    rep=d[d.repeat_of.notna()]; b=d[d.repeat_of.isna()].copy()
    m=rep.merge(b[['cid','clin']].rename(columns={'cid':'repeat_of','clin':'orig'}),on='repeat_of')
    if len(m): say(f'TEST-RETEST (clinician, n={len(m)} repeats): agreement {(m.clin==m.orig).mean():.3f}')
    say('CLINICIAN vs NON-CLINICIAN (base items)')
    agree(b,'clin','gold_noncl','clinician vs adjudicated non-clinician gold'); agree(b,'clin','ann1','clinician vs annotator 1'); agree(b,'clin','ann2','clinician vs annotator 2')
    say('  confusion (rows clinician, columns non-clinician gold):'); say(pd.crosstab(b.clin,b.gold_noncl).reindex(index=labs,columns=labs,fill_value=0).to_string())
    hi_=b[b.clin_conf>=2]; say(f'  excluding clinician confidence 1 (n={len(hi_)}): agreement {(hi_.clin==hi_.gold_noncl).mean():.3f}')
    if part=='A':
        say('CONSTRUCTION VALIDITY by clinician label (base items; Wilson 95% CI)')
        tgt={'CLEAN':'SUPPORTS','DRIFT':'NEITHER','NOISE':'NEITHER','CONTRADICTION':'CONTRADICTS'}
        for c,l in tgt.items():
            x=b[b.condition==c]; k=(x.clin==l).sum(); lo,hi=wil(k,len(x)); say(f'  {c}: {k}/{len(x)} = {k/len(x):.3f} labelled {l} (CI {lo:.3f}-{hi:.3f}); all: {x.clin.value_counts().to_dict()}')
        b['h']=(b.clin=='CONTRADICTS').astype(int)
        say('SYSTEM vs CLINICIAN (positive = clinician CONTRADICTS)')
        for nm,col,sg in (('NLI contradiction probability','contradiction',1),('-ERS','ERS',-1),('-similarity','similarity',-1)):
            fn=lambda s:roc_auc_score(s.h,sg*s[col]); lo,hi=boot(fn,b); say(f'  AUROC {nm}: {fn(b):.3f} (CI {lo:.3f}-{hi:.3f})')
        b['mis']=(b.could_mislead.str.upper()=='YES').astype(int)
        say(f'CLINICAL RISK: could_mislead YES in {b.mis.sum()}/{len(b)}; by condition: {b.groupby("condition").mis.mean().round(3).to_dict()}')
        for nm,col,sg in (('-ERS','ERS',-1),('NLI contradiction probability','contradiction',1),('-similarity','similarity',-1)):
            fn=lambda s:roc_auc_score(s.mis,sg*s[col])
            try: lo,hi=boot(fn,b); say(f'  AUROC for could_mislead, {nm}: {fn(b):.3f} (CI {lo:.3f}-{hi:.3f})')
            except Exception as e: say(f'  AUROC for could_mislead {nm}: n/a ({e})')
        for T in (0.0,0.2):
            fl=(b.ERS<T).astype(int); tp=((fl==1)&(b.mis==1)).sum(); say(f'  ERS<{T}: flags {fl.sum()}; precision for could_mislead {tp/max(fl.sum(),1):.3f}; recall {tp/max(b.mis.sum(),1):.3f}')
    else:
        b['stance_ok']=(b.clin.str.lower()==b.doc_stance.str.lower())
        say('STANCE FIDELITY of top-1 window vs the document stance (clinician): '+str(b.groupby('doc_role').stance_ok.mean().round(3).to_dict()))
        b['pos']=(b.doc_role=='correct').astype(int); fn=lambda s:roc_auc_score(s.pos,s.ERS_top1); lo,hi=boot(fn,b)
        say(f'  AUROC of top-1 ERS for correct vs incorrect document: {fn(b):.3f} (CI {lo:.3f}-{hi:.3f})')
        c=b[b.clin.isin(['YES','NO'])]; say(f'  windows carrying a clear YES/NO stance: {len(c)}/{len(b)}; by role: {c.doc_role.value_counts().to_dict()}')
open(f'{out}/clinician_analysis_report.txt','w').write('\n'.join(R))
