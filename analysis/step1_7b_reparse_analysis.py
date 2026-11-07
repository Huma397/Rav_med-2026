import json, re, pandas as pd, numpy as np, collections
from sklearn.metrics import f1_score
pd.set_option("display.width",220); pd.set_option("display.max_rows",100)
c=json.load(open("llm_cache_7b.json"))
lead=re.compile(r"^\W*(?:answer\s*:\s*)?\W*(yes|no|maybe)\b",re.I); rs=re.compile(r"reason\s*:\s*(.+)",re.I|re.S)
m={}
for v in c.values():
    a=lead.search(v.strip()); r=rs.search(v)
    if a and r: m[r.group(1).strip().split("\n")[0]]=a.group(1).lower()
R=pd.read_csv("step1_all_predictions.csv")
R["reason"]=R.reason.fillna("")
def fix(row):
    if row.abstained: return "maybe", True
    a=m.get(row.reason); return (a,True) if a else ("maybe",False)
tmp=R.apply(fix,axis=1); R["pred"]=[t[0] for t in tmp]; R["parsed"]=[t[1] for t in tmp]
R["correct"]=(R.pred==R.gold).astype(int)
print("parsed now:",round(R.parsed.mean(),4))
rng=np.random.default_rng(42); pubids=sorted(R.pubid.unique()); n=len(pubids)
M=R.pivot(index="pubid",columns="setting",values="correct").loc[pubids]; AB=R.pivot(index="pubid",columns="setting",values="abstained").loc[pubids]
bi=rng.integers(0,n,size=(2000,n))
def acc(col):
    v=M[col].values; b=v[bi].mean(1); return v.mean(),*np.percentile(b,[2.5,97.5])
def diff(a,b):
    d=(M[a]-M[b]).values; bs=d[bi].mean(1); return d.mean(),*np.percentile(bs,[2.5,97.5])
rows=[]
for s in M.columns:
    sub=R[R.setting==s].set_index("pubid").loc[pubids]; a,lo,hi=acc(s)
    rows.append(dict(setting=s,accuracy=a,ci_lo=lo,ci_hi=hi,macro_F1=f1_score(sub.gold,sub.pred,average="macro",labels=["yes","no","maybe"]),abstain_rate=AB[s].mean(),
        pct_yes=(sub.pred=="yes").mean(),pct_no=(sub.pred=="no").mean(),pct_maybe=(sub.pred=="maybe").mean()))
T=pd.DataFrame(rows).set_index("setting"); oc=["NO_RETRIEVAL","CLEAN"]+[x for x in T.index if x not in("NO_RETRIEVAL","CLEAN")]; T=T.loc[oc]
print(T.round(3)); T.to_csv("step1_7b_table_accuracy_reparsed.csv")
print("majority baseline:",round(R[R.setting=="CLEAN"].gold.value_counts(normalize=True).max(),3))
rows=[]
for nm in ["INJ_DRIFT","INJ_NOISE","INJ_CONTRADICTION","MIXED"]:
    rows.append(dict(contrast=f"{nm}: unfiltered - CLEAN (damage)",**dict(zip(["diff","lo","hi"],diff(nm,"CLEAN")))))
    for mode in ["proxy","oracle"]:
        f=f"{nm}+FILTER_{mode}_T0.2"
        rows.append(dict(contrast=f"{nm}: {mode} filter - unfiltered (recovery)",**dict(zip(["diff","lo","hi"],diff(f,nm)))))
        rows.append(dict(contrast=f"{nm}: {mode} filter - CLEAN (residual gap)",**dict(zip(["diff","lo","hi"],diff(f,"CLEAN")))))
D=pd.DataFrame(rows); print(D.round(3).to_string(index=False)); D.to_csv("step1_7b_table_paired_reparsed.csv",index=False)
st=[]
for s in ["CLEAN","MIXED","MIXED+FILTER_proxy_T0.2","MIXED+FILTER_oracle_T0.2"]:
    g=R[R.setting==s].groupby("gold").agg(n=("correct","size"),accuracy=("correct","mean"),abstain=("abstained","mean"),pred_yes=("pred",lambda x:(x=="yes").mean()),pred_no=("pred",lambda x:(x=="no").mean()))
    g.insert(0,"setting",s); st.append(g.reset_index())
S=pd.concat(st); print(S.round(3).to_string(index=False)); S.to_csv("step1_7b_table_by_gold_reparsed.csv",index=False)
R.to_csv("step1_7b_predictions_reparsed.csv",index=False)
# matched unsupported / contradicted (answer-independent)
def matched(a,b):
    A=R[R.setting==a].set_index("pubid"); B=R[R.setting==b].set_index("pubid")
    ok=A.index[(~A.abstained)&(~B.abstained)&A.unsupported.notna()&B.unsupported.notna()]; out={}
    for col in["unsupported","contradicted"]:
        d=(B.loc[ok,col]-A.loc[ok,col]).values; bs=d[rng.integers(0,len(d),(2000,len(d)))].mean(1)
        out[col]=(round(A.loc[ok,col].mean(),3),round(B.loc[ok,col].mean(),3),round(d.mean(),3),tuple(np.percentile(bs,[2.5,97.5]).round(3)))
    return len(ok),out
print("\nmatched-denominator unsupported/contradicted (before, after, diff, CI):")
for u in["INJ_DRIFT","INJ_NOISE","INJ_CONTRADICTION","MIXED"]:
    for md in["proxy","oracle"]: print(u,md,matched(u,f"{u}+FILTER_{md}_T0.2"))
print("CLEAN vs MIXED",matched("CLEAN","MIXED"))
