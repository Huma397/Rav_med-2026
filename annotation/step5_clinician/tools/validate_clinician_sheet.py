"""Usage: validate_clinician_sheet.py <returned.xlsx> [manifest_clinician.csv]   Exit 0 = PASS."""
import sys, os, hashlib, pandas as pd
f=sys.argv[1]; mp=sys.argv[2] if len(sys.argv)>2 else os.path.join(os.path.dirname(os.path.abspath(__file__)),'manifest_clinician.csv')
man=pd.read_csv(mp,dtype=str); err=[]; warn=[]
LAB={'A':{'CONTRADICTS','SUPPORTS','NEITHER'},'B':{'YES','NO','MIXED','NOT_RELEVANT'}}
x=pd.ExcelFile(f)
for part,qc in (('A','claim'),('B','question')):
    if 'Part'+part not in x.sheet_names: err.append('missing sheet Part'+part); continue
    d=x.parse('Part'+part,dtype=str).fillna(''); m=man[man.part==part]
    ids=list(d.item_id.str.strip())
    if set(ids)!=set(m.item_id) or len(ids)!=len(set(ids)): err.append(f'Part{part}: item_id set differs or duplicates')
    mh=dict(zip(m.item_id,m.sha256))
    for r in d.itertuples(index=False):
        i=r.item_id.strip(); rr=r._asdict()
        h=hashlib.sha256((rr[qc].strip()+'||'+rr['passage'].strip()).encode()).hexdigest()
        if mh.get(i)!=h: err.append(f'{i}: text altered or unknown item')
        lab=rr['label'].strip().upper().replace(' ','_')
        if lab not in LAB[part]: err.append(f'{i}: invalid/blank label "{rr["label"]}"')
        if rr['confidence'].strip().split('.')[0] not in {'1','2','3'}: err.append(f'{i}: confidence not 1/2/3')
        if part=='A' and rr['could_mislead'].strip().upper() not in {'YES','NO'}: err.append(f'{i}: could_mislead not YES/NO')
    vc=d.label.str.upper().value_counts(); print(f'Part{part} label counts:',dict(vc))
    if len(vc) and vc.iloc[0]/len(d)>0.9: warn.append(f'Part{part}: >90% one label')
    if ids!=list(m.item_id): warn.append(f'Part{part}: row order differs (harmless)')
p=x.parse('Annotator_profile',header=None,dtype=str).fillna('')
for i in range(1,len(p)):
    if 'optional' in p.iloc[i,0]: continue
    if p.iloc[i,1].strip()=='': err.append('profile blank: '+p.iloc[i,0][:40])
print(f'{len(err)} errors, {len(warn)} warnings')
for e in err[:40]: print('  ERROR',e)
for w in warn: print('  WARN ',w)
print('PASS' if not err else 'FAIL'); sys.exit(1 if err else 0)
