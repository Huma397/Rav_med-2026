"""Usage: validate_returned_sheet.py <A|B> <annotator 1|2> <returned.xlsx> [manifest.csv]
Exit code 0 = PASS, 1 = FAIL. Checks a completed workbook before analysis."""
import sys, hashlib, pandas as pd
part, ann, f = sys.argv[1], sys.argv[2], sys.argv[3]
man = pd.read_csv(sys.argv[4] if len(sys.argv)>4 else __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'manifest.csv'), dtype=str)
man = man[(man.part==part)&(man.annotator==ann)].reset_index(drop=True)
LAB = {'A':{'CONTRADICTS','SUPPORTS','NEITHER'}, 'B':{'YES','NO','MIXED','NOT_RELEVANT'}}[part]
err, warn = [], []
try:
    x = pd.ExcelFile(f)
except Exception as e:
    print('FAIL cannot open:', e); sys.exit(1)
need = {'Annotator_profile','Part'+part}
if not need <= set(x.sheet_names): err.append(f'missing sheets {need-set(x.sheet_names)}')
else:
    d = x.parse('Part'+part, dtype=str).fillna('')
    if list(d.columns[:6]) != list(pd.read_excel(f.replace(f,f),sheet_name='Part'+part,nrows=0).columns[:6]): pass
    if len(d)!=len(man): err.append(f'row count {len(d)} != expected {len(man)}')
    ids=list(d.iloc[:,0].str.strip())
    if len(set(ids))!=len(ids): err.append('duplicate item_ids: '+str([i for i in set(ids) if ids.count(i)>1]))
    if set(ids)!=set(man.item_id): err.append(f'item_id set differs: missing {sorted(set(man.item_id)-set(ids))[:5]} extra {sorted(set(ids)-set(man.item_id))[:5]}')
    elif ids!=list(man.item_id): warn.append('row ORDER differs from original (harmless if joined on item_id)')
    mh = dict(zip(man.item_id, man.sha256))
    for r in d.itertuples(index=False):
        h = hashlib.sha256((str(r[1]).strip()+'||'+str(r[2]).strip()).encode()).hexdigest()
        if r[0].strip() in mh and mh[r[0].strip()]!=h: err.append(f'{r[0]}: claim/passage text was ALTERED')
        lab=str(r[3]).strip().upper().replace(' ','_')
        if lab=='': err.append(f'{r[0]}: blank label')
        elif lab not in LAB: err.append(f'{r[0]}: invalid label "{r[3]}"')
        c=str(r[4]).strip()
        if c not in {'1','2','3','1.0','2.0','3.0'}: err.append(f'{r[0]}: confidence "{c}" not 1/2/3')
    lab = d.iloc[:,3].str.strip().str.upper().str.replace(' ','_')
    vc = lab.value_counts(); print('Label counts:',dict(vc))
    if len(vc) and vc.iloc[0]/len(d)>0.9: warn.append('>90% one label - confirm this is intended, not autofill')
    if d.iloc[:,3].nunique()==1: warn.append('single label used throughout')
    p = x.parse('Annotator_profile', header=None, dtype=str).fillna('')
    for i in range(1,len(p)):
        if 'optional' in str(p.iloc[i,0]): continue
        if str(p.iloc[i,1]).strip()=='' : err.append(f'profile blank: {p.iloc[i,0][:50]}')
print(f'\n{len(err)} errors, {len(warn)} warnings')
for e in err: print('  ERROR',e)
for w in warn: print('  WARN ',w)
print('PASS' if not err else 'FAIL'); sys.exit(1 if err else 0)
