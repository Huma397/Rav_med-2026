"""Build SHA-256 integrity manifest of the pristine annotator workbooks (run once, before sending)."""
import sys, hashlib, pandas as pd, glob, os
rows=[]
for f in sorted(glob.glob('out/RAV_Med_annotation_Part*_annotator*.xlsx')):
    part=os.path.basename(f).split('Part')[1][0]; ann=f.split('annotator')[1][0]
    sh='Part'+part; d=pd.read_excel(f,sheet_name=sh,dtype=str).fillna('')
    tcol=d.columns[1]
    for pos,r in enumerate(d.itertuples(index=False),1):
        txt=str(r[1]).strip()+'||'+str(r[2]).strip()
        rows.append(dict(part=part,annotator=ann,position=pos,item_id=r[0],
                         sha256=hashlib.sha256(txt.encode()).hexdigest()))
pd.DataFrame(rows).to_csv('out/manifest.csv',index=False); print(len(rows),'rows')
