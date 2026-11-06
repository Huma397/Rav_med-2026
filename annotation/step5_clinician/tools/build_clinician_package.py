"""Build the clinician labelling package (Step 5). Run once; outputs are fixed afterwards.
Reuses items already labelled by the two non-clinician annotators so the clinician is a blinded third rater.
Item IDs are re-generated (C001...) and shuffled, so the clinician cannot map items back to earlier sheets."""
import pandas as pd, numpy as np, hashlib, openpyxl
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Alignment, Font, PatternFill
S3='/mnt/user-data/outputs/step3_annotation/'; OUT='/mnt/user-data/outputs/step5_clinician/'
rng=np.random.default_rng(2026)
# ---- text sources (annotator 1 workbook has the same text as every other workbook)
tA=pd.read_excel(S3+'RAV_Med_annotation_PartA_annotator1.xlsx',sheet_name='PartA')[['item_id','claim','passage']]
tB=pd.read_excel(S3+'RAV_Med_annotation_PartB_annotator1.xlsx',sheet_name='PartB')[['item_id','question','passage']]
gA=pd.read_csv(S3+'returned_and_analysis/partA_ADJUDICATED_gold.csv'); gB=pd.read_csv(S3+'returned_and_analysis/partB_ADJUDICATED_gold.csv')
gA=gA[gA.repeat_of.isna()&(gA.stratum=='random')&gA.gold.notna()].merge(tA,on='item_id')
gB=gB[gB.repeat_of.isna()&gB.gold.notna()].merge(tB,on='item_id')
print('A pool',gA.condition.value_counts().to_dict(),'B pool',gB.doc_role.value_counts().to_dict())
# ---- sampling: Part A stratified by construction condition; Part B by document role
quota={'CLEAN':12,'CONTRADICTION':20,'DRIFT':14,'NOISE':14}
A=pd.concat([gA[gA.condition==c].sample(min(n,(gA.condition==c).sum()),random_state=2026) for c,n in quota.items()])
B=pd.concat([gB[gB.doc_role==r].sample(min(20,(gB.doc_role==r).sum()),random_state=2026) for r in ['correct','incorrect']])
A=A.sample(frac=1,random_state=1).reset_index(drop=True); B=B.sample(frac=1,random_state=2).reset_index(drop=True)
# repeats for test-retest (identical text, later position)
# simpler explicit construction (avoids the fragile mapping above)
def build(base,rep_src,k,prefix,seed):
    rows=base.to_dict('records'); 
    for r in rows: r['is_rep']=False; r['rep_of_orig']=None
    reps=rep_src.sample(k,random_state=seed).to_dict('records')
    for j,r in enumerate(reps):
        r=dict(r); r['is_rep']=True; r['rep_of_orig']=r['item_id']; rows.insert(int(rng.integers(len(base)//2+j,len(rows)+1)),r)
    df=pd.DataFrame(rows); df['cid']=[f'{prefix}{i+1:03d}' for i in range(len(df))]
    first={r.item_id:r.cid for r in df.itertuples() if not r.is_rep}
    df['repeat_of']=[first[o] if isinstance(o,str) else None for o in df.rep_of_orig]
    return df
A=build(A,A,4,'C',5)
B=build(B,B,3,'D',6)
print('rows A',len(A),'B',len(B))
# ---- private keys / manifest
keyA=A[['cid','item_id','repeat_of','condition','similarity','contradiction','ERS','label_1','label_2','gold']].rename(columns={'item_id':'step3_item_id','label_1':'ann1','label_2':'ann2','gold':'gold_noncl'})
keyB=B[['cid','item_id','repeat_of','topic_id','doc_role','evidence_answer','doc_stance','S_top1','C_top1','ERS_top1','label_1','label_2','gold']].rename(columns={'item_id':'step3_item_id','label_1':'ann1','label_2':'ann2','gold':'gold_noncl'})
keyA.to_csv(OUT+'private_do_not_share/ClinPartA_KEY.csv',index=False); keyB.to_csv(OUT+'private_do_not_share/ClinPartB_KEY.csv',index=False)
man=[]
for part,df,col in (('A',A,'claim'),('B',B,'question')):
    for r in df.itertuples():
        man.append(dict(part=part,item_id=r.cid,sha256=hashlib.sha256((str(getattr(r,col)).strip()+'||'+str(r.passage).strip()).encode()).hexdigest()))
pd.DataFrame(man).to_csv(OUT+'tools/manifest_clinician.csv',index=False)
# ---- workbook
HDR=PatternFill('solid',fgColor='DDE6F0'); wrap=Alignment(wrap_text=True,vertical='top')
wb=openpyxl.Workbook(); ws=wb.active; ws.title='Instructions'
nA,nB=len(A),len(B)
INS=[('RAV-Med clinician review','Thank you for helping. This is a blinded review of short medical texts. Your judgement as a clinician is the reference we compare an automatic system against.'),
('What you will do',f'Part A ({nA} items): for each CLAIM and PASSAGE, say how the passage relates to the claim. Part B ({nB} items): for each QUESTION and PASSAGE, say what answer the passage gives. Both parts ask for a label and a confidence (1 = unsure, 2 = fairly sure, 3 = very sure). Part A also asks one clinical question (see below). Comments are optional.'),
('Part A labels','CONTRADICTS: the passage says something that cannot be true if the claim is true, or states the opposite. SUPPORTS: the passage states the claim or clearly implies it. NEITHER: unrelated; related to the topic but not settling the claim; or mixed/inconclusive.'),
('Part A: could_mislead','After the relation label, answer YES or NO: "If a clinician or a patient relied on this passage as evidence about the claim, could it lead them to a wrong belief about the claim?" A passage that contradicts a TRUE claim, or supports a FALSE claim, would usually be YES. An irrelevant passage is usually NO. This uses your medical knowledge; the relation label does not.'),
('Part B labels','YES / NO: the passage indicates that the answer to the question is yes / no. MIXED: gives both answers or says the evidence is unclear. NOT_RELEVANT: does not address the question. Label what the PASSAGE says, not what you believe to be true.'),
('Rules','1) Part A relation label and Part B label judge the TEXT, not real-life truth. 2) Use only the text shown; do not search the web or use AI tools. 3) Work alone. 4) Do not change, sort or delete rows or the item_id column. 5) A few items repeat on purpose; label each row on its own merits. 6) Save often and send the file back with the same name and structure.'),
('Time',f'About 2 to 3 hours in total. Do the Practice sheets first (about 10 minutes) and compare with the Practice_key sheets.'),
('Before you start','Please complete the Annotator_profile sheet. It is only used to describe reviewers in the paper; you will not be named without your consent.')]
for i,(a,b) in enumerate(INS,1):
    ws.cell(i,1,a).font=Font(bold=True); ws.cell(i,2,b).alignment=wrap
ws.column_dimensions['A'].width=26; ws.column_dimensions['B'].width=120
ws=wb.create_sheet('Annotator_profile')
PROF=['Reviewer','Profession and highest qualification (e.g., MBBS, MD, PharmD, RN + MSc)','Years of clinical practice','Specialty / area (optional)','English: native / fluent / working','Have you read the RAV-Med paper or seen its results? (yes / no)','Consent to be named in the Acknowledgements? (yes / no)']
for i,p in enumerate(PROF,1): ws.cell(i,1,p).alignment=wrap
ws.cell(1,2,'Clinician reviewer'); ws.column_dimensions['A'].width=70; ws.column_dimensions['B'].width=50
# practice
def practice(name,rows,cols,keyrows):
    w=wb.create_sheet(name); w.append(cols)
    for r in rows: w.append(list(r)+[None]*(len(cols)-len(r)))
    w.column_dimensions['B'].width=50; w.column_dimensions['C'].width=70
    k=wb.create_sheet(name+'_key'); k.append(['id','expected label','why'])
    for r in keyrows: k.append(r)
    k.column_dimensions['C'].width=80
pa=[('P1','Aspirin reduces the risk of recurrent myocardial infarction in patients with prior MI.','In this randomized trial of 2,000 survivors of myocardial infarction, daily low-dose aspirin lowered reinfarction rates compared with placebo (relative risk 0.75).'),
('P2','Aspirin reduces the risk of recurrent myocardial infarction in patients with prior MI.','In this randomized trial of 2,000 survivors of myocardial infarction, aspirin did not reduce reinfarction rates compared with placebo (relative risk 1.01, no significant difference).'),
('P3','Aspirin reduces the risk of recurrent myocardial infarction in patients with prior MI.','Vitamin D deficiency is common in elderly people living at northern latitudes and is associated with reduced bone density.'),
('P4','Antibiotics cure viral infections.','Antibiotics are ineffective against viral infections.'),
('P5','Statin therapy reduces stroke risk.','Statins reduced stroke in some subgroups, but trials in elderly patients showed no benefit and the overall evidence remains inconclusive.')]
pka=[('P1','SUPPORTS / could_mislead NO','Results establish the claim, which is true.'),
('P2','CONTRADICTS / could_mislead YES','Opposite result for a claim that is true; relying on it would mislead.'),
('P3','NEITHER / could_mislead NO','Unrelated topic.'),
('P4','CONTRADICTS / could_mislead NO','The passage is accurate and the claim is false: the relation is CONTRADICTS, but nobody is misled by the passage.'),
('P5','NEITHER / could_mislead NO','Mixed message, no clear verdict.')]
w=wb.create_sheet('Practice_A'); w.append(['id','claim','passage','label','confidence','could_mislead'])
for r in pa: w.append(list(r)+[None]*3)
w.column_dimensions['B'].width=45; w.column_dimensions['C'].width=70
k=wb.create_sheet('Practice_A_key'); k.append(['id','expected','why'])
for r in pka: k.append(r)
k.column_dimensions['C'].width=80
pb=[('Q1','Does vitamin C cure the common cold?','Large trials show that vitamin C does not cure colds, and taking it after symptoms start does not shorten them.'),
('Q2','Is paracetamol safe in pregnancy?','Paracetamol is widely used in pregnancy, but some studies report associations with later childhood outcomes, and expert bodies disagree on how to interpret them.'),
('Q3','Is turmeric an effective treatment for arthritis?','Turmeric is a yellow spice commonly used in South Asian cooking and has been grown in India for centuries.'),
('Q4','Do statins lower LDL cholesterol?','Statin therapy reduced LDL cholesterol by 30 to 50 percent in the pooled analysis.')]
pkb=[('Q1','NO','The passage states it does not cure colds.'),('Q2','MIXED','Evidence presented as disputed.'),('Q3','NOT_RELEVANT','Same topic word, does not address efficacy.'),('Q4','YES','States it lowers LDL.')]
w=wb.create_sheet('Practice_B'); w.append(['id','question','passage','label','confidence'])
for r in pb: w.append(list(r)+[None]*2)
w.column_dimensions['B'].width=45; w.column_dimensions['C'].width=70
k=wb.create_sheet('Practice_B_key'); k.append(['id','expected','why'])
for r in pkb: k.append(r)
k.column_dimensions['C'].width=60
def mainsheet(name,df,qcol,cols,labs,extra):
    w=wb.create_sheet(name); w.append(cols)
    for c in w[1]: c.font=Font(bold=True); c.fill=HDR
    for r in df.itertuples(): w.append([r.cid,getattr(r,qcol),r.passage,None,None]+([None] if extra else [])+[None])
    n=len(df)+1; w.column_dimensions['A'].width=9; w.column_dimensions['B'].width=45; w.column_dimensions['C'].width=90
    for col in 'DEFG': w.column_dimensions[col].width=16
    w.column_dimensions['G' if extra else 'F'].width=40
    for row in w.iter_rows(min_row=2,max_row=n,max_col=3):
        for c in row: c.alignment=wrap
    w.freeze_panes='A2'
    dv=DataValidation(type='list',formula1='"'+labs+'"',allow_blank=True,showErrorMessage=True,error='Choose from the list'); dv.add(f'D2:D{n}'); w.add_data_validation(dv)
    dc=DataValidation(type='list',formula1='"1,2,3"',allow_blank=True,showErrorMessage=True); dc.add(f'E2:E{n}'); w.add_data_validation(dc)
    if extra:
        dm=DataValidation(type='list',formula1='"YES,NO"',allow_blank=True,showErrorMessage=True); dm.add(f'F2:F{n}'); w.add_data_validation(dm)
mainsheet('PartA',A,'claim',['item_id','claim','passage','label','confidence','could_mislead','comment'],'CONTRADICTS,SUPPORTS,NEITHER',True)
mainsheet('PartB',B,'question',['item_id','question','passage','label','confidence','comment'],'YES,NO,MIXED,NOT_RELEVANT',False)
wb.save(OUT+'for_clinician/RAV_Med_clinician_review.xlsx')
print('saved')
