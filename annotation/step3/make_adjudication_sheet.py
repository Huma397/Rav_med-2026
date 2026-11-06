"""Usage: python3 make_adjudication_sheet.py A|B annotator1.xlsx annotator2.xlsx output.xlsx
Lists only the items on which the two annotators disagreed (repeats excluded), with text and both labels, and a final_label column to fill in together."""
import sys, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
part,f1,f2,outp=sys.argv[1:5]; sheet='Part'+part
a=pd.read_excel(f1,sheet_name=sheet); b=pd.read_excel(f2,sheet_name=sheet)
tcol=[c for c in a.columns if c in('claim','question')][0]
m=a[['item_id',tcol,'passage','label','confidence']].merge(b[['item_id','label','confidence']],on='item_id',suffixes=('_1','_2'))
m=m[m.label_1!=m.label_2].sort_values('item_id')
wb=Workbook(); ws=wb.active; ws.title='Adjudication'
cols=['item_id',tcol,'passage','label_annotator1','conf_1','label_annotator2','conf_2','final_label','note']; widths=[9,40,90,16,7,16,7,16,30]; ws.append(cols)
for i,(c,w) in enumerate(zip(cols,widths),1):
    x=ws.cell(1,i); x.font=Font(bold=True,color='FFFFFF'); x.fill=PatternFill('solid',fgColor='1F3864'); ws.column_dimensions[x.column_letter].width=w
labs='"CONTRADICTS,SUPPORTS,NEITHER"' if part=='A' else '"YES,NO,MIXED,NOT_RELEVANT"'
dv=DataValidation(type='list',formula1=labs,allow_blank=True); ws.add_data_validation(dv)
for r,row in enumerate(m.itertuples(index=False),2):
    ws.append([row[0],row[1],row[2],row[3],row[4],row[5],row[6],None,None]); ws.row_dimensions[r].height=min(400,max(30,len(str(row[2]))//95*15+20))
    for c in (2,3): ws.cell(r,c).alignment=Alignment(wrap_text=True,vertical='top')
    ws.cell(r,8).fill=PatternFill('solid',fgColor='FFF2CC')
dv.add(f'H2:H{len(m)+1}'); ws.freeze_panes='A2'; wb.save(outp)
print(f'{len(m)} disagreements written to {outp}. After you decide each one, save a CSV with columns item_id, final_label (the analysis script reads it).')
