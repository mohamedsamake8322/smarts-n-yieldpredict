import pandas as pd
from pathlib import Path

SOURCE_ROOT = Path(r'C:\Downloads\Plantdataset')
XLSX = SOURCE_ROOT / 'class_distribution (1).xlsx'

df = pd.read_excel(XLSX, sheet_name=0)
df.columns = [c.strip() for c in df.columns]
current_crop=None
rows=[]
for _, r in df.iterrows():
    crop = r.get('Crop') if not pd.isna(r.get('Crop')) else None
    primary = r.get('Primary Class') if not pd.isna(r.get('Primary Class')) else None
    secondary = r.get('Secondary Class Abbrv.') if not pd.isna(r.get('Secondary Class Abbrv.')) else None
    full = r.get('Secondary Class Full Form') if not pd.isna(r.get('Secondary Class Full Form')) else None
    if crop: current_crop=crop
    if primary: current_primary=primary
    if pd.isna(r.get('Primary Class')) and pd.isna(r.get('Secondary Class Abbrv.')):
        continue
    if current_crop and current_primary:
        if pd.isna(secondary) and str(current_primary).strip().lower() == 'healthy':
            secondary='healthy'
            full='healthy'
        if secondary: rows.append((str(current_crop).strip(), str(current_primary).strip(), str(secondary).strip().lower()))

class_names = set([f"{c.lower().replace(' ','_')}__{s}" for c,p,s in rows])
folder_names = set()
for part in sorted(SOURCE_ROOT.iterdir()):
    if part.is_dir() and part.name.startswith('part_'):
        for d in sorted(part.iterdir()):
            if d.is_dir(): folder_names.add(d.name)

print('class_names from xlsx:', len(class_names))
print('folder_names found:', len(folder_names))
missing = class_names - folder_names
extra = folder_names - class_names
print('missing:', missing)
print('extra:', extra)
