import pandas as pd
xlsx = 'class_distribution (1).xlsx'
xl = pd.ExcelFile(xlsx)
print('Sheets:', xl.sheet_names)
s = pd.read_excel(xlsx, sheet_name=0)
print('Shape', s.shape)
print(s.head(30).to_string())
