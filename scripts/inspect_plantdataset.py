import os
from pathlib import Path
root = Path(r'C:\Downloads\Plantdataset')
print('Exists', root.exists())
for p in sorted(root.iterdir()):
    print(p.name, 'dir' if p.is_dir() else 'file')

print('\nSample folder contents:')
for part in sorted(root.iterdir()):
    if part.is_dir():
        sub = list(part.iterdir())
        print(part.name, '->', len(sub), 'items')
        if len(sub)>0:
            print('   first:', sub[0].name)
        break
