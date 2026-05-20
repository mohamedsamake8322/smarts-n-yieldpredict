import os
from pathlib import Path
root=Path(r'C:\Downloads\Plantdataset')
part=root/'part_1'
for d in sorted(part.iterdir()):
    print(d.name, 'dir' if d.is_dir() else 'file')
    if d.is_dir():
        subfiles = list(d.iterdir())
        print('   samples', len(subfiles), 'example', subfiles[:3])
        break
