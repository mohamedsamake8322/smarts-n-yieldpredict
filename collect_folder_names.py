from pathlib import Path
root = Path(r'C:\Downloads\Plantdataset')
all_folders = []
for part in sorted(root.iterdir()):
    if part.is_dir() and part.name.startswith('part_'):
        for d in sorted(part.iterdir()):
            if d.is_dir():
                all_folders.append(d.name)

all_folders = sorted(set(all_folders))
print('Total unique classes', len(all_folders))
for n in all_folders:
    print(n)
