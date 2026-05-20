from pathlib import Path
r=Path(r'C:\Downloads\Plantdataset_organized')
print('exists', r.exists())
tot=0
cls=0
for crop in sorted(r.iterdir()):
    if crop.is_dir():
        for primary in sorted(crop.iterdir()):
            for secund in sorted(primary.iterdir()):
                img_count=sum(1 for f in secund.iterdir() if f.is_file())
                if img_count>0:
                    cls+=1
                    tot+=img_count
print('classes dirs with images', cls, 'total images', tot)
for crop in sorted(r.iterdir()):
    if crop.is_dir():
        count_full = 0
        for primary in sorted(crop.iterdir()):
            for secund in sorted(primary.iterdir()):
                count_full += sum(1 for f in secund.iterdir() if f.is_file())
        print(crop.name, count_full)
