import pandas as pd
from pathlib import Path
import shutil

# Paths
base_path = Path(r"C:\Downloads\archive (6)")
images_path = base_path / "Images"
train_csv = base_path / "Train.csv"
test_csv = base_path / "SampleSubmission.csv"

# Output directories
train_path = base_path / "train"
test_path = base_path / "test"
train_0 = train_path / "0"
train_1 = train_path / "1"

# Create directories
train_0.mkdir(parents=True, exist_ok=True)
train_1.mkdir(parents=True, exist_ok=True)
test_path.mkdir(parents=True, exist_ok=True)

# Read CSVs
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

print(f"Train entries: {len(train_df)}")
print(f"Test entries: {len(test_df)}")

# Move train images
moved_train = 0
for _, row in train_df.iterrows():
    img_id = row['Image_id']
    label = row['Label']
    src = images_path / img_id
    if label == 0:
        dst = train_0 / img_id
    elif label == 1:
        dst = train_1 / img_id
    else:
        continue
    if src.exists():
        shutil.move(str(src), str(dst))
        moved_train += 1

# Move test images
moved_test = 0
for _, row in test_df.iterrows():
    img_id = row['Image_id']
    src = images_path / img_id
    dst = test_path / img_id
    if src.exists():
        shutil.move(str(src), str(dst))
        moved_test += 1

print(f"Moved train images: {moved_train}")
print(f"Moved test images: {moved_test}")
print(f"Remaining in Images: {len(list(images_path.iterdir()))}")
