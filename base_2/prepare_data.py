import os
import shutil
import random

# ======================
# Paths
# ======================
COMOFOD_PATH = "/home/23ucc569/DataSources/CoMoFoD_small_v2"
OUTPUT_PATH = "./data"

random.seed(42)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# ======================
# Clean old data
# ======================
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

# ======================
# Create folders
# ======================
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT_PATH}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/{split}/masks", exist_ok=True)

# ======================
# Group by base image ID
# ======================
files = sorted(os.listdir(COMOFOD_PATH))
valid_ext = [".jpg", ".png", ".tif"]

groups = {}

for f in files:

    ext = os.path.splitext(f)[1].lower()
    if ext not in valid_ext:
        continue

    if "_F" in f:

        base = f.split("_F")[0]

        img_path = os.path.join(COMOFOD_PATH, f)
        mask_path = os.path.join(COMOFOD_PATH, base + "_M" + ext)

        if os.path.exists(mask_path):

            if base not in groups:
                groups[base] = []

            groups[base].append((img_path, mask_path))

# ======================
# Split by base image
# ======================
bases = list(groups.keys())
random.shuffle(bases)

n_total = len(bases)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

train_bases = bases[:n_train]
val_bases = bases[n_train:n_train + n_val]
test_bases = bases[n_train + n_val:]

# ======================
# Copy files
# ======================
def copy_split(base_list, split_name):
    count = 0
    for base in base_list:
        for img_path, mask_path in groups[base]:
            shutil.copy(img_path, f"{OUTPUT_PATH}/{split_name}/images/")
            shutil.copy(mask_path, f"{OUTPUT_PATH}/{split_name}/masks/")
            count += 1
    return count

train_count = copy_split(train_bases, "train")
val_count = copy_split(val_bases, "val")
test_count = copy_split(test_bases, "test")

print("Train samples:", train_count)
print("Val samples:", val_count)
print("Test samples:", test_count)

print("\nCoMoFoD Train/Val/Test split complete (Leak-free).")
