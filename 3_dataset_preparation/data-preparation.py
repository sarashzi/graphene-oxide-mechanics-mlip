from ase.io import read, write
import numpy as np
import os
import glob

# ========================================
# SETTINGS

data_dir = "."
file_pattern = "dataset_GO*.extxyz"

data_files = sorted(
    glob.glob(os.path.join(data_dir, file_pattern))
)

if not data_files:
    raise RuntimeError("No dataset_GO*.extxyz files found!")

print("Found the following dataset files:")
for f in data_files:
    print("  -", f)

output_train = "train.extxyz"
output_valid = "valid.extxyz"

# ========================================
# READ AND MERGE ALL STRUCTURES

all_structures = []
for f in data_files:
    print(f"Reading {f} ...")
    frames = read(f, ":")  # read all frames
    print(f"  → {len(frames)} structures")
    all_structures.extend(frames)

n_total = len(all_structures)
print(f"\nTotal structures combined: {n_total}")

# ========================================
# SPLIT INTO TRAIN/VALID (90/10)

np.random.seed(42)  # reproducibility
indices = np.arange(n_total)
np.random.shuffle(indices)

n_train = int(0.9 * n_total)
train_idx = indices[:n_train]
valid_idx = indices[n_train:]

train_structures = [all_structures[i] for i in train_idx]
valid_structures = [all_structures[i] for i in valid_idx]

print(f"Train: {len(train_structures)} | Valid: {len(valid_structures)}")

# ========================================
# WRITE OUTPUT FILES

write(output_train, train_structures)
write(output_valid, valid_structures)

print(f"Done! Wrote {output_train} and {output_valid}")
