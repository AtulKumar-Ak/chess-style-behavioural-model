from torch.utils.data import DataLoader

from src.dataset import (
    ChessBehaviorDataset
)

# ======================================================

dataset = ChessBehaviorDataset(

    "dataset/hdf5/train.h5"
)

# ======================================================

loader = DataLoader(

    dataset,

    batch_size=32,

    shuffle=True
)

# ======================================================

batch = next(iter(loader))

# ======================================================

print("\nCONTEXT")

print(batch["context"].shape)

# ------------------------------------------------------

print("\nTARGET")

print(batch["target"].shape)

# ------------------------------------------------------

print("\nPLAYER")

print(batch["player"].shape)

# ------------------------------------------------------

print("\nSPEED")

print(batch["speed"].shape)

# ------------------------------------------------------

print("\nELO")

print(batch["elo"].shape)

# ======================================================

dataset.close()