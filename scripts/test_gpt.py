import math

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.dataset import (
    ChessBehaviorDataset
)

from src.models.gpt_model import (
    GPTBehaviorModel
)

# ======================================================
# DEVICE
# ======================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDEVICE:", device)

# ======================================================
# CONFIG
# ======================================================

VOCAB_SIZE = 4511

NUM_PLAYERS = 3133

BATCH_SIZE = 64

# ======================================================
# DATASET
# ======================================================

test_dataset = ChessBehaviorDataset(

    "dataset/hdf5/test.h5"
)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)

print(
    "\nTEST SAMPLES:",
    len(test_dataset)
)

# ======================================================
# MODEL
# ======================================================

model = GPTBehaviorModel(

    vocab_size=VOCAB_SIZE,

    max_seq_len=63,

    embed_dim=256,

    num_heads=4,

    num_layers=4,

    dropout=0.1,

    num_players=NUM_PLAYERS
)

# ======================================================
# LOAD CHECKPOINT
# ======================================================

checkpoint_path = (
    "checkpoints/gpt_latest.pt"
)

model.load_state_dict(

    torch.load(

        checkpoint_path,

        map_location=device
    )
)

model = model.to(device)

model.eval()

print("\nMODEL LOADED")


criterion = nn.CrossEntropyLoss()


total_loss = 0.0

total_batches = 0


with torch.no_grad():

    for batch in test_loader:

        input_ids = batch[
            "input_ids"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        player_ids = batch[
            "player"
        ].to(device)


        logits = model(

            input_ids,

            player_ids
        )


        logits = logits.reshape(

            -1,

            logits.size(-1)
        )

        labels = labels.reshape(-1)


        loss = criterion(

            logits,

            labels
        )

        total_loss += loss.item()

        total_batches += 1


avg_loss = (

    total_loss
    /
    total_batches
)

perplexity = math.exp(
    avg_loss
)


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"\nTEST LOSS: "
    f"{avg_loss:.4f}"
)

print(
    f"PERPLEXITY: "
    f"{perplexity:.4f}"
)


test_dataset.close()