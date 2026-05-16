import json

import torch
import torch.nn as nn
from pathlib import Path
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
# CHECKPOINT DIRECTORY
# ======================================================

CHECKPOINT_DIR = Path(

    "/content/drive/MyDrive/"
    "chess-style-model/checkpoints"
)

CHECKPOINT_DIR.mkdir(

    parents=True,

    exist_ok=True
)

# ======================================================
# PLAYER VOCAB
# ======================================================

with open(
    "dataset/vocab/player_vocab.json",
    encoding="utf-8"
) as f:

    player_vocab = json.load(f)

num_players = len(player_vocab)

print(
    "NUM PLAYERS:",
    num_players
)

# ======================================================
# LOAD VOCAB
# ======================================================

with open(
    "dataset/vocab/move_vocab.json",
    encoding="utf-8"
) as f:

    vocab = json.load(f)

vocab_size = len(vocab)

print(
    "VOCAB SIZE:",
    vocab_size
)

# ======================================================
# DATASETS
# ======================================================

train_dataset = ChessBehaviorDataset(

    "dataset/hdf5/train.h5"
)

val_dataset = ChessBehaviorDataset(

    "dataset/hdf5/val.h5"
)

# ======================================================
# DATALOADERS
# ======================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True,

    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(

    val_dataset,

    batch_size=32,

    shuffle=False,

    num_workers=2,
    pin_memory=True
)

# ======================================================
# MODEL
# ======================================================

model = GPTBehaviorModel(

    vocab_size=vocab_size,

    num_players=num_players,

    max_seq_len=127,

    embed_dim=256,

    num_heads=4,

    num_layers=4,

    dropout=0.1
)

model = model.to(device)

# ======================================================
# LOSS
# ======================================================

criterion = nn.CrossEntropyLoss()

# ======================================================
# OPTIMIZER
# ======================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=3e-4
)

# ======================================================
# VALIDATION
# ======================================================


def evaluate(model, loader):

    model.eval()

    total_loss = 0

    total_batches = 0

    with torch.no_grad():

        for batch in loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            player_ids = batch[
                "player"
            ].to(device)

            # ======================================
            # FORWARD
            # ======================================

            logits = model(
                input_ids,
                player_ids
            )

            # ======================================
            # RESHAPE
            # ======================================

            logits = logits.reshape(

                -1,
                vocab_size
            )

            labels = labels.reshape(-1)

            # ======================================

            loss = criterion(

                logits,
                labels
            )

            total_loss += loss.item()

            total_batches += 1

    return total_loss / total_batches


# ======================================================
# TRAIN LOOP
# ======================================================

EPOCHS = 5

# ======================================================

for epoch in range(EPOCHS):

    print("\n" + "=" * 70)
    print(f"EPOCH {epoch+1}")
    print("=" * 70)

    model.train()

    total_loss = 0

    # ==================================================

    for step, batch in enumerate(train_loader):

        input_ids = batch[
            "input_ids"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        player_ids = batch[
                "player"
            ].to(device)

        # ==============================================
        # FORWARD
        # ==============================================

        logits = model(
            input_ids,
            player_ids
        )

        # ==============================================
        # RESHAPE
        # ==============================================

        logits = logits.reshape(

            -1,
            vocab_size
        )

        labels = labels.reshape(-1)

        # ==============================================
        # LOSS
        # ==============================================

        loss = criterion(

            logits,
            labels
        )

        # ==============================================
        # BACKPROP
        # ==============================================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ==============================================

        total_loss += loss.item()

        # ==============================================

        if step % 100 == 0:

            avg_loss = (
                total_loss
                /
                (step + 1)
            )

            print(

                f"STEP {step} | "
                f"LOSS {avg_loss:.4f}"
            )

    # ==================================================
    # VALIDATION
    # ==================================================

    val_loss = evaluate(

        model,

        val_loader
    )

    print(

        f"\nVALIDATION LOSS: "
        f"{val_loss:.4f}"
    )

    # ==================================================
    # SAVE CHECKPOINT
    # ==================================================

    checkpoint_path = (

        CHECKPOINT_DIR /
        "gpt_latest.pt"
    )
    
    torch.save(
    
        model.state_dict(),
    
        checkpoint_path
    )
    
    print(
    
        f"\nCHECKPOINT SAVED:"
        f"\n{checkpoint_path}"
    )

# ======================================================

print("\nTRAINING COMPLETE")