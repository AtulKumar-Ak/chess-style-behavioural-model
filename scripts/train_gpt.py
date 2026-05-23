import json

import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from src.dataset import ChessBehaviorDataset
from src.models.gpt_model import GPTBehaviorModel

# ======================================================
# DEVICE
# ======================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDEVICE:", device)

# ======================================================
# CHECKPOINT DIRECTORY
# ======================================================

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/"
    "chess-style-model/checkpoints"
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# VOCABS
# ======================================================

with open("dataset/vocab/player_vocab.json", encoding="utf-8") as f:
    player_vocab = json.load(f)

num_players = len(player_vocab)
print("NUM PLAYERS:", num_players)

with open("dataset/vocab/move_vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

vocab_size = len(vocab)
print("VOCAB SIZE:", vocab_size)

# ======================================================
# DATASETS
# ======================================================

train_dataset = ChessBehaviorDataset("dataset/hdf5/train.h5")
val_dataset   = ChessBehaviorDataset("dataset/hdf5/val.h5")

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
    max_seq_len=63,
    embed_dim=384,
    num_heads=6,
    num_layers=6,
    dropout=0.1
)

model = model.to(device)

# ======================================================
# TRAINING CONFIG
# ======================================================

EPOCHS        = 15
LEARNING_RATE = 5e-5

# ======================================================
# LOSS, OPTIMIZER, SCHEDULER
# ======================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# Cosine decay from LEARNING_RATE down to eta_min.
# eta_min=1e-5 prevents LR from reaching near-zero
# by epoch 8, keeping updates meaningful in late epochs.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5
)

# ======================================================
# RESUME FROM CHECKPOINT
# ======================================================

start_epoch   = 0
best_val_loss = float("inf")

resume_path = CHECKPOINT_DIR / "gpt_resume.pt"

if resume_path.exists():

    print(f"\nRESUME CHECKPOINT FOUND: {resume_path}")

    checkpoint = torch.load(
        resume_path,
        map_location=device
    )

    # --------------------------------------------------
    # VOCAB SIZE GUARD
    # If vocab size changed (new dataset), the old
    # checkpoint is incompatible — skip and warn.
    # --------------------------------------------------

    ckpt_vocab_size = checkpoint.get("vocab_size", None)

    if ckpt_vocab_size is not None and ckpt_vocab_size != vocab_size:

        print(
            f"\nWARNING: checkpoint vocab size ({ckpt_vocab_size}) "
            f"!= current vocab size ({vocab_size}).\n"
            f"Checkpoint incompatible — starting fresh.\n"
            f"Delete gpt_resume.pt to suppress this warning.\n"
        )

    else:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch   = checkpoint["epoch"]
        best_val_loss = checkpoint.get(
            "best_val_loss", float("inf")
        )

        print(f"RESUMED FROM EPOCH  : {start_epoch}")
        print(f"BEST VAL LOSS SO FAR: {best_val_loss:.4f}")
        print(f"CURRENT LR          : {scheduler.get_last_lr()}")

else:

    print("\nNO RESUME CHECKPOINT — STARTING FRESH")

# ======================================================
# VALIDATION
# ======================================================

def evaluate(model, loader):

    model.eval()

    total_loss    = 0
    total_batches = 0

    with torch.no_grad():

        for batch in loader:

            input_ids  = batch["input_ids"].to(device)
            labels     = batch["labels"].to(device)
            player_ids = batch["player"].to(device)

            logits = model(input_ids, player_ids)

            logits = logits.reshape(-1, vocab_size)
            labels = labels.reshape(-1)

            loss = criterion(logits, labels)

            total_loss    += loss.item()
            total_batches += 1

    return total_loss / total_batches

# ======================================================
# TRAIN LOOP
# ======================================================

for epoch in range(start_epoch, EPOCHS):

    print("\n" + "=" * 70)
    print(f"EPOCH {epoch + 1} / {EPOCHS}")
    print(f"LR    {scheduler.get_last_lr()[0]:.2e}")
    print("=" * 70)

    model.train()

    total_loss = 0

    for step, batch in enumerate(train_loader):

        input_ids  = batch["input_ids"].to(device)
        labels     = batch["labels"].to(device)
        player_ids = batch["player"].to(device)

        logits = model(input_ids, player_ids)

        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            avg_loss = total_loss / (step + 1)
            print(f"STEP {step} | LOSS {avg_loss:.4f}")

    # ==================================================
    # SCHEDULER STEP
    # ==================================================

    scheduler.step()

    # ==================================================
    # VALIDATION
    # ==================================================

    val_loss = evaluate(model, val_loader)

    print(f"\nVALIDATION LOSS : {val_loss:.4f}")
    print(f"NEXT EPOCH LR   : {scheduler.get_last_lr()[0]:.2e}")

    # ==================================================
    # SAVE RESUME CHECKPOINT
    # vocab_size saved so future runs can detect
    # incompatible checkpoints automatically.
    # ==================================================

    torch.save(
        {
            "epoch":                epoch + 1,
            "vocab_size":           vocab_size,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss":             val_loss,
            "best_val_loss":        best_val_loss,
        },
        resume_path
    )

    print(f"\nRESUME CHECKPOINT SAVED : {resume_path}")

    # ==================================================
    # SAVE BEST MODEL
    # ==================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_path = CHECKPOINT_DIR / "gpt_best.pt"

        torch.save(model.state_dict(), best_path)

        print(f"BEST MODEL SAVED        : {best_path}")
        print(f"NEW BEST VAL LOSS       : {best_val_loss:.4f}")

    else:

        print(
            f"VAL LOSS DID NOT IMPROVE "
            f"(best: {best_val_loss:.4f})"
        )

# ======================================================

print("\nTRAINING COMPLETE")
print(f"BEST VAL LOSS: {best_val_loss:.4f}")