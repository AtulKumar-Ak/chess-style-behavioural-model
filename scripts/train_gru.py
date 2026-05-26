import json

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.dataset import (
    ChessBehaviorDataset
)

from src.models.gru_model import (
    GRUBehaviorModel
)


device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDEVICE:", device)


with open(
    "dataset/vocab/move_vocab.json",
    encoding="utf-8"
) as f:

    vocab = json.load(f)

vocab_size = len(vocab)

print("VOCAB SIZE:", vocab_size)


train_dataset = ChessBehaviorDataset(

    "dataset/hdf5/train.h5"
)

train_loader = DataLoader(

    train_dataset,

    batch_size=128,

    shuffle=True,

    num_workers=0
)

val_dataset = ChessBehaviorDataset(

    "dataset/hdf5/val.h5"
)

val_loader = DataLoader(

    val_dataset,

    batch_size=128,

    shuffle=False,

    num_workers=0
)


model = GRUBehaviorModel(

    vocab_size=vocab_size
)

model = model.to(device)


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-3
)


EPOCHS = 3


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

            logits = model(
                input_ids
            )

            logits = logits.reshape(
                -1,
                vocab_size
            )

            labels = labels.reshape(-1)

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

            total_batches += 1

    return total_loss / total_batches

for epoch in range(EPOCHS):


    print("\n" + "=" * 60)
    print(f"EPOCH {epoch+1}")
    print("=" * 60)

    model.train()

    total_loss = 0


    for step, batch in enumerate(train_loader):

        input_ids = batch[
            "input_ids"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)


        logits = model(
            input_ids
        )


        logits = logits.reshape(

            -1,
            vocab_size
        )

        labels = labels.reshape(-1)


        loss = criterion(

            logits,
            labels
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


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


    val_loss = evaluate(

        model,

        val_loader
    )

    print(

        f"\nVALIDATION LOSS: "
        f"{val_loss:.4f}"
    )

    torch.save(

        model.state_dict(),

        "checkpoints/gru_latest.pt"
    )


print("\nTRAINING COMPLETE")