import json

import torch
import torch.nn.functional as F

from src.models.gru_model import (
    GRUBehaviorModel
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
# LOAD VOCAB
# ======================================================

with open(
    "dataset/vocab/move_vocab.json",
    encoding="utf-8"
) as f:

    vocab = json.load(f)

# ======================================================
# REVERSE VOCAB
# ======================================================

id_to_token = {

    idx: token
    for token, idx in vocab.items()
}

vocab_size = len(vocab)

print("VOCAB SIZE:", vocab_size)

# ======================================================
# MODEL
# ======================================================

model = GRUBehaviorModel(

    vocab_size=vocab_size
)

# ======================================================
# LOAD CHECKPOINT
# ======================================================

model.load_state_dict(

    torch.load(

        "checkpoints/gru_latest.pt",

        map_location=device
    )
)

model = model.to(device)

model.eval()

# ======================================================
# SAMPLE CONTEXT
# ======================================================

sample_moves = [

    "P_e2e4",
    "P_e7e5",
    "N_g1f3",
    "N_b8c6",
    "B_f1b5",
    "P_a7a6",
    "B_b5a4",
    "N_g8f6",
    "K_e1g1",
    "B_f8e7",
    "R_f1e1",
    "P_b7b5",
    "B_a4b3",
    "P_d7d6",
    "P_c2c3"
]

# ======================================================
# ENCODE
# ======================================================

input_ids = [

    vocab.get(
        token,
        vocab["<UNK>"]
    )

    for token in sample_moves
]

input_tensor = torch.tensor(

    [input_ids],

    dtype=torch.long
).to(device)

# ======================================================
# GENERATE
# ======================================================

GENERATE_STEPS = 10

# ======================================================

generated = []

# ======================================================

with torch.no_grad():

    current_input = input_tensor

    for _ in range(GENERATE_STEPS):

        logits = model(
            current_input
        )

        # --------------------------------------
        # LAST POSITION
        # --------------------------------------

        next_logits = logits[
            0, -1
        ]

        # --------------------------------------
        # GREEDY
        # --------------------------------------

        next_id = torch.argmax(
            next_logits
        ).item()

        generated.append(next_id)

        # --------------------------------------
        # APPEND
        # --------------------------------------

        next_tensor = torch.tensor(

            [[next_id]],

            dtype=torch.long
        ).to(device)

        current_input = torch.cat(

            [
                current_input,
                next_tensor
            ],

            dim=1
        )

# ======================================================
# DECODE
# ======================================================

print("\nGENERATED MOVES\n")

for idx in generated:

    token = id_to_token[idx]

    print(token)