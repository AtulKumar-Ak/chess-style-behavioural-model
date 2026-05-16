import json

import chess

import torch

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
# LOAD MOVE VOCAB
# ======================================================

with open(
    "dataset/vocab/move_vocab.json",
    encoding="utf-8"
) as f:

    move_vocab = json.load(f)

# ======================================================
# LOAD PLAYER VOCAB
# ======================================================

with open(
    "dataset/vocab/player_vocab.json",
    encoding="utf-8"
) as f:

    player_vocab = json.load(f)

# ======================================================
# INVERSE VOCAB
# ======================================================

id_to_move = {

    idx: move

    for move, idx in move_vocab.items()
}

# ======================================================

vocab_size = len(move_vocab)

num_players = len(player_vocab)

print(
    "VOCAB SIZE:",
    vocab_size
)

print(
    "NUM PLAYERS:",
    num_players
)

# ======================================================
# MODEL
# ======================================================

model = GPTBehaviorModel(

    vocab_size=vocab_size,

    max_seq_len=63,

    embed_dim=256,

    num_heads=4,

    num_layers=4,

    dropout=0.1,

    num_players=num_players
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

# ======================================================
# CHOOSE PLAYER
# ======================================================

PLAYER_NAME = "alireza2003"

player_id = player_vocab[
    PLAYER_NAME
]

print(
    f"\nPLAYER: {PLAYER_NAME}"
)

# ======================================================
# PLAYER TENSOR
# ======================================================

player_tensor = torch.tensor(

    [player_id],

    dtype=torch.long,

    device=device
)

# ======================================================
# SEED MOVES
# ======================================================

seed_moves = [

    "P_e2e4",
    "P_e7e5",
    "N_g1f3",
    "N_b8c6"
]

# ======================================================
# CHESS BOARD
# ======================================================

board = chess.Board()

# ======================================================
# APPLY SEED MOVES
# ======================================================

for move_str in seed_moves:

    uci = move_str.split("_")[1]

    move = chess.Move.from_uci(uci)

    board.push(move)

# ======================================================
# ENCODE
# ======================================================

input_ids = [

    move_vocab[move]

    for move in seed_moves
]

# ======================================================
# SETTINGS
# ======================================================

MAX_NEW_MOVES = 40

TEMPERATURE = 0.8

# ======================================================
# GENERATION
# ======================================================

with torch.no_grad():

    for _ in range(MAX_NEW_MOVES):

        # ==============================================
        # CONTEXT WINDOW
        # ==============================================

        context = input_ids[-63:]

        x = torch.tensor(

            [context],

            dtype=torch.long,

            device=device
        )

        # ==============================================
        # FORWARD
        # ==============================================

        logits = model(

            x,

            player_tensor
        )

        logits = logits[:, -1, :]

        logits = logits / TEMPERATURE

        probs = torch.softmax(

            logits,

            dim=-1
        )

        # ==============================================
        # LEGAL MOVES
        # ==============================================

        legal_token_ids = []

        legal_moves = []

        for legal_move in board.legal_moves:

            uci = legal_move.uci()

            piece = board.piece_at(
                legal_move.from_square
            )

            piece_symbol = piece.symbol().upper()

            token = f"{piece_symbol}_{uci}"

            if token in move_vocab:

                token_id = move_vocab[token]

                legal_token_ids.append(
                    token_id
                )

                legal_moves.append(token)

        # ==============================================
        # FILTER LEGAL PROBS
        # ==============================================

        legal_probs = probs[
            0,
            legal_token_ids
        ]

        legal_probs = (
            legal_probs
            /
            legal_probs.sum()
        )

        TOP_K = 10

        # ==============================================
        # TOP-K FILTER
        # ==============================================

        k = min(
            TOP_K,
            len(legal_probs)
        )

        top_probs, top_indices = torch.topk(
        
            legal_probs,

            k=k
        )

        top_probs = top_probs / top_probs.sum()

        # ==============================================
        # SAMPLE
        # ==============================================

        sampled_relative_idx = torch.multinomial(

            top_probs,

            num_samples=1
        ).item()

        sampled_idx = top_indices[
            sampled_relative_idx
        ].item()

        next_token_id = legal_token_ids[
            sampled_idx
        ]

        next_move = legal_moves[
            sampled_idx
        ]

        # ==============================================
        # APPEND
        # ==============================================

        input_ids.append(
            next_token_id
        )

        # ==============================================
        # UPDATE BOARD
        # ==============================================

        uci = next_move.split("_")[1]

        board.push(

            chess.Move.from_uci(uci)
        )

# ======================================================
# DECODE
# ======================================================

generated_moves = [

    id_to_move[token_id]

    for token_id in input_ids
]

# ======================================================
# PRINT
# ======================================================

print("\nGENERATED GAME\n")

for move in generated_moves:

    print(move)

# ======================================================
# FINAL BOARD
# ======================================================

print("\nFINAL BOARD\n")

print(board)