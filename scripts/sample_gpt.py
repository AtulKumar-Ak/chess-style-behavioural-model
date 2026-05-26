# scripts/sample_gpt.py

import json

import chess
import torch

from src.models.gpt_model import GPTBehaviorModel
from src.dataset import board_to_torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)


with open("dataset/vocab/move_vocab.json", encoding="utf-8") as f:
    move_vocab = json.load(f)

with open("dataset/vocab/player_vocab.json", encoding="utf-8") as f:
    player_vocab = json.load(f)

id_to_move  = {idx: move for move, idx in move_vocab.items()}
vocab_size  = len(move_vocab)
num_players = len(player_vocab)

print("VOCAB SIZE :", vocab_size)
print("NUM PLAYERS:", num_players)


model = GPTBehaviorModel(
    vocab_size=vocab_size,
    max_seq_len=63,
    embed_dim=384,
    num_heads=6,
    num_layers=6,
    dropout=0.1,
    num_players=num_players
)

checkpoint_path = "checkpoints/gpt_best.pt"

ckpt = torch.load(checkpoint_path, map_location=device)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model = model.to(device)
model.eval()
print("\nMODEL LOADED")


PLAYER_NAME   = "MagnusCarlsen"
MAX_NEW_MOVES = 40
TEMPERATURE   = 0.7
TOP_K         = 5


if PLAYER_NAME not in player_vocab:
    print(f"\nERROR: '{PLAYER_NAME}' not in player vocab.")
    print("Available:", [p for p in player_vocab if not p.startswith("<")])
    exit()

player_id     = player_vocab[PLAYER_NAME]
player_tensor = torch.tensor([player_id], dtype=torch.long, device=device)
print(f"\nPLAYER: {PLAYER_NAME}  (id={player_id})")


seed_moves = [
    "P_e2e4",
    "P_e7e5",
    "N_g1f3",
    "N_b8c6",
]


board = chess.Board()

for move_str in seed_moves:
    board.push(chess.Move.from_uci(move_str.split("_")[1]))

input_ids = [move_vocab[m] for m in seed_moves]


print("\nGENERATING...\n")

with torch.no_grad():

    for step in range(MAX_NEW_MOVES):

        context = input_ids[-63:]
        x       = torch.tensor([context], dtype=torch.long, device=device)


        board_state = board_to_torch(board).unsqueeze(0).to(device)
        # shape: (1, 768)

        logits = model(x, player_tensor, board_state=board_state)
        logits = logits[:, -1, :] / TEMPERATURE
        probs  = torch.softmax(logits, dim=-1)

        # Legal move filter
        legal_token_ids = []
        legal_moves     = []

        for legal_move in board.legal_moves:
            uci          = legal_move.uci()
            piece        = board.piece_at(legal_move.from_square)
            piece_symbol = piece.symbol().upper()
            token        = f"{piece_symbol}_{uci}"
            if token in move_vocab:
                legal_token_ids.append(move_vocab[token])
                legal_moves.append(token)

        if not legal_token_ids:
            print("(no legal moves in vocab — stopping)")
            break

        legal_probs            = probs[0, legal_token_ids]
        legal_probs            = legal_probs / legal_probs.sum()
        k                      = min(TOP_K, len(legal_probs))
        top_probs, top_indices = torch.topk(legal_probs, k=k)
        top_probs              = top_probs / top_probs.sum()

        sampled_rel   = torch.multinomial(top_probs, num_samples=1).item()
        sampled_idx   = top_indices[sampled_rel].item()
        next_token_id = legal_token_ids[sampled_idx]
        next_move     = legal_moves[sampled_idx]

        input_ids.append(next_token_id)
        board.push(chess.Move.from_uci(next_move.split("_")[1]))


generated_moves = [id_to_move[tid] for tid in input_ids]

print("GENERATED GAME\n")
for move in generated_moves:
    print(move)

print("\nFINAL BOARD\n")
print(board)