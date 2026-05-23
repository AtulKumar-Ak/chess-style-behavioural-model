import json
import os
import statistics
from collections import defaultdict

import chess
import torch

from src.models.gpt_model import GPTBehaviorModel

# ======================================================
# SETTINGS
# ======================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "checkpoints/gpt_best.pt"
TEST_DIR        = "dataset/trajectories/test"

TOP_K     = 5
MAX_GAMES = 30

# Change to test any player
TARGET_PLAYER = "Hikaru"

# Test prediction accuracy at these move indices
TEST_AT_MOVES = [10, 20, 30, 40]

# ======================================================
# PLAYER FILE ALIASES
# ======================================================

PLAYER_FILE_ALIASES = {
    "MagnusCarlsen": ["magnus"],
    "Hikaru":        ["hikaru"],
    "alireza2003":   ["alireza"],
    "lachesisQ":     ["nepo"],
}

print(f"\nDEVICE      : {DEVICE}")
print(f"TARGET      : {TARGET_PLAYER}")

# ======================================================
# LOAD VOCABS
# ======================================================

with open("dataset/vocab/move_vocab.json", encoding="utf-8") as f:
    move_vocab = json.load(f)

with open("dataset/vocab/player_vocab.json", encoding="utf-8") as f:
    player_vocab = json.load(f)

print(f"VOCAB SIZE  : {len(move_vocab)}")
print(f"NUM PLAYERS : {len(player_vocab)}")

# ======================================================
# MODEL
# ======================================================

model = GPTBehaviorModel(
    vocab_size=len(move_vocab),
    num_players=len(player_vocab),
    max_seq_len=63,
    embed_dim=384,
    num_heads=6,
    num_layers=6,
    dropout=0.1
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()
print("MODEL LOADED")

# ======================================================
# PLAYER TENSOR
# ======================================================

if TARGET_PLAYER not in player_vocab:
    print(f"\nERROR: '{TARGET_PLAYER}' not in player_vocab.")
    print("Available players:")
    for p in sorted(player_vocab.keys()):
        if not p.startswith("<"):
            print(f"  {p}")
    exit()

player_id     = player_vocab[TARGET_PLAYER]
player_tensor = torch.tensor(
    [player_id], dtype=torch.long, device=DEVICE
)
print(f"PLAYER ID   : {player_id}")

# ======================================================
# PREDICTION FUNCTION
# ======================================================

def predict_at_position(token_moves, predict_at):
    """
    Feed token_moves[:predict_at] as context,
    predict the move at index predict_at.

    Returns (true_move, top_k_preds, rank) or None.
    rank is 1-indexed, or None if true_move not in top-k.
    """
    if len(token_moves) <= predict_at:
        return None

    context_tokens = token_moves[:predict_at]
    true_move      = token_moves[predict_at]

    # Replay board to get legal moves
    board = chess.Board()
    for token in context_tokens:
        try:
            uci  = token.split("_")[1]
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                return None
            board.push(move)
        except Exception:
            return None

    # Encode context
    input_ids = []
    for token in context_tokens:
        tid = move_vocab.get(token)
        if tid is None:
            return None
        input_ids.append(tid)

    if not input_ids:
        return None

    context = input_ids[-63:]
    x       = torch.tensor([context], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits = model(x, player_tensor)
        logits = logits[:, -1, :]
        probs  = torch.softmax(logits, dim=-1)

    # Legal move candidates
    legal_token_ids = []
    legal_moves     = []

    for legal_move in board.legal_moves:
        uci   = legal_move.uci()
        piece = board.piece_at(legal_move.from_square)
        if piece is None:
            continue
        token = f"{piece.symbol().upper()}_{uci}"
        if token in move_vocab:
            legal_token_ids.append(move_vocab[token])
            legal_moves.append(token)

    if not legal_token_ids:
        return None

    legal_probs        = probs[0, legal_token_ids]
    top_probs, top_idx = torch.topk(
        legal_probs, k=min(TOP_K, len(legal_probs))
    )
    predicted_moves = [legal_moves[i.item()] for i in top_idx]

    rank = None
    if true_move in predicted_moves:
        rank = predicted_moves.index(true_move) + 1

    return true_move, predicted_moves, rank

# ======================================================
# FIND TEST FILES
# ======================================================

target_files = []
aliases      = PLAYER_FILE_ALIASES[TARGET_PLAYER]

for filename in sorted(os.listdir(TEST_DIR)):
    if not filename.endswith(".jsonl"):
        continue
    # if "pgnmentor" in filename.lower():
    #     continue
    for alias in aliases:
        if alias in filename.lower():
            target_files.append(os.path.join(TEST_DIR, filename))
            break

print(f"\nFOUND FILES : {len(target_files)}")
for fp in target_files:
    print(f"  {fp}")

if not target_files:
    print("\nERROR: No test files found.")
    exit()

# ======================================================
# MAIN TEST LOOP
#
# Trajectory files store:
#   "player": canonical name (e.g. "MagnusCarlsen")
#   "moves":  piece-aware tokens
#
# The "player" field is already the label from
# build_trajectories.py — no metadata lookup needed.
# Windows labeled "opponent" are skipped since we only
# want to test prediction on target player moves.
#
# We can't determine is_white from trajectory files
# (metadata was stripped). Instead we determine it from
# move parity: in build_trajectories, a window at index i
# has predicted_move_idx = i + CONTEXT_SIZE - 1.
# But here we test at specific depths, so we check whether
# the move at that depth is consistent with the context —
# the board replay handles legality, and parity of the
# depth index tells us whose move it is globally.
# ======================================================

depth_metrics = defaultdict(lambda: {
    "top1": 0, "top5": 0, "total": 0, "ranks": []
})

games_processed = 0
games_skipped   = 0

for filepath in target_files:
    if games_processed >= MAX_GAMES:
        break

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if games_processed >= MAX_GAMES:
                break

            record = json.loads(line)
            moves  = record["moves"]

            # Only use windows labeled as target player
            # (opponent-labeled windows are skipped)
            if record.get("player") != TARGET_PLAYER:
                games_skipped += 1
                continue

            any_valid = False

            for depth in TEST_AT_MOVES:

                result = predict_at_position(moves, predict_at=depth)
                if result is None:
                    continue

                true_move, predicted, rank = result
                any_valid = True

                m = depth_metrics[depth]
                m["total"] += 1

                if predicted[0] == true_move:
                    m["top1"] += 1

                if true_move in predicted:
                    m["top5"] += 1
                    if rank:
                        m["ranks"].append(rank)

                # Print first 5 games in detail
                if games_processed < 5:
                    print(
                        f"\n[Game {games_processed+1:>2} | Move {depth:>2}]"
                    )
                    print(f"  True       : {true_move}")
                    print(f"  Top-{TOP_K} pred: {predicted}")
                    print(
                        f"  Rank       : "
                        f"{rank if rank else f'not in top-{TOP_K}'}"
                    )

            if any_valid:
                games_processed += 1
            else:
                games_skipped += 1

# ======================================================
# RESULTS
# ======================================================

print("\n" + "=" * 60)
print(f"MOVE PREDICTION ACCURACY — {TARGET_PLAYER}")
print("=" * 60)

print(f"\n{'Depth':<8} {'Top-1':>8} {'Top-5':>8} {'Avg rank':>10} {'n':>6}")
print("-" * 45)

all_top1  = []
all_top5  = []
all_ranks = []

for depth in TEST_AT_MOVES:
    m = depth_metrics[depth]
    if m["total"] == 0:
        print(f"{depth:<8} {'—':>8} {'—':>8} {'—':>10} {'0':>6}")
        continue

    t1 = m["top1"] / m["total"]
    t5 = m["top5"] / m["total"]
    ar = statistics.mean(m["ranks"]) if m["ranks"] else float("nan")

    all_top1.append(t1)
    all_top5.append(t5)
    all_ranks.extend(m["ranks"])

    print(f"{depth:<8} {t1:>8.3f} {t5:>8.3f} {ar:>10.2f} {m['total']:>6}")

print("-" * 45)
if all_top1:
    print(
        f"{'Overall':<8} "
        f"{sum(all_top1)/len(all_top1):>8.3f} "
        f"{sum(all_top5)/len(all_top5):>8.3f} "
        f"{statistics.mean(all_ranks) if all_ranks else float('nan'):>10.2f}"
    )

print(f"\nGames processed : {games_processed}")
print(f"Games skipped   : {games_skipped}  "
      f"(opponent-labeled windows or too short)")
print("\nDONE.")