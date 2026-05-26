import csv
import json
import os
import random
import statistics
from collections import defaultdict

import chess
import chess.engine
import torch

from src.models.gpt_model import GPTBehaviorModel
from src.dataset import board_to_torch

PIECE_MAP = {
    chess.PAWN:   "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK:   "R",
    chess.QUEEN:  "Q",
    chess.KING:   "K",
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)


STOCKFISH_PATH          = r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"
CHECKPOINT_PATH         = "checkpoints/gpt_best.pt"
TRAJECTORY_DIR          = "dataset/splits/test"

GENERATED_ROLLOUTS      = 80
SEED_CONTEXT_MOVES      = 20
MAX_NEW_MOVES           = 40
TEMPERATURE             = 0.5
TOP_K                   = 5
OUTLIER_SWING_THRESHOLD = 1000

PLAYERS = [
    "MagnusCarlsen",
    "Hikaru",
    "alireza2003",
    "lachesisQ"
]

PLAYER_FILE_ALIASES = {
    "MagnusCarlsen": ["magnus"],
    "Hikaru":        ["hikaru"],
    "alireza2003":   ["alireza"],
    "lachesisQ":     ["nepo"],
}

PLAYER_ACCOUNTS = {
    "MagnusCarlsen": [
        "magnuscarlsen",
        "drnykterstein",
    ],
    "Hikaru": [
        "hikaru",
    ],
    "alireza2003": [
        "firouzja2003",
        "alireza2003",
    ],
    "lachesisQ": [
        "lachesisq",
        "nepo",
    ],
}


with open("dataset/vocab/move_vocab.json", encoding="utf-8") as f:
    move_vocab = json.load(f)

with open("dataset/vocab/player_vocab.json", encoding="utf-8") as f:
    player_vocab = json.load(f)


ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
state_dict = (
    ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt
    else ckpt
)

ckpt_vocab_size  = state_dict["token_embedding.weight"].shape[0]
ckpt_num_players = state_dict["player_embedding.weight"].shape[0]

if ckpt_vocab_size != len(move_vocab):
    print(
        f"\nNOTE: checkpoint vocab ({ckpt_vocab_size}) != "
        f"move_vocab.json ({len(move_vocab)}). Using checkpoint size.\n"
    )

model = GPTBehaviorModel(
    vocab_size=ckpt_vocab_size,
    max_seq_len=63,
    embed_dim=384,
    num_heads=6,
    num_layers=6,
    dropout=0.1,
    num_players=ckpt_num_players
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(f"MODEL LOADED  (vocab={ckpt_vocab_size}, players={ckpt_num_players})")


engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)


def get_player_tensor(name):
    pid = player_vocab.get(name, player_vocab.get("opponent", 0))
    pid = min(pid, ckpt_num_players - 1)
    return torch.tensor([pid], dtype=torch.long, device=device)

opponent_tensor = get_player_tensor("opponent")


def get_player_side(metadata, target_player):
    white_lower = metadata.get("white", "").lower()
    black_lower = metadata.get("black", "").lower()
    for account in PLAYER_ACCOUNTS.get(target_player, []):
        if account in white_lower:
            return True
        if account in black_lower:
            return False
    return None


def analyze_moves(move_tokens, start_board):
    """
    Analyzes a sequence of piece-aware tokens from start_board.

    Queen trade definition: a move that CAPTURES a queen.
    This counts both sides — if white captures black's queen
    or black captures white's queen, it counts as a queen trade.
    This is consistent and symmetric.
    """
    board        = start_board.copy()
    evaluations  = []
    captures     = 0
    queen_trades = 0

    for token in move_tokens:
        try:
            uci  = token.split("_")[1]
            move = chess.Move.from_uci(uci)
        except Exception:
            continue

        if move not in board.legal_moves:
            break

        if board.is_capture(move):
            captures += 1

            # Count captures OF a queen (symmetric definition)
            captured_piece = board.piece_at(move.to_square)
            if (captured_piece is not None and
                    captured_piece.piece_type == chess.QUEEN):
                queen_trades += 1

        board.push(move)

        info    = engine.analyse(board, chess.engine.Limit(depth=10))
        score   = info["score"].white()
        eval_cp = 10000 if score.is_mate() else (score.score() or 0)
        evaluations.append(eval_cp)

    if len(evaluations) < 2:
        return None

    swings    = [abs(evaluations[i] - evaluations[i-1])
                 for i in range(1, len(evaluations))]
    max_swing = max(swings)

    return {
        "avg_eval":     statistics.mean(evaluations),
        "volatility":   statistics.stdev(evaluations),
        "captures":     captures,
        "queen_trades": queen_trades,
        "max_swing":    max_swing,
        "outlier":      max_swing > OUTLIER_SWING_THRESHOLD,
    }


def build_board_from_tokens(move_tokens):
    board = chess.Board()
    for token in move_tokens:
        try:
            uci  = token.split("_")[1]
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                return None, False
            board.push(move)
        except Exception:
            return None, False
    return board, True


def load_game_seeds(player_name, n):
    """
    Loads n seeds from test split.
    Stores is_white so generation can alternate embeddings.
    """
    aliases = PLAYER_FILE_ALIASES[player_name]
    seeds   = []

    for filename in sorted(os.listdir(TRAJECTORY_DIR)):
        if not filename.endswith(".jsonl"):
            continue
        if "pgnmentor" in filename.lower():
            continue
        if not any(a in filename.lower() for a in aliases):
            continue

        with open(
            os.path.join(TRAJECTORY_DIR, filename),
            encoding="utf-8"
        ) as f:
            for line in f:
                if len(seeds) >= n:
                    break

                record    = json.loads(line)
                metadata  = record.get("metadata", {})
                raw_moves = record.get("moves", [])

                is_white = get_player_side(metadata, player_name)
                if is_white is None:
                    continue

                board       = chess.Board()
                token_moves = []
                valid       = True

                for uci in raw_moves:
                    try:
                        move  = chess.Move.from_uci(uci)
                        piece = board.piece_at(move.from_square)
                        if piece is None:
                            valid = False
                            break
                        token = f"{PIECE_MAP[piece.piece_type]}_{uci}"
                        token_moves.append(token)
                        board.push(move)
                    except Exception:
                        valid = False
                        break

                if not valid or len(token_moves) < SEED_CONTEXT_MOVES + 10:
                    continue

                seeds.append({
                    "seed_tokens":      token_moves[:SEED_CONTEXT_MOVES],
                    "post_seed_tokens": token_moves[
                        SEED_CONTEXT_MOVES:
                        SEED_CONTEXT_MOVES + MAX_NEW_MOVES
                    ],
                    "is_white": is_white,
                })

        if len(seeds) >= n:
            break

    random.shuffle(seeds)
    return seeds[:n]


def generate_from_seed(seed_tokens, player_name, is_white):
    """
    Generates MAX_NEW_MOVES moves from the seed position.

    Alternates player embeddings per turn:
      target player turns  → target player embedding
      opponent turns       → opponent embedding

    This produces a realistic two-player game rather than
    one style playing against itself.

    is_white: True  = target player plays white
              False = target player plays black
    """
    if player_name not in player_vocab:
        return None

    player_id = player_vocab[player_name]
    if player_id >= ckpt_num_players:
        print(f"  WARNING: {player_name} id={player_id} exceeds checkpoint")
        return None

    target_tensor = get_player_tensor(player_name)

    board, ok = build_board_from_tokens(seed_tokens)
    if not ok:
        return None

    try:
        input_ids = [move_vocab[m] for m in seed_tokens]
    except KeyError:
        return None

    generated_tokens = []

    # First generated move is at index SEED_CONTEXT_MOVES
    # White moves at even indices (0,2,4,...), black at odd
    move_index = SEED_CONTEXT_MOVES

    with torch.no_grad():
        for _ in range(MAX_NEW_MOVES):

            # Alternate embedding based on whose turn it is
            white_to_move = (move_index % 2 == 0)
            current_tensor = (
                target_tensor
                if white_to_move == is_white
                else opponent_tensor
            )

            context = input_ids[-63:]
            x       = torch.tensor(
                [context], dtype=torch.long, device=device
            )

            # Board state conditioning
            board_state = board_to_torch(board).unsqueeze(0).to(device)

            logits = model(x, current_tensor, board_state=board_state)
            logits = logits[:, -1, :] / TEMPERATURE
            probs  = torch.softmax(logits, dim=-1)

            legal_token_ids = []
            legal_moves     = []

            for legal_move in board.legal_moves:
                uci          = legal_move.uci()
                piece        = board.piece_at(legal_move.from_square)
                piece_symbol = piece.symbol().upper()
                token        = f"{piece_symbol}_{uci}"
                if token in move_vocab and move_vocab[token] < ckpt_vocab_size:
                    legal_token_ids.append(move_vocab[token])
                    legal_moves.append(token)

            if not legal_token_ids:
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
            generated_tokens.append(next_move)
            board.push(chess.Move.from_uci(next_move.split("_")[1]))
            move_index += 1

    return generated_tokens


real_results      = defaultdict(list)
generated_results = defaultdict(list)

for player_name in PLAYERS:

    print(f"\n{'='*60}")
    print(f"PLAYER: {player_name}")
    print(f"{'='*60}")

    seeds = load_game_seeds(player_name, n=GENERATED_ROLLOUTS)
    print(f"  Seeds loaded: {len(seeds)}")

    if not seeds:
        print("  NO SEEDS FOUND — skipping")
        continue

    for i, seed in enumerate(seeds):

        seed_tokens      = seed["seed_tokens"]
        post_seed_tokens = seed["post_seed_tokens"]
        is_white         = seed["is_white"]

        seed_board, ok = build_board_from_tokens(seed_tokens)
        if not ok:
            continue

        real_metrics = analyze_moves(post_seed_tokens, seed_board)
        if real_metrics is not None:
            real_results[player_name].append(real_metrics)

        gen_tokens = generate_from_seed(
            seed_tokens, player_name, is_white
        )
        if gen_tokens:
            gen_metrics = analyze_moves(gen_tokens, seed_board)
            if gen_metrics is not None:
                generated_results[player_name].append(gen_metrics)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(seeds)} done")

    print(f"  Real      : {len(real_results[player_name])}")
    print(f"  Generated : {len(generated_results[player_name])}")


engine.quit()


def summarize(records, label):
    if not records:
        return {}
    clean = [r for r in records if not r["outlier"]]

    def m(lst, key):
        vals = [x[key] for x in lst]
        return round(statistics.mean(vals), 2) if vals else None

    return {
        "type":              label,
        "n_total":           len(records),
        "n_clean":           len(clean),
        "mean_eval_all":     m(records, "avg_eval"),
        "mean_eval_clean":   m(clean,   "avg_eval"),
        "mean_vol_all":      m(records, "volatility"),
        "mean_vol_clean":    m(clean,   "volatility"),
        "mean_captures":     m(records, "captures"),
        "mean_queen_trades": m(records, "queen_trades"),
    }

rows = []
for player in PLAYERS:
    rs = summarize(real_results[player],      "real")
    gs = summarize(generated_results[player], "generated")
    if rs:
        rows.append({"player": player, **rs})
    if gs:
        rows.append({"player": player, **gs})

fieldnames = [
    "player", "type", "n_total", "n_clean",
    "mean_eval_all", "mean_eval_clean",
    "mean_vol_all",  "mean_vol_clean",
    "mean_captures", "mean_queen_trades",
]

output_csv = "real_vs_generated.csv"
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 60)
print("REAL vs GENERATED SUMMARY")
print(f"Seed       : first {SEED_CONTEXT_MOVES} moves of real game")
print(f"Eval       : next {MAX_NEW_MOVES} moves (same game phase)")
print(f"Generation : target embedding on target turns,")
print(f"             opponent embedding on opponent turns")
print("Queen trade: any capture of a queen (symmetric)")
print("=" * 60)

for player in PLAYERS:
    print(f"\n{player}")
    for row in rows:
        if row["player"] != player:
            continue
        print(
            f"  [{row['type']:>10}]  "
            f"n={row['n_total']} (clean={row['n_clean']})  "
            f"eval(clean)={row['mean_eval_clean']:>8}  "
            f"vol(clean)={row['mean_vol_clean']:>8}  "
            f"captures={row['mean_captures']:>5}  "
            f"queen_trades={row['mean_queen_trades']:>4}"
        )

print(f"\nSAVED: {output_csv}")
print("\nDONE.")