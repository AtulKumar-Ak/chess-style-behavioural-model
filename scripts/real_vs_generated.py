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

PIECE_MAP = {
    chess.PAWN:   "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK:   "R",
    chess.QUEEN:  "Q",
    chess.KING:   "K",
}

# ======================================================
# DEVICE
# ======================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)

# ======================================================
# SETTINGS
# ======================================================

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

# All account name variants across chesscom / lichess / pgnmentor
PLAYER_ACCOUNTS = {
    "MagnusCarlsen": [
        "magnuscarlsen",    # chesscom
        "drnykterstein",    # lichess
        "carlsen,magnus",   # pgnmentor full
        "carlsen,m",        # pgnmentor short
    ],
    "Hikaru": [
        "hikaru",           # chesscom + lichess
        "nakamura,hikaru",  # pgnmentor full
        "nakamura,h",       # pgnmentor short
    ],
    "alireza2003": [
        "firouzja2003",     # chesscom
        "alireza2003",      # lichess
        "firouzja,alireza", # pgnmentor full
        "firouzja,a",       # pgnmentor short
    ],
    "lachesisQ": [
        "lachesisq",            # chesscom
        "nepo",                 # lichess
        "nepomniachtchi,ian",   # pgnmentor full
        "nepomniachtchi,i",     # pgnmentor short
    ],
}

# ======================================================
# LOAD VOCABS
# ======================================================

with open("dataset/vocab/move_vocab.json", encoding="utf-8") as f:
    move_vocab = json.load(f)

with open("dataset/vocab/player_vocab.json", encoding="utf-8") as f:
    player_vocab = json.load(f)

# ======================================================
# MODEL
# ======================================================

model = GPTBehaviorModel(
    vocab_size=len(move_vocab),
    max_seq_len=63,
    embed_dim=384,
    num_heads=6,
    num_layers=6,
    dropout=0.1,
    num_players=len(player_vocab)
)

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model = model.to(device)
model.eval()
print("MODEL LOADED")

# ======================================================
# STOCKFISH
# ======================================================

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# ======================================================
# HELPERS
# ======================================================

def get_player_side(metadata, target_player):
    white_lower = metadata["white"].lower()
    black_lower = metadata["black"].lower()
    for account in PLAYER_ACCOUNTS.get(target_player, []):
        if account in white_lower:
            return True
        if account in black_lower:
            return False
    return None


def analyze_moves(move_tokens, start_board):
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

        piece = board.piece_at(move.from_square)
        if piece is not None and piece.piece_type == chess.QUEEN:
            if board.is_capture(move):
                queen_trades += 1

        board.push(move)

        info  = engine.analyse(board, chess.engine.Limit(depth=10))
        score = info["score"].white()

        if score.is_mate():
            eval_cp = 10000
        else:
            eval_cp = score.score() or 0

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
    Load n game seeds from the test split.
    Seeds come from all sources including pgnmentor.
    Each seed contains the first SEED_CONTEXT_MOVES moves
    as context and the next MAX_NEW_MOVES moves as the
    real continuation to compare against.
    """
    aliases = PLAYER_FILE_ALIASES[player_name]
    seeds   = []

    for filename in sorted(os.listdir(TRAJECTORY_DIR)):
        if not filename.endswith(".jsonl"):
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

                # Convert raw UCI to piece-aware tokens
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
                })

        if len(seeds) >= n:
            break

    random.shuffle(seeds)
    return seeds[:n]


def generate_from_seed(seed_tokens, player_name):
    if player_name not in player_vocab:
        return None

    player_id     = player_vocab[player_name]
    player_tensor = torch.tensor(
        [player_id], dtype=torch.long, device=device
    )

    board, ok = build_board_from_tokens(seed_tokens)
    if not ok:
        return None

    try:
        input_ids = [move_vocab[m] for m in seed_tokens]
    except KeyError:
        return None

    generated_tokens = []

    with torch.no_grad():
        for _ in range(MAX_NEW_MOVES):

            context = input_ids[-63:]
            x       = torch.tensor([context], dtype=torch.long, device=device)
            logits  = model(x, player_tensor)
            logits  = logits[:, -1, :] / TEMPERATURE
            probs   = torch.softmax(logits, dim=-1)

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

    return generated_tokens

# ======================================================
# MAIN LOOP
# ======================================================

real_results      = defaultdict(list)
generated_results = defaultdict(list)

for player_name in PLAYERS:

    print(f"\n{'='*60}")
    print(f"PLAYER: {player_name}")
    print(f"{'='*60}")

    seeds = load_game_seeds(player_name, n=GENERATED_ROLLOUTS)
    print(f"  Loaded {len(seeds)} game seeds")

    if not seeds:
        print("  NO SEEDS FOUND — skipping")
        continue

    for i, seed in enumerate(seeds):

        seed_tokens      = seed["seed_tokens"]
        post_seed_tokens = seed["post_seed_tokens"]

        seed_board, ok = build_board_from_tokens(seed_tokens)
        if not ok:
            continue

        real_metrics = analyze_moves(post_seed_tokens, seed_board)
        if real_metrics is not None:
            real_results[player_name].append(real_metrics)

        gen_tokens = generate_from_seed(seed_tokens, player_name)
        if gen_tokens:
            gen_metrics = analyze_moves(gen_tokens, seed_board)
            if gen_metrics is not None:
                generated_results[player_name].append(gen_metrics)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(seeds)} done")

    print(f"  Real results     : {len(real_results[player_name])}")
    print(f"  Generated results: {len(generated_results[player_name])}")

# ======================================================
# CLEANUP
# ======================================================

engine.quit()

# ======================================================
# SUMMARIZE
# ======================================================

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

# ======================================================
# PRINT
# ======================================================

print("\n" + "=" * 60)
print("REAL vs GENERATED SUMMARY")
print(f"Seed context : first {SEED_CONTEXT_MOVES} moves of real game")
print(f"Evaluated on : next {MAX_NEW_MOVES} moves (same game phase)")
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

print(f"\nRESULTS SAVED TO: {output_csv}")
print("\nDONE.")