import csv
import json
import statistics

import chess
import chess.engine

import torch

from src.models.gpt_model import GPTBehaviorModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDEVICE:", device)


STOCKFISH_PATH = (
    r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"
)


ROLLOUTS_PER_PLAYER = 80
MAX_NEW_MOVES = 40
TEMPERATURE = 0.7
TOP_K = 10
OUTLIER_SWING_THRESHOLD = 1000


with open("dataset/vocab/move_vocab.json", encoding="utf-8") as f:
    move_vocab = json.load(f)

with open("dataset/vocab/player_vocab.json", encoding="utf-8") as f:
    player_vocab = json.load(f)

id_to_move = {idx: move for move, idx in move_vocab.items()}

vocab_size  = len(move_vocab)
num_players = len(player_vocab)


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

model.load_state_dict(
    torch.load(checkpoint_path, map_location=device)
)

model = model.to(device)
model.eval()

print("\nMODEL LOADED")


PLAYERS = [
    "MagnusCarlsen",
    "Hikaru",
    "alireza2003",
    "lachesisQ"
]


seed_moves = [
    "P_e2e4",
    "P_e7e5",
    "N_g1f3",
    "N_b8c6"
]


csv_path     = "behavioral_results.csv"
summary_path = "behavioral_summary.csv"

csv_file = open(csv_path, "w", newline="", encoding="utf-8")
writer   = csv.writer(csv_file)
writer.writerow([
    "player",
    "rollout",
    "avg_eval",
    "eval_volatility",
    "max_swing",       # largest single-move eval change
    "outlier_flagged"  # 1 if max_swing > threshold
])


engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)


player_records = {p: [] for p in PLAYERS}


for player_name in PLAYERS:

    print("\n" + "=" * 60)
    print("PLAYER:", player_name)
    print("=" * 60)

    if player_name not in player_vocab:
        print("PLAYER NOT FOUND IN VOCAB — SKIPPING")
        continue

    player_id     = player_vocab[player_name]
    player_tensor = torch.tensor([player_id], dtype=torch.long, device=device)

    for rollout_idx in range(ROLLOUTS_PER_PLAYER):

        print(f"  ROLLOUT {rollout_idx + 1}/{ROLLOUTS_PER_PLAYER}")

        board     = chess.Board()
        input_ids = []

        # Apply seed moves
        for move_str in seed_moves:
            uci = move_str.split("_")[1]
            board.push(chess.Move.from_uci(uci))
            input_ids.append(move_vocab[move_str])

        evaluations = []


        with torch.no_grad():

            for _ in range(MAX_NEW_MOVES):

                context = input_ids[-127:]
                x       = torch.tensor([context], dtype=torch.long, device=device)
                logits  = model(x, player_tensor)
                logits  = logits[:, -1, :] / TEMPERATURE
                probs   = torch.softmax(logits, dim=-1)

                # Build legal-move candidate list
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

                # Restrict to legal moves and re-normalise
                legal_probs = probs[0, legal_token_ids]
                legal_probs = legal_probs / legal_probs.sum()

                # Top-k among legal moves
                k           = min(TOP_K, len(legal_probs))
                top_probs, top_indices = torch.topk(legal_probs, k=k)
                top_probs   = top_probs / top_probs.sum()

                sampled_rel = torch.multinomial(top_probs, num_samples=1).item()
                sampled_idx = top_indices[sampled_rel].item()

                next_token_id = legal_token_ids[sampled_idx]
                next_move     = legal_moves[sampled_idx]

                input_ids.append(next_token_id)
                board.push(chess.Move.from_uci(next_move.split("_")[1]))

                # Stockfish eval after this move
                info  = engine.analyse(board, chess.engine.Limit(depth=12))
                score = info["score"].white()

                if score.is_mate():
                    eval_cp = 10000
                else:
                    eval_cp = score.score() or 0

                evaluations.append(eval_cp)


        if len(evaluations) < 2:
            continue

        avg_eval        = statistics.mean(evaluations)
        eval_volatility = statistics.stdev(evaluations)

        # Largest move-to-move eval swing in this rollout
        swings    = [abs(evaluations[i] - evaluations[i-1])
                     for i in range(1, len(evaluations))]
        max_swing = max(swings) if swings else 0

        is_outlier = int(max_swing > OUTLIER_SWING_THRESHOLD)

        writer.writerow([
            player_name,
            rollout_idx + 1,
            round(avg_eval,        2),
            round(eval_volatility, 2),
            round(max_swing,       2),
            is_outlier
        ])

        player_records[player_name].append({
            "avg_eval":        avg_eval,
            "eval_volatility": eval_volatility,
            "max_swing":       max_swing,
            "is_outlier":      bool(is_outlier)
        })


engine.quit()
csv_file.close()

summary_file = open(summary_path, "w", newline="", encoding="utf-8")
summary_writer = csv.writer(summary_file)
summary_writer.writerow([
    "player",
    "n_total",
    "n_clean",        # rollouts below threshold
    "n_outliers",
    "mean_eval_all",
    "mean_eval_clean",
    "median_eval_clean",
    "std_eval_clean",
    "mean_vol_all",
    "mean_vol_clean",
    "median_vol_clean",
    "std_vol_clean",
])

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for player_name in PLAYERS:

    records = player_records[player_name]
    if not records:
        continue

    all_evals  = [r["avg_eval"]        for r in records]
    all_vols   = [r["eval_volatility"] for r in records]
    clean      = [r for r in records if not r["is_outlier"]]
    clean_evals = [r["avg_eval"]        for r in clean]
    clean_vols  = [r["eval_volatility"] for r in clean]

    n_total    = len(records)
    n_clean    = len(clean)
    n_outliers = n_total - n_clean

    def safe_median(lst):
        return round(statistics.median(lst), 2) if lst else None

    def safe_mean(lst):
        return round(statistics.mean(lst), 2) if lst else None

    def safe_std(lst):
        return round(statistics.stdev(lst), 2) if len(lst) >= 2 else None

    row = [
        player_name,
        n_total,
        n_clean,
        n_outliers,
        safe_mean(all_evals),
        safe_mean(clean_evals),
        safe_median(clean_evals),
        safe_std(clean_evals),
        safe_mean(all_vols),
        safe_mean(clean_vols),
        safe_median(clean_vols),
        safe_std(clean_vols),
    ]

    summary_writer.writerow(row)

    print(f"\n{player_name}")
    print(f"  total rollouts : {n_total}  |  clean: {n_clean}  |  outliers: {n_outliers}")
    print(f"  eval  — mean(all): {safe_mean(all_evals):>8}  mean(clean): {safe_mean(clean_evals):>8}  "
          f"median(clean): {safe_median(clean_evals):>8}  std: {safe_std(clean_evals):>8}")
    print(f"  vol   — mean(all): {safe_mean(all_vols):>8}  mean(clean): {safe_mean(clean_vols):>8}  "
          f"median(clean): {safe_median(clean_vols):>8}  std: {safe_std(clean_vols):>8}")

summary_file.close()

print(f"\nRAW CSV    : {csv_path}")
print(f"SUMMARY CSV: {summary_path}")
print("\nDONE.")