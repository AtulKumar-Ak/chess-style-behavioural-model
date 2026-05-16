
import csv
import json
import statistics
from collections import defaultdict

import chess
import chess.engine

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
# STOCKFISH
# ======================================================

STOCKFISH_PATH = (
    r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"
)

# ======================================================
# LOAD VOCABS
# ======================================================

with open(
    "dataset/vocab/move_vocab.json",
    encoding="utf-8"
) as f:

    move_vocab = json.load(f)

with open(
    "dataset/vocab/player_vocab.json",
    encoding="utf-8"
) as f:

    player_vocab = json.load(f)

# ======================================================
# MODEL CONFIG
# ======================================================

VOCAB_SIZE = len(move_vocab)

NUM_PLAYERS = len(player_vocab)

# ======================================================
# MODEL
# ======================================================

model = GPTBehaviorModel(

    vocab_size=VOCAB_SIZE,

    max_seq_len=63,

    embed_dim=256,

    num_heads=4,

    num_layers=4,

    dropout=0.1,

    num_players=NUM_PLAYERS
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
# PLAYERS
# ======================================================

PLAYERS = [

    "MagnusCarlsen",

    "Hikaru",

    "alireza2003",

    "lachesisQ"
]

# ======================================================
# SETTINGS
# ======================================================

ROLLOUTS_PER_PLAYER = 30

MAX_NEW_MOVES = 40

TEMPERATURE = 0.8

TOP_K = 10

# ======================================================
# SEED
# ======================================================

seed_moves = [

    "P_e2e4",
    "P_e7e5",
    "N_g1f3",
    "N_b8c6"
]

# ======================================================
# STOCKFISH
# ======================================================

engine = chess.engine.SimpleEngine.popen_uci(
    STOCKFISH_PATH
)

# ======================================================
# METRIC STORAGE
# ======================================================

results = defaultdict(list)

# ======================================================
# GENERATION LOOP
# ======================================================

for player_name in PLAYERS:

    print("\n" + "=" * 60)
    print("PLAYER:", player_name)
    print("=" * 60)

    if player_name not in player_vocab:

        print("PLAYER NOT FOUND")
        continue

    player_id = player_vocab[player_name]

    player_tensor = torch.tensor(

        [player_id],

        dtype=torch.long,

        device=device
    )

    # ==================================================

    for rollout_idx in range(
        ROLLOUTS_PER_PLAYER
    ):

        board = chess.Board()

        for move_str in seed_moves:

            uci = move_str.split("_")[1]

            board.push(
                chess.Move.from_uci(uci)
            )

        input_ids = [

            move_vocab[m]

            for m in seed_moves
        ]

        evaluations = []

        queen_trades = 0

        captures = 0

        # ==============================================

        with torch.no_grad():

            for _ in range(MAX_NEW_MOVES):

                context = input_ids[-63:]

                x = torch.tensor(

                    [context],

                    dtype=torch.long,

                    device=device
                )

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

                # ======================================
                # LEGAL MOVES
                # ======================================

                legal_token_ids = []

                legal_moves = []

                legal_chess_moves = []

                for legal_move in board.legal_moves:

                    uci = legal_move.uci()

                    piece = board.piece_at(
                        legal_move.from_square
                    )

                    piece_symbol = (
                        piece.symbol().upper()
                    )

                    token = (
                        f"{piece_symbol}_{uci}"
                    )

                    if token in move_vocab:

                        token_id = move_vocab[
                            token
                        ]

                        legal_token_ids.append(
                            token_id
                        )

                        legal_moves.append(
                            token
                        )

                        legal_chess_moves.append(
                            legal_move
                        )

                if len(legal_token_ids) == 0:

                    break

                # ======================================

                legal_probs = probs[
                    0,
                    legal_token_ids
                ]

                legal_probs = (
                    legal_probs
                    /
                    legal_probs.sum()
                )

                k = min(
                    TOP_K,
                    len(legal_probs)
                )

                top_probs, top_indices = torch.topk(

                    legal_probs,

                    k=k
                )

                top_probs = (
                    top_probs
                    /
                    top_probs.sum()
                )

                sampled_relative_idx = (
                    torch.multinomial(

                        top_probs,

                        num_samples=1
                    ).item()
                )

                sampled_idx = top_indices[
                    sampled_relative_idx
                ].item()

                next_token_id = (
                    legal_token_ids[
                        sampled_idx
                    ]
                )

                next_move = legal_moves[
                    sampled_idx
                ]

                next_chess_move = legal_chess_moves[
                    sampled_idx
                ]

                # ======================================
                # METRICS
                # ======================================

                if board.is_capture(
                    next_chess_move
                ):

                    captures += 1

                piece = board.piece_at(
                    next_chess_move.from_square
                )

                if piece is not None:

                    if piece.piece_type == chess.QUEEN:

                        if board.is_capture(
                            next_chess_move
                        ):

                            queen_trades += 1

                # ======================================
                # UPDATE
                # ======================================

                input_ids.append(
                    next_token_id
                )

                board.push(
                    next_chess_move
                )

                # ======================================
                # STOCKFISH ANALYSIS
                # ======================================

                info = engine.analyse(

                    board,

                    chess.engine.Limit(
                        depth=10
                    )
                )

                score = info[
                    "score"
                ].white()

                if score.is_mate():

                    eval_cp = 10000

                else:

                    eval_cp = score.score()

                    if eval_cp is None:

                        eval_cp = 0

                evaluations.append(
                    eval_cp
                )

        # ==================================================

        if len(evaluations) < 2:

            continue

        avg_eval = statistics.mean(
            evaluations
        )

        volatility = statistics.stdev(
            evaluations
        )

        results[player_name].append({

            "avg_eval": avg_eval,

            "volatility": volatility,

            "captures": captures,

            "queen_trades": queen_trades
        })

# ======================================================
# SAVE AGGREGATED RESULTS
# ======================================================

output_csv = "style_similarity_results.csv"

with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([

        "player",

        "mean_eval",

        "mean_volatility",

        "mean_captures",

        "mean_queen_trades"
    ])

    # ==============================================

    for player_name, player_results in results.items():

        mean_eval = statistics.mean([
            x["avg_eval"]
            for x in player_results
        ])

        mean_volatility = statistics.mean([
            x["volatility"]
            for x in player_results
        ])

        mean_captures = statistics.mean([
            x["captures"]
            for x in player_results
        ])

        mean_queen_trades = statistics.mean([
            x["queen_trades"]
            for x in player_results
        ])

        writer.writerow([

            player_name,

            round(mean_eval, 2),

            round(mean_volatility, 2),

            round(mean_captures, 2),

            round(mean_queen_trades, 2)
        ])

# ======================================================
# CLEANUP
# ======================================================

engine.quit()

print("\nDONE")
print(
    f"\nRESULTS SAVED TO: {output_csv}"
)
