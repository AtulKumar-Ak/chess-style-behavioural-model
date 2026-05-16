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

engine = chess.engine.SimpleEngine.popen_uci(
    STOCKFISH_PATH
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
# MODEL
# ======================================================

VOCAB_SIZE = len(move_vocab)

NUM_PLAYERS = len(player_vocab)

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

GENERATED_ROLLOUTS = 20

MAX_NEW_MOVES = 40

TEMPERATURE = 0.8

TOP_K = 10

# ======================================================
# RESULTS
# ======================================================

real_results = defaultdict(list)

generated_results = defaultdict(list)

# ======================================================
# ANALYSIS FUNCTION
# ======================================================

def analyze_game(move_list):

    board = chess.Board()

    evaluations = []

    captures = 0

    queen_trades = 0

    for token in move_list:

        try:

            uci = token.split("_")[1]

            move = chess.Move.from_uci(uci)

        except:

            continue

        if move not in board.legal_moves:

            break

        # ==============================================
        # METRICS
        # ==============================================

        if board.is_capture(move):

            captures += 1

        piece = board.piece_at(
            move.from_square
        )

        if piece is not None:

            if piece.piece_type == chess.QUEEN:

                if board.is_capture(move):

                    queen_trades += 1

        # ==============================================
        # APPLY MOVE
        # ==============================================

        board.push(move)

        # ==============================================
        # STOCKFISH
        # ==============================================

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

        evaluations.append(eval_cp)

    # ==================================================

    if len(evaluations) < 2:

        return None

    return {

        "avg_eval":
            statistics.mean(evaluations),

        "volatility":
            statistics.stdev(evaluations),

        "captures":
            captures,

        "queen_trades":
            queen_trades
    }

# ======================================================
# REAL GAME ANALYSIS
# ======================================================

print("\n" + "=" * 60)
print("ANALYZING REAL GAMES")
print("=" * 60)

trajectory_base = (
    "dataset/trajectories/train"
)

import os

for filename in os.listdir(
    trajectory_base
):

    if not filename.endswith(".jsonl"):

        continue

    matched_player = None

    for player in PLAYERS:

        if player.lower() in filename.lower():

            matched_player = player

            break

    if matched_player is None:

        continue

    print("\nFILE:", filename)

    path = os.path.join(
        trajectory_base,
        filename
    )

    game_count = 0

    with open(
        path,
        encoding="utf-8"
    ) as f:

        for line in f:

            if game_count >= 100:

                break

            record = json.loads(line)

            moves = record["moves"]

            metrics = analyze_game(
                moves
            )

            if metrics is not None:

                real_results[
                    matched_player
                ].append(metrics)

                game_count += 1

# ======================================================
# GENERATED ANALYSIS
# ======================================================

print("\n" + "=" * 60)
print("ANALYZING GENERATED GAMES")
print("=" * 60)

seed_moves = [

    "P_e2e4",
    "P_e7e5",
    "N_g1f3",
    "N_b8c6"
]

for player_name in PLAYERS:

    print("\nPLAYER:", player_name)

    if player_name not in player_vocab:

        continue

    player_id = player_vocab[
        player_name
    ]

    player_tensor = torch.tensor(

        [player_id],

        dtype=torch.long,

        device=device
    )

    # ==================================================

    for rollout_idx in range(
        GENERATED_ROLLOUTS
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

        generated_moves = list(
            seed_moves
        )

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

                logits = (
                    logits
                    / TEMPERATURE
                )

                probs = torch.softmax(

                    logits,

                    dim=-1
                )

                legal_token_ids = []

                legal_moves = []

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

                if len(legal_token_ids) == 0:

                    break

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

                input_ids.append(
                    next_token_id
                )

                generated_moves.append(
                    next_move
                )

                uci = next_move.split("_")[1]

                board.push(
                    chess.Move.from_uci(uci)
                )

        # ==================================================

        metrics = analyze_game(
            generated_moves
        )

        if metrics is not None:

            generated_results[
                player_name
            ].append(metrics)

# ======================================================
# SAVE CSV
# ======================================================

output_csv = (
    "real_vs_generated.csv"
)

with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([

        "player",

        "type",

        "mean_eval",

        "mean_volatility",

        "mean_captures",

        "mean_queen_trades"
    ])

    # ==================================================

    for player in PLAYERS:

        # ==============================================
        # REAL
        # ==============================================

        if len(real_results[player]) > 0:

            rr = real_results[player]

            writer.writerow([

                player,

                "real",

                round(statistics.mean(
                    x["avg_eval"]
                    for x in rr
                ), 2),

                round(statistics.mean(
                    x["volatility"]
                    for x in rr
                ), 2),

                round(statistics.mean(
                    x["captures"]
                    for x in rr
                ), 2),

                round(statistics.mean(
                    x["queen_trades"]
                    for x in rr
                ), 2)
            ])

        # ==============================================
        # GENERATED
        # ==============================================

        if len(generated_results[player]) > 0:

            gr = generated_results[player]

            writer.writerow([

                player,

                "generated",

                round(statistics.mean(
                    x["avg_eval"]
                    for x in gr
                ), 2),

                round(statistics.mean(
                    x["volatility"]
                    for x in gr
                ), 2),

                round(statistics.mean(
                    x["captures"]
                    for x in gr
                ), 2),

                round(statistics.mean(
                    x["queen_trades"]
                    for x in gr
                ), 2)
            ])

# ======================================================
# CLEANUP
# ======================================================

engine.quit()

print("\nDONE")

print(
    f"\nRESULTS SAVED TO: {output_csv}"
)