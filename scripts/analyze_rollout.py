import json
import statistics

import chess
import chess.engine

import torch

from src.models.gpt_model import (
    GPTBehaviorModel
)


device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDEVICE:", device)


STOCKFISH_PATH = (
    r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"
)


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


id_to_move = {

    idx: move

    for move, idx in move_vocab.items()
}


vocab_size = len(move_vocab)

num_players = len(player_vocab)


model = GPTBehaviorModel(

    vocab_size=vocab_size,

    max_seq_len=63,

    embed_dim=256,

    num_heads=4,

    num_layers=4,

    dropout=0.1,

    num_players=num_players
)


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


PLAYER_NAME = "alireza2003"

player_id = player_vocab[
    PLAYER_NAME
]

player_tensor = torch.tensor(

    [player_id],

    dtype=torch.long,

    device=device
)

print(
    f"\nPLAYER: {PLAYER_NAME}"
)


seed_moves = [

    "P_e2e4",
    "P_e7e5",
    "N_g1f3",
    "N_b8c6"
]


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


MAX_NEW_MOVES = 40

TEMPERATURE = 0.8

TOP_K = 10


engine = chess.engine.SimpleEngine.popen_uci(
    STOCKFISH_PATH
)


evaluations = []

generated_moves = list(seed_moves)


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


        info = engine.analyse(

            board,

            chess.engine.Limit(depth=12)
        )

        score = info["score"].white()

        if score.is_mate():

            eval_cp = 10000

        else:

            eval_cp = score.score()

            if eval_cp is None:

                eval_cp = 0

        evaluations.append(eval_cp)


engine.quit()


print("\n" + "=" * 60)
print("GENERATED GAME")
print("=" * 60)

for move in generated_moves:

    print(move)


print("\n" + "=" * 60)
print("STOCKFISH METRICS")
print("=" * 60)

avg_eval = statistics.mean(evaluations)

eval_std = statistics.stdev(evaluations)

print(
    f"\nAVERAGE EVAL: "
    f"{avg_eval:.2f} cp"
)

print(
    f"EVAL VOLATILITY: "
    f"{eval_std:.2f}"
)


print("\nFINAL BOARD\n")

print(board)