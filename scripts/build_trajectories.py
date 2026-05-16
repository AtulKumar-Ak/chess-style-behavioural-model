from pathlib import Path
import json

import chess

# ======================================================
# INPUT / OUTPUT
# ======================================================

SPLITS_DIR = Path(
    "dataset/splits"
)

OUTPUT_BASE_DIR = Path(
    "dataset/trajectories"
)

# ======================================================

CONTEXT_SIZE = 128
STRIDE = 4
# ======================================================
# PIECE PREFIXES
# ======================================================

PIECE_MAP = {

    chess.PAWN: "P",

    chess.KNIGHT: "N",

    chess.BISHOP: "B",

    chess.ROOK: "R",

    chess.QUEEN: "Q",

    chess.KING: "K",
}

# ======================================================


def move_to_token(board, move):

    piece = board.piece_at(
        move.from_square
    )

    if piece is None:
        return None

    prefix = PIECE_MAP[
        piece.piece_type
    ]

    return f"{prefix}_{move.uci()}"


# ======================================================
# PROCESS SPLITS
# ======================================================

for split in ["train", "val", "test"]:

    print("\n" + "=" * 70)
    print("PROCESSING SPLIT:", split)
    print("=" * 70)

    input_dir = SPLITS_DIR / split

    output_dir = (
        OUTPUT_BASE_DIR / split
    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True
    )

    # ==================================================

    for jsonl_file in input_dir.glob("*.jsonl"):

        print("\n" + "-" * 60)
        print("FILE:", jsonl_file.name)
        print("-" * 60)

        output_path = (
            output_dir /
            jsonl_file.name
        )

        total_samples = 0

        skipped_games = 0

        # ==================================================

        with open(
            jsonl_file,
            encoding="utf-8"
        ) as in_f, open(
            output_path,
            "w",
            encoding="utf-8"
        ) as out_f:

            for line in in_f:

                record = json.loads(line)

                raw_moves = record["moves"]

                metadata = record["metadata"]

                # ==========================================
                # REPLAY BOARD
                # ==========================================

                board = chess.Board()

                token_moves = []

                valid_game = True

                # ==========================================

                for uci in raw_moves:

                    try:

                        move = chess.Move.from_uci(
                            uci
                        )

                        token = move_to_token(
                            board,
                            move
                        )

                        if token is None:

                            valid_game = False
                            break

                        token_moves.append(
                            token
                        )

                        board.push(move)

                    except:

                        valid_game = False
                        break

                # ==========================================

                if not valid_game:

                    skipped_games += 1
                    continue

                # ==========================================
                # NOT ENOUGH MOVES
                # ==========================================

                if len(token_moves) < CONTEXT_SIZE:
                    continue

                # ==========================================
                # SLIDING WINDOWS
                # ==========================================

                for i in range(

                    0,
                
                    len(token_moves)
                    -
                    CONTEXT_SIZE
                    + 1,
                
                    STRIDE
                ):

                    window = token_moves[
                        i :
                        i + CONTEXT_SIZE
                    ]

                    # ======================================
                    # FINAL SAMPLE
                    # ======================================

                    sample = {

                        "player":
                            metadata["white"],

                        "moves":
                            window,

                        "opening":
                            metadata["opening"],

                        "speed":
                            metadata["speed"],

                        "avg_elo":
                            (
                                metadata["white_elo"]
                                +
                                metadata["black_elo"]
                            ) / 2
                    }

                    out_f.write(
                        json.dumps(sample)
                        + "\n"
                    )

                    total_samples += 1

        # ==================================================

        print(
            f"TOTAL SAMPLES : "
            f"{total_samples}"
        )

        print(
            f"SKIPPED GAMES : "
            f"{skipped_games}"
        )

print("\nDONE.")