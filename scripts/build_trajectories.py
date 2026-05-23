from pathlib import Path
import json

import chess

# ======================================================
# INPUT / OUTPUT
# ======================================================

SPLITS_DIR      = Path("dataset/splits")
OUTPUT_BASE_DIR = Path("dataset/trajectories")

# ======================================================
# CONTEXT SIZE
# ======================================================

CONTEXT_SIZE = 64

# ======================================================
# PHASE-AWARE STRIDE
# ======================================================

OPENING_CUTOFF = 40
OPENING_STRIDE = 8
MIDGAME_STRIDE = 2

# ======================================================
# PIECE PREFIXES
# ======================================================

PIECE_MAP = {
    chess.PAWN:   "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK:   "R",
    chess.QUEEN:  "Q",
    chess.KING:   "K",
}

# ======================================================
# PLAYER FILE MAP
# ======================================================

PLAYER_FILE_MAP = {
    "magnus":  "MagnusCarlsen",
    "hikaru":  "Hikaru",
    "alireza": "alireza2003",
    "nepo":    "lachesisQ",
}

# ======================================================
# ACCOUNT NAMES
# All known account name variants per canonical player,
# across chesscom / lichess / pgnmentor.
# ======================================================

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
# OPPONENT LABEL
# ======================================================

OPPONENT_LABEL = "opponent"

# ======================================================

def move_to_token(board, move):
    piece = board.piece_at(move.from_square)
    if piece is None:
        return None
    return f"{PIECE_MAP[piece.piece_type]}_{move.uci()}"


def get_player_side(metadata, target_player):
    white_lower = metadata["white"].lower()
    black_lower = metadata["black"].lower()
    for account in PLAYER_ACCOUNTS.get(target_player, []):
        if account in white_lower:
            return True
        if account in black_lower:
            return False
    return None

# ======================================================
# PROCESS SPLITS
# ======================================================

for split in ["train", "val", "test"]:

    print("\n" + "=" * 70)
    print("PROCESSING SPLIT:", split)
    print("=" * 70)

    input_dir  = SPLITS_DIR / split
    output_dir = OUTPUT_BASE_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    for jsonl_file in input_dir.glob("*.jsonl"):
        if "pgnmentor" in jsonl_file.name:
            print(f"  SKIP {jsonl_file.name} (pgnmentor excluded)")
            continue

        print("\n" + "-" * 60)
        print("FILE:", jsonl_file.name)
        print("-" * 60)

        filename_lower = jsonl_file.name.lower()
        target_player  = None

        for alias, player_name in PLAYER_FILE_MAP.items():
            if alias in filename_lower:
                target_player = player_name
                break

        if target_player is None:
            print("COULD NOT DETERMINE PLAYER — SKIPPING")
            continue

        print("TARGET PLAYER:", target_player)

        output_path           = output_dir / jsonl_file.name
        total_samples         = 0
        target_player_samples = 0
        opponent_samples      = 0
        skipped_games         = 0

        with open(jsonl_file, encoding="utf-8") as in_f, \
             open(output_path, "w", encoding="utf-8") as out_f:

            for line in in_f:

                record    = json.loads(line)
                raw_moves = record["moves"]
                metadata  = record["metadata"]

                is_white = get_player_side(metadata, target_player)

                if is_white is None:
                    skipped_games += 1
                    continue

                board       = chess.Board()
                token_moves = []
                valid_game  = True

                for uci in raw_moves:
                    try:
                        move  = chess.Move.from_uci(uci)
                        token = move_to_token(board, move)
                        if token is None:
                            valid_game = False
                            break
                        token_moves.append(token)
                        board.push(move)
                    except Exception:
                        valid_game = False
                        break

                if not valid_game:
                    skipped_games += 1
                    continue

                if len(token_moves) < CONTEXT_SIZE:
                    continue

                i = 0

                while i <= len(token_moves) - CONTEXT_SIZE:

                    window = token_moves[i : i + CONTEXT_SIZE]

                    predicted_move_idx      = i + CONTEXT_SIZE - 1
                    target_is_white_to_move = (predicted_move_idx % 2 == 0)

                    if is_white == target_is_white_to_move:
                        player_label = target_player
                        target_player_samples += 1
                    else:
                        player_label = OPPONENT_LABEL
                        opponent_samples += 1

                    valid_elos = [
                        e for e in [
                            metadata["white_elo"],
                            metadata["black_elo"],
                        ]
                        if e > 0
                    ]

                    avg_elo = (
                        sum(valid_elos) / len(valid_elos)
                        if valid_elos else 0
                    )

                    sample = {
                        "player":  player_label,
                        "moves":   window,
                        "opening": metadata["opening"],
                        "speed":   metadata["speed"],
                        "avg_elo": avg_elo,
                    }

                    out_f.write(json.dumps(sample) + "\n")
                    total_samples += 1

                    if i < OPENING_CUTOFF:
                        i += OPENING_STRIDE
                    else:
                        i += MIDGAME_STRIDE

        print(f"TOTAL SAMPLES          : {total_samples}")
        print(f"  target player samples: {target_player_samples}")
        print(f"  opponent samples     : {opponent_samples}")
        print(f"SKIPPED GAMES          : {skipped_games}")

print("\nDONE.")