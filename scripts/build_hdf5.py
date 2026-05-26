# scripts/build_hdf5.py

from pathlib import Path
import json
import random

import chess
import h5py
import numpy as np


TOKENIZED_BASE_DIR = Path("dataset/tokenized")
TRAJ_BASE_DIR      = Path("dataset/trajectories")   # needed for board replay
OUTPUT_DIR         = Path("dataset/hdf5")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 64


PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def board_to_vector(board):
    """
    Converts chess.Board → (768,) float32 numpy array.
    12 piece planes × 64 squares, binary encoding.
    """
    planes = np.zeros((12, 64), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            planes[plane, square] = 1.0
    return planes.flatten()


def compute_board_state(move_tokens):
    """
    Replays piece-aware tokens (e.g. 'P_e2e4') to get
    the board state after the LAST move in the window.
    This is the board state at the prediction point.
    Returns (768,) float32 array, or None if replay fails.
    """
    board = chess.Board()
    for token in move_tokens:
        try:
            uci  = token.split("_")[1]
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                return None
            board.push(move)
        except Exception:
            return None
    return board_to_vector(board)



PLAYER_CAPS = {
    "train": {
        "hikaru":  50_000,
        "alireza": 50_000,
        "magnus":  None,
        "nepo":    None,
    }
}

PLAYER_FILE_MAP = {
    "hikaru":  "hikaru",
    "alireza": "alireza",
    "magnus":  "magnus",
    "nepo":    "nepo",
}


def get_player_key(filename):
    lower = filename.lower()
    for alias, key in PLAYER_FILE_MAP.items():
        if alias in lower:
            return key
    return None


def load_jsonl_with_board(tokenized_path, traj_path, cap=None):
    """
    Loads tokenized records and computes board states from
    the corresponding trajectory file (which has move tokens).

    tokenized_path : dataset/tokenized/{split}/{file}.jsonl
    traj_path      : dataset/trajectories/{split}/{file}.jsonl

    Both files are aligned line-by-line — same order, same games.
    """
    records = []

    # Load trajectory moves for board replay
    traj_moves = []
    if traj_path.exists():
        with open(traj_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                traj_moves.append(rec.get("moves", []))
    else:
        print(f"  WARNING: trajectory file not found: {traj_path}")

    with open(tokenized_path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            record = json.loads(line)

            if len(record["move_ids"]) != SEQUENCE_LENGTH:
                continue

            # Compute board state from trajectory moves
            board_state = None
            if line_idx < len(traj_moves):
                board_state = compute_board_state(
                    traj_moves[line_idx]
                )

            # If board replay failed use zeros
            # (model will ignore it — no useful signal)
            if board_state is None:
                board_state = np.zeros(768, dtype=np.float32)

            record["board_state"] = board_state
            records.append(record)

    if cap is not None and len(records) > cap:
        records = random.sample(records, cap)

    return records


random.seed(42)

for split in ["train", "val", "test"]:

    print("\n" + "=" * 70)
    print("PROCESSING SPLIT:", split)
    print("=" * 70)

    tokenized_dir = TOKENIZED_BASE_DIR / split
    traj_dir      = TRAJ_BASE_DIR      / split
    caps          = PLAYER_CAPS.get(split, {})

    player_budgets = {}
    player_loaded  = {}

    for player_key, cap in caps.items():
        player_budgets[player_key] = cap
        player_loaded[player_key]  = 0

    all_records = []

    for jsonl_file in sorted(tokenized_dir.glob("*.jsonl")):

        if "pgnmentor" in jsonl_file.name:
            print(f"  SKIP {jsonl_file.name} (pgnmentor excluded)")
            continue

        player_key = get_player_key(jsonl_file.name)
        traj_path  = traj_dir / jsonl_file.name

        if split == "train" and player_key is not None:

            budget = player_budgets.get(player_key)

            if budget is not None:
                already   = player_loaded.get(player_key, 0)
                remaining = budget - already

                if remaining <= 0:
                    print(f"  SKIP {jsonl_file.name} (player cap reached)")
                    continue

                records = load_jsonl_with_board(
                    jsonl_file, traj_path, cap=remaining
                )
                player_loaded[player_key] = already + len(records)

            else:
                records = load_jsonl_with_board(
                    jsonl_file, traj_path, cap=None
                )

        else:
            records = load_jsonl_with_board(
                jsonl_file, traj_path, cap=None
            )

        print(f"  {jsonl_file.name:<40} {len(records):>7,} samples")
        all_records.extend(records)

    random.shuffle(all_records)
    total_samples = len(all_records)

    print(f"\nTOTAL AFTER CAPPING : {total_samples:,}")

    if split == "train":
        print("\nPER-PLAYER TOTALS:")
        for pk in sorted(player_loaded.keys()):
            loaded  = player_loaded[pk]
            cap     = player_budgets.get(pk)
            cap_str = f"{cap:,}" if cap else "no cap"
            print(f"  {pk:<12} {loaded:>7,}  (cap: {cap_str})")


    h5_path = OUTPUT_DIR / f"{split}.h5"

    with h5py.File(h5_path, "w") as h5f:

        moves_ds = h5f.create_dataset(
            "move_ids",
            shape=(total_samples, SEQUENCE_LENGTH),
            maxshape=(None, SEQUENCE_LENGTH),
            dtype=np.int32
        )
        player_ds = h5f.create_dataset(
            "player_ids",
            shape=(total_samples,),
            maxshape=(None,),
            dtype=np.int32
        )
        speed_ds = h5f.create_dataset(
            "speed_ids",
            shape=(total_samples,),
            maxshape=(None,),
            dtype=np.int32
        )
        elo_ds = h5f.create_dataset(
            "avg_elo",
            shape=(total_samples,),
            maxshape=(None,),
            dtype=np.float32
        )

        # NEW: board states — 12×64 binary encoding
        board_ds = h5f.create_dataset(
            "board_states",
            shape=(total_samples, 768),
            maxshape=(None, 768),
            dtype=np.float32
        )

        for idx, record in enumerate(all_records):
            moves_ds[idx]  = np.array(record["move_ids"], dtype=np.int32)
            player_ds[idx] = record["player_id"]
            speed_ds[idx]  = record["speed_id"]
            elo_ds[idx]    = record["avg_elo"]
            board_ds[idx]  = record["board_state"]

        moves_ds.resize(idx + 1,  axis=0)
        player_ds.resize(idx + 1, axis=0)
        speed_ds.resize(idx + 1,  axis=0)
        elo_ds.resize(idx + 1,    axis=0)
        board_ds.resize(idx + 1,  axis=0)

    print(f"\nSAVED : {h5_path}")

print("\nDONE.")