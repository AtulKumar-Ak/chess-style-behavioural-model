from pathlib import Path
import json

# ======================================================
# PATHS
# ======================================================

TRAJ_BASE_DIR = Path(
    "dataset/trajectories"
)

VOCAB_DIR = Path(
    "dataset/vocab"
)

OUTPUT_BASE_DIR = Path(
    "dataset/tokenized"
)

# ======================================================
# LOAD VOCABS
# ======================================================

with open(
    VOCAB_DIR / "move_vocab.json",
    encoding="utf-8"
) as f:

    move_vocab = json.load(f)

with open(
    VOCAB_DIR / "player_vocab.json",
    encoding="utf-8"
) as f:

    player_vocab = json.load(f)

with open(
    VOCAB_DIR / "speed_vocab.json",
    encoding="utf-8"
) as f:

    speed_vocab = json.load(f)

# ======================================================
# SPECIAL TOKENS
# ======================================================

MOVE_UNK_ID = move_vocab["<UNK>"]

PLAYER_UNK_ID = player_vocab[
    "<UNK_PLAYER>"
]

SPEED_UNK_ID = speed_vocab[
    "<UNK_SPEED>"
]

# ======================================================
# ENCODERS
# ======================================================


def encode_moves(tokens):

    return [

        move_vocab.get(
            token,
            MOVE_UNK_ID
        )

        for token in tokens
    ]


# ======================================================


def encode_player(player):

    return player_vocab.get(

        player,

        PLAYER_UNK_ID
    )


# ======================================================


def encode_speed(speed):

    return speed_vocab.get(

        speed,

        SPEED_UNK_ID
    )


# ======================================================
# PROCESS SPLITS
# ======================================================

for split in ["train", "val", "test"]:

    print("\n" + "=" * 70)
    print("PROCESSING SPLIT:", split)
    print("=" * 70)

    input_dir = (
        TRAJ_BASE_DIR / split
    )

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

                # ==========================================
                # TOKENIZE MOVES
                # ==========================================

                move_ids = encode_moves(
                    record["moves"]
                )

                # ==========================================
                # TOKENIZE METADATA
                # ==========================================

                player_id = encode_player(
                    record["player"]
                )

                speed_id = encode_speed(
                    record["speed"]
                )

                # ==========================================
                # FINAL RECORD
                # ==========================================

                tokenized_record = {

                    "move_ids":
                        move_ids,

                    "player_id":
                        player_id,

                    "speed_id":
                        speed_id,

                    "avg_elo":
                        record["avg_elo"]
                }

                out_f.write(
                    json.dumps(tokenized_record)
                    + "\n"
                )

                total_samples += 1

        # ==================================================

        print(
            f"TOTAL SAMPLES : "
            f"{total_samples}"
        )

print("\nDONE.")