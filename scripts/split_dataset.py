from pathlib import Path
import json
import hashlib

# ======================================================

INPUT_DIR = Path(
    "dataset/final"
)

OUTPUT_DIR = Path(
    "dataset/splits"
)

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

# ======================================================

TRAIN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ======================================================

TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

# ======================================================


def get_split(move_string):

    # ------------------------------------------
    # DETERMINISTIC HASH
    # ------------------------------------------

    hash_value = hashlib.md5(

        move_string.encode()

    ).hexdigest()

    # ------------------------------------------
    # CONVERT TO NUMBER
    # ------------------------------------------

    numeric = int(
        hash_value,
        16
    )

    ratio = (
        numeric % 10000
    ) / 10000

    # ------------------------------------------

    if ratio < TRAIN_RATIO:
        return "train"

    elif ratio < (
        TRAIN_RATIO + VAL_RATIO
    ):
        return "val"

    else:
        return "test"


# ======================================================

counts = {

    "train": 0,
    "val": 0,
    "test": 0
}

# ======================================================

for jsonl_file in INPUT_DIR.glob("*.jsonl"):

    print("\n" + "=" * 60)
    print("PROCESSING:", jsonl_file.name)
    print("=" * 60)

    train_out = open(

        TRAIN_DIR / jsonl_file.name,

        "w",
        encoding="utf-8"
    )

    val_out = open(

        VAL_DIR / jsonl_file.name,

        "w",
        encoding="utf-8"
    )

    test_out = open(

        TEST_DIR / jsonl_file.name,

        "w",
        encoding="utf-8"
    )

    # ==================================================

    with open(
        jsonl_file,
        encoding="utf-8"
    ) as f:

        for line in f:

            record = json.loads(line)

            moves = record["moves"]

            move_string = " ".join(moves)

            split = get_split(
                move_string
            )

            # ------------------------------------------

            if split == "train":

                train_out.write(line)

            elif split == "val":

                val_out.write(line)

            else:

                test_out.write(line)

            counts[split] += 1

    # ==================================================

    train_out.close()
    val_out.close()
    test_out.close()

# ======================================================

print("\n" + "=" * 60)
print("FINAL SPLIT COUNTS")
print("=" * 60)

print(f"TRAIN : {counts['train']}")
print(f"VAL   : {counts['val']}")
print(f"TEST  : {counts['test']}")