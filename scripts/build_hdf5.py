#scripts/build_hdf5.py
from pathlib import Path
import json
import random

import h5py
import numpy as np

# ======================================================
# PATHS
# ======================================================

TOKENIZED_BASE_DIR = Path("dataset/tokenized")
OUTPUT_DIR         = Path("dataset/hdf5")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# FIXED SEQUENCE LENGTH
# ======================================================

SEQUENCE_LENGTH = 64

# ======================================================
# PER-PLAYER CAPS  (train split only)
#
# Caps apply to target-player labeled samples only.
# "opponent" labeled samples are NEVER capped —
# they come from the same games and capping them
# independently would break the game position
# distribution balance.
#
# Hikaru  → 50k  (was 192k raw, dominant without cap)
# Alireza → 50k  (was 78k raw, minor reduction)
# Magnus  → None (natural ceiling ~48k)
# Nepo    → None (natural ceiling ~44k)
# ======================================================

PLAYER_CAPS = {
    "train": {
        "hikaru":  50_000,
        "alireza": 50_000,
        "magnus":  None,
        "nepo":    None,
    }
}

# Maps filename substrings → player key used in caps.
# "opponent" is intentionally absent — never capped.
PLAYER_FILE_MAP = {
    "hikaru":  "hikaru",
    "alireza": "alireza",
    "magnus":  "magnus",
    "nepo":    "nepo",
}

# ======================================================
# HELPERS
# ======================================================

def get_player_key(filename):
    lower = filename.lower()
    for alias, key in PLAYER_FILE_MAP.items():
        if alias in lower:
            return key
    return None


def load_jsonl(path, cap=None):
    """
    Load records from a tokenized jsonl file.
    If cap is set, randomly sample down to cap size.
    cap applies to TARGET PLAYER records only —
    opponent records within the same file are always
    loaded in full to preserve position distribution.
    """
    target_records   = []
    opponent_records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if len(record["move_ids"]) != SEQUENCE_LENGTH:
                continue
            # We don't have the player name here, only player_id.
            # The cap is applied at file level — all records in
            # a target-player file are target records, and all
            # records in the same file with opponent_id are
            # opponent records. Since both come from the same
            # jsonl, we load all and return together.
            target_records.append(record)

    if cap is not None and len(target_records) > cap:
        target_records = random.sample(target_records, cap)

    return target_records


# ======================================================
# PROCESS SPLITS
# ======================================================

random.seed(42)

for split in ["train", "val", "test"]:

    print("\n" + "=" * 70)
    print("PROCESSING SPLIT:", split)
    print("=" * 70)

    input_dir = TOKENIZED_BASE_DIR / split
    caps      = PLAYER_CAPS.get(split, {})

    player_budgets = {}
    player_loaded  = {}

    for player_key, cap in caps.items():
        player_budgets[player_key] = cap
        player_loaded[player_key]  = 0

    all_records    = []
    opponent_total = 0

    for jsonl_file in sorted(input_dir.glob("*.jsonl")):

        # # Skip pgnmentor
        if "pgnmentor" in jsonl_file.name:
            print(f"  SKIP {jsonl_file.name} (pgnmentor excluded)")
            continue

        player_key = get_player_key(jsonl_file.name)

        if split == "train" and player_key is not None:

            budget = player_budgets.get(player_key)

            if budget is not None:
                already_loaded = player_loaded.get(player_key, 0)
                remaining      = budget - already_loaded

                if remaining <= 0:
                    print(f"  SKIP {jsonl_file.name} (player cap reached)")
                    continue

                records = load_jsonl(jsonl_file, cap=remaining)
                player_loaded[player_key] = already_loaded + len(records)

            else:
                records = load_jsonl(jsonl_file, cap=None)

        else:
            # Val / test — no capping
            records = load_jsonl(jsonl_file, cap=None)

        # Count how many of these are opponent-labeled
        # player_id for opponent was set during tokenization
        # We can't easily check by name here, so just report total
        print(f"  {jsonl_file.name:<40} {len(records):>7,} samples")
        all_records.extend(records)

    # Shuffle so sources are interleaved
    random.shuffle(all_records)

    total_samples = len(all_records)

    print(f"\nTOTAL AFTER CAPPING : {total_samples:,}")

    if split == "train":
        print("\nPER-PLAYER TOTALS (target players only):")
        for pk in sorted(player_loaded.keys()):
            loaded  = player_loaded[pk]
            cap     = player_budgets.get(pk)
            cap_str = f"{cap:,}" if cap else "no cap"
            print(f"  {pk:<12} {loaded:>7,}  (cap: {cap_str})")
        print(f"\n  NOTE: opponent-labeled samples are included in")
        print(f"  total but not tracked separately here.")
        print(f"  Check tokenize output for per-file opponent counts.")

    # ==================================================
    # WRITE HDF5
    # ==================================================

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

        for idx, record in enumerate(all_records):
            moves_ds[idx]  = np.array(record["move_ids"], dtype=np.int32)
            player_ds[idx] = record["player_id"]
            speed_ds[idx]  = record["speed_id"]
            elo_ds[idx]    = record["avg_elo"]

        moves_ds.resize(idx + 1,  axis=0)
        player_ds.resize(idx + 1, axis=0)
        speed_ds.resize(idx + 1,  axis=0)
        elo_ds.resize(idx + 1,    axis=0)

    print(f"\nSAVED : {h5_path}")

print("\nDONE.")