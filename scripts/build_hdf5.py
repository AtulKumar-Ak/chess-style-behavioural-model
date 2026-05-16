from pathlib import Path
import json

import h5py
import numpy as np

# ======================================================
# PATHS
# ======================================================

TOKENIZED_BASE_DIR = Path(
    "dataset/tokenized"
)

OUTPUT_DIR = Path(
    "dataset/hdf5"
)

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True
)

# ======================================================
# FIXED SEQUENCE LENGTH
# ======================================================

SEQUENCE_LENGTH = 128

# ======================================================
# PROCESS SPLITS
# ======================================================

for split in ["train", "val", "test"]:

    print("\n" + "=" * 70)
    print("PROCESSING SPLIT:", split)
    print("=" * 70)

    input_dir = (
        TOKENIZED_BASE_DIR / split
    )

    # ==================================================
    # COUNT TOTAL SAMPLES
    # ==================================================

    total_samples = 0

    for jsonl_file in input_dir.glob("*.jsonl"):

        with open(
            jsonl_file,
            encoding="utf-8"
        ) as f:

            for _ in f:

                total_samples += 1

    print(
        f"\nTOTAL SAMPLES : "
        f"{total_samples}"
    )

    # ==================================================
    # CREATE HDF5
    # ==================================================

    h5_path = (
        OUTPUT_DIR /
        f"{split}.h5"
    )

    with h5py.File(
        h5_path,
        "w"
    ) as h5f:

        # ==============================================
        # DATASETS
        # ==============================================

        moves_ds = h5f.create_dataset(

            "move_ids",

            shape=(
                total_samples,
                SEQUENCE_LENGTH
            ),

            dtype=np.int32
        )

        player_ds = h5f.create_dataset(

            "player_ids",

            shape=(total_samples,),

            dtype=np.int32
        )

        speed_ds = h5f.create_dataset(

            "speed_ids",

            shape=(total_samples,),

            dtype=np.int32
        )

        elo_ds = h5f.create_dataset(

            "avg_elo",

            shape=(total_samples,),

            dtype=np.float32
        )

        # ==============================================
        # WRITE DATA
        # ==============================================

        idx = 0

        for jsonl_file in input_dir.glob("*.jsonl"):

            print("\nFILE:", jsonl_file.name)

            with open(
                jsonl_file,
                encoding="utf-8"
            ) as f:

                for line in f:

                    record = json.loads(line)

                    # ==================================
                    # MOVE IDS
                    # ==================================

                    move_ids = record[
                        "move_ids"
                    ]

                    # ==================================
                    # VALIDATE LENGTH
                    # ==================================

                    if len(move_ids) != SEQUENCE_LENGTH:

                        continue

                    # ==================================
                    # WRITE MOVE IDS
                    # ==================================

                    moves_ds[idx] = np.array(

                        move_ids,

                        dtype=np.int32
                    )

                    # ==================================
                    # WRITE METADATA
                    # ==================================

                    player_ds[idx] = record[
                        "player_id"
                    ]

                    speed_ds[idx] = record[
                        "speed_id"
                    ]

                    elo_ds[idx] = record[
                        "avg_elo"
                    ]

                    idx += 1

        # ==============================================
        # FINAL COUNT
        # ==============================================

        print(
            f"\nWRITTEN SAMPLES : "
            f"{idx}"
        )

    # ==================================================

    print(
        f"\nSAVED : {h5_path}"
    )

print("\nDONE.")