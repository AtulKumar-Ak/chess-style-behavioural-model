from pathlib import Path
import json
from collections import Counter

# ======================================================

TRAJ_DIR = Path(
    "dataset/trajectories/train"
)

OUTPUT_DIR = Path(
    "dataset/vocab"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ======================================================

token_counter = Counter()

# ======================================================

for jsonl_file in TRAJ_DIR.glob("*.jsonl"):

    print("\nPROCESSING:", jsonl_file.name)

    with open(
        jsonl_file,
        encoding="utf-8"
    ) as f:

        for line in f:

            record = json.loads(line)

            # ==========================================
            # CONTEXT TOKENS
            # ==========================================

            for token in record["context"]:

                token_counter[token] += 1

            # ==========================================
            # TARGET TOKENS
            # ==========================================

            for token in record["target"]:

                token_counter[token] += 1

# ======================================================
# SPECIAL TOKENS
# ======================================================

vocab = {

    "<PAD>": 0,
    "<UNK>": 1
}

# ======================================================
# SORT BY FREQUENCY
# ======================================================

sorted_tokens = sorted(

    token_counter.items(),

    key=lambda x: x[1],

    reverse=True
)

# ======================================================
# BUILD VOCAB
# ======================================================

for token, count in sorted_tokens:

    vocab[token] = len(vocab)

# ======================================================

print("\nVOCAB SIZE:", len(vocab))

# ======================================================
# SAVE
# ======================================================

output_path = (
    OUTPUT_DIR /
    "move_vocab.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        vocab,
        f,
        indent=2
    )

print("\nSAVED:", output_path)