from pathlib import Path
import json

# ======================================================
# PATHS
# ======================================================

TRAJ_BASE_DIR   = Path("dataset/trajectories")
VOCAB_DIR       = Path("dataset/vocab")
OUTPUT_BASE_DIR = Path("dataset/tokenized")

# ======================================================
# LOAD VOCABS
# ======================================================

with open(VOCAB_DIR / "move_vocab.json", encoding="utf-8") as f:
    move_vocab = json.load(f)

with open(VOCAB_DIR / "player_vocab.json", encoding="utf-8") as f:
    player_vocab = json.load(f)

with open(VOCAB_DIR / "speed_vocab.json", encoding="utf-8") as f:
    speed_vocab = json.load(f)

print(f"Move vocab size  : {len(move_vocab)}")
print(f"Player vocab size: {len(player_vocab)}")
print(f"Speed vocab size : {len(speed_vocab)}")

# Sanity check — opponent must be in player vocab
if "opponent" not in player_vocab:
    print("\nERROR: 'opponent' missing from player_vocab.")
    print("Rebuild vocab before tokenizing.")
    exit()

print(f"Opponent id      : {player_vocab['opponent']}")

# ======================================================
# SPECIAL TOKENS
# ======================================================

MOVE_UNK_ID   = move_vocab["<UNK>"]
PLAYER_UNK_ID = player_vocab["<UNK_PLAYER>"]
SPEED_UNK_ID  = speed_vocab["<UNK_SPEED>"]

# ======================================================
# ENCODERS
# ======================================================

def encode_moves(tokens):
    return [move_vocab.get(token, MOVE_UNK_ID) for token in tokens]

def encode_player(player):
    return player_vocab.get(player, PLAYER_UNK_ID)

def encode_speed(speed):
    return speed_vocab.get(speed, SPEED_UNK_ID)

# ======================================================
# PROCESS SPLITS
# ======================================================

for split in ["train", "val", "test"]:

    print("\n" + "=" * 70)
    print("PROCESSING SPLIT:", split)
    print("=" * 70)

    input_dir  = TRAJ_BASE_DIR / split
    output_dir = OUTPUT_BASE_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    for jsonl_file in input_dir.glob("*.jsonl"):

        # Skip pgnmentor — excluded from pipeline
        if "chesscom" in jsonl_file.name.lower() or "lichess" in jsonl_file.name.lower():
            print(f"\nSKIP (chesscom or lichess excluded): {jsonl_file.name}")
            continue

        print("\n" + "-" * 60)
        print("FILE:", jsonl_file.name)
        print("-" * 60)

        output_path     = output_dir / jsonl_file.name
        total_samples   = 0
        unk_move_hits   = 0
        unk_player_hits = 0
        unk_speed_hits  = 0
        opponent_count  = 0

        with open(jsonl_file, encoding="utf-8") as in_f, \
             open(output_path, "w", encoding="utf-8") as out_f:

            for line in in_f:

                record = json.loads(line)

                # ==========================================
                # TOKENIZE MOVES
                # ==========================================

                move_ids = encode_moves(record["moves"])

                if len(move_ids) != 64:
                    print(f"WARNING: Expected 64 tokens, got {len(move_ids)} — skipping")
                    continue

                unk_move_hits += sum(
                    1 for mid in move_ids if mid == MOVE_UNK_ID
                )

                # ==========================================
                # TOKENIZE METADATA
                # ==========================================

                player_id = encode_player(record["player"])
                speed_id  = encode_speed(record["speed"])

                if player_id == PLAYER_UNK_ID:
                    unk_player_hits += 1

                if speed_id == SPEED_UNK_ID:
                    unk_speed_hits += 1

                if record["player"] == "opponent":
                    opponent_count += 1

                # ==========================================
                # WRITE TOKENIZED RECORD
                # ==========================================

                out_f.write(json.dumps({
                    "move_ids":  move_ids,
                    "player_id": player_id,
                    "speed_id":  speed_id,
                    "avg_elo":   record["avg_elo"],
                    "seq_len":   len(move_ids),
                }) + "\n")

                total_samples += 1

        target_count = total_samples - opponent_count

        print(f"TOTAL SAMPLES    : {total_samples}")
        print(f"  target player  : {target_count}")
        print(f"  opponent       : {opponent_count}")
        print(f"UNK MOVE TOKENS  : {unk_move_hits}")
        print(f"UNK PLAYERS      : {unk_player_hits}")
        print(f"UNK SPEEDS       : {unk_speed_hits}")

        if unk_player_hits > 0:
            print("  → Check PLAYER_ACCOUNTS in build_trajectories.py")
        if unk_speed_hits > 0:
            print("  → Check speed values in source data")

print("\nDONE.")