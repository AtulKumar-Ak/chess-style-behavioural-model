#build_vocab.py
from pathlib import Path
import json
from collections import Counter


TRAJ_DIR   = Path("dataset/trajectories/train")
OUTPUT_DIR = Path("dataset/vocab")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


move_counter = Counter()
players      = set()
speeds       = set()


for jsonl_file in TRAJ_DIR.glob("*.jsonl"):

    # Skip pgnmentor — excluded from pipeline
    if "pgnmentor" in jsonl_file.name.lower():
        print(f"SKIP (pgnmentor excluded): {jsonl_file.name}")
        continue

    print("\n" + "=" * 60)
    print("PROCESSING:", jsonl_file.name)
    print("=" * 60)

    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            for token in record["moves"]:
                move_counter[token] += 1

            # "opponent" will be picked up here automatically
            players.add(record["player"])

            speeds.add(record["speed"])


move_vocab = {"<PAD>": 0, "<UNK>": 1}

for token, _ in sorted(
    move_counter.items(), key=lambda x: x[1], reverse=True
):
    move_vocab[token] = len(move_vocab)


player_vocab = {"<UNK_PLAYER>": 0}

for player in sorted(players):
    player_vocab[player] = len(player_vocab)


speed_vocab = {"<UNK_SPEED>": 0}

for speed in sorted(speeds):
    speed_vocab[speed] = len(speed_vocab)


print("\n" + "=" * 60)
print("VOCAB SUMMARY")
print("=" * 60)
print(f"\nMOVE VOCAB SIZE   : {len(move_vocab)}")
print(f"PLAYER VOCAB SIZE : {len(player_vocab)}")
print(f"SPEED VOCAB SIZE  : {len(speed_vocab)}")

print("\nPLAYERS:")
for player in sorted(players):
    print(f"  {player}")

print("\nSPEEDS:")
for speed in sorted(speeds):
    print(f"  {speed}")

# Sanity check — opponent must be present
if "opponent" not in player_vocab:
    print("\nWARNING: 'opponent' not found in player vocab.")
    print("Check that build_trajectories.py ran correctly.")
else:
    print(f"\nOpponent label OK — id: {player_vocab['opponent']}")


with open(OUTPUT_DIR / "move_vocab.json", "w", encoding="utf-8") as f:
    json.dump(move_vocab, f, indent=2)

with open(OUTPUT_DIR / "player_vocab.json", "w", encoding="utf-8") as f:
    json.dump(player_vocab, f, indent=2)

with open(OUTPUT_DIR / "speed_vocab.json", "w", encoding="utf-8") as f:
    json.dump(speed_vocab, f, indent=2)

print(f"\nSAVED TO: {OUTPUT_DIR}")
print("\nDONE.")