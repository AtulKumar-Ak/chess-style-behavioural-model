from pathlib import Path
import json
import hashlib

INPUT_DIR = Path("dataset/filtered")

OUTPUT_DIR = Path("dataset/final")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


seen_hashes = set()

total_games = 0
duplicate_games = 0
kept_games = 0


for jsonl_file in INPUT_DIR.glob("*.jsonl"):
    print("\n" + "=" * 60)
    print("PROCESSING:", jsonl_file.name)
    print("=" * 60)

    output_path = (
        OUTPUT_DIR /
        jsonl_file.name
    )

    with open(
        jsonl_file,
        encoding="utf-8"
    ) as in_f, open(
        output_path,
        "w",
        encoding="utf-8"
    ) as out_f:

        for line in in_f:

            total_games += 1

            record = json.loads(line)

            moves = record["moves"]


            move_string = " ".join(moves)

            game_hash = hashlib.md5(
                move_string.encode()
            ).hexdigest()


            if game_hash in seen_hashes:

                duplicate_games += 1
                continue

            seen_hashes.add(game_hash)

            kept_games += 1

            out_f.write(line)


print("\n" + "=" * 60)
print("DEDUPLICATION COMPLETE")
print("=" * 60)

print(f"TOTAL GAMES      : {total_games}")
print(f"DUPLICATES       : {duplicate_games}")
print(f"FINAL KEPT       : {kept_games}")