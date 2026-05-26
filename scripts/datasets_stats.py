from pathlib import Path
import json
from collections import Counter

FILTERED_DIR = Path("dataset/final")


for jsonl_file in FILTERED_DIR.glob("*.jsonl"):

    print("\n" + "=" * 70)
    print("FILE:", jsonl_file.name)
    print("=" * 70)

    total_games = 0
    total_moves = 0

    speed_counter = Counter()
    opening_counter = Counter()

    move_lengths = []

    elo_sum = 0
    elo_count = 0


    with open(jsonl_file, encoding="utf-8") as f:

        for line in f:

            record = json.loads(line)

            metadata = record["metadata"]

            total_games += 1


            move_count = metadata["move_count"]

            total_moves += move_count

            move_lengths.append(move_count)


            speed_counter[
                metadata["speed"]
            ] += 1


            opening = metadata["opening"]

            if opening:

                opening_counter[
                    opening
                ] += 1


            avg_elo = (
                metadata["white_elo"]
                +
                metadata["black_elo"]
            ) / 2

            elo_sum += avg_elo
            elo_count += 1


    avg_moves = (
        total_moves / total_games
        if total_games > 0 else 0
    )

    avg_elo = (
        elo_sum / elo_count
        if elo_count > 0 else 0
    )


    print(f"\nTOTAL GAMES      : {total_games}")
    print(f"TOTAL MOVES      : {total_moves}")
    print(f"AVERAGE MOVES    : {avg_moves:.2f}")
    print(f"AVERAGE ELO      : {avg_elo:.2f}")


    print("\nSPEED DISTRIBUTION")

    for speed, count in speed_counter.items():

        pct = (
            count / total_games
        ) * 100

        print(
            f"{speed:12}"
            f"{count:8}"
            f" ({pct:.2f}%)"
        )


    print("\nTOP OPENINGS")

    for opening, count in opening_counter.most_common(10):

        print(
            f"{count:6}  {opening}"
        )


    if move_lengths:

        print("\nMOVE LENGTHS")

        print(
            f"MIN : {min(move_lengths)}"
        )

        print(
            f"MAX : {max(move_lengths)}"
        )

        print(
            f"AVG : "
            f"{sum(move_lengths)/len(move_lengths):.2f}"
        )

print("\nDONE.")