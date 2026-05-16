from pathlib import Path
import json

from src.parse_pgn import iter_games

from src.metadata import (
    extract_metadata
)

from src.filters.lichess_filter import (
    lichess_filter
)

from src.filters.chesscom_filter import (
    chesscom_filter
)

# =====================================================

RAW_DIR = Path("dataset/raw_pgns")

OUTPUT_DIR = Path("dataset/filtered")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================

for pgn_file in RAW_DIR.rglob("*.pgn"):

    print("\n" + "=" * 60)
    print("PROCESSING:", pgn_file)
    print("=" * 60)

    # -------------------------------------------------

    if "lichess" in str(pgn_file).lower():

        platform = "lichess"

    elif "chesscom" in str(pgn_file).lower():

        platform = "chesscom"

    else:

        print("UNKNOWN PLATFORM")
        continue

    # -------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        f"{platform}_{pgn_file.stem}.jsonl"
    )

    total_games = 0
    valid_games = 0

    # =================================================

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as out_f:

        for game in iter_games(pgn_file):

            total_games += 1

            metadata = extract_metadata(
                game,
                platform
            )

            # -----------------------------------------
            # FILTERING
            # -----------------------------------------

            if platform == "lichess":

                keep = lichess_filter(
                    metadata
                )

            else:

                keep = chesscom_filter(
                    metadata
                )

            if not keep:
                continue

            valid_games += 1

            # -----------------------------------------
            # UCI MOVES
            # -----------------------------------------

            moves = [

                move.uci()

                for move in game.mainline_moves()
            ]

            # -----------------------------------------
            # FINAL RECORD
            # -----------------------------------------

            record = {

                "platform":
                    platform,

                "metadata":
                    metadata,

                "moves":
                    moves
            }

            out_f.write(
                json.dumps(record)
                + "\n"
            )

    # =================================================

    print(f"TOTAL GAMES : {total_games}")
    print(f"VALID GAMES : {valid_games}")

    if total_games > 0:

        keep_rate = (
            valid_games
            /
            total_games
        ) * 100

        print(
            f"KEEP RATE : "
            f"{keep_rate:.2f}%"
        )

print("\nDONE.")