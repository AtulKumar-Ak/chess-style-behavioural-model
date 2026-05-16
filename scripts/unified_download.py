from pathlib import Path

from src.player_registery import PLAYER_REGISTRY

from scripts.downloaders.lichess_downloader import (
    download_lichess_games
)

from scripts.downloaders.chesscom_downloader import (
    download_chesscom_games
)

# =========================================================

LICHESS_DIR = Path("dataset/raw_pgns/lichess")
CHESSCOM_DIR = Path("dataset/raw_pgns/chesscom")

LICHESS_DIR.mkdir(parents=True, exist_ok=True)
CHESSCOM_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================

for player_id, player_data in PLAYER_REGISTRY.items():

    print("\n" + "=" * 50)
    print(f"PLAYER → {player_id}")
    print("=" * 50)

    # ---------------------------
    # LICHESS
    # ---------------------------

    # lichess_accounts = player_data["accounts"]["lichess"]

    # for username in lichess_accounts:

    #     output_path = LICHESS_DIR / f"{player_id}.pgn"

    #     download_lichess_games(
    #         username=username,
    #         output_path=output_path
    #     )

    # ---------------------------
    # CHESS.COM
    # ---------------------------

    chesscom_accounts = player_data["accounts"]["chesscom"]

    for username in chesscom_accounts:

        output_path = CHESSCOM_DIR / f"{player_id}.pgn"

        download_chesscom_games(
            username=username,
            output_path=output_path
        )

print("\nDONE.")