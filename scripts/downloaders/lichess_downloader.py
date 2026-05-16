import requests
from pathlib import Path


def download_lichess_games(username, output_path, max_games=3000):

    print(f"\n[Lichess] Downloading {username}")

    url = f"https://lichess.org/api/games/user/{username}"

    params = {
        "max": max_games,
        "opening": "true",
        "clocks": "false",
        "evals": "false"
    }

    headers = {
        "Accept": "application/x-chess-pgn"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:

        print(f"Failed → {response.status_code}")
        return False

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Saved → {output_path}")

    return True