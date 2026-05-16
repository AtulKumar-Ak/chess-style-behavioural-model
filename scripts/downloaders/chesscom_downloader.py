"""
Fixed Chess.com downloader.

Three likely causes of empty file:
  1. pgn field is None or empty string for some/all games (filter wasn't catching it)
  2. Chess.com returns games in JSON format without PGN — need /pgn endpoint instead
  3. Rate limiting silently returning empty responses mid-loop

This version handles all three and gives verbose diagnostics.
"""

import cloudscraper
import time
from tqdm import tqdm
from pathlib import Path


scraper = cloudscraper.create_scraper()


def download_chesscom_games(username, output_path, max_months=None):

    print(f"\n[Chess.com] Downloading: {username}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Get archive list ───────────────────────────────────────────────────
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    r = scraper.get(archives_url)

    if r.status_code != 200:
        print(f"  Failed to get archives: {r.status_code}")
        print(f"  {r.text[:300]}")
        return False

    archives = r.json().get("archives", [])
    if not archives:
        print("  No archives found for this user.")
        return False

    print(f"  Found {len(archives)} archive months.")

    # Most recent months first (better data, more likely to have PGN)
    archives = list(reversed(archives))
    if max_months:
        archives = archives[:max_months]
        print(f"  Limiting to {max_months} most recent months.")

    # ── 2. Try /pgn endpoint first (most reliable) ────────────────────────────
    # Chess.com has a dedicated /pgn endpoint per month that returns
    # raw PGN text directly — much more reliable than the JSON games endpoint
    print("\n  Strategy A: trying /pgn endpoint per month...")
    pgn_chunks = _download_via_pgn_endpoint(archives)

    # ── 3. Fallback: JSON games endpoint ─────────────────────────────────────
    if not pgn_chunks:
        print("\n  Strategy A yielded nothing.")
        print("  Strategy B: trying JSON games endpoint with strict field check...")
        pgn_chunks = _download_via_json_endpoint(archives)

    # ── 4. Write output ───────────────────────────────────────────────────────
    if not pgn_chunks:
        print("\n  Both strategies failed — no PGN data found.")
        print("  Run debug_chesscom.py to inspect raw API response structure.")
        return False

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(pgn_chunks))

    total_games = len(pgn_chunks)
    total_bytes = output_path.stat().st_size
    print(f"\n  Saved {total_games} games → {output_path}")
    print(f"  File size: {total_bytes / 1024:.1f} KB")

    return True


def _download_via_pgn_endpoint(archives):
    """
    Chess.com exposes: GET /pub/player/{user}/games/{year}/{month}/pgn
    Returns raw multi-game PGN text directly.
    This is more reliable than extracting pgn field from JSON.
    """
    all_chunks = []
    failed = 0

    for archive_url in tqdm(archives, desc="  /pgn endpoint"):
        pgn_url = archive_url + "/pgn"
        r = scraper.get(pgn_url)

        if r.status_code == 200 and r.text.strip():
            # PGN starts with [Event " — basic sanity check
            if "[Event " in r.text:
                all_chunks.append(r.text.strip())
            else:
                print(f"\n  Warning: response from {pgn_url} doesn't look like PGN")
                print(f"  Preview: {r.text[:100]}")
                failed += 1
        elif r.status_code == 429:
            print(f"\n  Rate limited (429). Sleeping 10s...")
            time.sleep(10)
            r2 = scraper.get(pgn_url)
            if r2.status_code == 200 and "[Event " in r2.text:
                all_chunks.append(r2.text.strip())
        else:
            failed += 1

        time.sleep(0.3)  # polite delay

    print(f"  /pgn endpoint: {len(all_chunks)} months OK, {failed} failed")
    return all_chunks


def _download_via_json_endpoint(archives):
    """
    Fallback: fetch JSON games and extract 'pgn' field.
    Includes strict validation — empty/None pgn values are logged,
    not silently dropped.
    """
    all_pgns = []
    stats = {"total": 0, "has_pgn": 0, "empty_pgn": 0, "no_pgn_key": 0}

    for archive_url in tqdm(archives, desc="  JSON endpoint"):
        r = scraper.get(archive_url)

        if r.status_code == 429:
            print(f"\n  Rate limited. Sleeping 10s...")
            time.sleep(10)
            r = scraper.get(archive_url)

        if r.status_code != 200:
            continue

        games = r.json().get("games", [])
        stats["total"] += len(games)

        for game in games:
            if "pgn" not in game:
                stats["no_pgn_key"] += 1
                continue

            pgn = game["pgn"]

            if not pgn or not pgn.strip():
                stats["empty_pgn"] += 1
                continue

            if "[Event " not in pgn:
                # Not a valid PGN block
                stats["empty_pgn"] += 1
                continue

            stats["has_pgn"] += 1
            all_pgns.append(pgn.strip())

        time.sleep(0.3)

    print(f"\n  JSON endpoint stats:")
    print(f"    Total game objects:  {stats['total']}")
    print(f"    Had valid PGN:       {stats['has_pgn']}")
    print(f"    Empty/null PGN:      {stats['empty_pgn']}")
    print(f"    Missing 'pgn' key:   {stats['no_pgn_key']}")

    if stats["total"] > 0 and stats["has_pgn"] == 0:
        print("\n  !! All games are missing PGN data.")
        print("  This usually means the account's games are set to private,")
        print("  or Chess.com changed their API response format.")
        print("  Run debug_chesscom.py to see the raw game structure.")

    return all_pgns


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    players = {
        "magnus":   "magnuscarlsen",
        "hikaru":   "hikaru",
        "caruana":  "fabianocaruana",
        "firouzja": "firouzja2003",
    }

    output_dir = Path("dataset/raw_pgns/chesscom")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, username in players.items():
        out = output_dir / f"{name}.pgn"
        success = download_chesscom_games(
            username=username,
            output_path=out,
            max_months=24,   # last 2 years to start — increase once confirmed working
        )
        if not success:
            print(f"  FAILED for {name} — check debug_chesscom.py output")
        print()