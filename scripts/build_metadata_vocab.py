from pathlib import Path
import json


TRAIN_TRAJ_DIR = Path(
    "dataset/trajectories/train"
)

OUTPUT_DIR = Path(
    "dataset/vocab"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


players = set()
speeds = set()


for jsonl_file in TRAIN_TRAJ_DIR.glob("*.jsonl"):

    print("\nPROCESSING:", jsonl_file.name)

    with open(
        jsonl_file,
        encoding="utf-8"
    ) as f:

        for line in f:

            record = json.loads(line)

            players.add(
                record["player"]
            )

            speeds.add(
                record["speed"]
            )


player_vocab = {

    "<UNK_PLAYER>": 0
}

for player in sorted(players):

    player_vocab[player] = len(
        player_vocab
    )


speed_vocab = {

    "<UNK_SPEED>": 0
}

for speed in sorted(speeds):

    speed_vocab[speed] = len(
        speed_vocab
    )


with open(
    OUTPUT_DIR / "player_vocab.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        player_vocab,
        f,
        indent=2
    )


with open(
    OUTPUT_DIR / "speed_vocab.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        speed_vocab,
        f,
        indent=2
    )


print("\n" + "=" * 60)
print("VOCAB BUILD COMPLETE")
print("=" * 60)

print(
    f"\nTOTAL PLAYERS : "
    f"{len(player_vocab)}"
)

print(
    f"TOTAL SPEEDS  : "
    f"{len(speed_vocab)}"
)