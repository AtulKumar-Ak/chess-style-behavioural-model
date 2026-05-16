import chess.pgn


def iter_games(pgn_path):

    with open(pgn_path, encoding="utf-8") as f:

        while True:

            game = chess.pgn.read_game(f)

            if game is None:
                break

            yield game