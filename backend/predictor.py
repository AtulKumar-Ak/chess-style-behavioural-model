# backend/predictor.py

import json
from pathlib import Path
import sys
import chess
import torch
sys.path.append(str(Path(__file__).parent.parent))
from src.models.gpt_model import GPTBehaviorModel
from src.dataset import board_to_torch

# ======================================================
# PIECE MAP
# ======================================================

PIECE_MAP = {
    chess.PAWN:   "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK:   "R",
    chess.QUEEN:  "Q",
    chess.KING:   "K",
}

# ======================================================
# PREDICTOR
# ======================================================

class ChessPredictor:
    """
    Wraps the GPT behavior model for inference.
    Loaded once at FastAPI startup and reused for all requests.
    """

    def __init__(
        self,
        checkpoint_path: str,
        move_vocab_path: str,
        player_vocab_path: str,
        device: str = None,
        temperature: float = 0.7,
        top_k: int = 5,
    ):
        self.device = torch.device(
            device if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.temperature = temperature
        self.top_k       = top_k

        # --------------------------------------------------
        # LOAD VOCABS
        # --------------------------------------------------

        with open(move_vocab_path, encoding="utf-8") as f:
            self.move_vocab = json.load(f)

        with open(player_vocab_path, encoding="utf-8") as f:
            self.player_vocab = json.load(f)

        self.id_to_move = {
            v: k for k, v in self.move_vocab.items()
        }

        # --------------------------------------------------
        # LOAD MODEL — read sizes from checkpoint
        # --------------------------------------------------

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )

        vocab_size  = state_dict["token_embedding.weight"].shape[0]
        num_players = state_dict["player_embedding.weight"].shape[0]

        self.model = GPTBehaviorModel(
            vocab_size=vocab_size,
            max_seq_len=63,
            embed_dim=384,
            num_heads=6,
            num_layers=6,
            dropout=0.1,
            num_players=num_players,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.vocab_size  = vocab_size
        self.num_players = num_players

        print(f"[Predictor] loaded  vocab={vocab_size}  players={num_players}  device={self.device}")

    # ==================================================

    def get_available_players(self) -> list[str]:
        """Returns all non-special player names from vocab."""
        return [
            p for p in self.player_vocab
            if not p.startswith("<") and p != "opponent"
        ]

    # ==================================================

    def predict(
        self,
        moves: list[str],   # list of UCI strings e.g. ["e2e4", "e7e5"]
        player_name: str,
        top_k: int = None,
    ) -> list[dict]:
        """
        Given a list of UCI moves and a player name,
        returns the top-k predicted next moves with probabilities.

        Returns list of dicts:
          {
            "move": "e2e4",        # UCI
            "token": "P_e2e4",     # piece-aware token
            "probability": 0.342,  # normalised over legal moves
            "rank": 1
          }
        """
        top_k = top_k or self.top_k

        if player_name not in self.player_vocab:
            raise ValueError(
                f"Player '{player_name}' not in vocab. "
                f"Available: {self.get_available_players()}"
            )

        player_id = self.player_vocab[player_name]
        if player_id >= self.num_players:
            raise ValueError(f"Player id {player_id} out of range")

        player_tensor = torch.tensor(
            [player_id], dtype=torch.long, device=self.device
        )

        # --------------------------------------------------
        # REPLAY BOARD + BUILD TOKEN SEQUENCE
        # --------------------------------------------------

        board      = chess.Board()
        token_list = []

        for uci in moves:
            try:
                move  = chess.Move.from_uci(uci)
                piece = board.piece_at(move.from_square)
                if piece is None:
                    raise ValueError(f"No piece at from-square for move {uci}")
                token = f"{PIECE_MAP[piece.piece_type]}_{uci}"
                token_list.append(token)
                board.push(move)
            except Exception as e:
                raise ValueError(f"Invalid move '{uci}': {e}")

        # --------------------------------------------------
        # ENCODE CONTEXT
        # --------------------------------------------------

        input_ids = [
            self.move_vocab.get(t, self.move_vocab["<UNK>"])
            for t in token_list
        ]

        context = input_ids[-63:]

        if not context:
            # No moves yet — use a dummy start token
            context = [self.move_vocab.get("<PAD>", 0)]

        x = torch.tensor([context], dtype=torch.long, device=self.device)

        # --------------------------------------------------
        # BOARD STATE
        # --------------------------------------------------

        board_state = board_to_torch(board).unsqueeze(0).to(self.device)

        # --------------------------------------------------
        # FORWARD
        # --------------------------------------------------

        with torch.no_grad():
            logits = self.model(x, player_tensor, board_state=board_state)
            logits = logits[:, -1, :] / self.temperature
            probs  = torch.softmax(logits, dim=-1)

        # --------------------------------------------------
        # LEGAL MOVE FILTERING
        # --------------------------------------------------

        legal_token_ids = []
        legal_tokens    = []
        legal_ucis      = []

        for legal_move in board.legal_moves:
            uci          = legal_move.uci()
            piece        = board.piece_at(legal_move.from_square)
            piece_symbol = piece.symbol().upper()
            token        = f"{piece_symbol}_{uci}"

            if token in self.move_vocab and self.move_vocab[token] < self.vocab_size:
                legal_token_ids.append(self.move_vocab[token])
                legal_tokens.append(token)
                legal_ucis.append(uci)

        if not legal_token_ids:
            return []

        # --------------------------------------------------
        # RANK BY PROBABILITY
        # --------------------------------------------------

        legal_probs = probs[0, legal_token_ids]
        legal_probs = legal_probs / legal_probs.sum()

        k = min(top_k, len(legal_probs))
        top_probs, top_indices = torch.topk(legal_probs, k=k)
        top_probs = (top_probs / top_probs.sum()).cpu().tolist()

        results = []
        for rank, (rel_idx, prob) in enumerate(
            zip(top_indices.cpu().tolist(), top_probs), start=1
        ):
            results.append({
                "move":        legal_ucis[rel_idx],
                "token":       legal_tokens[rel_idx],
                "probability": round(prob, 4),
                "rank":        rank,
            })

        return results

    # ==================================================

    def moves_from_pgn(self, pgn_text: str) -> list[str]:
        """
        Parses a PGN string and returns a list of UCI move strings.
        Raises ValueError if parsing fails.
        """
        import chess.pgn
        import io

        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            raise ValueError("Could not parse PGN")

        board = chess.Board()
        ucis  = []

        for move in game.mainline_moves():
            ucis.append(move.uci())
            board.push(move)

        return ucis