# src/dataset.py

import chess
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset



PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def board_to_tensor(board):
    """
    Returns a (768,) float32 numpy array.
    Called during HDF5 building to precompute board states.
    """
    planes = np.zeros((12, 64), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            planes[plane, square] = 1.0

    return planes.flatten()   # (768,)


def board_to_torch(board):
    """
    Returns a (768,) float32 torch tensor.
    Called during generation (inference) where we don't
    want numpy — just compute directly into torch.
    """
    return torch.from_numpy(board_to_tensor(board))



class ChessBehaviorDataset(Dataset):

    def __init__(self, h5_path):

        self.h5_path = h5_path
        self.h5f     = h5py.File(h5_path, "r")

        self.move_ids    = self.h5f["move_ids"]
        self.player_ids  = self.h5f["player_ids"]
        self.speed_ids   = self.h5f["speed_ids"]
        self.avg_elo     = self.h5f["avg_elo"]

        # Board states are optional — older HDF5 files without
        # board_states will work without board conditioning.
        self.board_states = (
            self.h5f["board_states"]
            if "board_states" in self.h5f
            else None
        )

    def __len__(self):
        return len(self.move_ids)

    def __getitem__(self, idx):

        moves = torch.tensor(
            self.move_ids[idx], dtype=torch.long
        )

        # GPT-style shift: predict each token from previous
        input_ids = moves[:-1]
        labels    = moves[1:]

        player = torch.tensor(self.player_ids[idx], dtype=torch.long)
        speed  = torch.tensor(self.speed_ids[idx],  dtype=torch.long)
        elo    = torch.tensor(self.avg_elo[idx],     dtype=torch.float32)

        item = {
            "input_ids": input_ids,
            "labels":    labels,
            "player":    player,
            "speed":     speed,
            "elo":       elo,
        }

        # Include board state if available in HDF5
        if self.board_states is not None:
            item["board_state"] = torch.tensor(
                self.board_states[idx], dtype=torch.float32
            )

        return item

    def close(self):
        self.h5f.close()