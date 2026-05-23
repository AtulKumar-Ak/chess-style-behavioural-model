#src/dataset.py
import h5py
import torch

from torch.utils.data import Dataset


class ChessBehaviorDataset(Dataset):

    def __init__(self, h5_path):

        self.h5_path = h5_path

        self.h5f = h5py.File(
            h5_path,
            "r"
        )

        # ==========================================
        # DATASETS
        # ==========================================

        self.move_ids = self.h5f[
            "move_ids"
        ]

        self.player_ids = self.h5f[
            "player_ids"
        ]

        self.speed_ids = self.h5f[
            "speed_ids"
        ]

        self.avg_elo = self.h5f[
            "avg_elo"
        ]

    # ==============================================

    def __len__(self):

        return len(
            self.move_ids
        )

    # ==============================================

    def __getitem__(self, idx):

        # ==========================================
        # FULL MOVE WINDOW
        # ==========================================

        moves = torch.tensor(

            self.move_ids[idx],

            dtype=torch.long
        )

        # ==========================================
        # GPT SHIFTING
        # ==========================================

        input_ids = moves[:-1]

        labels = moves[1:]

        # ==========================================
        # METADATA
        # ==========================================

        player = torch.tensor(

            self.player_ids[idx],

            dtype=torch.long
        )

        speed = torch.tensor(

            self.speed_ids[idx],

            dtype=torch.long
        )

        elo = torch.tensor(

            self.avg_elo[idx],

            dtype=torch.float32
        )

        # ==========================================

        return {

            "input_ids":
                input_ids,

            "labels":
                labels,

            "player":
                player,

            "speed":
                speed,

            "elo":
                elo
        }

    # ==============================================

    def close(self):

        self.h5f.close()