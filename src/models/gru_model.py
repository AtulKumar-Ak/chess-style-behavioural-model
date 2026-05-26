import torch
import torch.nn as nn


class GRUBehaviorModel(nn.Module):

    def __init__(

        self,

        vocab_size,

        embed_dim=128,

        hidden_dim=256,

        num_layers=2
    ):

        super().__init__()


        self.embedding = nn.Embedding(

            num_embeddings=vocab_size,

            embedding_dim=embed_dim
        )


        self.gru = nn.GRU(

            input_size=embed_dim,

            hidden_size=hidden_dim,

            num_layers=num_layers,

            batch_first=True
        )


        self.output = nn.Linear(

            hidden_dim,

            vocab_size
        )


    def forward(self, x):


        x = self.embedding(x)


        out, hidden = self.gru(x)


        logits = self.output(out)


        return logits