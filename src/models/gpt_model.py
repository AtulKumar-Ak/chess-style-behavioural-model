# src/models/gpt_model.py

import torch
import torch.nn as nn



class BoardEncoder(nn.Module):

    def __init__(self, embed_dim, dropout):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.ln = nn.LayerNorm(embed_dim)

        # Conservative initialization — encoder starts near-zero
        # so it doesn't disrupt the transformer early in training.
        # Gradually learns to contribute as loss improves.
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, board_tensor):
        """
        board_tensor: (batch, 768) float32
        returns:      (batch, 1, embed_dim)
        """
        x = self.encoder(board_tensor)
        x = self.ln(x)
        return x.unsqueeze(1)



class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        attn_output, _ = self.attention(
            x, x, x,
            attn_mask=causal_mask
        )

        return self.dropout(attn_output)



class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):

        super().__init__()

        self.ln1       = nn.LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(embed_dim, num_heads, dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff  = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x



class GPTBehaviorModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embed_dim,
        num_heads,
        num_layers,
        dropout,
        num_players
    ):

        super().__init__()

        self.token_embedding    = nn.Embedding(vocab_size,  embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.player_embedding   = nn.Embedding(num_players, embed_dim)

        # Board encoder with conservative initialization
        self.board_encoder = BoardEncoder(embed_dim, dropout)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, player_ids, board_state=None):
        """
        input_ids  : (batch, seq_len)   long
        player_ids : (batch,)           long
        board_state: (batch, 768)       float  — optional
                     12×64 binary board encoding.
                     If None, board conditioning is skipped.
        """
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len, device=input_ids.device
        ).unsqueeze(0)

        token_emb  = self.token_embedding(input_ids)
        pos_emb    = self.position_embedding(positions)
        player_emb = self.player_embedding(player_ids).unsqueeze(1)

        x = token_emb + pos_emb + player_emb

        if board_state is not None:
            board_emb = self.board_encoder(board_state)
            x = x + board_emb

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        return self.head(x)