import torch
import torch.nn as nn


# ======================================================
# CAUSAL SELF-ATTENTION
# ======================================================

class CausalSelfAttention(nn.Module):

    def __init__(

        self,

        embed_dim,

        num_heads,

        dropout
    ):

        super().__init__()

        self.attention = nn.MultiheadAttention(

            embed_dim=embed_dim,

            num_heads=num_heads,

            dropout=dropout,

            batch_first=True
        )

        self.dropout = nn.Dropout(
            dropout
        )

    # ==================================================

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        # ==============================================
        # CAUSAL MASK
        # ==============================================

        causal_mask = torch.triu(

            torch.ones(

                seq_len,
                seq_len,

                device=x.device
            ),

            diagonal=1
        ).bool()

        # ==============================================

        attn_output, _ = self.attention(

            x,
            x,
            x,

            attn_mask=causal_mask
        )

        attn_output = self.dropout(
            attn_output
        )

        return attn_output


# ======================================================
# TRANSFORMER BLOCK
# ======================================================

class TransformerBlock(nn.Module):

    def __init__(

        self,

        embed_dim,

        num_heads,

        dropout
    ):

        super().__init__()

        # ==============================================
        # ATTENTION
        # ==============================================

        self.ln1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = CausalSelfAttention(

            embed_dim=embed_dim,

            num_heads=num_heads,

            dropout=dropout
        )

        # ==============================================
        # FEEDFORWARD
        # ==============================================

        self.ln2 = nn.LayerNorm(
            embed_dim
        )

        self.ff = nn.Sequential(

            nn.Linear(
                embed_dim,
                4 * embed_dim
            ),

            nn.GELU(),

            nn.Linear(
                4 * embed_dim,
                embed_dim
            ),

            nn.Dropout(dropout)
        )

    # ==================================================

    def forward(self, x):

        # ==============================================
        # ATTENTION RESIDUAL
        # ==============================================

        x = x + self.attention(
            self.ln1(x)
        )

        # ==============================================
        # FEEDFORWARD RESIDUAL
        # ==============================================

        x = x + self.ff(
            self.ln2(x)
        )

        return x


# ======================================================
# GPT MODEL
# ======================================================

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

        # ==============================================
        # TOKEN EMBEDDINGS
        # ==============================================

        self.token_embedding = nn.Embedding(

            vocab_size,

            embed_dim
        )

        # ==============================================
        # POSITION EMBEDDINGS
        # ==============================================

        self.position_embedding = nn.Embedding(

            max_seq_len,

            embed_dim
        )

        # ==============================================
        # PLAYER EMBEDDINGS
        # ==============================================

        self.player_embedding = nn.Embedding(
        
            num_players,

            embed_dim
        )

        # ==============================================
        # DROPOUT
        # ==============================================

        self.dropout = nn.Dropout(
            dropout
        )

        # ==============================================
        # TRANSFORMER BLOCKS
        # ==============================================

        self.blocks = nn.ModuleList([

            TransformerBlock(

                embed_dim=embed_dim,

                num_heads=num_heads,

                dropout=dropout
            )

            for _ in range(num_layers)
        ])

        # ==============================================
        # FINAL LAYER NORM
        # ==============================================

        self.ln_f = nn.LayerNorm(
            embed_dim
        )

        # ==============================================
        # OUTPUT HEAD
        # ==============================================

        self.head = nn.Linear(

            embed_dim,

            vocab_size
        )

    # ==================================================

    def forward(self, input_ids,player_ids):

        batch_size, seq_len = input_ids.shape

        # ==============================================
        # POSITION IDS
        # ==============================================

        positions = torch.arange(

            seq_len,

            device=input_ids.device
        )

        positions = positions.unsqueeze(0)

        # ==============================================
        # EMBEDDINGS
        # ==============================================

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        player_embeddings = self.player_embedding(
            player_ids
        )

        player_embeddings = player_embeddings.unsqueeze(1)

        x = (
            token_embeddings
            +
            position_embeddings
            +
            player_embeddings
        )

        x = self.dropout(x)

        # ==============================================
        # TRANSFORMER BLOCKS
        # ==============================================

        for block in self.blocks:

            x = block(x)

        # ==============================================
        # FINAL LAYER NORM
        # ==============================================

        x = self.ln_f(x)

        # ==============================================
        # OUTPUT LOGITS
        # ==============================================

        logits = self.head(x)

        return logits