import torch
import torch.nn as nn
from typing import Optional

class FusionDecoder(nn.Module):
    """
    Head2: a tiny Transformer decoder conditioned on fused latent z_{t+1}.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        fused_latent: torch.Tensor,
        history_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            fused_latent: (batch, hidden)
            history_hidden: (batch, seq, hidden)
            lm_head_weight: shared LM head weight from the base model.
            attention_mask: (batch, seq)
        Returns:
            logits: (batch, vocab)
            decoder_state: (batch, hidden)
        """
        tgt = fused_latent.unsqueeze(1)  # (batch, 1, hidden)
        memory = history_hidden
        # Use attention mask to ignore padding positions in memory
        memory_key_padding_mask = None
        if attention_mask is not None:
            memory_key_padding_mask = attention_mask == 0
        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        decoded_last = decoded[:, -1, :]
        logits = self.out_proj(decoded_last)
        return logits, decoded_last