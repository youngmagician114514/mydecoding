
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class FusionHeadPrefixTransformer(nn.Module):
    """
    fusion_head (Phase2..K):

    ✅ Input:
      - base_hidden_ctx: h_{1:t}      (B, Tctx, H)
      - z_prefix:        z_{1:i-1}    (B, Lz,  H)   (at least 1 at phase2)

    ✅ Output:
      - logits for next token         (B, V)
      - fusion_head_state             (B, H)  (last hidden state, used by belief_fusion to produce z_i)

    Design:
      - Concatenate [h_{1:t}, z_1..z_{i-1}] as a single causal sequence.
      - Add type embeddings (history vs latent) + learned positional embeddings.
      - Run a lightweight causal Transformer encoder (norm_first).
      - Use shared frozen base lm_head for vocabulary projection.

    This directly satisfies your requirement:
      head2 uses h_{1:t} AND z_1,z_2,... (instead of only z-prefix).
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        base_lm_head: Optional[nn.Module] = None,
        adapter_hidden_multiplier: int = 2,
        max_total_len: int = 2048,
    ):
        super().__init__()
        assert base_lm_head is not None, "FusionHeadPrefixTransformer requires base_lm_head for shared projection."

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # small adapter: stabilize + allow fusion_head to adjust to base hidden space
        mid = hidden_size * adapter_hidden_multiplier
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, mid, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, hidden_size, bias=True),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        self.out_norm = RMSNorm(hidden_size)

        # embeddings
        self.type_emb = nn.Embedding(2, hidden_size)  # 0=history, 1=latent
        self.pos_emb = nn.Embedding(max_total_len, hidden_size)

        # shared frozen lm_head
        self.lm_head = base_lm_head
        for p in self.lm_head.parameters():
            p.requires_grad = False

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_total_len = max_total_len

    def forward(
        self,
        base_hidden_ctx: torch.Tensor,                 # (B, Tctx, H)
        z_prefix: torch.Tensor,                        # (B, Lz, H)
        attention_mask_ctx: Optional[torch.Tensor] = None,  # (B, Tctx) 1=keep,0=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Tctx, H = base_hidden_ctx.shape
        
        param_dtype = next(self.parameters()).dtype
        if base_hidden_ctx.dtype != param_dtype:
            base_hidden_ctx = base_hidden_ctx.to(param_dtype)
        if z_prefix.dtype != param_dtype:
            z_prefix = z_prefix.to(param_dtype)
            
        _, Lz, _ = z_prefix.shape
        L = Tctx + Lz
        if L > self.max_total_len:
            raise ValueError(f"Total length {L} exceeds max_total_len={self.max_total_len}. Increase config.")

        # build sequence
        seq = torch.cat([base_hidden_ctx, z_prefix], dim=1)  # (B, L, H)

        # add type + pos embeddings
        device = seq.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B,L)
        type_ids = torch.cat(
            [
                torch.zeros((B, Tctx), device=device, dtype=torch.long),
                torch.ones((B, Lz), device=device, dtype=torch.long),
            ],
            dim=1,
        )
        seq = seq + self.pos_emb(pos_ids) + self.type_emb(type_ids)

        # causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1,
        )

        # key padding: only applies to history segment (ctx), latents are never padded
        src_key_padding_mask = None
        if attention_mask_ctx is not None:
            # attention_mask_ctx: 1 keep, 0 pad
            pad = (attention_mask_ctx == 0)
            if pad.ndim != 2 or pad.shape[0] != B or pad.shape[1] != Tctx:
                raise ValueError("attention_mask_ctx must be (B, Tctx)")
            # extend with False for latents
            pad2 = torch.cat([pad, torch.zeros((B, Lz), device=device, dtype=torch.bool)], dim=1)
            src_key_padding_mask = pad2  # (B,L)

        h = self.encoder(seq, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)  # (B,L,H)

        state = h[:, -1, :]  # last position corresponds to last latent z_{i-1}
        state = state + self.adapter(state)
        state = self.out_norm(state)

        lm_dtype = getattr(getattr(self.lm_head, "weight", None), "dtype", None)
        if lm_dtype is None:
            lm_dtype = next(self.lm_head.parameters()).dtype
        if state.dtype != lm_dtype:
            state = state.to(lm_dtype)

        logits = self.lm_head(state)  # (B,V)
        return logits, state
