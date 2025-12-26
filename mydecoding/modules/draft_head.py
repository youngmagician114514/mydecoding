import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """RMSNorm (like Qwen/LLaMA)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class CandidateDraftHead(nn.Module):
    """
    Hydra++-aligned draft head:

    1) one extra *causal self-attention block* (prefix layer)
    2) small MLP
    3) RMSNorm
    4) reuse frozen base lm_head
    """

    def __init__(
        self,
        hidden_size: int,
        num_candidates: int,
        num_heads: int,
        dropout: float,
        base_lm_head: nn.Module,
        ffn_multiplier: int = 4,
        num_mlp_layers: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_candidates = num_candidates

        # ✅ Prefix layer: causal self-attention only (NO cross-attn)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # 更接近现代 decoder block 行为（PreNorm）
        )
        self.prefix = nn.TransformerEncoder(enc_layer, num_layers=1)

        # ✅ MLP: H -> (H*ffn)*... -> H
        mlp_layers = []
        in_dim = hidden_size
        mid_dim = hidden_size * ffn_multiplier
        for _ in range(num_mlp_layers - 1):
            mlp_layers.append(nn.Linear(in_dim, mid_dim))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = mid_dim
        last = nn.Linear(in_dim, hidden_size)
        mlp_layers.append(last)
        self.mlp = nn.Sequential(*mlp_layers)

        # ✅ 关键稳定技巧：最后一层初始化为 0，让一开始 head 不会把 hidden 拉飞
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

        # ✅ RMSNorm before lm_head (对齐 base hidden 分布)
        self.out_norm = RMSNorm(hidden_size)

        # ✅ reuse frozen base lm_head
        self.lm_head = base_lm_head
        for p in self.lm_head.parameters():
            p.requires_grad = False

    def forward(
        self,
        base_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        base_hidden: (B, T, H)  last layer hidden from base model (ctx only)
        attention_mask: (B, T)  1=valid, 0=pad
        """
        B, T, H = base_hidden.shape
        assert H == self.hidden_size

        # ✅ causal mask: True means masked
        causal_mask = torch.triu(
            torch.ones(T, T, device=base_hidden.device, dtype=torch.bool),
            diagonal=1,
        )

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # True=pad

        # ✅ prefix self-attn
        h = self.prefix(
            base_hidden,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B,T,H)

        s_t1 = h[:, -1, :]  # (B,H)

        # ✅ MLP residual (非常重要：防止 collapse 到 uniform)
        delta = self.mlp(s_t1)     # (B,H)
        h_tilde = s_t1 + delta     # residual

        # ✅ RMSNorm对齐
        h_tilde = self.out_norm(h_tilde)

        # ✅ logits
        draft_logits = self.lm_head(h_tilde)  # (B,V)

        probs = F.softmax(draft_logits, dim=-1)
        belief_values, candidate_ids = torch.topk(probs, k=self.num_candidates, dim=-1)

        return draft_logits, candidate_ids, belief_values, h_tilde
