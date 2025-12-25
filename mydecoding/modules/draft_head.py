import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CandidateDraftHead(nn.Module):
    """Hydra++ / Qwen-style draft head (head1).

    1. 在 base_model 的最后隐状态 h_{1:t} 上再加一层 prefix self-attention decoder
       （结构类似 Qwen block：多头注意力 + FFN），得到 s_{t+1}。

    2. 再用一个小 MLP 把 s_{t+1} -> \tilde{h}_{t+1}（维度仍然是 hidden_size）。

    3. 最后复用 *冻结的* base_model.lm_head 把 \tilde{h}_{t+1} 投到词表上。
       这样 head1 的参数量主要来自 prefix decoder 和小 MLP，不再新建一个巨大的 vocab 线性层，
       也更贴近 Hydra++ 的设计。
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

        # 1) prefix self-attention decoder layer
        self.prefix_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.prefix_decoder = nn.TransformerDecoder(
            self.prefix_decoder_layer,
            num_layers=1,
        )

        # 2) small MLP: hidden_size -> ... -> hidden_size
        mlp_layers = []
        in_dim = hidden_size
        hidden_dim = hidden_size * ffn_multiplier
        for _ in range(num_mlp_layers - 1):
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(in_dim, hidden_size))
        self.mlp = nn.Sequential(*mlp_layers)

        # 3) reuse frozen base lm_head (hidden_size -> vocab)
        self.lm_head = base_lm_head
        for p in self.lm_head.parameters():
            p.requires_grad = False

        self.num_candidates = num_candidates

    def forward(
        self,
        base_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # base_hidden: (B, T, H)
        B, T, H = base_hidden.shape

        tgt = base_hidden
        memory = base_hidden

        memory_key_padding_mask = None
        if attention_mask is not None:
            memory_key_padding_mask = (attention_mask == 0)  # (B, T) True=pad

        # 因为我们用的是 TransformerDecoder，还是加一个 causal mask 保持自回归
        tgt_mask = torch.triu(
            torch.ones(T, T, device=base_hidden.device, dtype=torch.bool),
            diagonal=1,
        )

        decoded = self.prefix_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, T, H)

        # 取最后一个位置作为 s_{t+1}
        s_t1 = decoded[:, -1, :]  # (B, H)

        # 小 MLP 调整成 \tilde{h}_{t+1}
        h_tilde = self.mlp(s_t1)  # (B, H)

        # 通过 base 的 lm_head 得到对整个词表的 logits
        draft_logits = self.lm_head(h_tilde)  # (B, V)

        probs = F.softmax(draft_logits, dim=-1)
        belief_values, candidate_ids = torch.topk(
            probs, k=self.num_candidates, dim=-1
        )  # (B, K)

        # 返回的 state 用 h_tilde（也就是 head1 的“自己的隐藏状态”）
        return draft_logits, candidate_ids, belief_values, h_tilde
