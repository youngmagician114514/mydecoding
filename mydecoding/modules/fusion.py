import math
import torch
import torch.nn as nn


class BeliefFusion(nn.Module):
    """
    Fuse draft candidates with base history, conditioned on a *global* uncertainty/state vector s_{t+1}.

    Pipeline:
      candidates self-attn -> global gate from s_{t+1} -> candidate attention pooling
      -> history attention pooling (over last N tokens, query=s_{t+1})
      -> fuse pooled_candidate with pooled_history -> z_{t+1}.
    """

    def __init__(
        self,
        hidden_size: int,
        fusion_hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        history_pool_size: int = 32,   # ✅ 新增：history pooling window
    ):
        super().__init__()
        self.history_pool_size = int(history_pool_size)

        # Stage 2: candidate self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Stage 3: global conditioning gate from s_{t+1}
        self.state_to_gate = nn.Sequential(
            nn.Linear(hidden_size, fusion_hidden_size),
            nn.SiLU(),
            nn.Linear(fusion_hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        # Stage 4: candidate attention pooling scorer
        self.pool_scorer = nn.Sequential(
            nn.Linear(hidden_size, fusion_hidden_size),
            nn.SiLU(),
            nn.Linear(fusion_hidden_size, 1),
        )

        # ✅ NEW: history attention pooling (query = s_{t+1})
        self.hist_key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hist_query_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Stage 5: context fusion with base history summary
        self.context_proj = nn.Linear(hidden_size * 2, fusion_hidden_size)
        self.out_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(fusion_hidden_size, hidden_size),
        )

    def forward(
        self,
        candidate_embeddings: torch.Tensor,   # (B, K, H)
        s_t1: torch.Tensor,                  # (B, H)
        history_hidden: torch.Tensor,        # (B, T, H)
        candidate_padding_mask: torch.Tensor | None = None,  # (B, K), True=pad
    ) -> torch.Tensor:
        """
        Returns:
            z_{t+1}: (B, H)
        """
        param_dtype = next(self.parameters()).dtype
        if candidate_embeddings.dtype != param_dtype:
            candidate_embeddings = candidate_embeddings.to(param_dtype)
        if s_t1.dtype != param_dtype:
            s_t1 = s_t1.to(param_dtype)
        if history_hidden.dtype != param_dtype:
            history_hidden = history_hidden.to(param_dtype)
        
        # ---- Stage 2: candidate self-attention among K candidates ----
        attn_out, _ = self.self_attn(
            candidate_embeddings,
            candidate_embeddings,
            candidate_embeddings,
            key_padding_mask=candidate_padding_mask,
            need_weights=False,
        )  # (B, K, H)

        # ---- Stage 3: global gate from s_{t+1} ----
        gate = self.state_to_gate(s_t1).unsqueeze(1)  # (B, 1, H)
        gated = attn_out * gate                       # (B, K, H)

        # ---- Stage 4: attention pooling across candidates ----
        scores = self.pool_scorer(gated).squeeze(-1)  # (B, K)
        if candidate_padding_mask is not None:
            scores = scores.masked_fill(candidate_padding_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)          # (B, K)
        pooled_cand = (gated * alpha.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (B, 1, H)

        # ---- ✅ NEW: history attention pooling over last N tokens ----
        B, T, H = history_hidden.shape
        n = min(self.history_pool_size, T)
        hist = history_hidden[:, -n:, :]  # (B, T, H)

        # query = s_t1
        q = self.hist_query_proj(s_t1).unsqueeze(1)   # (B,1,H)
        k = self.hist_key_proj(hist)                  # (B,n,H)

        # scaled dot-product attention weights
        hist_scores = (k * q).sum(dim=-1) / math.sqrt(H)  # (B,n)
        hist_alpha = torch.softmax(hist_scores, dim=1)     # (B,n)
        pooled_hist = (hist * hist_alpha.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (B,1,H)

        # ---- Stage 5: fuse pooled_candidate with pooled_history ----
        fused_input = torch.cat([pooled_cand, pooled_hist], dim=-1)  # (B, 1, 2H)
        z_t1 = self.out_proj(self.context_proj(fused_input)).squeeze(1)  # (B, H)
        return z_t1
