import torch
import torch.nn as nn


class BeliefFusion(nn.Module):
    """
    Fuse draft candidates with base history, conditioned on a *global* uncertainty/state vector s_{t+1}.

    Here s_{t+1} is defined as the last hidden state from decoder1/head1 (global, one per step).
    Pipeline:
      candidates self-attn -> global gate from s_{t+1} -> attention pooling -> fuse with history_last -> z_{t+1}.
    """

    def __init__(
        self,
        hidden_size: int,
        fusion_hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Stage 2: candidate self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Stage 3: global conditioning gate from s_{t+1} (decoder1 last hidden state)
        # Input: s_{t+1} in R^H  -> Output: gate in (0,1)^H
        self.state_to_gate = nn.Sequential(
            nn.Linear(hidden_size, fusion_hidden_size),
            nn.SiLU(),
            nn.Linear(fusion_hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        # Attention pooling scorer: produce one scalar score per candidate
        self.pool_scorer = nn.Sequential(
            nn.Linear(hidden_size, fusion_hidden_size),
            nn.SiLU(),
            nn.Linear(fusion_hidden_size, 1),
        )

        # Stage 5: context fusion with base history_last
        self.context_proj = nn.Linear(hidden_size * 2, fusion_hidden_size)
        self.out_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(fusion_hidden_size, hidden_size),
        )

    def forward(
        self,
        candidate_embeddings: torch.Tensor,   # (B, K, H)
        s_t1: torch.Tensor,                  # (B, H)  decoder1 last hidden state = s_{t+1}
        history_hidden: torch.Tensor,        # (B, T, H) base hidden states (prefill)
        candidate_padding_mask: torch.Tensor | None = None,  # (B, K), True=invalid/pad
    ) -> torch.Tensor:
        """
        Returns:
            z_{t+1}: (B, H)
        """
        # Stage 2: candidate self-attention among K candidates
        attn_out, _ = self.self_attn(
            candidate_embeddings,
            candidate_embeddings,
            candidate_embeddings,
            key_padding_mask=candidate_padding_mask,
            need_weights=False,
        )  # (B, K, H)

        # Stage 3: global belief/state conditioning
        gate = self.state_to_gate(s_t1).unsqueeze(1)  # (B, 1, H)
        gated = attn_out * gate                       # (B, K, H) broadcast over K

        # Stage 4: attention pooling across candidates
        scores = self.pool_scorer(gated).squeeze(-1)  # (B, K)
        if candidate_padding_mask is not None:
            scores = scores.masked_fill(candidate_padding_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)          # (B, K)
        pooled = (gated * alpha.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (B, 1, H)

        # Stage 5: fuse with base history summary (last token)
        history_summary = history_hidden[:, -1:, :]   # (B, 1, H)
        fused_input = torch.cat([pooled, history_summary], dim=-1)  # (B, 1, 2H)
        z_t1 = self.out_proj(self.context_proj(fused_input)).squeeze(1)  # (B, H)
        return z_t1
