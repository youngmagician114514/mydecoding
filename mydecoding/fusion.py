import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefFusion(nn.Module):
    """
    Fuse draft candidates (tokens + belief/confidence) with the base hidden state.

    Pipeline mirrors the PDF: token embedding -> token self-attention -> belief
    conditioning -> context fusion (pooling with history).
    """

    def __init__(
        self,
        hidden_size: int,
        fusion_hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.belief_mlp = nn.Sequential(
            nn.Linear(1, fusion_hidden_size),
            nn.SiLU(),
            nn.Linear(fusion_hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.context_proj = nn.Linear(hidden_size * 2, fusion_hidden_size)
        self.out_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(fusion_hidden_size, hidden_size),
        )

    def forward(
        self,
        candidate_embeddings: torch.Tensor,
        belief: torch.Tensor,
        history_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            candidate_embeddings: (batch, k, hidden)
            belief: (batch, k) confidence for each candidate.
            history_hidden: (batch, seq, hidden) base hidden states (prefill).
        Returns:
            fused: (batch, hidden) fused latent z_{t+1}.
        """
        # Stage 2: token self-attention among candidates
        attn_out, _ = self.self_attn(candidate_embeddings, candidate_embeddings, candidate_embeddings)

        # Stage 3: belief conditioning as a gate on candidate channels
        belief_gate = self.belief_mlp(belief.unsqueeze(-1))  # (b, k, hidden)
        gated = attn_out * belief_gate

        # Context fusion: attend to history and pool candidates
        # Take the last hidden state as a compact history summary.
        history_summary = history_hidden[:, -1:, :]  # (b, 1, hidden)
        # Concatenate pooled candidate signal with history summary.
        pooled = gated.mean(dim=1, keepdim=True)  # (b, 1, hidden)
        fused_input = torch.cat([pooled, history_summary], dim=-1)
        fused = self.out_proj(self.context_proj(fused_input)).squeeze(1)
        return fused
