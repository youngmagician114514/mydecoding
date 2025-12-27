import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class DualDecoderOutput:
    # losses (None in infer_mode)
    loss: Optional[torch.Tensor]
    head1_loss: Optional[torch.Tensor]
    head2_loss: Optional[torch.Tensor]

    # logits
    draft_logits: torch.Tensor                 # head1 logits for phase1 (B,V)
    fusion_logits: torch.Tensor                # last head2 logits (B,V) or head1 logits
    head2_logits: torch.Tensor | None = None   # last head2 logits (B,V)
    head2_logits_all: torch.Tensor | None = None  # all head2 logits for phases 2..K, shape (B,K-1,V)

    # candidates / latent
    candidate_ids: torch.Tensor | None = None               # (B,K,max_k) per-phase candidate token ids
    fused_latent: torch.Tensor | None = None                # z1 (B,H)

    # optional teacher diagnostics (only when teacher_mode='base_greedy')
    base_greedy_tokens: torch.Tensor | None = None     # (B,K) tokens g1..gK
    teacher_head2_logits: torch.Tensor | None = None   # (B,K-1,V) teacher logits for phases 2..K
    teacher_phase1_logits: torch.Tensor | None = None  # (B,V) teacher logits for phase1
