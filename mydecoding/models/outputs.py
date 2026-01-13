
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class DualDecoderOutput:
    """
    Canonical names (recommended):
      - draft_head_*  : phase1 / proposal head
      - fusion_head_* : phase2..K / verifier-predictor head (conditioned on h_{1:t} + z_{1:i})

    Backward-compatible aliases are provided as @property for older scripts:
      head1_loss, head2_loss, draft_logits, fusion_logits, head2_logits, head2_logits_all
    """

    # losses (None in infer_mode)
    loss: Optional[torch.Tensor]
    draft_head_loss: Optional[torch.Tensor]
    fusion_head_loss: Optional[torch.Tensor]

    # logits
    draft_head_logits: torch.Tensor                 # phase1 logits (B,V)
    fusion_head_logits: torch.Tensor                # last fusion_head logits (B,V)
    fusion_head_logits_last: torch.Tensor | None = None  # alias-ish, optional
    fusion_head_logits_all: torch.Tensor | None = None   # (B,K-1,V) for phases 2..K

    # candidates / latent
    candidate_ids: torch.Tensor | None = None       # (B,K,max_k) per-phase candidate token ids
    fused_latent: torch.Tensor | None = None        # z1 (B,H)

    # optional teacher diagnostics (only when teacher_mode='base_greedy')
    base_greedy_tokens: torch.Tensor | None = None      # (B,K) tokens g1..gK
    teacher_fusion_head_logits: torch.Tensor | None = None   # (B,K-1,V) teacher logits for phases 2..K
    teacher_draft_head_logits: torch.Tensor | None = None    # (B,V) teacher logits for phase1

    # ---- Backward-compatible aliases ----
    @property
    def head1_loss(self):
        return self.draft_head_loss

    @property
    def head2_loss(self):
        return self.fusion_head_loss

    @property
    def draft_logits(self):
        return self.draft_head_logits

    @property
    def fusion_logits(self):
        return self.fusion_head_logits

    @property
    def head2_logits(self):
        return self.fusion_head_logits_last if self.fusion_head_logits_last is not None else self.fusion_head_logits

    @property
    def head2_logits_all(self):
        return self.fusion_head_logits_all

    @property
    def teacher_head2_logits(self):
        return self.teacher_fusion_head_logits

    @property
    def teacher_phase1_logits(self):
        return self.teacher_draft_head_logits
