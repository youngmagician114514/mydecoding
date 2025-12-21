import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class DualDecoderOutput:
    loss: Optional[torch.Tensor]
    head1_loss: Optional[torch.Tensor]
    head2_loss: Optional[torch.Tensor]
    draft_logits: torch.Tensor
    fusion_logits: torch.Tensor
    candidate_ids: torch.Tensor
    fused_latent: torch.Tensor