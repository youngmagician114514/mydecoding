"""
Belief-fused speculative decoding on top of a frozen Llama-3B base model.

The package mirrors Hydra's separation of a frozen base model plus lightweight
heads. Here we add a draft_head (phase1 proposer), a belief_fusion module, and a fusion_head
(phase2..K verifier-predictor) that can be trained or used for inference.
"""

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.models.dual_decoder import DualDecoderModel

__all__ = ["DualDecoderConfig", "DualDecoderModel"]
