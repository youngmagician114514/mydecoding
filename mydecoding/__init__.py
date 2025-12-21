"""
Belief-fused speculative decoding on top of a frozen Llama-3B base model.

The package mirrors Hydra's separation of a frozen base model plus lightweight
heads. Here we add a draft head (head1), a fusion module, and a belief-aware
decoder head (head2) that can be trained or used for inference.
"""

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.models.dual_decoder import DualDecoderModel

__all__ = ["DualDecoderConfig", "DualDecoderModel"]
