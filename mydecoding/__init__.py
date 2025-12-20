"""
Belief-fused speculative decoding on top of a frozen Llama-3B base model.

The package mirrors Hydra's separation of a frozen base model plus lightweight
heads. Here we add a draft head (head1), a fusion module, and a belief-aware
decoder head (head2) that can be trained or used for inference.
"""

from .config import DualDecoderConfig
from .model import DualDecoderModel
*** End Patch**"**"}},{
