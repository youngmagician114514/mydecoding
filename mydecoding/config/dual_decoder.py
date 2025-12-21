from transformers import PretrainedConfig


class DualDecoderConfig(PretrainedConfig):
    """
    Configuration for the dual-head speculative decoding model.

    The design follows the PDF: head1 drafts high-probability candidates,
    a fusion block combines candidate embeddings with belief/confidence and
    base hidden states, and head2 autoregressively scores the fused latent.
    """

    model_type = "llama-belief-fusion"

    def __init__(
        self,
        base_model_name_or_path: str = "meta-llama/Meta-Llama-3-8B",
        num_draft_candidates: int = 3,
        draft_hidden_size: int = 1024,
        draft_num_layers: int = 2,
        fusion_hidden_size: int = 1024,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.1,
        decoder_num_layers: int = 2,
        decoder_num_heads: int = 8,
        decoder_dropout: float = 0.1,
        draft_loss_weight: float = 1.0,
        fusion_loss_weight: float = 1.0,
        acceptance_threshold: float = 0.05,
        max_speculative_steps: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.num_draft_candidates = num_draft_candidates
        self.draft_hidden_size = draft_hidden_size
        self.draft_num_layers = draft_num_layers
        self.fusion_hidden_size = fusion_hidden_size
        self.fusion_num_heads = fusion_num_heads
        self.fusion_dropout = fusion_dropout
        self.decoder_num_layers = decoder_num_layers
        self.decoder_num_heads = decoder_num_heads
        self.decoder_dropout = decoder_dropout
        self.draft_loss_weight = draft_loss_weight
        self.fusion_loss_weight = fusion_loss_weight
        self.acceptance_threshold = acceptance_threshold
        self.max_speculative_steps = max_speculative_steps
        
