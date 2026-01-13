
from transformers import PretrainedConfig


class DualDecoderConfig(PretrainedConfig):
    """
    Configuration for the dual-head speculative decoding model.

    Naming convention:
      - draft_head_*  : phase1 / proposer
      - fusion_head_* : phase2..K / verifier-predictor (conditioned on h_{1:t} + z_{1:i})

    Backward compatibility:
      - num_draft_candidates, num_fusion_candidates
      - decoder_num_layers/heads/dropout (legacy fusion_head)
    """

    model_type = "llama-belief-fusion"

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen2.5-3B",

        # ---- draft_head ----
        draft_head_num_candidates: int = 3,
        draft_head_num_heads: int | None = None,
        draft_hidden_size: int = 1024,   # kept for old scripts (not strictly used if base hidden matches)
        draft_num_layers: int = 4,
        draft_dropout: float = 0.1,

        # ---- belief_fusion ----
        fusion_hidden_size: int = 1024,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.1,

        # ---- fusion_head ----
        fusion_head_num_candidates: int = 10,
        fusion_head_num_layers: int = 2,
        fusion_head_num_heads: int = 8,
        fusion_head_dropout: float = 0.1,
        fusion_head_adapter_hidden_multiplier: int = 2,
        fusion_head_max_total_len: int = 2048,

        # ---- training / decoding ----
        max_speculative_steps: int = 2,
        draft_loss_weight: float = 1.0,
        fusion_loss_weight: float = 1.0,
        acceptance_threshold: float = 0.05,

        # ---- distillation knobs (optional) ----
        fusion_head_teacher_topk: int = 10,
        fusion_head_soft_ce_weight: float = 1.0,
        fusion_head_kl_weight: float = 0.3,

        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path

        # canonical
        self.draft_head_num_candidates = draft_head_num_candidates
        self.draft_head_num_heads = draft_head_num_heads
        self.draft_hidden_size = draft_hidden_size
        self.draft_num_layers = draft_num_layers
        self.draft_dropout = draft_dropout

        self.fusion_hidden_size = fusion_hidden_size
        self.fusion_num_heads = fusion_num_heads
        self.fusion_dropout = fusion_dropout

        self.fusion_head_num_candidates = fusion_head_num_candidates
        self.fusion_head_num_layers = fusion_head_num_layers
        self.fusion_head_num_heads = fusion_head_num_heads
        self.fusion_head_dropout = fusion_head_dropout
        self.fusion_head_adapter_hidden_multiplier = fusion_head_adapter_hidden_multiplier
        self.fusion_head_max_total_len = fusion_head_max_total_len

        self.max_speculative_steps = max_speculative_steps
        self.draft_loss_weight = draft_loss_weight
        self.fusion_loss_weight = fusion_loss_weight
        self.acceptance_threshold = acceptance_threshold

        self.fusion_head_teacher_topk = fusion_head_teacher_topk
        self.fusion_head_soft_ce_weight = fusion_head_soft_ce_weight
        self.fusion_head_kl_weight = fusion_head_kl_weight

        # ---- legacy aliases ----
        # old names used across previous scripts
        self.num_draft_candidates = self.draft_head_num_candidates
        self.num_fusion_candidates = self.fusion_head_num_candidates

        self.decoder_num_layers = self.fusion_head_num_layers
        self.decoder_num_heads = self.fusion_head_num_heads
        self.decoder_dropout = self.fusion_head_dropout

        # old head2 knobs
        self.head2_teacher_topk = self.fusion_head_teacher_topk
        self.head2_soft_ce_weight = self.fusion_head_soft_ce_weight
        self.head2_kl_weight = self.fusion_head_kl_weight
