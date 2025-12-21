import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.modules.draft_head import CandidateDraftHead
from mydecoding.modules.fusion import BeliefFusion
from mydecoding.modules.head2 import FusionDecoder
from mydecoding.models.outputs import DualDecoderOutput

class DualDecoderModel(nn.Module):
    """
    Wrap a frozen Llama-3B base model with head1 (draft) + fusion + head2 (belief-aware decoder).
    """

    def __init__(
        self,
        config: DualDecoderConfig,
        base_model: Optional[AutoModelForCausalLM] = None,
    ):
        super().__init__()
        self.config = config
        if base_model is None:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        # Freeze base parameters; we only train the heads.
        for p in self.base_model.parameters():
            p.requires_grad = False
        hidden_size = self.base_model.config.hidden_size
        vocab_size = self.base_model.config.vocab_size

        self.draft_head = CandidateDraftHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            hidden_proj=config.draft_hidden_size,
            num_mlp_layers=config.draft_num_layers,
            num_candidates=config.num_draft_candidates,
            num_heads=config.decoder_num_heads,
            dropout=config.decoder_dropout,
        )
        self.fusion = BeliefFusion(
            hidden_size=hidden_size,
            fusion_hidden_size=config.fusion_hidden_size,
            num_heads=config.fusion_num_heads,
            dropout=config.fusion_dropout,
        )
        self.head2 = FusionDecoder(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=config.decoder_num_layers,
            num_heads=config.decoder_num_heads,
            dropout=config.decoder_dropout,
        )

    @property
    def lm_head_weight(self):
        # Share vocab projection with the frozen base LM head.
        return self.base_model.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> DualDecoderOutput:
        with torch.no_grad():
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        base_hidden = base_out.hidden_states[-1]

        draft_logits, candidate_ids, belief, s_t1 = self.draft_head(
            base_hidden=base_hidden,
            attention_mask=attention_mask,
        )

        candidate_embeds = self.base_model.get_input_embeddings()(candidate_ids)
        
        fused_latent = self.fusion(
            candidate_embeddings=candidate_embeds,
            s_t1=s_t1,
            history_hidden=base_hidden,
        )
        fusion_logits, _ = self.head2(
            fused_latent=fused_latent,
            history_hidden=base_hidden,
            attention_mask=attention_mask,
        )

        head1_loss = None
        head2_loss = None
        total_loss = None
        if labels is not None:
            # Teacher forcing: compare head1/head2 predictions against the real next token.
            next_labels = labels[:, -1]
            head1_loss = F.cross_entropy(draft_logits, next_labels)
            head2_loss = F.cross_entropy(fusion_logits, next_labels)
            total_loss = (
                self.config.draft_loss_weight * head1_loss
                + self.config.fusion_loss_weight * head2_loss
            )

        return DualDecoderOutput(
            loss=total_loss,
            head1_loss=head1_loss,
            head2_loss=head2_loss,
            draft_logits=draft_logits,
            fusion_logits=fusion_logits,
            candidate_ids=candidate_ids,
            fused_latent=fused_latent,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ):
        """
        Speculative decoding loop:
        1) Base Llama encodes history.
        2) Head1 proposes k candidates + belief.
        3) Fusion builds z_{t+1}; head2 proposes draft token.
        4) Accept draft if base model assigns it probability above threshold,
           otherwise fall back to base greedy choice.
        """
        device = self.base_model.device
        input_ids = input_ids.to(device)
        attention_mask = (
            attention_mask.to(device) if attention_mask is not None else None
        )

        for _ in range(max_new_tokens):
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            base_hidden = base_out.hidden_states[-1]
            base_logits = base_out.logits[:, -1, :]
            base_probs = F.softmax(base_logits / max(temperature, 1e-4), dim=-1)

            draft_logits, candidate_ids, belief, s_t1 = self.draft_head(
                base_hidden=base_hidden,
                attention_mask=attention_mask,
            )
            
            candidate_embeds = self.base_model.get_input_embeddings()(candidate_ids)
            fused_latent = self.fusion(
                candidate_embeddings=candidate_embeds,
                s_t1=s_t1,
                history_hidden=base_hidden,
            )
            fusion_logits, _ = self.head2(
                fused_latent=fused_latent,
                history_hidden=base_hidden,               
                attention_mask=attention_mask,
            )
            fusion_probs = F.softmax(fusion_logits / max(temperature, 1e-4), dim=-1)
            draft_token = torch.multinomial(fusion_probs, num_samples=1)

            draft_score = base_probs.gather(-1, draft_token)
            if (draft_score > self.config.acceptance_threshold).all():
                next_token = draft_token
            else:
                # fall back to base greedy token
                next_token = torch.argmax(base_probs, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))],
                    dim=-1,
                )
            yield next_token

    @classmethod
    def from_config(cls, config: DualDecoderConfig):
        return cls(config=config)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        config = DualDecoderConfig.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, **kwargs
        )
        model = cls(config=config, base_model=base)
        model.load_state_dict(torch.load(f"{model_path}/dual_decoder.pt"), strict=False)
        return model

    def save_pretrained(self, output_dir: str):
        # Saving state dict + config; no HF push to keep minimal in this repo.
        output_dir = str(output_dir)
        torch.save(self.state_dict(), f"{output_dir}/dual_decoder.pt")
        self.config.save_pretrained(output_dir)
