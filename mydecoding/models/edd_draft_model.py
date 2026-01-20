# edd_draft_model.py
# EDD draft model (only draft part; no tree/PCT)

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@dataclass
class EDDEncoding:
    """Outputs from frozen target encoder."""
    teacher_logits: torch.Tensor      # (B, T, V)
    enc_hidden: torch.Tensor          # (B, T, H)


class EDDDraftModel(nn.Module):
    def __init__(self, draft_lm: nn.Module):
        super().__init__()
        self.draft_lm = draft_lm  # AutoModelForCausalLM

    @classmethod
    def from_target(
        cls,
        target_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> Tuple["EDDDraftModel", nn.Module]:
        # Load target (frozen)
        target = AutoModelForCausalLM.from_pretrained(
            target_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)

        # Build draft config: only 1 layer
        draft_cfg = copy.deepcopy(target.config)
        if hasattr(draft_cfg, "num_hidden_layers"):
            draft_cfg.num_hidden_layers = 1
        elif hasattr(draft_cfg, "n_layer"):
            draft_cfg.n_layer = 1
        else:
            raise ValueError("Unknown config field for num layers; please adapt for your model family.")

        draft = AutoModelForCausalLM.from_config(draft_cfg, trust_remote_code=trust_remote_code)

        # Init embeddings + lm_head from target
        with torch.no_grad():
            draft.get_input_embeddings().weight.copy_(target.get_input_embeddings().weight)
            if hasattr(draft, "lm_head") and hasattr(target, "lm_head"):
                draft.lm_head.weight.copy_(target.lm_head.weight)
                if getattr(target.lm_head, "bias", None) is not None and getattr(draft.lm_head, "bias", None) is not None:
                    draft.lm_head.bias.copy_(target.lm_head.bias)

        return cls(draft), target

    @torch.no_grad()
    def encode_with_target(
        self,
        target_model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        enc_layer_index: int = -1,
    ) -> EDDEncoding:
        out = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("Target model did not return hidden_states. Ensure output_hidden_states=True.")
        enc_hidden = hidden_states[enc_layer_index]     # (B, T, H)
        teacher_logits = out.logits                     # (B, T, V)
        return EDDEncoding(teacher_logits=teacher_logits, enc_hidden=enc_hidden)

    def loss_one_block(
        self,
        input_ids: torch.Tensor,          # (B, T)
        attention_mask: torch.Tensor,     # (B, T)
        teacher_logits: torch.Tensor,     # (B, T, V)
        enc_hidden: torch.Tensor,         # (B, T, H)
        s: int,
        e: int,
        topk: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (loss_sum, weight_sum) for a single block [s:e).
        Return:
          - loss_sum: scalar tensor w/ grad
          - weight_sum: scalar tensor (no grad)
        """
        assert input_ids.dim() == 2
        B, T = input_ids.shape
        device = input_ids.device

        if s >= T - 1:
            # no next-token label
            z = torch.zeros((), device=device, dtype=torch.float32)
            return z, z + 1.0

        prompt_len = s
        tok_ids = input_ids[:, s:e]                 # (B, Lb)
        tok_mask = attention_mask[:, s:e]           # (B, Lb)

        if prompt_len > 0:
            prompt_embeds = enc_hidden[:, :prompt_len, :]    # (B, s, H)
            prompt_mask = attention_mask[:, :prompt_len]     # (B, s)
        else:
            prompt_embeds = enc_hidden[:, :0, :]
            prompt_mask = attention_mask[:, :0]

        tok_embeds = self.draft_lm.get_input_embeddings()(tok_ids)  # (B, Lb, H)
        inputs_embeds = torch.cat([prompt_embeds, tok_embeds], dim=1)  # (B, s+Lb, H)
        attn_mask = torch.cat([prompt_mask, tok_mask], dim=1)          # (B, s+Lb)

        # only run transformer body
        base = getattr(self.draft_lm, "model", None) or getattr(self.draft_lm, "transformer", None)
        if base is None:
            raise RuntimeError("Cannot find base model module: expected draft_lm.model or draft_lm.transformer")

        out = base(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state                      # (B, s+Lb, H)
        hidden_block = hidden[:, prompt_len:, :]            # (B, Lb, H)
        logits_block = self.draft_lm.lm_head(hidden_block)  # (B, Lb, V)

        # teacher block: (B, Lb, V)  (still huge overall, but only slice is used here)
        teacher_block = teacher_logits[:, s:e, :]

        # valid mask
        cur_valid = attention_mask[:, s:e]  # (B, Lb)
        if e < T:
            nxt_valid = attention_mask[:, s + 1 : e + 1]    # (B, Lb)
            valid = cur_valid * nxt_valid
        else:
            valid = cur_valid.clone()
            valid[:, -1] = 0

        # pos_has_label
        pos = torch.arange(s, e, device=device)
        pos_has_label = (pos < (T - 1)).view(1, -1).to(valid.dtype)
        valid = valid * pos_has_label
        weight = valid.float()
        weight_sum = weight.sum().clamp_min(1.0)

        # ---- topk distill WITHOUT full-vocab log_softmax ----
        # topk over teacher in its native dtype (avoid teacher_block.float() full copy)
        t_topv, t_topi = torch.topk(teacher_block, k=topk, dim=-1)   # (B, Lb, K)
        t_probs = F.softmax(t_topv.float(), dim=-1)                  # (B, Lb, K) small

        # student selected logits
        s_sel = logits_block.gather(-1, t_topi)                      # (B, Lb, K)

        # log P_s(i) = s_i - logsumexp(all_vocab)
        denom = torch.logsumexp(logits_block.float(), dim=-1, keepdim=True)  # (B, Lb, 1)
        s_logp = s_sel.float() - denom                                       # (B, Lb, K)

        # CE over teacher topk support
        ce = -(t_probs * s_logp).sum(dim=-1)                          # (B, Lb)

        loss_sum = (ce * weight).sum()
        return loss_sum, weight_sum

    # keep for compatibility (still ok for debugging/small T, but NOT memory-friendly)
    def kl_loss_dual_block(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        teacher_logits: torch.Tensor,
        enc_hidden: torch.Tensor,
        block_len: int,
    ) -> torch.Tensor:
        # NOTE: this version will keep graphs for all blocks, so it can OOM for long T.
        B, T = input_ids.shape
        device = input_ids.device
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_weight = torch.zeros((), device=device, dtype=torch.float32)

        for s in range(0, T, block_len):
            e = min(s + block_len, T)
            loss_sum, weight_sum = self.loss_one_block(
                input_ids=input_ids,
                attention_mask=attention_mask,
                teacher_logits=teacher_logits,
                enc_hidden=enc_hidden,
                s=s, e=e,
            )
            total_loss = total_loss + loss_sum
            total_weight = total_weight + weight_sum

        return total_loss / total_weight.clamp_min(1.0)

    def save_pretrained(self, output_dir: str):
        self.draft_lm.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_dir: str, trust_remote_code: bool = True) -> "EDDDraftModel":
        draft = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        return cls(draft)
