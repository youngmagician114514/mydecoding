# edd_draft_model.py
# EDD draft model (only draft part; no tree/PCT)
# - draft uses same structure as target but only 1 Transformer layer
# - use target hidden states as "soft prompt" (inputs_embeds prefix)
# - dual-block training: split into blocks; for block j, prompt = H[:j*L], tokens = T[j*L:(j+1)*L]
# - loss: KL(Pt || Pd)

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class EDDEncoding:
    """Outputs from frozen target encoder."""
    teacher_logits: torch.Tensor      # (B, T, V) logits predicting next token at each position
    enc_hidden: torch.Tensor          # (B, T, H) selected hidden states (e.g., 4th-to-last layer)


class EDDDraftModel(nn.Module):
    """
    Trainable EDD draft model:
      - same architecture family as target model
      - but num_hidden_layers = 1
      - embeddings + lm_head initialized by copying from target model
    """
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
        """
        Returns (edd_draft_model, frozen_target_model).

        The target model is loaded as a frozen encoder to provide:
          - teacher logits Pt
          - encoding hidden states H (we'll pick layer -4 by default in training script)
        """
        # Load target
        target = AutoModelForCausalLM.from_pretrained(
            target_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)

        # Build draft config: same as target but only 1 layer
        draft_cfg = copy.deepcopy(target.config)
        if hasattr(draft_cfg, "num_hidden_layers"):
            draft_cfg.num_hidden_layers = 1
        elif hasattr(draft_cfg, "n_layer"):
            draft_cfg.n_layer = 1
        else:
            raise ValueError("Unknown config field for num layers; please adapt for your model family.")

        draft = AutoModelForCausalLM.from_config(draft_cfg, trust_remote_code=trust_remote_code)

        # Copy embeddings + LM head weights for initialization
        with torch.no_grad():
            draft.get_input_embeddings().weight.copy_(target.get_input_embeddings().weight)
            # Some models tie weights; we still copy explicitly
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
        """
        Runs frozen target model once:
          - teacher logits
          - selected hidden states as encoding result
        """
        out = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out.hidden_states  # tuple: len = num_layers+1
        if hidden_states is None:
            raise RuntimeError("Target model did not return hidden_states. Ensure output_hidden_states=True is supported.")
        enc_hidden = hidden_states[enc_layer_index]  # (B, T, H)
        teacher_logits = out.logits  # (B, T, V)
        return EDDEncoding(teacher_logits=teacher_logits, enc_hidden=enc_hidden)

    def kl_loss_dual_block(
        self,
        input_ids: torch.Tensor,          # (B, T)
        attention_mask: torch.Tensor,     # (B, T)
        teacher_logits: torch.Tensor,     # (B, T, V)
        enc_hidden: torch.Tensor,         # (B, T, H)
        block_len: int,
    ) -> torch.Tensor:
        """
        Dual-block mask training (efficient implementation):
          - Split T into blocks of length L.
          - For block starting at s=j*L:
                prompt = H[:s]
                tokens = T[s:s+L]
            Run one forward on concatenated embeds [prompt; tokens] with causal mask (standard).
            Compute KL on logits for positions s..e-1 (predicting next token).
        """
        assert input_ids.dim() == 2
        B, T = input_ids.shape
        device = input_ids.device

        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_weight = torch.zeros((), device=device, dtype=torch.float32)

        # We need next-token targets => last position has no label
        max_pos_for_loss = T - 1
        if max_pos_for_loss <= 0:
            return total_loss

        for s in range(0, T, block_len):
            e = min(s + block_len, T)
            if s >= max_pos_for_loss:
                break  # no next-token label

            # prompt length equals s (all previous positions)
            prompt_len = s

            # token block
            tok_ids = input_ids[:, s:e]  # (B, Lb)
            tok_mask = attention_mask[:, s:e]  # (B, Lb)

            # Build embeddings: [H[:s], embed(tok_ids)]
            if prompt_len > 0:
                prompt_embeds = enc_hidden[:, :prompt_len, :]  # (B, s, H)
                prompt_mask = attention_mask[:, :prompt_len]   # (B, s)
            else:
                prompt_embeds = enc_hidden[:, :0, :]
                prompt_mask = attention_mask[:, :0]

            tok_embeds = self.draft_lm.get_input_embeddings()(tok_ids)  # (B, Lb, H)
            inputs_embeds = torch.cat([prompt_embeds, tok_embeds], dim=1)  # (B, s+Lb, H)
            attn_mask = torch.cat([prompt_mask, tok_mask], dim=1)          # (B, s+Lb)

            # Forward draft
            out = self.draft_lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                use_cache=False,
            )
            logits_block = out.logits[:, prompt_len:, :]  # (B, Lb, V) aligns to positions s..e-1

            # Align teacher logits for same positions
            # teacher_logits[:, pos, :] predicts token at pos+1
            teacher_block = teacher_logits[:, s:e, :]  # (B, Lb, V)

            # Valid positions: need current token + next token not padded, and pos <= T-2
            # For position pos, label is input_ids[:, pos+1], so require attention_mask[pos] & attention_mask[pos+1]
            pos = torch.arange(s, e, device=device)  # (Lb,)
            pos_has_label = pos < (T - 1)  # (Lb,)
            # Build per-batch valid mask
            cur_valid = attention_mask[:, s:e]  # (B, Lb)
            nxt_valid = attention_mask[:, s+1:e+1] if e < T else attention_mask[:, s+1:T]  # (B, Lb') maybe shorter

            if e < T:
                valid = cur_valid * nxt_valid  # (B, Lb)
            else:
                # last block reaches T; nxt_valid is shorter by 1
                # so we ignore the last position in this block (no next token)
                valid = cur_valid.clone()
                valid[:, -1] = 0

            # additionally enforce pos_has_label
            valid = valid * pos_has_label.view(1, -1).to(valid.dtype)

            # # KL(Pt || Pd)
            # pt = F.softmax(teacher_block.float(), dim=-1)               # (B, Lb, V)
            # log_pd = F.log_softmax(logits_block.float(), dim=-1)        # (B, Lb, V)
            # log_pt = torch.log(pt.clamp_min(1e-9))
            # kl = (pt * (log_pt - log_pd)).sum(dim=-1)                   # (B, Lb)


            #使用topk近似
            topk = 32  # 32/64 都行，先从 32 开始
            t_logits = teacher_block.float()              # (B, Lb, V)
            s_logits = logits_block.float()               # (B, Lb, V)

            t_topv, t_topi = torch.topk(t_logits, k=topk, dim=-1)  # (B, Lb, K)

            # teacher probs over topk (renormalized)
            t_probs = F.softmax(t_topv, dim=-1)  # (B, Lb, K)

            # student log-probs at those indices
            s_logp = F.log_softmax(s_logits, dim=-1).gather(-1, t_topi)  # (B, Lb, K)

            # cross-entropy over teacher topk support:  -E_{t}[log s]
            ce = -(t_probs * s_logp).sum(dim=-1)  # (B, Lb)
            kl = ce  # (approx) since teacher entropy term is dropped/constant for training


            weight = valid.float()
            loss_sum = (kl * weight).sum()
            weight_sum = weight.sum().clamp_min(1.0)

            total_loss = total_loss + loss_sum
            total_weight = total_weight + weight_sum

        return total_loss / total_weight

    def save_pretrained(self, output_dir: str):
        self.draft_lm.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_dir: str, trust_remote_code: bool = True) -> "EDDDraftModel":
        draft = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        return cls(draft)
