# edd_draft_model_masked.py
# EDD draft model (draft only; no tree/PCT)
# Faster dual-block training: single forward with dual-block attention mask (Figure 3(a) in ECNU-EDD).
#
# Key idea:
#   Build a combined sequence of length 2T: [H_1..H_T, t_1..t_T] as inputs_embeds,
#   then build a 4D additive attention mask that enforces:
#     - hidden queries (H part): causal inside H only
#     - token queries (T part): can attend to
#         (i) hidden keys up to start of its block (jL)
#         (ii) token keys inside its own block up to itself (causal inside block)
#       and cannot attend to tokens from earlier blocks.
#
# Loss: KL(P_teacher || P_draft) over token positions only (the second half).

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
    teacher_logits: torch.Tensor  # (B, T, V)
    enc_hidden: torch.Tensor      # (B, T, H)


class EDDDraftModel(nn.Module):
    def __init__(self, draft_lm: nn.Module):
        super().__init__()
        self.draft_lm = draft_lm

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

        # Draft config: same as target but 1 layer
        draft_cfg = copy.deepcopy(target.config)
        if hasattr(draft_cfg, "num_hidden_layers"):
            draft_cfg.num_hidden_layers = 1
        elif hasattr(draft_cfg, "n_layer"):
            draft_cfg.n_layer = 1
        else:
            raise ValueError("Unknown config field for num layers; adapt for your model family.")

        draft = AutoModelForCausalLM.from_config(draft_cfg, trust_remote_code=trust_remote_code)

        # Init: copy embeddings + lm_head
        with torch.no_grad():
            draft.get_input_embeddings().weight.copy_(target.get_input_embeddings().weight)
            if hasattr(draft, "lm_head") and hasattr(target, "lm_head"):
                draft.lm_head.weight.copy_(target.lm_head.weight)
                if getattr(target.lm_head, "bias", None) is not None and getattr(draft.lm_head, "bias", None) is not None:
                    draft.lm_head.bias.copy_(target.lm_head.bias)
            if hasattr(draft_cfg, "layer_types") and isinstance(draft_cfg.layer_types, (list, tuple)):
                draft_cfg.layer_types = list(draft_cfg.layer_types)[:draft_cfg.num_hidden_layers]

        return cls(draft), target

    @torch.no_grad()
    def encode_with_target(
        self,
        target_model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        enc_layer_index: int = -4,
    ) -> EDDEncoding:
        out = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("Target did not return hidden_states; ensure output_hidden_states=True works.")
        enc_hidden = hs[enc_layer_index]  # (B,T,H)
        teacher_logits = out.logits       # (B,T,V)
        return EDDEncoding(teacher_logits=teacher_logits, enc_hidden=enc_hidden)

    @staticmethod
    def _dual_block_attn_bias(
        attention_mask: torch.Tensor,  # (B,T) 1=valid
        block_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Return additive attention bias (B,1,S,S) where S=2T.
        0 means allowed; -inf means masked.
        """
        device = attention_mask.device
        B, T = attention_mask.shape
        S = 2 * T
        neg_inf = torch.finfo(dtype).min

        # Build indices
        idx = torch.arange(T, device=device)  # (T,)
        block_id = idx // block_len
        block_start = block_id * block_len  # (T,)

        # Allowed for token queries:
        # token query at i can attend to:
        #   - hidden keys < block_start[i]
        #   - token keys in same block AND <= i
        #
        # We'll build two boolean matrices (T,T):
        #   allow_token_to_hidden[i,k] where i=query token idx, k=hidden idx
        #   allow_token_to_token[i,k]  where i=query token idx, k=token idx
        k = idx.view(1, T)  # (1,T)
        i = idx.view(T, 1)  # (T,1)

        allow_token_to_hidden = k < block_start.view(T, 1)  # (T,T)
        allow_token_to_token = (block_id.view(T, 1) == block_id.view(1, T)) & (k <= i)  # same block, causal inside block

        # Hidden queries: causal inside hidden
        allow_hidden_to_hidden = k <= i  # (T,T)

        # Compose full allow matrix for SxS (hidden+token)
        allow = torch.zeros((S, S), device=device, dtype=torch.bool)

        # hidden queries -> hidden keys
        allow[:T, :T] = allow_hidden_to_hidden
        # hidden queries -> token keys (disallow; already False)

        # token queries -> hidden keys
        allow[T:, :T] = allow_token_to_hidden
        # token queries -> token keys
        allow[T:, T:] = allow_token_to_token

        # Key padding: both hidden and token positions share the same validity from attention_mask
        valid_1 = attention_mask.to(torch.bool)                  # (B,T)
        valid_2 = torch.cat([valid_1, valid_1], dim=1)           # (B,2T)

        # Base bias from allow (same for all batch)
        bias = torch.where(allow, torch.zeros((), device=device, dtype=dtype), torch.full((), neg_inf, device=device, dtype=dtype))
        bias = bias.view(1, 1, S, S).expand(B, 1, S, S).contiguous()

        # Mask out padded keys (for all queries)
        # If a key position is invalid, set bias[..., :, key] = -inf
        key_invalid = ~valid_2  # (B,S)
        if key_invalid.any():
            bias = bias.masked_fill(key_invalid.view(B, 1, 1, S), neg_inf)

        # Avoid rows with all -inf (can create NaNs) for padded queries:
        # For invalid queries, allow attending to key=0 (if it is valid; otherwise key=0 is also pad but we still force it).
        query_invalid = ~valid_2  # (B,S)
        if query_invalid.any():
            bias[query_invalid.view(B, 1, S, 1).expand(B, 1, S, S)] = neg_inf
            bias[query_invalid, 0, 0] = 0  # (advanced indexing) allow key 0

        return bias

    @staticmethod
    def _kl_full_vocab_chunked(
        teacher_logits: torch.Tensor,   # (B,T,V) float/half ok
        student_logits: torch.Tensor,   # (B,T,V)
        valid: torch.Tensor,            # (B,T) float 0/1
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """Exact KL(Pt || Pd) computed in vocab chunks to reduce peak memory.
        KL = sum_x p_t(x) * (log p_t(x) - log p_d(x)).
        We avoid materializing full softmax by using logsumexp for denominators and chunked exp.
        """
        # (B,T)
        logZ_t = torch.logsumexp(teacher_logits.float(), dim=-1)
        logZ_s = torch.logsumexp(student_logits.float(), dim=-1)

        B, T, V = teacher_logits.shape
        kl_bt = torch.zeros((B, T), device=teacher_logits.device, dtype=torch.float32)

        for v0 in range(0, V, chunk_size):
            v1 = min(v0 + chunk_size, V)
            t_chunk = teacher_logits[:, :, v0:v1].float()  # (B,T,C)
            s_chunk = student_logits[:, :, v0:v1].float()  # (B,T,C)

            logp_t = t_chunk - logZ_t.unsqueeze(-1)
            logp_s = s_chunk - logZ_s.unsqueeze(-1)
            p_t = torch.exp(logp_t)  # (B,T,C)

            kl_bt += (p_t * (logp_t - logp_s)).sum(dim=-1)

        loss_sum = (kl_bt * valid).sum()
        weight_sum = valid.sum().clamp_min(1.0)
        return loss_sum / weight_sum

    def kl_loss_dual_block_masked(
        self,
        input_ids: torch.Tensor,      # (B,T)
        attention_mask: torch.Tensor, # (B,T)
        teacher_logits: torch.Tensor, # (B,T,V)
        enc_hidden: torch.Tensor,     # (B,T,H)
        block_len: int,
        kl_mode: str = "topk",        # "topk" or "full"
        topk: int = 32,
        kl_chunk_size: int = 4096,
    ) -> torch.Tensor:
        """
        Single-forward dual-block training (Eq.3 dual-block mask + KL loss Eq.4):

        1) inputs_embeds = concat([enc_hidden, embed(input_ids)])   # length 2T
        2) pass 4D additive mask enforcing Figure-3(a) visibility
        3) compute KL(P_teacher || P_draft) on token half only

        Note on KL:
        - kl_mode="full": exact full-vocab KL via chunked computation (recommended for faithful paper reproduction).
        - kl_mode="topk": teacher-topk approximation (faster/less compute but can hurt acceptance).
        """
        assert input_ids.dim() == 2
        B, T = input_ids.shape

        tok_embeds = self.draft_lm.get_input_embeddings()(input_ids)  # (B,T,H)
        inputs_embeds = torch.cat([enc_hidden, tok_embeds], dim=1)    # (B,2T,H)

        # 4D attention bias (dual-block visibility)
        attn_bias = self._dual_block_attn_bias(attention_mask, block_len, dtype=inputs_embeds.dtype)

        # Run transformer body once; apply lm_head only to token half
        out = self.draft_lm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_bias,   # 4D additive mask
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state           # (B,2T,H)
        hidden_tok = hidden[:, T:, :]            # (B,T,H)
        logits_tok = self.draft_lm.lm_head(hidden_tok)  # (B,T,V)

        # Valid positions for next-token loss: need pos and pos+1 both valid, and pos <= T-2
        cur_valid = attention_mask[:, :T]                       # (B,T)
        nxt_valid = torch.zeros_like(cur_valid)
        nxt_valid[:, :-1] = attention_mask[:, 1:]
        valid = (cur_valid * nxt_valid).float()                 # (B,T)
        valid[:, -1] = 0.0

        if kl_mode == "full":
            return self._kl_full_vocab_chunked(
                teacher_logits=teacher_logits,
                student_logits=logits_tok,
                valid=valid,
                chunk_size=kl_chunk_size,
            )

        # ---- top-k approx (teacher support only) ----
        t_logits = teacher_logits.float()
        s_logits = logits_tok.float()

        t_topv, t_topi = torch.topk(t_logits, k=min(topk, t_logits.size(-1)), dim=-1)  # (B,T,K)
        t_probs = torch.softmax(t_topv, dim=-1)  # (B,T,K)
        s_logp = torch.log_softmax(s_logits, dim=-1).gather(-1, t_topi)  # (B,T,K)
        ce = -(t_probs * s_logp).sum(dim=-1)  # (B,T)

        loss_sum = (ce * valid).sum()
        weight_sum = valid.sum().clamp_min(1.0)
        return loss_sum / weight_sum

    def save_pretrained(self, output_dir: str):
        self.draft_lm.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_dir: str, trust_remote_code: bool = True) -> "EDDDraftModel":
        draft = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        return cls(draft)
