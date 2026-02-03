# edd_draft_model_effective.py
# Effective Draft Decoder (Faster-SD style) adapted to generic HF causal LMs (Llama3.x, Qwen2, etc.)
# - No hard-coded vocab size
# - Uses one (or N) transformer blocks copied from teacher for architecture compatibility
# - Copies embedding + lm_head weights from teacher
# - Implements Faster-SD dual-block mask + position_ids scheme
#
# Expected training call (see train_edd_draft_effective_single_gpu.py):
#   loss_ce, loss_kl = draft(encoder_out=teacher_hidden, labels=input_ids, llm_logits=teacher_logits_tail)

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import AutoModelForCausalLM, AutoConfig


def _get_attr(obj: Any, names: Tuple[str, ...]) -> Optional[Any]:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

def ce_from_teacher_chunked(teacher_logits, student_logits, chunk=16384):
    # teacher_logits, student_logits: (B, L, V)
    t = teacher_logits.float()
    s = student_logits.float()
    B, L, V = t.shape

    t_logZ = torch.logsumexp(t, dim=-1, keepdim=True)  # (B,L,1)
    s_logZ = torch.logsumexp(s, dim=-1, keepdim=True)  # (B,L,1)

    ce = 0.0
    for st in range(0, V, chunk):
        ed = min(V, st + chunk)
        t_chunk = t[..., st:ed]
        s_chunk = s[..., st:ed]

        p_t = torch.exp(t_chunk - t_logZ)     # teacher probs chunk
        logp_s = s_chunk - s_logZ             # student log-probs chunk
        ce = ce + (p_t * (-logp_s)).sum(dim=-1)  # (B,L)

    return ce  # (B,L)



def _get_base_and_layers(teacher: nn.Module):
    """
    Return (base_model, layers_modulelist, norm_module_or_None).
    Works for most HF decoder-only LMs:
      - Llama/Llama2/Llama3: teacher.model.layers, teacher.model.norm
      - Qwen2: teacher.model.layers, teacher.model.norm (usually)
      - GPT-like: teacher.transformer.h, teacher.transformer.ln_f
    """
    # Common: .model.layers
    if hasattr(teacher, "model") and hasattr(teacher.model, "layers"):
        base = teacher.model
        layers = teacher.model.layers
        norm = _get_attr(base, ("norm", "final_layernorm", "ln_f"))
        return base, layers, norm

    # GPT-NeoX style: .gpt_neox.layers
    if hasattr(teacher, "gpt_neox") and hasattr(teacher.gpt_neox, "layers"):
        base = teacher.gpt_neox
        layers = teacher.gpt_neox.layers
        norm = _get_attr(base, ("final_layer_norm", "ln_f"))
        return base, layers, norm

    # GPT-2 style: .transformer.h
    if hasattr(teacher, "transformer") and hasattr(teacher.transformer, "h"):
        base = teacher.transformer
        layers = teacher.transformer.h
        norm = _get_attr(base, ("ln_f", "norm", "final_layernorm"))
        return base, layers, norm

    raise ValueError(
        "Unsupported teacher model structure: cannot find transformer layers. "
        "Expected teacher.model.layers or teacher.transformer.h."
    )


@dataclass
class DraftConfig:
    target_model_name_or_path: str
    num_layers: int
    block_len_min: int
    block_len_max: int
    hidden_layer_index: int


class Effective_Draft_Decoder(nn.Module):
    """
    Faster-SD style 'Effective Draft Decoder', but made architecture-agnostic by copying
    decoder blocks from the teacher model.

    Key points (matching Faster-SD model.py):
      - Inputs: encoder_out (teacher hidden states), labels (full input_ids), llm_logits (teacher logits tail)
      - Random block length in [block_len_min, block_len_max]
      - Rebuild position_ids so the token-stream positions match original sequence positions
      - Dual-block mask: token-stream blocks cannot attend to the corresponding encoder_out slice
      - Loss: returns (CE loss, KL loss); training typically uses KL only
    """

    def __init__(
        self,
        teacher: AutoModelForCausalLM,
        num_layers: int = 1,
        block_len_min: int = 5,
        block_len_max: int = 10,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if block_len_min < 1 or block_len_max < block_len_min:
            raise ValueError("Invalid block_len range")

        self.block_len_min = int(block_len_min)
        self.block_len_max = int(block_len_max)

        cfg = teacher.config
        vocab_size = int(getattr(cfg, "vocab_size", teacher.get_input_embeddings().weight.shape[0]))
        hidden_size = int(getattr(cfg, "hidden_size", teacher.get_input_embeddings().weight.shape[1]))

        # Embedding + LM head: same shapes as teacher; copy weights outside (from_teacher()).
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Copy N decoder blocks from teacher (deepcopy to make them trainable & independent).
        base, layers, norm = _get_base_and_layers(teacher)
        # Rotary embedding (for newer Transformers: layers expect precomputed (cos, sin) via position_embeddings)
        self.rotary_emb = copy.deepcopy(getattr(base, "rotary_emb", None))
        if len(layers) < num_layers:
            raise ValueError(f"Teacher has only {len(layers)} layers, but num_layers={num_layers}")
        self.decoder_layers = nn.ModuleList([copy.deepcopy(layers[i]) for i in range(num_layers)])

        # Final norm (copy if teacher has it, else Identity)
        self.norm = copy.deepcopy(norm) if norm is not None else nn.Identity()

    @classmethod
    def from_teacher(
        cls,
        teacher: AutoModelForCausalLM,
        num_layers: int = 1,
        block_len_min: int = 5,
        block_len_max: int = 10,
    ) -> "Effective_Draft_Decoder":
        draft = cls(teacher, num_layers=num_layers, block_len_min=block_len_min, block_len_max=block_len_max)

        # Copy embedding + lm_head weights from teacher (Faster-SD train.py does this).
        with torch.no_grad():
            # input embedding
            emb = teacher.get_input_embeddings()
            draft.embedding_layer.weight.copy_(emb.weight)

            # output embedding / lm_head
            out_emb = teacher.get_output_embeddings()
            if out_emb is None and hasattr(teacher, "lm_head"):
                out_emb = teacher.lm_head
            if out_emb is None:
                raise ValueError("Teacher model has no output embeddings / lm_head.")
            draft.lm_head.weight.copy_(out_emb.weight)

        return draft

    def _build_position_ids(self, enc_len: int, inp_len: int, label_len: int, device) -> torch.Tensor:
        """
        Match Faster-SD behavior:
          - encoder_out positions: 0..enc_len-1
          - token-stream positions: inp_len..inp_len+label_len-1  (NOT enc_len..enc_len+label_len-1)
        """
        pos0 = torch.arange(enc_len, dtype=torch.long, device=device)
        pos1 = torch.arange(inp_len, inp_len + label_len, dtype=torch.long, device=device)
        pos = torch.cat([pos0, pos1], dim=0)[None, :]
        return pos

    def _build_dual_block_mask(self, enc_len: int, total_len: int, inp_len: int, pred_len: int, device, dtype):
        """
        Build additive attention mask (float, -inf for masked) of shape (1,1,S,S).
        This matches Faster-SD model.py logic:
          - start from standard causal mask over total_len tokens
          - additionally, for each token-stream block [i:i+pred_len], forbid attending to
            encoder_out slice [inp_len + i - enc_len : i].
        """
        S = total_len
        causal_mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        attn = torch.zeros((S, S), device=device, dtype=torch.float32)
        attn[causal_mask] = float("-inf")

        # Apply block restriction for token-stream part
        for i in range(enc_len, S, pred_len):
            # columns to block are within encoder_out region
            col_start = inp_len + i - enc_len
            col_end = i
            if col_start < 0:
                col_start = 0
            if col_start < col_end:
                attn[i : min(i + pred_len, S), col_start:col_end] = float("-inf")

        mask = attn.to(dtype=dtype)[None, None, :, :]
        return mask

    def _compute_position_embeddings(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        Newer Transformers (e.g. recent Llama) expect precomputed rotary position embeddings
        to be passed into each decoder layer as `position_embeddings=(cos, sin)`.
        We try to compute them from `self.rotary_emb` if available; otherwise return None.
        """
        if getattr(self, "rotary_emb", None) is None:
            return None
        try:
            return self.rotary_emb(x, position_ids)
        except TypeError:
            # Some implementations may not take `x`
            try:
                return self.rotary_emb(position_ids)
            except Exception:
                return None

    def forward(self, encoder_out: torch.Tensor, labels: torch.Tensor, llm_logits: torch.Tensor):
        """
        encoder_out: (B, seq_len, H)    teacher hidden states for the *full* input (prompt+answer)
        labels:      (B, seq_len)       full input_ids (prompt+answer)
        llm_logits:  (B, label_len, V)  teacher logits for the answer tail (same len as answer)

        Returns:
          (ce_loss, kl_loss)
        """
        B, enc_len, H = encoder_out.shape
        if labels.dim() != 2 or labels.shape[0] != B:
            raise ValueError("labels must be (B, seq_len)")
        label_len = llm_logits.shape[1]
        if label_len <= 1:
            raise ValueError("label_len must be > 1 for shifted next-token KL/CE")

        inp_len = enc_len - label_len  # prompt length
        if inp_len < 0:
            raise ValueError("encoder_out length must be >= label_len")

        # Random block length (pred_len in Faster-SD code)
        pred_len = random.randint(self.block_len_min, self.block_len_max)

        # Token-stream embeddings use the answer token ids (suffix)
        input_ids = labels[:, inp_len:]  # (B, label_len)
        input_embeds = self.embedding_layer(input_ids)  # (B, label_len, H)

        # Concatenate teacher hidden + token-stream embeddings
        hidden_states = torch.cat([encoder_out, input_embeds], dim=1)  # (B, enc_len + label_len, H)
        total_len = hidden_states.shape[1]

        # position_ids & attention_mask
        position_ids = self._build_position_ids(enc_len=enc_len, inp_len=inp_len, label_len=label_len, device=hidden_states.device)
        # expand to batch (some HF layers expect (B, S))
        if position_ids.size(0) != B:
            position_ids = position_ids.expand(B, -1)

        attention_mask = self._build_dual_block_mask(
            enc_len=enc_len,
            total_len=total_len,
            inp_len=inp_len,
            pred_len=pred_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        # expand to batch (broadcasting works for most models; this makes it explicit)
        if attention_mask.size(0) != B:
            attention_mask = attention_mask.expand(B, -1, -1, -1)

        # Run N decoder layers. Each layer returns tuple (hs, ...)
        position_embeddings = self._compute_position_embeddings(hidden_states, position_ids)
        x = hidden_states
        for layer in self.decoder_layers:
            try:
                out = layer(x, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)
            except TypeError:
                # Older implementations may not support `position_embeddings`
                out = layer(x, attention_mask=attention_mask, position_ids=position_ids)
            x = out[0] if isinstance(out, (tuple, list)) else out

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, total_len, V)

        # Align to Faster-SD training:
        # student logits: token-stream positions, shifted (exclude last)
        student = logits[:, -label_len:-1, :].float().contiguous().view(-1, logits.size(-1))
        # teacher logits: answer tail, shifted (exclude last)
        teacher = llm_logits[:, :-1, :].float().contiguous().view(-1, logits.size(-1))

        # --- chunked full-vocab KL with reduction="batchmean" (memory-safe) ---
        # student: (N, V) float
        # teacher: (N, V) float
        # NOTE: This computes KL(pt||ps) with the same scaling as F.kl_div(..., reduction="batchmean")

        chunk = getattr(self, "kl_chunk_size", 16384)

        # Precompute log-normalizers
        t_logZ = torch.logsumexp(teacher, dim=-1, keepdim=True)  # (N, 1)
        s_logZ = torch.logsumexp(student, dim=-1, keepdim=True)  # (N, 1)

        # KL = sum_v p_t(v) * (log p_t(v) - log p_s(v))
        # We'll compute it in vocab chunks to avoid allocating full softmax tensors.
        kl_per_row = torch.zeros((student.size(0),), device=student.device, dtype=student.dtype)  # (N,)

        V = student.size(-1)
        for st in range(0, V, chunk):
            ed = min(V, st + chunk)
            t_chunk = teacher[:, st:ed]  # (N, C)
            s_chunk = student[:, st:ed]  # (N, C)

            logp_t = t_chunk - t_logZ    # (N, C)
            logp_s = s_chunk - s_logZ    # (N, C)
            p_t = torch.exp(logp_t).detach()  # teacher distribution, no grad

            kl_per_row += (p_t * (logp_t - logp_s)).sum(dim=-1)  # (N,)

        # reduction="batchmean": sum over all rows, divided by batch size B
        kl_loss = kl_per_row.sum() / max(1, student.size(0))
        # --- end chunked KL ---


        # CE (optional, not usually used)
        tgt = labels[:, -label_len + 1 :].contiguous().view(-1)
        ce_loss = CrossEntropyLoss()(student, tgt)

        return ce_loss, kl_loss

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd, strict=True)
        return self


def load_teacher(
    target_model: str,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool = True,
):
    teacher = AutoModelForCausalLM.from_pretrained(
        target_model,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=None,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.to(device)
    return teacher