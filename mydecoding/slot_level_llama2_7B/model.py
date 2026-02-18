from typing import Any, List, Optional

import random
import torch
from torch import nn

from modeling_llama import LlamaDecoderLayer, LlamaRMSNorm


class TimeEmbed(nn.Module):
    """
    Compress token-time axis into slot axis with patch pooling.
    Input:  (B, K, T, H)
    Output: (B, K, S, H), S = ceil(T / slot_size)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_size)
        self.pre_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.post_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, slot_size: int) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"TimeEmbed expects (B,K,T,H), got {tuple(x.shape)}")
        slot_size = max(int(slot_size), 1)
        if slot_size == 1:
            return self.post_proj(self.pre_proj(self.pre_norm(x)))

        bsz, num_layers, tok_len, hidden = x.shape
        # Keep only full slots: tail shorter than slot_size is dropped.
        num_slots = tok_len // slot_size
        if num_slots == 0:
            return x.new_zeros((bsz, num_layers, 0, hidden))

        valid_len = num_slots * slot_size
        x = x[:, :, :valid_len, :]
        x = x.view(bsz, num_layers, num_slots, slot_size, hidden)
        x = self.pre_proj(self.pre_norm(x))
        x = x.mean(dim=3)
        x = self.post_proj(x)
        return x


class LayerFusion(nn.Module):
    """
    iTransformer-style cross-layer fusion over slot-aligned layer tokens.
    Input:  (B, K, S, H)
    Output: (B, S, H)
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_layers, num_slots, hidden = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz * num_slots, num_layers, hidden)
        x = self.norm(x)
        y, _ = self.attn(x, x, x, need_weights=False)
        x = x + y
        x = x.mean(dim=1)
        return x.reshape(bsz, num_slots, hidden)


class ChunkKLDivergenceBatchMean(torch.autograd.Function):
    """
    Memory-efficient batchmean KL(Q || P).
    """

    @staticmethod
    def forward(ctx, draft_logits, teacher_logits, chunk_size: int = 8192):
        teacher_logits = teacher_logits.detach()

        n_rows, vocab_size = draft_logits.shape
        ctx.chunk_size = chunk_size
        ctx.n_rows = n_rows

        draft_f = draft_logits.float()
        teacher_f = teacher_logits.float()

        lse_p = torch.logsumexp(draft_f, dim=-1)
        lse_q = torch.logsumexp(teacher_f, dim=-1)

        kl_per_row = torch.zeros((n_rows,), device=draft_logits.device, dtype=torch.float32)
        for start in range(0, vocab_size, chunk_size):
            end = min(vocab_size, start + chunk_size)
            log_p = draft_f[:, start:end] - lse_p[:, None]
            log_q = teacher_f[:, start:end] - lse_q[:, None]
            q = torch.exp(log_q)
            kl_per_row += (q * (log_q - log_p)).sum(dim=-1)

        kl = kl_per_row.mean()
        ctx.save_for_backward(draft_logits, teacher_logits, lse_p, lse_q)
        return kl

    @staticmethod
    def backward(ctx, grad_out):
        draft_logits, teacher_logits, lse_p, lse_q = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        n_rows = ctx.n_rows
        _, vocab_size = draft_logits.shape

        scale = grad_out / n_rows
        grad_draft = torch.empty_like(draft_logits)

        for start in range(0, vocab_size, chunk_size):
            end = min(vocab_size, start + chunk_size)
            draft_chunk = draft_logits[:, start:end].float()
            teacher_chunk = teacher_logits[:, start:end].float()

            log_p = draft_chunk - lse_p[:, None]
            log_q = teacher_chunk - lse_q[:, None]
            p = torch.exp(log_p)
            q = torch.exp(log_q)

            grad_chunk = (p - q) * scale
            grad_draft[:, start:end] = grad_chunk.to(grad_draft.dtype)

        return grad_draft, None, None


def chunk_kl_batchmean(draft_logits, teacher_logits, chunk_size=8192):
    return ChunkKLDivergenceBatchMean.apply(draft_logits, teacher_logits, chunk_size)


class TreeNode:
    def __init__(self, value: Any):
        self.value = value
        self.children = []
        self.parent = None
        self.idx = -1

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def __repr__(self):
        return f"TreeNode({self.value})"

    def has_child(self, value: Any) -> bool:
        return any(child.value == value for child in self.children)

    def get_child(self, value: Any) -> Optional["TreeNode"]:
        for child in self.children:
            if child.value == value:
                return child
        return None

    def flatten(self, seq: List[int], position_id: List[int] = None):
        queue = [self]
        layer = 0
        num = 0
        while queue:
            level = []
            next_queue = []
            for node in queue:
                level.append(node)
                next_queue.extend(node.children)
            for node in level:
                if node.value == -1:
                    break
                node.idx = num
                num += 1
                seq += [node.value]
                position_id += [layer]
            layer += 1
            queue = next_queue

    def get_mask(self, mask, parent_ids):
        if self.value != -1:
            mask[self.idx, self.idx] = False
            mask[self.idx, parent_ids] = False
            new_parent_ids = parent_ids + [self.idx]
        else:
            new_parent_ids = parent_ids

        for child in self.children:
            child.get_mask(mask, new_parent_ids)


def Flatten_tree(lst):
    device = lst[0].device
    root = TreeNode(-1)
    for i in range(len(lst)):
        lst[i] = lst[i].cpu()

    for ts in lst:
        for row in range(ts.shape[0]):
            parent = root
            for col in range(ts.shape[1]):
                token = ts[row, col]
                if parent.has_child(token):
                    parent = parent.get_child(token)
                else:
                    new_node = TreeNode(token.item())
                    parent.add_child(new_node)
                    parent = new_node

    seq, position_ids = [], []
    root.flatten(seq, position_ids)
    attention_mask = torch.ones((len(seq), len(seq)), dtype=torch.bool)
    root.get_mask(attention_mask, [])

    seq = torch.tensor(seq, dtype=torch.long, device=device)
    position_ids = torch.tensor(position_ids, dtype=torch.long, device=device)
    attention_mask = attention_mask.to(device)
    return seq, attention_mask, root, position_ids


class Effective_Draft_Decoder(nn.Module):
    """
    INV-v2 style draft decoder:
    - time-axis compression to slot memory (TimeEmbed)
    - cross-layer fusion on slot-aligned layer tokens
    - block-wise cross-attention visibility using slot index
    """

    def __init__(
        self,
        hidden_size,
        dim_feedforward,
        head_num,
        num_layers,
        config,
        layer_dropout: float = 0.0,
        slot_dropout: float = 0.0,
        slot_size: int = 5,
    ):
        super().__init__()
        vocab_size = config.vocab_size
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.decoder = LlamaDecoderLayer(config)

        self.time_embed = TimeEmbed(hidden_size)
        self.fusion = LayerFusion(hidden_size, head_num)

        self.cross_attn = nn.MultiheadAttention(hidden_size, head_num, batch_first=True)
        self.cross_norm = LlamaRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.norm = LlamaRMSNorm(hidden_size)

        self.layer_dropout = float(layer_dropout)
        self.slot_dropout = float(slot_dropout)
        self.slot_size = max(int(slot_size), 1)

    def _prepare_layers_and_self(self, encoder_out: torch.Tensor):
        if isinstance(encoder_out, (list, tuple)):
            encoder_out = torch.stack(encoder_out, dim=1)

        if encoder_out.dim() == 4:
            layers = encoder_out
            encoder_self = encoder_out[:, -1, :, :]
        elif encoder_out.dim() == 3:
            layers = encoder_out.unsqueeze(1)
            encoder_self = encoder_out
        else:
            raise ValueError(f"encoder_out must be rank-3 or rank-4, got {encoder_out.dim()}")
        return layers, encoder_self

    def _apply_fusion_dropout(self, slots: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self.layer_dropout <= 0 and self.slot_dropout <= 0):
            return slots

        out = slots
        _, num_layers, num_slots, _ = out.shape

        if self.layer_dropout > 0 and num_layers > 1:
            keep_prob = max(1.0 - self.layer_dropout, 1e-6)
            layer_mask = (torch.rand((1, num_layers, 1, 1), device=out.device) < keep_prob).to(out.dtype)
            if layer_mask.sum() == 0:
                idx = torch.randint(0, num_layers, (1,), device=out.device)
                layer_mask[:, idx, :, :] = 1.0
            out = out * layer_mask / keep_prob

        if self.slot_dropout > 0 and num_slots > 1:
            keep_prob = max(1.0 - self.slot_dropout, 1e-6)
            slot_mask = (torch.rand((1, 1, num_slots, 1), device=out.device) < keep_prob).to(out.dtype)
            if slot_mask.sum() == 0:
                idx = torch.randint(0, num_slots, (1,), device=out.device)
                slot_mask[:, :, idx, :] = 1.0
            out = out * slot_mask / keep_prob

        return out

    def init_slot_cache(self, encoder_out: torch.Tensor, slot_size: Optional[int] = None, apply_dropout: bool = False):
        slot_size = self.slot_size if slot_size is None else max(int(slot_size), 1)
        layers, encoder_self = self._prepare_layers_and_self(encoder_out)
        bsz, num_layers, tok_len, hidden = layers.shape

        full_tokens = (tok_len // slot_size) * slot_size
        if full_tokens > 0:
            full_layers = layers[:, :, :full_tokens, :]
            slots = self.time_embed(full_layers, slot_size=slot_size)
            if apply_dropout:
                slots = self._apply_fusion_dropout(slots)
            memory = self.fusion(slots) if slots.shape[2] > 0 else layers.new_zeros((bsz, 0, hidden))
        else:
            memory = layers.new_zeros((bsz, 0, hidden))

        tail_layers = layers[:, :, full_tokens:, :]
        return {
            "slot_size": slot_size,
            "encoder_self": encoder_self,
            "memory": memory,
            "tail_layers": tail_layers,
            "num_layers": num_layers,
        }

    def update_slot_cache(self, slot_cache: dict, accepted_layers: torch.Tensor, apply_dropout: bool = False):
        if accepted_layers.dim() == 3:
            accepted_layers = accepted_layers.unsqueeze(1)

        if accepted_layers.dim() != 4:
            raise ValueError(f"accepted_layers must be rank-4 or rank-3, got {accepted_layers.dim()}")

        slot_size = int(slot_cache["slot_size"])
        encoder_self = slot_cache["encoder_self"]
        memory = slot_cache["memory"]
        tail_layers = slot_cache["tail_layers"]

        if accepted_layers.shape[1] != tail_layers.shape[1]:
            raise ValueError(
                f"accepted_layers layer dim ({accepted_layers.shape[1]}) "
                f"!= cache layer dim ({tail_layers.shape[1]})"
            )

        encoder_self = torch.cat([encoder_self, accepted_layers[:, -1, :, :]], dim=1)
        cat_layers = torch.cat([tail_layers, accepted_layers], dim=2)

        full_tokens = (cat_layers.shape[2] // slot_size) * slot_size
        if full_tokens > 0:
            new_full_layers = cat_layers[:, :, :full_tokens, :]
            new_slots = self.time_embed(new_full_layers, slot_size=slot_size)
            if apply_dropout:
                new_slots = self._apply_fusion_dropout(new_slots)
            if new_slots.shape[2] > 0:
                new_memory = self.fusion(new_slots)
                memory = torch.cat([memory, new_memory], dim=1)

        tail_layers = cat_layers[:, :, full_tokens:, :]

        slot_cache["encoder_self"] = encoder_self
        slot_cache["memory"] = memory
        slot_cache["tail_layers"] = tail_layers
        return slot_cache

    def split_from_slot_cache(self, slot_cache: dict):
        return slot_cache["encoder_self"], slot_cache["memory"]

    def _split_encoder_out(self, encoder_out: torch.Tensor, slot_size: int = 1, apply_dropout: bool = False):
        layers, encoder_self = self._prepare_layers_and_self(encoder_out)

        slots = self.time_embed(layers, slot_size=max(int(slot_size), 1))
        if apply_dropout:
            slots = self._apply_fusion_dropout(slots)
        memory = self.fusion(slots)
        return encoder_self, memory

    def forward(self, encoder_out, labels, llm_logits):
        # In training, block size L is sampled for draft self-attn masking,
        # while slot_size is fixed (self.slot_size).
        pred_len = random.randint(5, 10)

        encoder_self, memory = self._split_encoder_out(
            encoder_out,
            slot_size=self.slot_size,
            apply_dropout=self.training,
        )

        inp_len = encoder_self.shape[1] - llm_logits.shape[1]
        label_len = llm_logits.shape[1]

        input_ids = labels[:, inp_len:]
        input_embeds = self.embedding_layer(input_ids)

        hidden_states = torch.cat([encoder_self, input_embeds], dim=1)

        position_ids = torch.arange(encoder_self.shape[1], dtype=torch.long, device=hidden_states.device)
        position_ids = torch.cat(
            [
                position_ids,
                torch.arange(inp_len, inp_len + label_len, dtype=torch.long, device=hidden_states.device),
            ],
            dim=0,
        )[None, :]

        mask_min = torch.finfo(hidden_states.dtype).min
        causal_mask = torch.triu(
            torch.ones(hidden_states.shape[1], hidden_states.shape[1], device=hidden_states.device), diagonal=1
        ).bool()
        attention_mask = torch.zeros_like(causal_mask, dtype=hidden_states.dtype)
        attention_mask[causal_mask] = mask_min
        for start in range(encoder_self.shape[1], hidden_states.shape[1], pred_len):
            left = inp_len + start - encoder_self.shape[1]
            attention_mask[start : start + pred_len, left:start] = mask_min

        hidden_states = self.decoder(
            hidden_states,
            attention_mask=attention_mask[None, None, :, :],
            position_ids=position_ids,
        )[0]

        token_states = hidden_states[:, -label_len:, :]

        if memory.shape[1] > 0:
            mem_len = memory.shape[1]
            i = torch.arange(label_len, device=hidden_states.device).unsqueeze(1)
            j = torch.arange(mem_len, device=hidden_states.device).unsqueeze(0)

            # Per-token visibility with fixed slot size.
            # token absolute index (1-based) is inp_len + i + 1, so visible full slots are floor((inp_len + i) / slot_size).
            visible_slots = (inp_len + i) // self.slot_size
            mem_mask = j >= visible_slots

            cross_out, _ = self.cross_attn(
                self.cross_norm(token_states),
                memory,
                memory,
                attn_mask=mem_mask,
                need_weights=False,
            )
            token_states = self.norm(token_states + cross_out)
        else:
            token_states = self.norm(token_states)
        logits = self.lm_head(token_states)

        labels_flat = labels[:, -label_len + 1 :].reshape(-1).contiguous()
        draft_logits = logits[:, :-1, :].reshape(-1, logits.size(-1)).contiguous()
        teacher_logits = llm_logits[:, :-1, :].reshape(-1, llm_logits.size(-1)).contiguous().detach()

        kl_loss = chunk_kl_batchmean(draft_logits, teacher_logits, chunk_size=8192)

        with torch.no_grad():
            loss = torch.nn.functional.cross_entropy(draft_logits.float(), labels_flat)

        return loss, kl_loss

    def generate(
        self,
        encoder_out,
        decoder_inp_token,
        max_length=10,
        top_k=5,
        threshold=0.1,
        slot_cache: Optional[dict] = None,
    ):
        if slot_cache is not None:
            encoder_self, memory_base = self.split_from_slot_cache(slot_cache)
        else:
            encoder_self, memory_base = self._split_encoder_out(
                encoder_out,
                slot_size=self.slot_size,
                apply_dropout=False,
            )

        cur_p = torch.tensor([1.0], device=encoder_self.device).unsqueeze(-1)
        all_candidates_new = []
        past_key_values = None

        for _ in range(max_length):
            if past_key_values is None:
                dec_inp = torch.cat([encoder_self, self.embedding_layer(decoder_inp_token)], dim=1)
                hidden_states = self.decoder(dec_inp, use_cache=True)
                past_key_values = tuple(kv[:, :, : encoder_self.shape[1], :] for kv in hidden_states[1])
            else:
                cur_past_key_values = [kv.repeat(decoder_inp_token.shape[0], 1, 1, 1) for kv in past_key_values]
                position_ids = torch.arange(
                    encoder_self.shape[1],
                    encoder_self.shape[1] + decoder_inp_token.shape[1],
                    dtype=torch.long,
                    device=encoder_self.device,
                )[None, :]
                dec_inp = self.embedding_layer(decoder_inp_token)
                hidden_states = self.decoder(dec_inp, past_key_value=cur_past_key_values, position_ids=position_ids)

            if memory_base.shape[0] != decoder_inp_token.shape[0]:
                memory = memory_base.repeat(decoder_inp_token.shape[0], 1, 1)
            else:
                memory = memory_base

            token_state = hidden_states[0][:, -1:, :]
            if memory.shape[1] > 0:
                cross_out, _ = self.cross_attn(
                    self.cross_norm(token_state),
                    memory,
                    memory,
                    attn_mask=None,
                    need_weights=False,
                )
                token_state = self.norm(token_state + cross_out)
            else:
                token_state = self.norm(token_state)

            logits = self.lm_head(token_state)
            top_scores, top_indices = logits[:, -1, :].softmax(-1).topk(top_k, dim=-1)

            cur_p = cur_p * top_scores
            mask = cur_p > threshold
            if mask.sum().item() == 0:
                decoder_inp_token = torch.cat([decoder_inp_token, top_indices[:, 0].unsqueeze(-1)], dim=1)
                all_candidates_new.append(decoder_inp_token[:, 1:])
                break

            decoder_inp_token = torch.cat(
                [
                    decoder_inp_token.unsqueeze(1).repeat(1, top_k, 1),
                    top_indices.unsqueeze(-1),
                ],
                dim=2,
            )
            all_candidates_new.append(decoder_inp_token[~mask][:, 1:])
            decoder_inp_token, cur_p = decoder_inp_token[mask], cur_p[mask].unsqueeze(-1)

        seq, attention_mask, division, position_ids = Flatten_tree(all_candidates_new)
        return seq.unsqueeze(0), attention_mask, division, position_ids
