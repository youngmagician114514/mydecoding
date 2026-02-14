from modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
import torch
import torch.nn.functional as F
from torch import nn
import random
from torch.nn import CrossEntropyLoss
from typing import List, Dict, Any, Optional


import torch


class LayerFusion(nn.Module):
    """
    iTransformer-style cross-layer fusion.
    Input: (B, K, T, H) where K=layers, T=slots/tokens.
    Output: (B, T, H) memory tokens.
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, t, h = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * t, k, h)
        x = self.norm(x)
        y, _ = self.attn(x, x, x, need_weights=False)
        x = x + y
        x = x.mean(dim=1)
        return x.reshape(b, t, h)

class ChunkKLDivergenceBatchMean(torch.autograd.Function):
    """
    计算 batchmean KL(Q || P):
      KL = (1/N) * sum_i sum_v q_iv * (log q_iv - log p_iv)

    - forward: 分块累加 KL 值
    - backward: 利用 d/dlogits KL = (p - q)/N 分块算梯度
    """
    @staticmethod
    def forward(ctx, draft_logits, teacher_logits, chunk_size: int = 8192):
        # draft_logits: (N,V) 需要梯度
        # teacher_logits: (N,V) 不需要梯度（建议提前 detach）
        teacher_logits = teacher_logits.detach()

        N, V = draft_logits.shape
        ctx.chunk_size = chunk_size
        ctx.N = N

        # 为了“不要降精度”：softmax 的数值运算用 float32
        draft_f = draft_logits.float()
        teacher_f = teacher_logits.float()

        lse_p = torch.logsumexp(draft_f, dim=-1)   # (N,)
        lse_q = torch.logsumexp(teacher_f, dim=-1) # (N,)

        kl_per_row = torch.zeros((N,), device=draft_logits.device, dtype=torch.float32)

        for s in range(0, V, chunk_size):
            e = min(V, s + chunk_size)
            log_p = draft_f[:, s:e] - lse_p[:, None]
            log_q = teacher_f[:, s:e] - lse_q[:, None]
            q = torch.exp(log_q)
            kl_per_row += (q * (log_q - log_p)).sum(dim=-1)

        kl = kl_per_row.mean()  # batchmean

        # backward 需要 teacher_logits + lse_p/lse_q
        ctx.save_for_backward(draft_logits, teacher_logits, lse_p, lse_q)
        return kl

    @staticmethod
    def backward(ctx, grad_out):
        draft_logits, teacher_logits, lse_p, lse_q = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        N = ctx.N
        _, V = draft_logits.shape

        # grad_out 是标量，batchmean => /N
        scale = (grad_out / N)

        grad_draft = torch.empty_like(draft_logits)

        # 分块计算 (p - q)
        for s in range(0, V, chunk_size):
            e = min(V, s + chunk_size)

            # float32 做 softmax 相关计算
            d = draft_logits[:, s:e].float()
            t = teacher_logits[:, s:e].float()

            log_p = d - lse_p[:, None]
            log_q = t - lse_q[:, None]
            p = torch.exp(log_p)
            q = torch.exp(log_q)

            g = (p - q) * scale
            grad_draft[:, s:e] = g.to(grad_draft.dtype)

        return grad_draft, None, None


def chunk_kl_batchmean(draft_logits, teacher_logits, chunk_size=8192):
    return ChunkKLDivergenceBatchMean.apply(draft_logits, teacher_logits, chunk_size)


class TreeNode:
    """树节点类"""
    def __init__(self, value: Any):
        self.value = value
        self.children = []
        self.parent = None
        self.idx = -1 # 展平后的idx
    
    def add_child(self, child_node):
        """添加子节点"""
        child_node.parent = self
        self.children.append(child_node)
    
    def __repr__(self):
        return f"TreeNode({self.value})"
    
    def has_child(self, value: Any) -> bool:
        """检查是否有指定值的子节点"""
        return any(child.value == value for child in self.children)
    
    def get_child(self, value: Any) -> Optional['TreeNode']:
        """获取指定值的子节点"""
        for child in self.children:
            if child.value == value:
                return child
        return None
    
    def flatten(self, seq: List[int],  position_id: List[int] = None):
        """通过层次遍历将树展平为一维序列"""
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
        """获取当前节点的mask"""
        if self.value != -1:
            mask[self.idx, self.idx] = False
            mask[self.idx, parent_ids] = False
            new_parent_ids = parent_ids + [self.idx]
        for child in self.children:
            child.get_mask(mask, new_parent_ids if self.value != -1 else parent_ids) 

    

def Flatten_tree(lst):
    """
    将树展平为一维序列。
    """
    device = lst[0].device
    root = TreeNode(-1) 
    for i in range(len(lst)):
        lst[i] = lst[i].cpu()
    for ts in lst:
        for r in range(ts.shape[0]):
            parent = root
            for c in range(ts.shape[1]):
                if parent.has_child(ts[r, c]):
                # 如果当前节点的值已经存在于子节点中，则无需添加
                    parent = parent.get_child(ts[r, c])
                    continue
                # 如果当前节点的值不存在于子节点中，则添加新节点
                else:
                    new_node = TreeNode(ts[r, c].item())
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




# 以LLM编码的结果为输入，构造NAR模型
class Effective_Draft_Decoder(nn.Module):
    def __init__(self, hidden_size, dim_feedforward, head_num, num_layers, config):
        super(Effective_Draft_Decoder, self).__init__()
        vocab_size = config.vocab_size
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.decoder = LlamaDecoderLayer(config)
        self.fusion = LayerFusion(hidden_size, head_num)
        self.cross_attn = nn.MultiheadAttention(hidden_size, head_num, batch_first=True)
        self.cross_norm = LlamaRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.norm = LlamaRMSNorm(hidden_size)

    def _split_encoder_out(self, encoder_out: torch.Tensor):
        if isinstance(encoder_out, (list, tuple)):
            encoder_out = torch.stack(encoder_out, dim=1)
        if encoder_out.dim() == 4:
            memory = self.fusion(encoder_out)
            encoder_self = encoder_out[:, -1, :, :]
            return encoder_self, memory
        return encoder_out, encoder_out

    def forward(self, encoder_out, labels, llm_logits):
        '''
            encoder_out: LLM编码的结果，shape为(batch_size, seq_len, hidden_size)
            labels: 目标token的ID，shape为(batch_size, seq_len)
            llm_logits: LLM的输出，shape为(batch_size, seq_len, vocab_size)
        '''
        pred_len = random.randint(5, 10) # 随机划分block的长度
        encoder_self, memory = self._split_encoder_out(encoder_out)
        inp_len = encoder_self.shape[1] - llm_logits.shape[1]
        label_len = llm_logits.shape[1]

        # 获得输入的embedding
        input_ids = labels[:, inp_len:]#*prompt后面的所有id,也即answer部分
        input_embeds = self.embedding_layer(input_ids)

        # 将LLM输出的向量作为soft-prompt输入，用于后续token的生成
        hidden_states = torch.cat([encoder_self, input_embeds], dim=1)
        # 重新构造position_ids
        position_ids = torch.arange(encoder_self.shape[1], dtype=torch.long, device=hidden_states.device)
        position_ids = torch.cat([position_ids, torch.arange(inp_len, inp_len + label_len, dtype=torch.long, device=hidden_states.device)], dim=0)[None, :]

        # block attention mask
        causal_mask = torch.triu(torch.ones(hidden_states.shape[1], hidden_states.shape[1]), diagonal=1).bool()
        attention_mask = torch.zeros_like(causal_mask).float()
        attention_mask[causal_mask==1] = float('-inf')
        for i in range(encoder_self.shape[1], hidden_states.shape[1], pred_len):
            attention_mask[i:i+pred_len, inp_len+i-encoder_self.shape[1]:i] = float('-inf')
        attention_mask = attention_mask.bfloat16().to(hidden_states.device)

        # self-attn
        hidden_states = self.decoder(hidden_states, attention_mask=attention_mask[None, None, :, :], position_ids=position_ids)
        hidden_states = hidden_states[0]

        # cross-attn (F as memory, C as query)
        token_states = hidden_states[:, -label_len:, :]
        mem_len = memory.shape[1]
        i = torch.arange(label_len, device=hidden_states.device).unsqueeze(1)
        j = torch.arange(mem_len, device=hidden_states.device).unsqueeze(0)
        visible = inp_len + (i // pred_len) * pred_len
        mem_mask = (j >= visible)  # True means masked
        cross_out, _ = self.cross_attn(
            self.cross_norm(token_states), memory, memory, attn_mask=mem_mask, need_weights=False
        )
        token_states = token_states + cross_out

        token_states = self.norm(token_states)
        logits = self.lm_head(token_states)

        # 计算loss(不要把 labels/logits 覆盖成1D后又做2D)

        labels_flat = labels[:, -label_len+1:].reshape(-1).contiguous()  # (N,)

        draft_logits = logits[:, :-1, :].reshape(-1, logits.size(-1)).contiguous()  # (N, V)
        teacher_logits = llm_logits[:, :-1, :].reshape(-1, llm_logits.size(-1)).contiguous().detach()  # (N, V)

        kl_loss = chunk_kl_batchmean(draft_logits, teacher_logits, chunk_size=8192)

        # 在 train.py 里最终只反传 kl_loss(loss=kl_loss)，CE 仅做日志可选
        with torch.no_grad():
            loss = torch.nn.functional.cross_entropy(draft_logits.float(), labels_flat)

        return loss, kl_loss

    # 用于Speculative Decoding
    def generate(self, encoder_out, decoder_inp_token, max_length=10, top_k=5, threshold=0.1):
        '''
            encoder_out: LLM编码的结果，shape为(batch_size, seq_len, hidden_size)
            decoder_inp_token: 当前解码器输入的token ID，shape为(batch_size, cur_length)
            max_length: 最大生成长度
            threshold: 用于剪枝的阈值
        '''
        encoder_self, memory_base = self._split_encoder_out(encoder_out)
        cur_p = torch.tensor([1.0], device=encoder_self.device).unsqueeze(-1)  # 初始概率
        all_candidates_new = []
        past_key_values = None
        for _ in range(max_length):
            # 构造hidden_states
            if past_key_values is None:
                dec_inp = torch.cat([encoder_self, self.embedding_layer(decoder_inp_token)], dim=1)
                hidden_states = self.decoder(dec_inp, use_cache=True)
                past_key_values = tuple(kv[:, :, :encoder_self.shape[1], :] for kv in hidden_states[1])
            else:
                cur_past_key_values = [kv.repeat(decoder_inp_token.shape[0], 1, 1, 1) for kv in past_key_values] # 复制soft prompt
                position_ids = torch.arange(encoder_self.shape[1], encoder_self.shape[1]+decoder_inp_token.shape[1], dtype=torch.long, device=encoder_self.device)[None, :]
                dec_inp = self.embedding_layer(decoder_inp_token)
                hidden_states = self.decoder(dec_inp, past_key_value=cur_past_key_values, position_ids=position_ids)

            # cross-attn (memory is same for all candidates)
            if memory_base.shape[0] != decoder_inp_token.shape[0]:
                memory = memory_base.repeat(decoder_inp_token.shape[0], 1, 1)
            else:
                memory = memory_base
            token_state = hidden_states[0][:, -1:, :]
            cross_out, _ = self.cross_attn(self.cross_norm(token_state), memory, memory, attn_mask=None, need_weights=False)
            token_state = self.norm(token_state + cross_out)

            # 计算logits并筛选top_k
            logits = self.lm_head(token_state)
            top_scores, top_indices = logits[:, -1, :].softmax(-1).topk(top_k, dim=-1)

            # 更新概率并筛选有效候选
            cur_p = cur_p * top_scores
            mask = cur_p > threshold
            if mask.sum().item() == 0: # 全部低于阈值，终止生成，并将本次生成概率最大的token拼接到后面
                decoder_inp_token = torch.cat([decoder_inp_token, top_indices[:, 0].unsqueeze(-1)], dim=1)
                all_candidates_new.append(decoder_inp_token[:, 1:])
                break

            decoder_inp_token = torch.cat([decoder_inp_token.unsqueeze(1).repeat(1, top_k, 1), top_indices.unsqueeze(-1)], dim=2)
            # 将概率低于阈值的序列添加到候选序列中
            all_candidates_new.append(decoder_inp_token[~mask][:, 1:])
            # 仅保留概率高于阈值的序列
            decoder_inp_token, cur_p = decoder_inp_token[mask], cur_p[mask].unsqueeze(-1)

        seq, attention_mask, division, position_ids = Flatten_tree(all_candidates_new)
        return seq.unsqueeze(0), attention_mask, division, position_ids
