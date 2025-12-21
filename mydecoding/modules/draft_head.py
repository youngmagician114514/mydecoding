import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateDraftHead(nn.Module):
    """
    Head1 (draft head):

    1) Use a 1-layer Transformer **decoder** ("prefix attention") to compress the full
       prefix hidden states into a single state vector s_{t+1}.
       - tgt:
       - memory: base_hidden (the frozen base model last hidden states)

    2) Use an **independent** MLP-to-vocab head (NOT tied to the base LM head)
       to propose top-k next tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        hidden_proj: int,
        num_mlp_layers: int,
        num_candidates: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # --- Prefix attention: 1-layer Transformer decoder ---
        
        self.prefix_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.prefix_decoder = nn.TransformerDecoder(self.prefix_decoder_layer, num_layers=1)

        # --- Independent MLP head (NOT shared with base LM head) ---
        mlp_layers = []
        in_dim = hidden_size
        # num_mlp_layers counts hidden blocks; final layer projects to vocab
        for _ in range(max(0, num_mlp_layers)):
            mlp_layers.append(nn.Linear(in_dim, hidden_proj))
            mlp_layers.append(nn.SiLU())
            in_dim = hidden_proj
        mlp_layers.append(nn.Linear(in_dim, vocab_size, bias=False))
        self.mlp_head = nn.Sequential(*mlp_layers)

        self.num_candidates = num_candidates


    def forward(self, base_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # base_hidden: (B, T, H)
        B, T, H = base_hidden.shape

        # tgt = 整段 prefix hidden（长度 T）
        tgt = base_hidden  # (B, T, H)
        memory = base_hidden  # 同样用 prefix 作为 memory（如果你坚持用 decoder layer 的结构）

        memory_key_padding_mask = None
        if attention_mask is not None:
            memory_key_padding_mask = (attention_mask == 0)  # (B, T) True=pad

        # causal mask（非常建议加：保持自回归性质）
        # PyTorch 的 tgt_mask: (T, T) 上三角为 True 表示禁止看未来
        tgt_mask = torch.triu(
            torch.ones(T, T, device=base_hidden.device, dtype=torch.bool),
            diagonal=1
        )

        decoded = self.prefix_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, T, H)

        s_t1 = decoded[:, -1, :]  # (B, H)

        draft_logits = self.mlp_head(s_t1)  # (B, V)  独立 vocab head
        probs = torch.softmax(draft_logits, dim=-1)
        belief_values, candidate_ids = torch.topk(probs, k=self.num_candidates, dim=-1)

        return draft_logits, candidate_ids, belief_values, s_t1
    '''
    draft_logits：head1 对整个词表的打分（未归一化）。

    softmax：转成概率。

    topk：取出 k 个最大概率 token：

        candidate_ids：token id

        belief_values：它们的概率
    '''
