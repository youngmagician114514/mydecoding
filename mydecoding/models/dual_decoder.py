from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.modules.draft_head import CandidateDraftHead
from mydecoding.modules.fusion import BeliefFusion
from mydecoding.models.outputs import DualDecoderOutput


def kl_to_teacher(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Distillation loss: KL( teacher || student ) with optional temperature.
    """
    T = max(float(temperature), 1e-6)
    teacher_prob = F.softmax(teacher_logits / T, dim=-1).detach()
    student_logprob = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(student_logprob, teacher_prob, reduction="batchmean") * (T * T)


class LatentARTransformerHead2(nn.Module):
    """
    Phase2..: Head2 / Decoder2

    - Autoregressive Transformer over latent prefix z_{<=i}
    - NO direct access to base history h_{<=t}

    input:  z_prefix (B, L, H)
    output: logits (B, V), state s_i (B, H)
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, z_prefix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_prefix: (B, L, H)
        returns:
          logits: (B, V)
          state:  (B, H)  (last hidden)
        """
        B, L, H = z_prefix.shape
        causal_mask = torch.triu(
            torch.ones(L, L, device=z_prefix.device, dtype=torch.bool),
            diagonal=1,
        )
        h = self.encoder(z_prefix, mask=causal_mask)  # (B, L, H)
        state = h[:, -1, :]                           # (B, H)
        logits = self.out_proj(state)                 # (B, V)
        return logits, state


class DualDecoderModel(nn.Module):
    """
    Matches your timing diagram:

      base_model (frozen): x_{1:t} -> h_{<=t}
      phase1 (head1):      h_{<=t} -> C_{t+1}, belief, s_{t+1}
      fusion:              (h_{<=t}, s_{t+1}, E(C_{t+1})) -> z_{t+1}

      phase2..N (head2):   z_{<=i} -> C_{t+i}, belief, s_{t+i}
      fusion:              (h_{<=t}, s_{t+i}, E(C_{t+i})) -> z_{t+i}

    Key constraint:
      - base history h_{<=t} is NEVER fed directly into head2.
      - h_{<=t} influences future only through Fusion.
    """

    def __init__(self, config: DualDecoderConfig, base_model: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        if base_model is None:
            self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,torch_dtype=torch.bfloat16, )
        else:
            self.base_model = base_model

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        # freeze base
        for p in self.base_model.parameters():
            p.requires_grad = False

        hidden_size = self.base_model.config.hidden_size
        vocab_size = self.base_model.config.vocab_size

        # Head1 (phase1)
        # config 映射：hidden_proj <- draft_hidden_size, num_mlp_layers <- draft_num_layers
        # num_heads：config 没有 head1_num_heads，用 decoder_num_heads 兜底（也可换成 fusion_num_heads）
        head1_num_heads = getattr(config, "head1_num_heads", None)
        if head1_num_heads is None:
            head1_num_heads = getattr(config, "decoder_num_heads", 8)

        self.head1 = CandidateDraftHead(
            hidden_size=hidden_size,
            num_candidates=config.num_draft_candidates,
            num_heads=head1_num_heads,
            dropout=getattr(config, "draft_dropout", 0.1),
            base_lm_head=self.base_model.lm_head,
            ffn_multiplier=4,  # 和 Qwen FFN 比例一致
            num_mlp_layers=config.draft_num_layers,  # 可以设成 4，模仿 Hydra++
        )

        # Fusion
        self.fusion = BeliefFusion(
            hidden_size=hidden_size,
            fusion_hidden_size=config.fusion_hidden_size,
            num_heads=config.fusion_num_heads,
            dropout=config.fusion_dropout,
        )

        # Head2 (phase2..): latent autoregressive transformer
        self.head2 = LatentARTransformerHead2(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=config.decoder_num_layers,
            num_heads=config.decoder_num_heads,
            dropout=config.decoder_dropout,
        )

        self.embedding = self.base_model.get_input_embeddings()

    def _topk_from_logits(self, logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)
        vals, ids = torch.topk(probs, k=k, dim=-1)
        return ids, vals  # (B,k), (B,k)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_phases: Optional[int] = None,
        temperature: float = 1.0,
        train_stage: str = "all",   # "head1" 或 "all"
    ) -> DualDecoderOutput:
        """
        Training (distillation) convention:
          - total phases = phase1(head1) + phase2..(head2 steps)
          - if num_phases is None: num_phases = 1 + config.max_speculative_steps
          - We assume input_ids contains enough future tokens for teacher logits.

        We split:
          context length = T - K, where K = num_phases
          phase1 aligns with base logits at position (ctx_len-1)
          phase j aligns with base logits at position (ctx_len-1 + (j-1))
        """
        if num_phases is None:
            num_phases = 1 + int(self.config.max_speculative_steps)
        num_phases = max(int(num_phases), 1)

        B, T = input_ids.shape
        K = num_phases
        if T < K + 1:
            # not enough tokens to distill multiple phases -> fall back to 1 phase
            K = 1

        # context / future split for teacher
        ctx_len = T - K
        input_ctx = input_ids[:, :ctx_len]
        attn_ctx = attention_mask[:, :ctx_len] if attention_mask is not None else None

        # base teacher on full sequence
        with torch.no_grad():
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        base_hidden_full = base_out.hidden_states[-1]  # (B,T,H)
        base_logits_full = base_out.logits             # (B,T,V)

        # base history for fusion conditioning (h_{<=t})
        base_hidden_ctx = base_hidden_full[:, :ctx_len, :]  # (B,ctx_len,H)

        # --------------------------
        # Phase 1: head1 -> C_{t+1}
        # --------------------------
        head1_logits, C1_ids, C1_belief, s1 = self.head1(base_hidden_ctx, attn_ctx)

        # fusion -> z_{t+1}
        C1_emb = self.embedding(C1_ids)  # (B,k1,H)
        z1 = self.fusion(
            candidate_embeddings=C1_emb,
            s_t1=s1,
            history_hidden=base_hidden_ctx,
        )  # (B,H)

        # collect groups (each phase one group)
        candidate_groups: List[torch.Tensor] = [C1_ids]       # phase1 group
        belief_groups: List[torch.Tensor] = [C1_belief]
        state_groups: List[torch.Tensor] = [s1]
        latent_list: List[torch.Tensor] = [z1]

        # head1 distill target: base logits at ctx_len-1 predicts token at ctx_len
        head1_loss = kl_to_teacher(head1_logits, base_logits_full[:, ctx_len - 1, :], temperature)

        # ========== Stage A: 只训练 head1 的分支 ==========
        if train_stage == "head1":
            # 不展开 head2，自然也没有 head2_loss
            total_loss = self.config.draft_loss_weight * head1_loss

            # pack candidate_groups 成 tensor，和原来保持接口一致
            max_k = max(g.shape[1] for g in candidate_groups)
            padded = []
            for g in candidate_groups:
                if g.shape[1] < max_k:
                    pad = torch.full((B, max_k - g.shape[1]), -1, device=g.device, dtype=g.dtype)
                    padded.append(torch.cat([g, pad], dim=1))
                else:
                    padded.append(g)
            candidate_groups_tensor = torch.stack(padded, dim=1)  # (B,1,max_k) 这里 K=1

            return DualDecoderOutput(
                loss=total_loss,
                head1_loss=head1_loss,
                head2_loss=None,
                draft_logits=head1_logits,
                fusion_logits=head1_logits,      # 暂时用 head1_logits 占位
                candidate_ids=candidate_groups_tensor,
                fused_latent=z1,
            )
        
        # --------------------------
        # Phase 2..K: head2 loop
        # input: z_{<=i}
        # output: C_{t+i}, belief, s_{t+i}
        # --------------------------
        head2_losses: List[torch.Tensor] = []
        last_head2_logits = None

        z_prefix = z1.unsqueeze(1)  # (B,1,H)

        for phase_idx in range(2, K + 1):
            head2_logits, s_i = self.head2(z_prefix)  # (B,V), (B,H)
            last_head2_logits = head2_logits

            # head2 candidates per your diagram: fixed top-3
            Ck_ids, Ck_belief = self._topk_from_logits(head2_logits, k=3)

            candidate_groups.append(Ck_ids)
            belief_groups.append(Ck_belief)
            state_groups.append(s_i)

            # fusion -> z_{t+phase_idx}
            Ck_emb = self.embedding(Ck_ids)  # (B,3,H)
            z_next = self.fusion(
                candidate_embeddings=Ck_emb,
                s_t1=s_i,
                history_hidden=base_hidden_ctx,
            )  # (B,H)

            latent_list.append(z_next)
            z_prefix = torch.cat([z_prefix, z_next.unsqueeze(1)], dim=1)

            # distill to teacher base distribution
            teacher_pos = (ctx_len - 1) + (phase_idx - 1)
            teacher_pos = min(teacher_pos, T - 1)
            head2_losses.append(kl_to_teacher(head2_logits, base_logits_full[:, teacher_pos, :], temperature))

        head2_loss = torch.stack(head2_losses).mean() if len(head2_losses) > 0 else None

        total_loss = self.config.draft_loss_weight * head1_loss
        if head2_loss is not None:
            total_loss = total_loss + self.config.fusion_loss_weight * head2_loss

        # pack candidate groups into (B, K, max_k) tensor for backward compatibility
        max_k = max(g.shape[1] for g in candidate_groups)
        padded = []
        for g in candidate_groups:
            if g.shape[1] < max_k:
                pad = torch.full((B, max_k - g.shape[1]), -1, device=g.device, dtype=g.dtype)
                padded.append(torch.cat([g, pad], dim=1))
            else:
                padded.append(g)
        candidate_groups_tensor = torch.stack(padded, dim=1)  # (B,K,max_k)

        return DualDecoderOutput(
            loss=total_loss,
            head1_loss=head1_loss,
            head2_loss=head2_loss,
            draft_logits=head1_logits,
            fusion_logits=last_head2_logits if last_head2_logits is not None else head1_logits,
            candidate_ids=candidate_groups_tensor,  # now contains per-phase candidate groups
            fused_latent=z1,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        num_phases: Optional[int] = None,
        temperature: float = 1.0,
        acceptance_threshold: Optional[float] = None,
    ):
        """
        Generation:
          per decoding step:
            base -> h_{<=t}
            phase1: head1 -> C_{t+1}, s1 -> fusion -> z1
            phase2..: head2(z_prefix) -> C, s -> fusion -> z_next
            choose actual next token using head2 (phase2) with optional base acceptance
            return (next_token, candidate_groups_per_phase)
        """
        if num_phases is None:
            num_phases = 1 + int(self.config.max_speculative_steps)
        num_phases = max(int(num_phases), 1)

        if acceptance_threshold is None:
            acceptance_threshold = float(self.config.acceptance_threshold)

        device = self.base_model.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        for _ in range(max_new_tokens):
            # base history
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            base_hidden = base_out.hidden_states[-1]       # (B,T,H)
            base_next_logits = base_out.logits[:, -1, :]   # (B,V)
            base_probs = F.softmax(base_next_logits / max(float(temperature), 1e-6), dim=-1)

            # phase1
            head1_logits, C1_ids, C1_belief, s1 = self.head1(base_hidden, attention_mask)
            C1_emb = self.embedding(C1_ids)
            z1 = self.fusion(candidate_embeddings=C1_emb, s_t1=s1, history_hidden=base_hidden)

            candidate_groups: List[torch.Tensor] = [C1_ids]

            # head2 rollout phases
            z_prefix = z1.unsqueeze(1)
            decided_token = None

            for phase_idx in range(2, num_phases + 1):
                head2_logits, s_i = self.head2(z_prefix)
                probs = F.softmax(head2_logits / max(float(temperature), 1e-6), dim=-1)

                # group for this phase (top-3)
                Ck_ids = torch.topk(probs, k=3, dim=-1).indices
                candidate_groups.append(Ck_ids)

                # decide actual next token at phase2 only
                if phase_idx == 2:
                    draft_token = torch.multinomial(probs, num_samples=1)  # (B,1)
                    draft_score = base_probs.gather(-1, draft_token)
                    if (draft_score > float(acceptance_threshold)).all():
                        decided_token = draft_token
                    else:
                        decided_token = torch.argmax(base_probs, dim=-1, keepdim=True)

                # fusion update latent for next phase
                Ck_emb = self.embedding(Ck_ids)
                z_next = self.fusion(candidate_embeddings=Ck_emb, s_t1=s_i, history_hidden=base_hidden)
                z_prefix = torch.cat([z_prefix, z_next.unsqueeze(1)], dim=1)

            if decided_token is None:
                # if only phase1 exists
                decided_token = torch.argmax(base_probs, dim=-1, keepdim=True)

            # append
            input_ids = torch.cat([input_ids, decided_token], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(decided_token)], dim=1)

            yield decided_token, candidate_groups
