from __future__ import annotations

from typing import Optional, List, Tuple, Dict

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


def soft_ce_topk_to_teacher(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    topk: int = 10,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Soft CE on teacher topK only:
      p = softmax(teacher/T) restricted to topK and renormalized
      loss = - sum_{i in topK} p_i * log q_i   (q = softmax(student/T))
    """
    T = max(float(temperature), 1e-6)
    # teacher probs
    with torch.no_grad():
        teacher_prob_full = F.softmax(teacher_logits / T, dim=-1)  # (B,V)
        topv, topi = torch.topk(teacher_prob_full, k=topk, dim=-1)  # (B,K)
        p = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # renorm (B,K)

    # student logprob
    student_logprob_full = F.log_softmax(student_logits / T, dim=-1)  # (B,V)
    logq = torch.gather(student_logprob_full, dim=-1, index=topi)     # (B,K)

    loss = -(p * logq).sum(dim=-1).mean()  # average over batch
    return loss * (T * T)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


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
        base_lm_head: Optional[nn.Module] = None,
        adapter_hidden_multiplier: int = 2,
    ):
        super().__init__()
        assert base_lm_head is not None, "LatentARTransformerHead2 requires base_lm_head for shared projection."

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # ✅ adapter: latent-state -> base-hidden-like
        mid = hidden_size * adapter_hidden_multiplier
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, mid, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, hidden_size, bias=True),
        )
        # ✅ zero-init last layer for stability
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        self.out_norm = RMSNorm(hidden_size)

        # ✅ shared lm_head (frozen)
        self.lm_head = base_lm_head
        for p in self.lm_head.parameters():
            p.requires_grad = False

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

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

        # ✅ adapter refine + residual + RMSNorm
        state2 = state + self.adapter(state)
        state2 = self.out_norm(state2)

        logits = self.lm_head(state2)                 # (B, V)
        return logits, state2



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
            base_lm_head=self.base_model.lm_head,               # ✅共享
            adapter_hidden_multiplier=2,
        )

        self.embedding = self.base_model.get_input_embeddings()

    def _topk_from_logits(self, logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)
        vals, ids = torch.topk(probs, k=k, dim=-1)
        return ids, vals  # (B,k), (B,k)


    @torch.no_grad()
    def _base_greedy_rollout(
        self,
        input_ctx: torch.Tensor,
        attn_ctx: Optional[torch.Tensor],
        num_phases: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Base-model greedy rollout aligned to speculative setting.

        Given a *context* x<=t (input_ctx), compute greedy tokens:
        g1 = argmax P_base(. | x<=t)                 (token at t+1)
        g2 = argmax P_base(. | x<=t, g1)             (token at t+2)
        ...
        gK = argmax P_base(. | x<=t, g1..g_{K-1})    (token at t+K)

        Also return teacher logits for head2 phases:
        teacher_head2_logits[i-2] = logits for token g_i, i=2..K
        (i.e., distribution after feeding g_{i-1})

        Returns dict with:
        base_next_logits: (B,V)         logits for g1 distribution
        base_greedy_tokens: (B,K)       [g1..gK]
        teacher_head2_logits: (B,K-1,V) logits for g2..gK
        """
        device = input_ctx.device
        B, ctx_len = input_ctx.shape
        K = max(int(num_phases), 1)

        # Ensure attention mask exists
        if attn_ctx is None:
            attn = torch.ones((B, ctx_len), device=device, dtype=torch.long)
        else:
            attn = attn_ctx.to(device)

        # Prefill on context
        out = self.base_model(
            input_ids=input_ctx,
            attention_mask=attn,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        past = out.past_key_values
        base_next_logits = out.logits[:, -1, :]  # (B,V) for g1
        g1 = torch.argmax(base_next_logits, dim=-1, keepdim=True)  # (B,1)

        greedy_tokens = [g1]         # g1..gK (B,1) tensors
        teacher_logits_list = []     # logits for g2..gK

        cur = g1
        cur_attn = attn

        # Rollout remaining phases to get g2..gK
        for _step in range(2, K + 1):
            one = torch.ones((B, 1), device=device, dtype=cur_attn.dtype)
            cur_attn = torch.cat([cur_attn, one], dim=1)

            step_out = self.base_model(
                input_ids=cur,
                attention_mask=cur_attn,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            past = step_out.past_key_values
            logits = step_out.logits[:, -1, :]  # dist for next token g_i
            teacher_logits_list.append(logits)
            g = torch.argmax(logits, dim=-1, keepdim=True)
            greedy_tokens.append(g)
            cur = g

        base_greedy_tokens = torch.cat(greedy_tokens, dim=1)  # (B,K)
        teacher_head2_logits = (
            torch.stack(teacher_logits_list, dim=1)
            if len(teacher_logits_list) > 0
            else torch.empty((B, 0, base_next_logits.shape[-1]), device=device, dtype=base_next_logits.dtype)
        )
        return {
            "base_next_logits": base_next_logits,
            "base_greedy_tokens": base_greedy_tokens,
            "teacher_head2_logits": teacher_head2_logits,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_phases: Optional[int] = None,
        temperature: float = 1.0,
        train_stage: str = "all",   # "head1" 或 "all"
        infer_mode: bool = False,   # True: 只用 prefix，不依赖 future tokens，不回传 loss
        teacher_mode: str = "teacher_forced",  # "teacher_forced" | "base_greedy"
    ) -> DualDecoderOutput:
        """
        Phase definitions (num_phases=K):
        phase1: head1 predicts token at t+1 (candidate set C1)
        phase2..K: head2 autoregresses over latent z_{<=i} to predict token at t+i (candidate sets C2..CK)

        teacher_mode:
        - teacher_forced: use base_model logits aligned to *gold* future tokens in input_ids (training-only).
        - base_greedy:    use base_model greedy rollout:
                g1 = argmax P_base(.|x<=t)
                g2 = argmax P_base(.|x<=t,g1)
                ...
            and distill head2 to P_base(.|x<=t,g1..g_{i-1}).

        Key guarantee for analysis/eval:
        When K=3, a **single forward** returns:
            head1_logits (phase1),
            head2_logits_all[:,0] (phase2),
            head2_logits_all[:,1] (phase3),
        plus corresponding candidate groups in candidate_ids[:,phase,:].
        """
        if num_phases is None:
            num_phases = 1 + int(self.config.max_speculative_steps)
        K = max(int(num_phases), 1)

        B, T_full = input_ids.shape

        # -------------------------
        # Determine context / split
        # -------------------------
        if infer_mode:
            # input_ids is prefix-only: x<=t
            ctx_len = T_full
            input_ctx = input_ids
            attn_ctx = attention_mask if attention_mask is not None else None
        else:
            # training: input_ids ends with K future tokens for teacher-forced alignment
            if T_full < K + 1:
                K = 1
            ctx_len = T_full - K
            input_ctx = input_ids[:, :ctx_len]
            attn_ctx = attention_mask[:, :ctx_len] if attention_mask is not None else None

        # -------------------------
        # Base forward (for fusion history)
        # -------------------------
        # For base_greedy we only need prefix; for teacher_forced we need full sequence logits.
        base_input_ids = input_ctx if (infer_mode or teacher_mode == "base_greedy") else input_ids
        base_attn = attn_ctx if (infer_mode or teacher_mode == "base_greedy") else attention_mask

        with torch.no_grad():
            base_out = self.base_model(
                input_ids=base_input_ids,
                attention_mask=base_attn,
                output_hidden_states=True,
                use_cache=(teacher_mode == "base_greedy"),
                return_dict=True,
            )
        base_hidden_full = base_out.hidden_states[-1]  # (B, T_base, H)
        base_logits_full = base_out.logits             # (B, T_base, V)

        # base history for fusion conditioning (h_{<=t})
        base_hidden_ctx = base_hidden_full[:, :ctx_len, :]  # (B,ctx_len,H)

        # -------------------------
        # Optional base-greedy teacher rollout
        # -------------------------
        base_next_logits = None
        base_greedy_tokens = None
        teacher_head2_logits = None

        if teacher_mode == "base_greedy":
            rollout = self._base_greedy_rollout(input_ctx=input_ctx, attn_ctx=attn_ctx, num_phases=K)
            base_next_logits = rollout["base_next_logits"]             # (B,V)
            base_greedy_tokens = rollout["base_greedy_tokens"]         # (B,K)  tokens g1..gK
            teacher_head2_logits = rollout["teacher_head2_logits"]     # (B,K-1,V) for g2..gK
        else:
            base_next_logits = base_logits_full[:, ctx_len - 1, :]

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

        # collect groups
        candidate_groups: List[torch.Tensor] = [C1_ids]
        belief_groups: List[torch.Tensor] = [C1_belief]
        state_groups: List[torch.Tensor] = [s1]
        latent_list: List[torch.Tensor] = [z1]

        head1_loss = None
        if not infer_mode:
            head1_loss = kl_to_teacher(head1_logits, base_next_logits, temperature)

        # ========== Stage A: only train head1 ==========
        if (not infer_mode) and train_stage == "head1":
            total_loss = self.config.draft_loss_weight * head1_loss

            max_k = max(g.shape[1] for g in candidate_groups)
            padded = []
            for g in candidate_groups:
                if g.shape[1] < max_k:
                    pad = torch.full((B, max_k - g.shape[1]), -1, device=g.device, dtype=g.dtype)
                    padded.append(torch.cat([g, pad], dim=1))
                else:
                    padded.append(g)
            candidate_groups_tensor = torch.stack(padded, dim=1)  # (B,1,max_k)

            return DualDecoderOutput(
                loss=total_loss,
                head1_loss=head1_loss,
                head2_loss=None,
                draft_logits=head1_logits,
                fusion_logits=head1_logits,
                candidate_ids=candidate_groups_tensor,
                fused_latent=z1,
                head2_logits=None,
                head2_logits_all=None,
                base_greedy_tokens=base_greedy_tokens,
                teacher_head2_logits=teacher_head2_logits,
                teacher_phase1_logits=base_next_logits,
            )

        # --------------------------
        # Phase 2..K: head2 loop (single forward produces all phase2..K logits)
        # --------------------------
        head2_losses: List[torch.Tensor] = []
        head2_logits_list: List[torch.Tensor] = []
        last_head2_logits = None

        z_prefix = z1.unsqueeze(1)  # (B,1,H)

        for phase_idx in range(2, K + 1):
            head2_logits, s_i = self.head2(z_prefix)  # (B,V), (B,H)
            last_head2_logits = head2_logits
            head2_logits_list.append(head2_logits)

            cand_k = int(getattr(self.config, "num_fusion_candidates", 10))
            Ck_ids, Ck_belief = self._topk_from_logits(head2_logits, k=cand_k)
            candidate_groups.append(Ck_ids)
            belief_groups.append(Ck_belief)
            state_groups.append(s_i)

            Ck_emb = self.embedding(Ck_ids)
            z_next = self.fusion(
                candidate_embeddings=Ck_emb,
                s_t1=s_i,
                history_hidden=base_hidden_ctx,
            )
            latent_list.append(z_next)
            z_prefix = torch.cat([z_prefix, z_next.unsqueeze(1)], dim=1)

            if not infer_mode:
                if teacher_mode == "base_greedy":
                    teacher_logits = teacher_head2_logits[:, phase_idx - 2, :]
                else:
                    teacher_pos = (ctx_len - 1) + (phase_idx - 1)
                    teacher_pos = min(teacher_pos, base_logits_full.shape[1] - 1)
                    teacher_logits = base_logits_full[:, teacher_pos, :]

                topk_ce = soft_ce_topk_to_teacher(
                    student_logits=head2_logits,
                    teacher_logits=teacher_logits,
                    topk=getattr(self.config, "head2_teacher_topk", 10),
                    temperature=temperature,
                )
                kl = kl_to_teacher(
                    student_logits=head2_logits,
                    teacher_logits=teacher_logits,
                    temperature=temperature,
                )
                w_ce = float(getattr(self.config, "head2_soft_ce_weight", 1.0))
                w_kl = float(getattr(self.config, "head2_kl_weight", 0.3))
                head2_losses.append(w_ce * topk_ce + w_kl * kl)

        head2_loss = None
        if not infer_mode and len(head2_losses) > 0:
            head2_loss = torch.stack(head2_losses).mean()

        total_loss = None
        if not infer_mode:
            total_loss = self.config.draft_loss_weight * head1_loss
            if head2_loss is not None:
                total_loss = total_loss + self.config.fusion_loss_weight * head2_loss

        max_k = max(g.shape[1] for g in candidate_groups)
        padded = []
        for g in candidate_groups:
            if g.shape[1] < max_k:
                pad = torch.full((B, max_k - g.shape[1]), -1, device=g.device, dtype=g.dtype)
                padded.append(torch.cat([g, pad], dim=1))
            else:
                padded.append(g)
        candidate_groups_tensor = torch.stack(padded, dim=1)  # (B,K,max_k)

        head2_logits_all = torch.stack(head2_logits_list, dim=1) if len(head2_logits_list) > 0 else None

        return DualDecoderOutput(
            loss=total_loss,
            head1_loss=head1_loss,
            head2_loss=head2_loss,
            draft_logits=head1_logits,
            fusion_logits=last_head2_logits if last_head2_logits is not None else head1_logits,
            candidate_ids=candidate_groups_tensor,
            fused_latent=z1,
            head2_logits=last_head2_logits,
            head2_logits_all=head2_logits_all,
            base_greedy_tokens=base_greedy_tokens,
            teacher_head2_logits=teacher_head2_logits,
            teacher_phase1_logits=base_next_logits,
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
