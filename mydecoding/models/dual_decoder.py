
from __future__ import annotations

from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.modules.draft_head import CandidateDraftHead
from mydecoding.modules.fusion import BeliefFusion
from mydecoding.modules.fusion_head import FusionHeadPrefixTransformer
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
    with torch.no_grad():
        teacher_prob_full = F.softmax(teacher_logits / T, dim=-1)  # (B,V)
        topv, topi = torch.topk(teacher_prob_full, k=topk, dim=-1)  # (B,K)
        p = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # (B,K)

    student_logprob_full = F.log_softmax(student_logits / T, dim=-1)  # (B,V)
    logq = torch.gather(student_logprob_full, dim=-1, index=topi)     # (B,K)

    loss = -(p * logq).sum(dim=-1).mean()
    return loss * (T * T)


class DualDecoderModel(nn.Module):
    """
    Draft+Fusion model (your paper diagram, but cleaned naming):

      base_model (frozen): x_{1:t} -> h_{1:t}

      phase1 (draft_head):
          h_{1:t} -> draft_head_logits -> C1_ids, C1_belief, draft_head_state(s1)

      belief_fusion:
          (h_{1:t}, s1, E(C1_ids)*belief) -> z1

      phase2..K (fusion_head):
          (h_{1:t}, z_{1:i-1}) -> fusion_head_logits -> C_i_ids, C_i_belief, fusion_head_state(s_i)

      belief_fusion:
          (h_{1:t}, s_i, E(C_i_ids)*belief) -> z_i

    ✅ Key change vs old code:
      fusion_head now sees BOTH h_{1:t} AND z_1..z_{i-1},
      satisfying your requirement: "head2 input改为h1:t以及z1,z2等".
    """

    def __init__(self, config: DualDecoderConfig, base_model: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        if base_model is None:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
            )
        else:
            self.base_model = base_model

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        # freeze base
        for p in self.base_model.parameters():
            p.requires_grad = False

        hidden_size = self.base_model.config.hidden_size
        vocab_size = self.base_model.config.vocab_size

        # ---- draft_head (phase1) ----
        head1_num_heads = getattr(config, "draft_head_num_heads", None)
        if head1_num_heads is None:
            head1_num_heads = getattr(config, "decoder_num_heads", 8)

        self.draft_head = CandidateDraftHead(
            hidden_size=hidden_size,
            num_candidates=getattr(config, "draft_head_num_candidates", getattr(config, "num_draft_candidates", 3)),
            num_heads=head1_num_heads,
            dropout=getattr(config, "draft_dropout", 0.1),
            base_lm_head=self.base_model.lm_head,
            ffn_multiplier=4,
            num_mlp_layers=getattr(config, "draft_num_layers", 4),
        )

        # ---- belief_fusion ----
        self.belief_fusion = BeliefFusion(
            hidden_size=hidden_size,
            fusion_hidden_size=getattr(config, "fusion_hidden_size", hidden_size),
            num_heads=getattr(config, "fusion_num_heads", 4),
            dropout=getattr(config, "fusion_dropout", 0.1),
        )

        # ---- fusion_head (phase2..K) ----
        self.fusion_head = FusionHeadPrefixTransformer(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=getattr(config, "fusion_head_num_layers", getattr(config, "decoder_num_layers", 2)),
            num_heads=getattr(config, "fusion_head_num_heads", getattr(config, "decoder_num_heads", 8)),
            dropout=getattr(config, "fusion_head_dropout", getattr(config, "decoder_dropout", 0.1)),
            base_lm_head=self.base_model.lm_head,
            adapter_hidden_multiplier=getattr(config, "fusion_head_adapter_hidden_multiplier", 2),
            max_total_len=getattr(config, "fusion_head_max_total_len", 2048),
        )

        self.embedding = self.base_model.get_input_embeddings()

    def _topk_from_logits(self, logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)
        vals, ids = torch.topk(probs, k=k, dim=-1)
        return ids, vals

    @torch.no_grad()
    def _base_greedy_rollout(
        self,
        input_ctx: torch.Tensor,
        attn_ctx: Optional[torch.Tensor],
        num_phases: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Greedy rollout of base model aligned to speculative setting:
          g1 = argmax P_base(.|x<=t)
          g2 = argmax P_base(.|x<=t,g1)
          ...
          gK

        Also returns teacher logits for fusion_head phases:
          teacher_fusion_head_logits[:, i-2, :] = logits for g_i (i=2..K)
        """
        device = input_ctx.device
        B, ctx_len = input_ctx.shape
        K = max(int(num_phases), 1)

        if attn_ctx is None:
            attn = torch.ones((B, ctx_len), device=device, dtype=torch.long)
        else:
            attn = attn_ctx.to(device)

        out = self.base_model(
            input_ids=input_ctx,
            attention_mask=attn,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        past = out.past_key_values
        base_next_logits = out.logits[:, -1, :]  # (B,V)
        g1 = torch.argmax(base_next_logits, dim=-1, keepdim=True)

        greedy_tokens = [g1]
        teacher_logits_list = []

        cur = g1
        cur_attn = attn

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
            logits = step_out.logits[:, -1, :]
            teacher_logits_list.append(logits)
            g = torch.argmax(logits, dim=-1, keepdim=True)
            greedy_tokens.append(g)
            cur = g

        base_greedy_tokens = torch.cat(greedy_tokens, dim=1)  # (B,K)
        teacher_fusion_head_logits = (
            torch.stack(teacher_logits_list, dim=1)
            if len(teacher_logits_list) > 0
            else torch.empty((B, 0, base_next_logits.shape[-1]), device=device, dtype=base_next_logits.dtype)
        )
        return {
            "base_next_logits": base_next_logits,
            "base_greedy_tokens": base_greedy_tokens,
            "teacher_fusion_head_logits": teacher_fusion_head_logits,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_phases: Optional[int] = None,
        temperature: float = 1.0,
        train_stage: str = "all",   # "draft_head" | "all"
        infer_mode: bool = False,
        teacher_mode: str = "teacher_forced",  # "teacher_forced" | "base_greedy"
    ) -> DualDecoderOutput:
        if num_phases is None:
            num_phases = 1 + int(self.config.max_speculative_steps)
        K = max(int(num_phases), 1)

        B, T_full = input_ids.shape

        # ---- Determine context split ----
        if infer_mode:
            ctx_len = T_full
            input_ctx = input_ids
            attn_ctx = attention_mask if attention_mask is not None else None
        else:
            if T_full < K + 1:
                K = 1
            ctx_len = T_full - K
            input_ctx = input_ids[:, :ctx_len]
            attn_ctx = attention_mask[:, :ctx_len] if attention_mask is not None else None

        # ---- Base forward (frozen) ----
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

        base_hidden_ctx = base_hidden_full[:, :ctx_len, :]  # (B,ctx_len,H)

        # ---- Teachers ----
        base_next_logits = None
        base_greedy_tokens = None
        teacher_fusion_head_logits = None

        if teacher_mode == "base_greedy":
            rollout = self._base_greedy_rollout(input_ctx=input_ctx, attn_ctx=attn_ctx, num_phases=K)
            base_next_logits = rollout["base_next_logits"]
            base_greedy_tokens = rollout["base_greedy_tokens"]
            teacher_fusion_head_logits = rollout["teacher_fusion_head_logits"]
        else:
            base_next_logits = base_logits_full[:, ctx_len - 1, :]

        # ==========================
        # Phase1: draft_head
        # ==========================
        draft_head_logits, C1_ids, C1_belief, draft_head_state = self.draft_head(base_hidden_ctx, attn_ctx)

        # belief-weighted candidate embeddings -> z1
        C1_emb = self.embedding(C1_ids) * C1_belief.unsqueeze(-1)
        z1 = self.belief_fusion(
            candidate_embeddings=C1_emb,
            s_t1=draft_head_state,
            history_hidden=base_hidden_ctx,
        )

        candidate_groups: List[torch.Tensor] = [C1_ids]
        belief_groups: List[torch.Tensor] = [C1_belief]
        state_groups: List[torch.Tensor] = [draft_head_state]
        latent_list: List[torch.Tensor] = [z1]

        draft_head_loss = None
        if not infer_mode:
            draft_head_loss = kl_to_teacher(draft_head_logits, base_next_logits, temperature)

        # ---- StageA: train only draft_head ----
        if (not infer_mode) and train_stage in ("draft_head", "head1"):
            total_loss = float(self.config.draft_loss_weight) * draft_head_loss

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
                draft_head_loss=draft_head_loss,
                fusion_head_loss=None,
                draft_head_logits=draft_head_logits,
                fusion_head_logits=draft_head_logits,
                fusion_head_logits_last=None,
                fusion_head_logits_all=None,
                candidate_ids=candidate_groups_tensor,
                fused_latent=z1,
                base_greedy_tokens=base_greedy_tokens,
                teacher_fusion_head_logits=teacher_fusion_head_logits,
                teacher_draft_head_logits=base_next_logits,
            )

        # ==========================
        # Phase2..K: fusion_head loop
        # ==========================
        fusion_head_losses: List[torch.Tensor] = []
        fusion_head_logits_list: List[torch.Tensor] = []
        last_fusion_head_logits = None

        z_prefix = z1.unsqueeze(1)  # (B,1,H)

        for phase_idx in range(2, K + 1):
            fusion_head_logits, fusion_head_state = self.fusion_head(
                base_hidden_ctx=base_hidden_ctx,
                z_prefix=z_prefix,
                attention_mask_ctx=attn_ctx,
            )
            last_fusion_head_logits = fusion_head_logits
            fusion_head_logits_list.append(fusion_head_logits)

            cand_k = int(getattr(self.config, "fusion_head_num_candidates", getattr(self.config, "num_fusion_candidates", 10)))
            Ci_ids, Ci_belief = self._topk_from_logits(fusion_head_logits, k=cand_k)
            candidate_groups.append(Ci_ids)
            belief_groups.append(Ci_belief)
            state_groups.append(fusion_head_state)

            Ci_emb = self.embedding(Ci_ids) * Ci_belief.unsqueeze(-1)
            z_next = self.belief_fusion(
                candidate_embeddings=Ci_emb,
                s_t1=fusion_head_state,
                history_hidden=base_hidden_ctx,
            )
            latent_list.append(z_next)
            z_prefix = torch.cat([z_prefix, z_next.unsqueeze(1)], dim=1)

            if not infer_mode:
                if teacher_mode == "base_greedy":
                    teacher_logits = teacher_fusion_head_logits[:, phase_idx - 2, :]
                else:
                    teacher_pos = (ctx_len - 1) + (phase_idx - 1)
                    teacher_pos = min(teacher_pos, base_logits_full.shape[1] - 1)
                    teacher_logits = base_logits_full[:, teacher_pos, :]

                topk_ce = soft_ce_topk_to_teacher(
                    student_logits=fusion_head_logits,
                    teacher_logits=teacher_logits,
                    topk=int(getattr(self.config, "fusion_head_teacher_topk", getattr(self.config, "head2_teacher_topk", 10))),
                    temperature=temperature,
                )
                kl = kl_to_teacher(
                    student_logits=fusion_head_logits,
                    teacher_logits=teacher_logits,
                    temperature=temperature,
                )
                w_ce = float(getattr(self.config, "fusion_head_soft_ce_weight", getattr(self.config, "head2_soft_ce_weight", 1.0)))
                w_kl = float(getattr(self.config, "fusion_head_kl_weight", getattr(self.config, "head2_kl_weight", 0.3)))
                fusion_head_losses.append(w_ce * topk_ce + w_kl * kl)

        fusion_head_loss = None
        if not infer_mode and len(fusion_head_losses) > 0:
            fusion_head_loss = torch.stack(fusion_head_losses).mean()

        total_loss = None
        if not infer_mode:
            total_loss = float(self.config.draft_loss_weight) * draft_head_loss
            if fusion_head_loss is not None:
                total_loss = total_loss + float(self.config.fusion_loss_weight) * fusion_head_loss

        max_k = max(g.shape[1] for g in candidate_groups)
        padded = []
        for g in candidate_groups:
            if g.shape[1] < max_k:
                pad = torch.full((B, max_k - g.shape[1]), -1, device=g.device, dtype=g.dtype)
                padded.append(torch.cat([g, pad], dim=1))
            else:
                padded.append(g)
        candidate_groups_tensor = torch.stack(padded, dim=1)  # (B,K,max_k)

        fusion_head_logits_all = torch.stack(fusion_head_logits_list, dim=1) if len(fusion_head_logits_list) > 0 else None

        return DualDecoderOutput(
            loss=total_loss,
            draft_head_loss=draft_head_loss,
            fusion_head_loss=fusion_head_loss,
            draft_head_logits=draft_head_logits,
            fusion_head_logits=last_fusion_head_logits if last_fusion_head_logits is not None else draft_head_logits,
            fusion_head_logits_last=last_fusion_head_logits,
            fusion_head_logits_all=fusion_head_logits_all,
            candidate_ids=candidate_groups_tensor,
            fused_latent=z1,
            base_greedy_tokens=base_greedy_tokens,
            teacher_fusion_head_logits=teacher_fusion_head_logits,
            teacher_draft_head_logits=base_next_logits,
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
        A simple speculative-like generator.

        NOTE:
          This is kept minimal to avoid mixing "research code" with "evaluation code".
          For speedup evaluation, use mydecoding/eval_speedup.py.
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
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            base_hidden = base_out.hidden_states[-1]       # (B,T,H)
            base_next_logits = base_out.logits[:, -1, :]   # (B,V)
            base_probs = F.softmax(base_next_logits / max(float(temperature), 1e-6), dim=-1)

            # phase1 draft_head
            draft_head_logits, C1_ids, C1_belief, draft_head_state = self.draft_head(base_hidden, attention_mask)
            C1_emb = self.embedding(C1_ids) * C1_belief.unsqueeze(-1)
            z1 = self.belief_fusion(candidate_embeddings=C1_emb, s_t1=draft_head_state, history_hidden=base_hidden)

            z_prefix = z1.unsqueeze(1)

            decided_token = None

            for phase_idx in range(2, num_phases + 1):
                fusion_head_logits, fusion_head_state = self.fusion_head(
                    base_hidden_ctx=base_hidden,
                    z_prefix=z_prefix,
                    attention_mask_ctx=attention_mask,
                )
                probs = F.softmax(fusion_head_logits / max(float(temperature), 1e-6), dim=-1)

                if phase_idx == 2:
                    # propose token and accept if base agrees
                    draft_token = torch.multinomial(probs, num_samples=1)  # (B,1)
                    draft_score = base_probs.gather(-1, draft_token)
                    if (draft_score > float(acceptance_threshold)).all():
                        decided_token = draft_token
                    else:
                        decided_token = torch.argmax(base_probs, dim=-1, keepdim=True)

                # update latent
                Ck_ids = torch.topk(probs, k=3, dim=-1).indices  # (B,3)
                Ck_belief = torch.gather(probs, -1, Ck_ids)
                Ck_emb = self.embedding(Ck_ids) * Ck_belief.unsqueeze(-1)
                z_next = self.belief_fusion(candidate_embeddings=Ck_emb, s_t1=fusion_head_state, history_hidden=base_hidden)
                z_prefix = torch.cat([z_prefix, z_next.unsqueeze(1)], dim=1)

            if decided_token is None:
                decided_token = torch.argmax(base_probs, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, decided_token], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(decided_token)], dim=1)

            yield decided_token
