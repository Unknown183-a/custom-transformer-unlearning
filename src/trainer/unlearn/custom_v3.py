"""CustomV3: V2 objective + in-training gradient diagnostics + switchable mechanisms.

Defaults (forget_loss_type=npo, grad_mode=sum, diag_every=0, kl_fp32=false) are intended to be
mathematically identical to CustomV2 (NPO + retain NLL + masked retain KL), except that the retain
forward pass is reused for the KL term instead of being recomputed (less compute/memory).

Switches (each is a single config override, so ablations change one thing at a time):
  forget_loss_type : "npo" (reference-relative, sequence-sum) | "simnpo" (length-normalised, reference-free)
  grad_mode        : "sum"    -> ordinary summed loss
                     "pcgrad" -> if cos(g_F, g_R) < 0, remove from g_F its component along g_R
                                 (asymmetric: protects the retain direction), then step on g_F' + g_R
  diag_every       : >0 logs ||g_F||, ||g_R||, cos(g_F, g_R) (global + per layer) every N optimizer steps
                     to <output_dir>/grad_diag.jsonl. No weights are changed by measuring.

g_F = grad of gamma * forget_loss ; g_R = grad of (alpha * retain_nll + kl_alpha * masked_kl).

NOTE: diagnostics / pcgrad use torch.autograd.grad, which is not compatible with *reentrant*
gradient checkpointing. Run with trainer.args.gradient_checkpointing=false (0.5B fits on one A6000).
"""
import json
import os
import re
from collections import defaultdict

import torch
import torch.nn.functional as F

from trainer.unlearn.custom_v2 import CustomV2
from trainer.utils import compute_batch_nll, compute_dpo_loss

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


class CustomV3(CustomV2):
    def __init__(
        self,
        forget_loss_type="npo",
        simnpo_beta=4.5,
        simnpo_delta=0.0,
        grad_mode="sum",
        diag_every=0,
        diag_layerwise=True,
        kl_fp32=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert forget_loss_type in ("npo", "simnpo"), forget_loss_type
        assert grad_mode in ("sum", "pcgrad"), grad_mode
        self.forget_loss_type = forget_loss_type
        self.simnpo_beta = simnpo_beta
        self.simnpo_delta = simnpo_delta
        self.grad_mode = grad_mode
        self.diag_every = int(diag_every)
        self.diag_layerwise = diag_layerwise
        self.kl_fp32 = kl_fp32

    # ---------------------------------------------------------------- losses
    def _forget_loss_v3(self, model, forget_inputs):
        if self.forget_loss_type == "npo":
            loss, outputs = compute_dpo_loss(
                model=model,
                ref_model=self.ref_model,
                win_inputs=None,
                lose_inputs=forget_inputs,
                beta=self.beta,
            )
            return loss, outputs[1]
        nll, outputs = compute_batch_nll(model, forget_inputs)  # per-sequence SUM nll
        n_tok = (forget_inputs["labels"][..., 1:] != -100).sum(-1).clamp(min=1)
        x = nll / n_tok - self.simnpo_delta
        loss = -F.logsigmoid(self.simnpo_beta * x).mean() * 2 / self.simnpo_beta
        return loss, outputs

    def _masked_kl_from_logits(self, cur_logits, inputs):
        with torch.no_grad():
            ref_logits = self.ref_model(**inputs).logits
        cur = cur_logits[:, :-1, :]
        ref = ref_logits[:, :-1, :]
        if self.kl_fp32:
            cur, ref = cur.float(), ref.float()
        mask = inputs["labels"][:, 1:].ne(-100)
        tok_kl = F.kl_div(
            F.log_softmax(cur, dim=-1),
            F.log_softmax(ref, dim=-1),
            reduction="none",
            log_target=True,
        ).sum(dim=-1)  # KL(ref || current) per token
        m = mask.to(tok_kl.dtype)
        return (tok_kl * m).sum() / m.sum().clamp(min=1)

    # ---------------------------------------------------------------- main
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        keys = ("input_ids", "attention_mask", "labels")
        forget_inputs = {k: inputs["forget"][k] for k in keys}
        retain_inputs = {k: inputs["retain"][k] for k in keys}

        forget_loss, forget_outputs = self._forget_loss_v3(model, forget_inputs)
        retain_outputs = model(**retain_inputs)
        retain_nll = retain_outputs.loss
        if self.kl_alpha > 0:
            retain_kl = self._masked_kl_from_logits(retain_outputs.logits, retain_inputs)
        else:
            retain_kl = retain_nll.new_zeros(())

        forget_term = self.gamma * forget_loss
        retain_term = self.alpha * retain_nll + self.kl_alpha * retain_kl
        total = forget_term + retain_term

        step = int(self.state.global_step)
        do_diag = (
            self.diag_every > 0 and model.training and step % self.diag_every == 0
        )
        if not (model.training and (do_diag or self.grad_mode == "pcgrad")):
            return (total, forget_outputs) if return_outputs else total

        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        plist = [p for _, p in named]
        keep_graph = self.grad_mode == "sum"  # sum mode still back-propagates `total`
        gF = torch.autograd.grad(forget_term, plist, retain_graph=keep_graph, allow_unused=True)
        gR = torch.autograd.grad(retain_term, plist, retain_graph=keep_graph, allow_unused=True)
        gF = [torch.zeros_like(p) if g is None else g for g, p in zip(gF, plist)]
        gR = [torch.zeros_like(p) if g is None else g for g, p in zip(gR, plist)]

        dot, nf2, nr2 = self._global_dot(gF, gR)
        conflict = bool((dot < 0).item())

        if do_diag:
            self._write_diag(
                step, named, gF, gR, dot, nf2, nr2, conflict,
                float(forget_loss.detach()), float(retain_nll.detach()), float(retain_kl.detach()),
            )

        if self.grad_mode == "sum":
            return (total, forget_outputs) if return_outputs else total

        # ---- pcgrad: protect retain direction
        if conflict:
            coef = dot / (nr2 + 1e-12)
            gF = [a - coef.to(a.dtype) * b for a, b in zip(gF, gR)]
        g_comb = [a + b for a, b in zip(gF, gR)]
        # surrogate whose VALUE equals the real loss and whose GRADIENT equals g_comb
        sur = sum((g.detach().to(p.dtype) * p).sum() for g, p in zip(g_comb, plist))
        surrogate = sur - sur.detach() + total.detach()
        return (surrogate, forget_outputs) if return_outputs else surrogate

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _global_dot(gF, gR):
        dev = gF[0].device
        dot = torch.zeros((), device=dev, dtype=torch.float32)
        nf2 = torch.zeros_like(dot)
        nr2 = torch.zeros_like(dot)
        for a, b in zip(gF, gR):
            a32, b32 = a.float(), b.float()
            dot += (a32 * b32).sum()
            nf2 += (a32 * a32).sum()
            nr2 += (b32 * b32).sum()
        return dot, nf2, nr2

    def _write_diag(self, step, named, gF, gR, dot, nf2, nr2, conflict, lf, lr, lkl):
        if not self.accelerator.is_main_process:
            return
        eps = 1e-12
        rec = {
            "step": step,
            "grad_mode": self.grad_mode,
            "forget_loss": lf,
            "retain_nll": lr,
            "retain_kl": lkl,
            "norm_F": float(nf2.sqrt()),
            "norm_R": float(nr2.sqrt()),
            "cos": float(dot / (nf2.sqrt() * nr2.sqrt() + eps)),
            "conflict": conflict,
            "n_params_measured": int(sum(p.numel() for _, p in named)),
        }
        rec["norm_ratio_F_over_R"] = rec["norm_F"] / (rec["norm_R"] + eps)
        if self.diag_layerwise:
            acc = defaultdict(lambda: [0.0, 0.0, 0.0])
            tmp = defaultdict(lambda: [None, None, None])
            for (n, _), a, b in zip(named, gF, gR):
                m = _LAYER_RE.search(n)
                key = f"layer_{int(m.group(1)):02d}" if m else "other"
                a32, b32 = a.float(), b.float()
                vals = [(a32 * b32).sum(), (a32 * a32).sum(), (b32 * b32).sum()]
                t = tmp[key]
                for i in range(3):
                    t[i] = vals[i] if t[i] is None else t[i] + vals[i]
            layers = {}
            for key, (d, f2, r2) in sorted(tmp.items()):
                d, f2, r2 = float(d), float(f2), float(r2)
                layers[key] = {
                    "norm_F": f2 ** 0.5,
                    "norm_R": r2 ** 0.5,
                    "cos": d / ((f2 ** 0.5) * (r2 ** 0.5) + eps),
                }
            rec["layers"] = layers
        path = os.path.join(self.args.output_dir, "grad_diag.jsonl")
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
