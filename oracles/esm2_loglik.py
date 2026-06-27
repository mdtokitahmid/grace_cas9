"""
ESM2LogLikOracle
────────────────
Zero-shot constraint oracle: measures evolutionary plausibility of a sequence
under ESM-2's learned protein language model.

Score = Σ_i  Σ_k  P(k, i) · log P_ESM2(AA=k | context)_i

where:
  P(k, i)              = current SeqProp distribution at position i, amino acid k
  P_ESM2(AA=k | ...)_i = ESM-2's predicted probability for AA k at position i
                         given the rest of the sequence as context

This oracle requires NO training data — ESM-2 provides the signal directly.

Usage as a GRACE constraint:
    Constraining this oracle to not decrease during optimization prevents
    the optimizer from producing thermostable but evolutionarily implausible
    (i.e. never observed in nature) sequences.

    Biologically: keeps optimized sequences in the "natural protein" manifold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseOracle, OracleConfig
from .esm2_backbone import ESM2SoftBackbone


class ESM2LogLikOracle(BaseOracle):
    """
    ESM-2 pseudo-log-likelihood as a zero-shot constraint oracle.

    Architecture:
        soft one-hot (B, 20, L)
        → ESM2SoftBackbone (frozen)
        → LM head (tied to ESM-2 word embeddings, frozen)
        → log-softmax over 20 AAs
        → inner product with soft distribution x
        → summed over positions
        → (B,) log-likelihood scores

    Higher score = more "natural" / evolutionarily plausible sequence.
    """

    def __init__(self, backbone: ESM2SoftBackbone):
        super().__init__(OracleConfig(name="esm2_loglik", maximize=True))
        self.backbone = backbone

        # LM head: ties weights to ESM-2's word embeddings (same as ESM-2 training)
        # This is a linear map: d_model → vocab_size, weight-tied, no training needed.
        d_model    = backbone.d_model
        vocab_size = backbone.esm.config.vocab_size

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = backbone.esm.embeddings.word_embeddings.weight

        # Freeze — this is a zero-shot oracle
        for p in self.lm_head.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute expected log-likelihood under ESM-2.

        Args:
            x: (B, 20, L) soft one-hot sequences

        Returns:
            loglik: (B,) — higher = more evolutionarily plausible
        """
        B, K, L = x.shape

        # ── ESM-2 hidden states ───────────────────────────────────────────────
        h = self.backbone.soft_forward(x)           # (B, L+2, d_model)

        # ── LM logits → log-probs over vocabulary ────────────────────────────
        seq_h   = h[:, 1:-1, :]                     # (B, L, d_model) — drop BOS/EOS
        logits  = self.lm_head(seq_h)               # (B, L, vocab_size)

        # Restrict to the 20 canonical AA token IDs
        aa_ids    = self.backbone.aa_token_ids       # (20,) long
        aa_logits = logits[:, :, aa_ids]             # (B, L, 20)

        # Log-softmax over the 20 AAs
        log_probs = F.log_softmax(aa_logits, dim=-1) # (B, L, 20)

        # ── Expected log-likelihood under the soft distribution x ─────────────
        # x: (B, 20, L) → permute → (B, L, 20)
        x_t = x.permute(0, 2, 1)                    # (B, L, 20)

        # Element-wise: x_t * log_probs, then sum over AAs and positions
        per_pos_ll = (x_t * log_probs).sum(dim=-1)  # (B, L)
        total_ll   = per_pos_ll.sum(dim=-1)          # (B,)

        return total_ll
