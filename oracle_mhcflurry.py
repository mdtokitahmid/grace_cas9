"""
oracle_mhcflurry.py — differentiable MHCFlurry-based immunogenicity oracle for GRACE.

Uses both MHCFlurry ensembles (binding affinity + antigen processing) to compute
a differentiable presentation score, then applies a per-allele percentile
normalization (fitted as a smooth differentiable function) to produce IRS-like
scores that weight alleles comparably.

The forward pass for each allele:
  1. Affinity network:  soft one-hot → BLOSUM62 → pad → pan-allele net → sigmoid (affinity)
  2. Processing network: soft one-hot → BLOSUM62 → pad_middle → processing net → sigmoid (processing)
  3. Presentation:  logistic_regression(affinity_score, processing_score) → presentation_score
  4. Normalization:  differentiable percentile approximation (linear interp of pre-fitted CDF)
  5. IRS (immunogeNN):  freq-weighted sum of min(1/percentile, cap) across all alleles
"""

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

GRACE_AA = "ACDEFGHIKLMNPQRSTVWY"
MHCF_AA = "ACDEFGHIKLMNPQRSTWYVX"

GRACE_TO_MHCF = [MHCF_AA.index(aa) for aa in GRACE_AA]


def _differentiable_percentile(scores, bin_edges, cdf):
    """Differentiable approximation of percentile transform via linear interpolation."""
    idx_float = torch.searchsorted(bin_edges, scores.contiguous()).float()
    idx_float = idx_float.clamp(1, len(cdf) - 2)
    idx_lo = idx_float.long() - 1
    idx_hi = idx_lo + 1
    frac = scores - bin_edges[idx_lo.clamp(0, len(bin_edges) - 1)]
    bin_width = bin_edges[idx_hi.clamp(0, len(bin_edges) - 1)] - bin_edges[idx_lo.clamp(0, len(bin_edges) - 1)]
    bin_width = bin_width.clamp(min=1e-8)
    alpha = (frac / bin_width).clamp(0, 1)
    return cdf[idx_lo + 1] * (1 - alpha) + cdf[idx_hi + 1] * alpha


class MHCFlurryOracle(nn.Module):
    """Differentiable immunogenicity oracle using MHCFlurry's full presentation pipeline.

    Parameters
    ----------
    hla_a, hla_b, hla_c : str
        Paths to CSVs with columns (allele, frequency).
    device : str
        "cuda" or "cpu".
    """

    def __init__(self, hla_a: str, hla_b: str, hla_c: str,
                 device: str = "cuda", irs_cap: float = 5.0,
                 max_batch: int = 4096):
        super().__init__()
        from mhcflurry import Class1PresentationPredictor
        from mhcflurry.amino_acid import BLOSUM62_MATRIX

        self.irs_cap = irs_cap
        self.max_batch = max_batch
        self._pres_predictor = Class1PresentationPredictor.load()
        self._affinity_predictor = self._pres_predictor.affinity_predictor

        alleles, freqs, locus_ids = self._load_alleles(hla_a, hla_b, hla_c)
        self._allele_names = alleles
        self._n_loci = 3

        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))
        self.register_buffer("locus_ids", torch.tensor(locus_ids, dtype=torch.long))

        blosum = torch.tensor(BLOSUM62_MATRIX.values.astype(np.float32))
        self.register_buffer("blosum", blosum)

        perm = torch.tensor(GRACE_TO_MHCF, dtype=torch.long)
        self.register_buffer("perm", perm)

        x_blosum = torch.zeros(21, dtype=torch.float32)
        x_blosum[MHCF_AA.index("X")] = 1.0
        x_encoding = (x_blosum @ blosum).unsqueeze(0)  # (1, 21)
        self.register_buffer("x_encoding", x_encoding)

        # LR weights for presentation score
        w = self._pres_predictor.weights_dataframe.loc["without_flanks"]
        self.register_buffer("lr_intercept", torch.tensor(w["intercept"], dtype=torch.float32))
        self.register_buffer("lr_w_affinity", torch.tensor(w["affinity_score"], dtype=torch.float32))
        self.register_buffer("lr_w_processing", torch.tensor(w["processing_score"], dtype=torch.float32))

        # Percentile transform (differentiable linear interpolation)
        prt = self._pres_predictor.percent_rank_transform
        self.register_buffer("prt_bin_edges", torch.tensor(prt.bin_edges.astype(np.float64), dtype=torch.float32))
        self.register_buffer("prt_cdf", torch.tensor(prt.cdf.astype(np.float64), dtype=torch.float32))

        self._affinity_max_length = 15
        self._processing_max_length = 15
        self._pep_len = 9

        self._setup_models(alleles, device)

    def _load_alleles(self, hla_a, hla_b, hla_c):
        alleles, freqs, locus_ids = [], [], []
        for locus_id, path in enumerate([hla_a, hla_b, hla_c]):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                allele = row["allele"]
                supported = self._affinity_predictor.supported_alleles
                resolved = None
                for candidate in [allele, "HLA-" + allele]:
                    if candidate in supported:
                        resolved = candidate
                        break
                if resolved is not None:
                    alleles.append(resolved)
                    freqs.append(float(row["frequency"]))
                    locus_ids.append(locus_id)
                else:
                    print(f"[oracle_mhcflurry] WARNING: allele {allele} not supported")
        return alleles, freqs, locus_ids

    def _setup_models(self, alleles, device):
        from mhcflurry.allele_encoding import AlleleEncoding

        # --- Affinity model ---
        pan_models = self._affinity_predictor.class1_pan_allele_models
        if not pan_models:
            raise RuntimeError("No pan-allele affinity models found")
        aff_model_obj = pan_models[0]

        allele_encoding = AlleleEncoding(
            alleles=alleles,
            allele_to_sequence=self._affinity_predictor.allele_to_sequence,
        )
        allele_input, allele_reps = aff_model_obj.allele_encoding_to_network_input(
            allele_encoding)
        aff_model_obj.set_allele_representations(allele_reps)
        self.affinity_network = aff_model_obj.network(borrow=True)
        self.affinity_network.to(device)
        self.affinity_network.eval()

        self.register_buffer(
            "allele_input",
            torch.tensor(np.array(allele_input), dtype=torch.long))

        # --- Processing models (ensemble of 8) ---
        proc_predictor = self._pres_predictor.processing_predictor_without_flanks
        if proc_predictor is None:
            raise RuntimeError("No processing predictor (without flanks) found")
        self.processing_networks = nn.ModuleList()
        for proc_model_obj in proc_predictor.models:
            net = proc_model_obj.network()
            net.to(device)
            net.eval()
            self.processing_networks.append(net)

    def _grace_to_mhcf_onehot(self, x):
        """(B, 20, 9) GRACE -> (B, 21, 9) MHCFlurry AA order, with X=0."""
        B = x.shape[0]
        x_mhcf = torch.zeros(B, 21, self._pep_len, device=x.device, dtype=x.dtype)
        x_mhcf[:, self.perm, :] = x
        return x_mhcf

    def _soft_blosum(self, x_mhcf):
        """(B, 21, L) one-hot -> (B, L, 21) BLOSUM62 encoded."""
        return torch.einsum("bap,ae->bpe", x_mhcf, self.blosum)

    def _pad_affinity(self, encoded):
        """(B, 9, 21) -> (B, 45, 21) left_pad_centered_right_pad for affinity model."""
        B = encoded.shape[0]
        ml = self._affinity_max_length
        padded = self.x_encoding.expand(B, ml * 3, -1).clone()

        # Left-aligned: positions 0..8
        padded[:, :self._pep_len, :] = encoded
        # Centered: offset = ml + floor((ml-pep)/2)
        pad_left = math.floor((ml - self._pep_len) / 2)
        c = ml + pad_left
        padded[:, c:c + self._pep_len, :] = encoded
        # Right-aligned: offset = 2*ml + (ml-pep)
        r = 2 * ml + (ml - self._pep_len)
        padded[:, r:r + self._pep_len, :] = encoded
        return padded

    def _pad_processing(self, encoded):
        """(B, 9, 21) -> (B, 15, 21) left-aligned padding for processing model."""
        B = encoded.shape[0]
        ml = self._processing_max_length
        padded = self.x_encoding.expand(B, ml, -1).clone()
        padded[:, :self._pep_len, :] = encoded
        return padded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 20, 9) soft one-hot 9-mers
        Returns : (B,) IRS-like immunogenicity score (higher = more immunogenic)
        """
        if x.shape[0] > self.max_batch:
            return torch.cat([self._forward_chunk(chunk)
                              for chunk in x.split(self.max_batch, dim=0)], dim=0)
        return self._forward_chunk(x)

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_mhcf = self._grace_to_mhcf_onehot(x)  # (B, 21, 9)
        blosum_encoded = self._soft_blosum(x_mhcf)  # (B, 9, 21)

        aff_input = self._pad_affinity(blosum_encoded)  # (B, 45, 21)
        proc_input = self._pad_processing(blosum_encoded)  # (B, 15, 21)

        # Processing score (allele-independent, ensemble average)
        proc_inputs = {
            "sequence": proc_input,
            "peptide_length": torch.full((B, 1), self._pep_len,
                                         device=x.device, dtype=torch.float32),
        }
        proc_out = torch.stack(
            [net(proc_inputs) for net in self.processing_networks], dim=0
        ).mean(dim=0)  # (B,)

        n_alleles = len(self._allele_names)
        irs = torch.zeros(B, device=x.device, dtype=x.dtype)

        for ai in range(n_alleles):
            allele_idx = self.allele_input[ai:ai + 1].expand(B)
            aff_out = self.affinity_network({"peptide": aff_input, "allele": allele_idx})
            affinity_score = aff_out.mean(dim=-1)  # (B,) ensemble average

            # Presentation score = LR(affinity_score, processing_score)
            logit = self.lr_intercept + self.lr_w_affinity * affinity_score + self.lr_w_processing * proc_out
            presentation_score = torch.sigmoid(logit)  # (B,)

            # Differentiable percentile: cdf_value is raw CDF in [0, 100].
            # MHCFlurry's presentation_percentile = 100 - cdf_value.
            cdf_value = _differentiable_percentile(
                presentation_score, self.prt_bin_edges, self.prt_cdf)
            pres_percentile = 100.0 - cdf_value  # (B,) lower = more immunogenic

            # immunogeNN: freq * min(1 / max(percentile, eps), cap)
            inv_rank = torch.clamp(1.0 / pres_percentile.clamp(min=1e-4), max=self.irs_cap)
            irs = irs + self.freqs[ai] * inv_rank
        return irs


def load_mhcflurry_oracle(hla_a: str, hla_b: str, hla_c: str,
                           device: str = "cuda",
                           max_batch: int = 4096) -> nn.Module:
    """Load the MHCFlurry-based differentiable oracle."""
    oracle = MHCFlurryOracle(hla_a, hla_b, hla_c, device=device,
                             max_batch=max_batch)
    oracle.to(device)
    oracle.eval()
    print(f"[oracle_mhcflurry] loaded with {len(oracle._allele_names)} alleles: "
          f"{oracle._allele_names}")
    return oracle
