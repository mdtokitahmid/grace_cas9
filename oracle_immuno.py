"""
oracle_immuno.py — the immunogenicity oracle (the thing you swap in).

CONTRACT
--------
The oracle is a torch.nn.Module that scores ONE 9-mer:

    input :  (B, 20, 9)   soft one-hot 9-mers   (20 amino acids, alphabetical: ACDEFGHIKLMNPQRSTVWY)
    output:  (B,)         immunogenicity score   (higher = more immunogenic)

It must be differentiable w.r.t. the input (so gradients can flow back to the
sequence). That's the only requirement. Your real model can be an ensemble,
have a regression head, whatever — as long as it takes (B,20,9) and returns (B,).

Until your real oracle exists, `ImmunoStub` lets the whole pipeline run.
Swap it out by pointing `--oracle /path/to/model.pt` at your saved module.
"""

import torch
import torch.nn as nn


class ImmunoStub(nn.Module):
    """
    Placeholder 9-mer immunogenicity predictor. A tiny MLP with fixed random
    weights — produces a smooth, differentiable score so you can test the
    optimizer end-to-end. REPLACE with your real ensemble.
    """

    def __init__(self, hidden: int = 64, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Flatten(),                 # (B, 20*9)
            nn.Linear(20 * 9, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # deterministic init so the stub gives repeatable scores
        for p in self.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.1
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, 20, 9)
        return self.net(x).squeeze(-1)                    # (B,)


def load_immuno_oracle(path: str, device: str,
                       hla_a: str = None, hla_b: str = None,
                       hla_c: str = None,
                       max_batch: int = 4096) -> nn.Module:
    """
    Load the immunogenicity oracle.

    path == "stub"      -> ImmunoStub (placeholder)
    path == "mhcflurry" -> MHCFlurryOracle (requires --hla-a/b/c)
    otherwise torch.load(path):
        * a full nn.Module                       -> used directly
        * a dict with "model" being an nn.Module -> that module
    Anything else raises, with a reminder of the contract.
    """
    if not path or path == "stub":
        print(f"[oracle] using ImmunoStub (path={path!r})")
        return ImmunoStub().to(device).eval()

    if path == "mhcflurry":
        from oracle_mhcflurry import load_mhcflurry_oracle
        if not (hla_a and hla_b and hla_c):
            raise ValueError(
                "--oracle mhcflurry requires --hla-a, --hla-b, --hla-c "
                "allele frequency CSVs")
        return load_mhcflurry_oracle(hla_a, hla_b, hla_c, device=device,
                                     max_batch=max_batch)

    if not __import__("os").path.exists(path):
        print(f"[oracle] file not found, using ImmunoStub (path={path!r})")
        return ImmunoStub().to(device).eval()

    obj = torch.load(path, map_location=device)
    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict) and isinstance(obj.get("model"), nn.Module):
        model = obj["model"]
    else:
        raise ValueError(
            f"Could not load an immunogenicity oracle from {path!r}.\n"
            "Expected a torch nn.Module (or a dict with key 'model' = nn.Module) "
            "that maps a (B,20,9) 9-mer to a (B,) score."
        )
    print(f"[oracle] loaded {type(model).__name__} from {path}")
    return model.to(device).eval()
