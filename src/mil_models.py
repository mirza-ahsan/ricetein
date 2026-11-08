"""The tile decision-makers (methodology "Aggregator"; Ilse et al. 2018).

Every tile description h_i is projected, g_i = ReLU(W_p h_i + b_p) (width D), with dropout. Then:
- ``attention``: e_i = w^T (tanh(V g_i) * sigmoid(U g_i)), a = softmax(e) over the bag's real tiles,
  z = sum_i a_i g_i (V, U: 128 x D; w has no bias, as in the Stage 0 skeleton);
- ``mean``: z = average of g_i;  ``max``: z = element-wise max of g_i.
All three share the projection and the classifier softmax(W_c z + b_c). Padded tiles are masked out.
"""

from __future__ import annotations

import torch
from torch import nn

ATTENTION_DIM = 128


class MILModel(nn.Module):
    def __init__(self, in_dim: int, width: int, dropout: float, aggregator: str, n_classes: int) -> None:
        super().__init__()
        if aggregator not in ("attention", "mean", "max"):
            raise ValueError(f"unknown aggregator {aggregator}")
        self.aggregator = aggregator
        self.project = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout))
        if aggregator == "attention":
            self.v = nn.Linear(width, ATTENTION_DIM)
            self.u = nn.Linear(width, ATTENTION_DIM)
            self.w = nn.Linear(ATTENTION_DIM, 1, bias=False)
        self.classify = nn.Linear(width, n_classes)

    def forward(self, tiles: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(B,N,d) tiles and a real-tile mask -> class logits and optional per-tile attention."""
        g = self.project(tiles)
        if self.aggregator == "attention":
            scores = self.w(torch.tanh(self.v(g)) * torch.sigmoid(self.u(g))).squeeze(-1)
            weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
            return self.classify((weights.unsqueeze(-1) * g).sum(1)), weights
        if self.aggregator == "mean":
            z = (g * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        else:
            z = g.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
        return self.classify(z), None
