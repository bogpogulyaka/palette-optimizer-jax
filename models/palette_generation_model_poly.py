import random

import torch
import torch.nn.functional as F
from torch import nn

from models.positional_encoding import PositionalEncoding


class PaletteGenerationModelPoly(nn.Module):
    def __init__(self, max_color_clusters=256, palette_size=16, polynomial_degree=3, hidden_dim=128, num_layers=4, num_heads=8, stack_mult=1):
        super().__init__()

        self.stack_mult = stack_mult
        self.palette_size = palette_size
        self.polynomial_degree = polynomial_degree

        self.fc_in = nn.Linear(4 * self.stack_mult, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim, 0, max_color_clusters)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                activation=F.gelu,
                norm_first=True,
                batch_first=True,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])
        self.fc_palette = nn.Linear(hidden_dim, palette_size * 3 * self.polynomial_degree)

    def forward(self, colors, weights, loss_ratios):
        x = torch.cat([colors, weights.unsqueeze(-1)], dim=-1)
        x = x - 0.5
        x = x.view(x.shape[0], x.shape[1] // self.stack_mult, x.shape[2] * self.stack_mult)

        x = self.fc_in(x)
        x = self.pe(x)

        # encoder layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)

        x = self.fc_palette(x)
        x = x.permute(0, 2, 1)
        x = torch.mean(x, dim=2).view(x.shape[0], -1, 3, self.polynomial_degree)

        coeffs = x
        x = evaluate_polynomial(loss_ratios.unsqueeze(-1).unsqueeze(-1), coeffs)
        # x = (F.tanh(x) + 1.0) * 0.5

        return x, coeffs


def evaluate_polynomial(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(p.shape[:-1], device=x.device)
    n = p.shape[-1]

    for i in range(n):
        result += x ** (n - 1 - i) * p[..., i]

    return result
    # return p.squeeze(-1) + 0.5
