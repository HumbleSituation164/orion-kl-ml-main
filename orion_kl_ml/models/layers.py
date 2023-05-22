import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        norm: bool = True,
        activation: str = "SiLU",
        activation_args: str = "",
    ):
        super().__init__()
        linear = nn.Linear(input_dim, output_dim)
        act_func = eval(f"nn.{activation}({activation_args})")
        drop_layer = nn.Dropout(dropout)
        if norm:
            norm = nn.BatchNorm1d(output_dim)
        else:
            norm = nn.Identity()
        self.layers = nn.Sequential(linear, norm, act_func, drop_layer)

    def forward(self, X: torch.Tensor):
        return self.layers(X)
