import torch
import torch.nn as nn
from ldm.modules.attention import PSTransformerBlock, zero_module

class Generator(nn.Module):
    def __init__(self, context_dim=4096, hidden_dim=128, depth=4, device=None, opt=None):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.opt = opt

        self.proj_in = nn.Linear(3, hidden_dim)

        self.transformer_blocks = nn.ModuleList(
            [PSTransformerBlock(hidden_dim, n_heads=1, d_head=hidden_dim, dropout=0., context_dim=context_dim,
                                disable_self_attn=False, patch_size=self.opt.patch_size)
             for d in range(depth)]
        )

        self.shape_transform = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.Softplus(),
                                             nn.Linear(hidden_dim, 1))

        self.shape_transform[-1].bias.data[0] = -2.0
        self.color_transform = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.Softplus(),
                                             nn.Linear(hidden_dim, 3))

    def forward(self, input, z):
        B, N, _ = input.shape
        x = self.proj_in(input)
        for block in self.transformer_blocks:
            x = block(input, x, context=z)
        shape = self.shape_transform(torch.cat([x, input], dim=-1))
        color = self.color_transform(torch.cat([x, input], dim=-1))
        return torch.cat([shape, color], dim=-1)


