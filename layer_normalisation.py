import torch
from torch import nn


class LayerNormalization(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 5e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift


# emb_dim = 5
# sample_input = torch.randn(2, emb_dim)
# sample_network = nn.Sequential(nn.Linear(emb_dim, 6), nn.ReLU())
# ln = LayerNormalization(6)
# out = sample_network(sample_input)
# print("Output layer before layer normalisation")
# print(out)
# print("Mean=", out.mean(dim=-1, keepdim=True))
# print("variance=", out.var(dim=-1, keepdim=True))
# print("Output layer after layer normalisation")
# out = ln(out)
# print(out)
# print("Mean=", out.mean(dim=-1, keepdim=True))
# print("variance=", out.var(dim=-1, keepdim=True))
