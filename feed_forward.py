import torch
from torch import nn
from config import GPTConfig


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        result = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.0044715 * (x ** 3))))
        return result


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(config.emb_dim, 4 * config.emb_dim), GeLU(), nn.Linear(4 * config.emb_dim, config.emb_dim))

    def forward(self, x):
        return self.layers(x)
