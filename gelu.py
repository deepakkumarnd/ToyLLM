import torch
from torch import nn

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.0044715 * torch.pow(x, 3))))

gelu = GeLU()

layer = torch.randn(3, 4)
print(gelu(layer))