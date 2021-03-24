import torch
import math
from torch import Tensor


class CutoffCosine(torch.nn.Module):

    def __init__(self, cutoff: float):
        super().__init__()
        self.register_buffer('cutoff', torch.tensor(cutoff))

    def forward(self, distances: Tensor) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return 0.5 * torch.cos((distances / self.cutoff) * math.pi) + 0.5


class CutoffSmooth(torch.nn.Module):

    def __init__(self, cutoff: float, eps: float = 1e-10):
        super().__init__()
        self.register_buffer('cutoff', torch.tensor(cutoff))
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, distances: Tensor) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        e = 1 - 1 / (1 - (distances / self.cutoff) ** 2).clamp(min=self.eps)
        return torch.exp(e)
