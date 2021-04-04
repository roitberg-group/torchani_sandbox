import torch
import math
import sys
from torch import Tensor

if sys.version_info[:2] < (3, 7):

    class FakeFinal:
        def __getitem__(self, x):
            return x

    Final = FakeFinal()
else:
    from torch.jit import Final


class CutoffCosine(torch.nn.Module):

    def __init__(self, cutoff: float):
        super().__init__()
        self.register_buffer('cutoff', torch.tensor(cutoff))

    def forward(self, distances: Tensor) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return 0.5 * torch.cos((distances / self.cutoff) * math.pi) + 0.5


class CutoffSmooth(torch.nn.Module):

    order: Final[int]

    def __init__(self, cutoff: float, eps: float = 1e-10, order: int = 2):
        super().__init__()
        self.register_buffer('cutoff', torch.tensor(cutoff))
        self.register_buffer('eps', torch.tensor(eps))
        self.order = order

    def forward(self, distances: Tensor) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        e = 1 - 1 / (1 - (distances / self.cutoff).pow(self.order)).clamp(min=self.eps)
        return torch.exp(e)
