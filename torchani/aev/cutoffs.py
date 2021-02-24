import torch
import math
from torch import Tensor
import sys

if sys.version_info[:2] < (3, 7):

    class FakeFinal:
        def __getitem__(self, x):
            return x

    Final = FakeFinal()
else:
    from torch.jit import Final


class CutoffCosine(torch.nn.Module):

    cutoff: Final[float]

    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return 0.5 * torch.cos(distances * (math.pi / self.cutoff)) + 0.5


class CutoffSmooth(torch.nn.Module):

    cutoff: Final[float]
    eps: Final[float]
    power: Final[int]

    def __init__(self, cutoff, power=2, eps=1e-10):
        super().__init__()
        self.cutoff = cutoff
        self.power = power
        self.eps = eps

    def forward(self, distances: Tensor) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        e = 1 - 1 / (1 - (distances / self.cutoff)**self.power).clamp(min=self.eps)
        return torch.exp(e)
