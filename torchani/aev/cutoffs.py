import torch
import math
from torch import Tensor
from ..compat import Final


class CutoffCosine(torch.nn.Module):

    def __init__(self, cutoff: float):
        super().__init__()
        self.register_buffer('cutoff', torch.tensor(cutoff))

    def forward(self, distances: Tensor) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return 0.5 * torch.cos((distances / self.cutoff) * math.pi ) + 0.5


class CutoffSmooth(torch.nn.Module):

    order: Final[int]

    def __init__(self, cutoff: float, eps: float = 1e-10, order: int = 2):
        super().__init__()
        # higher orders make the cutoff more similar to 1
        # for a wider range of distances, before the cutoff.
        # lower orders distort the underlying function more
        assert order > 0, "order must be a positive integer greater than zero"
        assert order % 2 == 0, "Order must be even"
        self.order = order
        self.register_buffer('cutoff', torch.tensor(cutoff))
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, distances: Tensor) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        e = 1 - 1 / (1 - (distances / self.cutoff) ** self.order).clamp(min=self.eps)
        return torch.exp(e)
