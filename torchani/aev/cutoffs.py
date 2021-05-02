import torch
import math
from torch import Tensor
from ..compat import Final


def _parse_cutoff_fn(cutoff_fn):
    # currently only cosine, smooth and custom cutoffs are supported
    if cutoff_fn == 'cosine':
        cutoff_fn = CutoffCosine()
    elif cutoff_fn == 'smooth':
        cutoff_fn = CutoffSmooth()
    elif cutoff_fn == 'physnet':
        cutoff_fn = CutoffPhysNet()
    elif cutoff_fn == 'hip':
        cutoff_fn = CutoffHIP()
    elif cutoff_fn is None:
        cutoff_fn = CutoffDummy()
    else:
        assert isinstance(cutoff_fn, torch.nn.Module)
    return cutoff_fn


class CutoffCosine(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return 0.5 * torch.cos(distances * (math.pi / cutoff)) + 0.5


class CutoffHIP(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return torch.cos((math.pi / 2) * (distances / cutoff)) ** 2


class CutoffPhysNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return 1 - 6 * (distances / cutoff) ** 5 + 15 * (distances / cutoff) ** 4 - 10 * (distances / cutoff) ** 3


class CutoffDummy(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return torch.ones_like(distances)


class CutoffSmooth(torch.nn.Module):

    order: Final[int]
    eps: Final[float]

    def __init__(self, eps: float = 1e-10, order: int = 2):
        super().__init__()
        # higher orders make the cutoff more similar to 1
        # for a wider range of distances, before the cutoff.
        # lower orders distort the underlying function more
        assert order > 0, "order must be a positive integer greater than zero"
        assert order % 2 == 0, "Order must be even"

        self.order = order
        self.eps = eps

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        e = 1 - 1 / (1 - (distances / cutoff) ** self.order).clamp(min=self.eps)
        return torch.exp(e)
