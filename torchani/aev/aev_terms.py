import torch
import math
from torch import Tensor
import sys
from .cutoffs import CutoffCosine, CutoffSmooth

if sys.version_info[:2] < (3, 7):
    class FakeFinal:
        def __getitem__(self, x):
            return x
    Final = FakeFinal()
else:
    from torch.jit import Final


class RadialTerms(torch.nn.Module):

    cutoff: Final[float]

    def __init__(self, EtaR : Tensor, ShfR : Tensor, cutoff : float, cutoff_function='smooth'):
        super().__init__()
        self.cutoff = cutoff

        self.register_buffer('EtaR', EtaR.view(-1, 1))
        self.register_buffer('ShfR', ShfR.view(1, -1))

        if cutoff_function == 'smooth':
            self.cutoff_function = CutoffSmooth(cutoff)
        elif cutoff_function == 'cosine':
            self.cutoff_function = CutoffCosine(cutoff)
        else:
            raise ValueError(f"The cutoff function {cutoff_function} is not implemented")

    def sublength(self) -> int:
        return self.EtaR.numel() * self.ShfR.numel()

    def length(self, num_species: int) -> int:
        return self.sublength() * num_species

    def forward(self, distances: Tensor) -> Tensor:
        """Compute the radial subAEV terms of the center atom given neighbors

        This correspond to equation (3) in the `ANI paper`_. This function just
        compute the terms. The sum in the equation is not computed.
        The input tensor have shape (conformations, atoms, N), where ``N``
        is the number of neighbor atoms within the cutoff radius and output
        tensor should have shape
        (conformations, atoms, ``self.radial_sublength()``)

        .. _ANI paper:
            http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
        """
        distances = distances.view(-1, 1, 1)
        fc = self.cutoff_function(distances)
        # Note that in the equation in the paper there is no 0.25
        # coefficient, but in NeuroChem there is such a coefficient.
        # We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-self.EtaR * (distances - self.ShfR)**2) * fc
        # At this point, ret now has shape
        # (conformations x atoms, ?, ?) where ? depend on constants.
        # We then should flat the last 2 dimensions to view the subAEV as a two
        # dimensional tensor (onnx doesn't support negative indices in flatten)
        return ret.flatten(start_dim=1)



class AngularTerms(torch.nn.Module):

    cutoff: Final[float]

    def __init__(self, EtaA : Tensor, Zeta : Tensor, ShfA : Tensor, ShfZ : Tensor, cutoff : float, cutoff_function='smooth'):
        super().__init__()
        self.cutoff = cutoff

        self.register_buffer('EtaA', EtaA.view(-1, 1, 1, 1))
        self.register_buffer('Zeta', Zeta.view(1, -1, 1, 1))
        self.register_buffer('ShfA', ShfA.view(1, 1, -1, 1))
        self.register_buffer('ShfZ', ShfZ.view(1, 1, 1, -1))

        if cutoff_function == 'smooth':
            self.cutoff_function = CutoffSmooth(cutoff)
        elif cutoff_function == 'cosine':
            self.cutoff_function = CutoffCosine(cutoff)
        else:
            raise ValueError(f"The cutoff function {cutoff_function} is not implemented")

    def sublength(self) -> int:
        return self.EtaA.numel() * self.Zeta.numel() * self.ShfA.numel() * self.ShfZ.numel()

    def length(self, num_species: int) -> int:
        return self.sublength() * num_species * (num_species + 1) // 2

    def forward(self, vectors12: Tensor) -> Tensor:
        """Compute the angular subAEV terms of the center atom given neighbor pairs.

        This correspond to equation (4) in the `ANI paper`_. This function just
        compute the terms. The sum in the equation is not computed.
        The input tensor have shape (conformations, atoms, N), where N
        is the number of neighbor atom pairs within the cutoff radius and
        output tensor should have shape
        (conformations, atoms, ``self.angular_sublength()``)

        .. _ANI paper:
            http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
        """
        vectors12 = vectors12.view(2, -1, 3, 1, 1, 1, 1)
        distances12 = vectors12.norm(2, dim=-5)
        cos_angles = vectors12.prod(0).sum(1) / torch.clamp(distances12.prod(0), min=1e-10)
        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
        angles = torch.acos(0.95 * cos_angles)

        fcj12 = self.cutoff_function(distances12)
        factor1 = ((1 + torch.cos(angles - self.ShfZ)) / 2) ** self.Zeta
        factor2 = torch.exp(-self.EtaA * (distances12.sum(0) / 2 - self.ShfA) ** 2)
        ret = 2 * factor1 * factor2 * fcj12.prod(0)
        # At this point, ret now has shape
        # (conformations x atoms, ?, ?, ?, ?) where ? depend on constants.
        # We then should flat the last 4 dimensions to view the subAEV as a two
        # dimensional tensor (onnx doesn't support negative indices in flatten)
        return ret.flatten(start_dim=1)
