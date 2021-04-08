import torch
from torch import Tensor
from .cutoffs import _parse_cutoff_fn


def _parse_angular_terms(angular_terms):
    # currently only ANI-1 style angular terms or custom are supported
    if angular_terms == 'ani1':
        angular_terms = AngularTermsANI1
    else:
        assert issubclass(angular_terms, torch.nn.Module)
    return angular_terms


def _parse_radial_terms(radial_terms):
    # currently only ANI-1 style radial terms or custom are supported
    if radial_terms == 'ani1':
        radial_terms = RadialTermsANI1
    else:
        assert issubclass(radial_terms, torch.nn.Module)
    return radial_terms


class RadialTermsANI1(torch.nn.Module):
    """Compute the radial subAEV terms of the center atom given neighbors

    This correspond to equation (3) in the `ANI paper`_. This function just
    computes the terms. The sum in the equation is not computed.  The input
    tensor has shape (conformations, atoms, N), where ``N`` is the number of
    neighbor atoms within the cutoff radius and the output tensor should have
    shape (conformations, atoms, ``self.sublength()``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """

    def __init__(self,
                 EtaR: Tensor,
                 ShfR: Tensor,
                 cutoff: float,
                 cutoff_fn='cosine',
                 cutoff_fn_kwargs=None):

        super().__init__()
        # initialize the cutoff function
        cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        cut_args = (cutoff,)
        cut_kwargs = dict() if cutoff_fn_kwargs is None else cutoff_fn_kwargs
        self.cutoff_fn = cutoff_fn(*cut_args, **cut_kwargs)

        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (..., EtaR, ShfR)
        self.register_buffer('EtaR', EtaR.view(-1, 1))
        self.register_buffer('ShfR', ShfR.view(1, -1))
        self.EtaR: Tensor
        self.ShfR: Tensor

    def get_cutoff(self) -> Tensor:
        return self.cutoff_fn.cutoff

    def sublength(self) -> int:
        return self.EtaR.numel() * self.ShfR.numel()

    def forward(self, distances: Tensor) -> Tensor:
        distances = distances.view(-1, 1, 1)
        fc = self.cutoff_fn(distances)
        # Note that in the equation in the paper there is no 0.25
        # coefficient, but in NeuroChem there is such a coefficient.
        # We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-self.EtaR * (distances - self.ShfR)**2) * fc
        # At this point, ret now has shape
        # (conformations x atoms, ?, ?) where ? depend on constants.
        # We then should flat the last 2 dimensions to view the subAEV as a two
        # dimensional tensor (onnx doesn't support negative indices in flatten)
        return ret.flatten(start_dim=1)


class AngularTermsANI1(torch.nn.Module):
    """Compute the angular subAEV terms of the center atom given neighbor pairs.

    This correspond to equation (4) in the `ANI paper`_. This function just
    compute the terms. The sum is not computed.  The input tensor has shape
    (conformations, atoms, N), where N is the number of neighbor atom pairs
    within the cutoff radius and the output tensor should have shape
    (conformations, atoms, ``self.sublength()``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """

    def __init__(self,
                 EtaA: Tensor,
                 Zeta: Tensor,
                 ShfA: Tensor,
                 ShfZ: Tensor,
                 cutoff: float,
                 cutoff_fn='cosine',
                 cutoff_fn_kwargs=None):
        super().__init__()
        # initialize the cutoff function
        cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        cut_args = (cutoff,)
        cut_kwargs = dict() if cutoff_fn_kwargs is None else cutoff_fn_kwargs
        self.cutoff_fn = cutoff_fn(*cut_args, **cut_kwargs)

        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (..., EtaA, Zeta, ShfA, ShfZ)
        self.register_buffer('EtaA', EtaA.view(-1, 1, 1, 1))
        self.register_buffer('Zeta', Zeta.view(1, -1, 1, 1))
        self.register_buffer('ShfA', ShfA.view(1, 1, -1, 1))
        self.register_buffer('ShfZ', ShfZ.view(1, 1, 1, -1))
        self.EtaA: Tensor
        self.Zeta: Tensor
        self.ShfA: Tensor
        self.ShfZ: Tensor

    def get_cutoff(self) -> Tensor:
        return self.cutoff_fn.cutoff

    def sublength(self) -> int:
        return self.EtaA.numel() * self.Zeta.numel() * self.ShfA.numel() * self.ShfZ.numel()

    def forward(self, vectors12: Tensor) -> Tensor:
        vectors12 = vectors12.view(2, -1, 3, 1, 1, 1, 1)
        distances12 = vectors12.norm(2, dim=-5)
        cos_angles = vectors12.prod(0).sum(1) / torch.clamp(
            distances12.prod(0), min=1e-10)
        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
        angles = torch.acos(0.95 * cos_angles)

        fcj12 = self.cutoff_fn(distances12)
        factor1 = ((1 + torch.cos(angles - self.ShfZ)) / 2)**self.Zeta
        factor2 = torch.exp(-self.EtaA * (distances12.sum(0) / 2 - self.ShfA)**2)
        ret = 2 * factor1 * factor2 * fcj12.prod(0)
        # At this point, ret now has shape
        # (conformations x atoms, ?, ?, ?, ?) where ? depend on constants.
        # We then should flat the last 4 dimensions to view the subAEV as a two
        # dimensional tensor (onnx doesn't support negative indices in flatten)
        return ret.flatten(start_dim=1)
