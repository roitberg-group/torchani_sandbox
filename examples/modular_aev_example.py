import torch
import torchani
import torch.utils.tensorboard
from torch import Tensor

# this example is meant to show how to take advantage of the modular AEV implementation

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# coordinates and species
coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           device=device)
species = torch.tensor([[1, 0, 0, 0, 0]], device=device)

# suppose we want to make an AEV computer essentially like the 1x aev computer,
# but using a different cutoff function, such as a smooth cutoff
# we can then call:
aev_computer_smooth = torchani.AEVComputer.like_1x(
    cutoff_fn='smooth').to(device)

# and use this AEV Computer normally:
species, aevs = aev_computer_smooth((species, coordinates))

# WARNING: Be very careful, if a model has not been trained using this cutoff function
# then using this aev computer with it will give nonsensical results

# Lets say now we want to experiment with a different cutoff function, such as
# a biweight cutoff (this is a bad idea, biweight does not have a continuous
# second derivative at the cutoff radius value, this is done just as an
# example)

# since biweight is not coded in Torchani we can code it ourselves and pass it
# to the AEVComputer


class CutoffBiweight(torch.nn.Module):
    def __init__(self, cutoff: float):
        super().__init__()
        self.register_buffer('cutoff', torch.tensor(cutoff))
        self.cutoff: Tensor

    def forward(self, distances: Tensor) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return (self.cutoff**2 - distances**2)**2 / self.cutoff**4


cutoff_fn = CutoffBiweight()
aev_computer_bw = torchani.AEVComputer.like_1x(cutoff_fn=cutoff_fn).to(device)
species, aevs = aev_computer_smooth((species, coordinates))


# Now lets try something a bit more complicated. I want to experiment with
# different angular terms that have a form of exp(-alpha * (cos(theta) -
# cos(theta0))**2) how can I do that? I can pass this function to 
# torchani, as long as it exposes the same API as StandardAngular


class AngularCosDiff(torchani.aev.StandardAngular):
    EtaA: Tensor
    Zeta: Tensor
    ShfA: Tensor
    ShfZ: Tensor
    sublength: Final[int]
    cutoff: Final[float]

    def __init__(self,
                 EtaA: Tensor,
                 Zeta: Tensor,
                 ShfA: Tensor,
                 ShfZ: Tensor,
                 cutoff: float,
                 cutoff_fn='cosine'):
        super().__init__()
        # initialize the cutoff function
        self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)

        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (..., EtaA, Zeta, ShfA, ShfZ)
        self.register_buffer('EtaA', EtaA.view(-1, 1, 1, 1))
        self.register_buffer('Zeta', Zeta.view(1, -1, 1, 1))
        self.register_buffer('ShfA', ShfA.view(1, 1, -1, 1))
        self.register_buffer('ShfZ', ShfZ.view(1, 1, 1, -1))
        self.sublength = self.EtaA.numel() * self.Zeta.numel() * self.ShfA.numel() * self.ShfZ.numel()
        self.cutoff = cutoff

    def forward(self, vectors12: Tensor) -> Tensor:
        vectors12 = vectors12.view(2, -1, 3, 1, 1, 1, 1)
        distances12 = vectors12.norm(2, dim=-5)
        cos_angles = vectors12.prod(0).sum(1) / torch.clamp(
            distances12.prod(0), min=1e-10)
        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
        angles = torch.acos(0.95 * cos_angles)

        fcj12 = self.cutoff_fn(distances12, self.cutoff)
        factor1 = ((1 + torch.cos(angles - self.ShfZ)) / 2)**self.Zeta
        factor2 = torch.exp(-self.EtaA * (distances12.sum(0) / 2 - self.ShfA)**2)
        ret = 2 * factor1 * factor2 * fcj12.prod(0)
        # At this point, ret now has shape
        # (conformations x atoms, ?, ?, ?, ?) where ? depend on constants.
        # We then should flat the last 4 dimensions to view the subAEV as a two
        # dimensional tensor (onnx doesn't support negative indices in flatten)
        return ret.flatten(start_dim=1)

    @classmethod
    def cover_linearly(cls, eta: float, num_shifts: int, zeta: float,
            num_angle_sections: int, start: float = 0.9, cutoff: float = 5.2, cutoff_fn='cosine'):
        r""" Builds angular terms by linearly subdividing space in the angular
        dimension and in the radial one up to a cutoff

        "num_shifts" are created, starting from "start" until "cutoff",
        excluding it. "num_angle_sections" does a similar thing for the angles.
        This is the way angular and radial shifts were originally created in
        ANI.

        To reproduce ANI-1x angular terms the signature cutoff=3.5, eta=16.0,
        num_angle_sections=8, num_shifts=4
        """
        EtaA = torch.tensor([eta], dtype=torch.float)
        ShfA = torch.linspace(start, cutoff, int(num_shifts) + 1)[:-1].to(torch.float)
        Zeta = torch.tensor([zeta], dtype=torch.float)
        angle_start = math.pi / (2 * int(num_angle_sections))
        ShfZ = (torch.linspace(0, math.pi, int(num_angle_sections) + 1) + angle_start)[:-1].to(torch.float)
        return cls(EtaA, Zeta, ShfA, ShfZ, cutoff, cutoff_fn)

    @classmethod
    def like_1x(cls, **kwargs):
        return cls.cover_linearly(cutoff=3.5, eta=12.5, zeta=32.0, num_shifts=4, num_angle_sections=8, **kwargs)

    @classmethod
    def like_2x(cls, **kwargs):
        return cls.cover_linearly(cutoff=3.5, eta=19.7, num_shifts=8, zeta=14.1, num_angle_sections=4, **kwargs)


# for legacy aev computer initialization the parameters for the angular and
# radial terms are passed directly to the aev computer and we forward them
# here, otherwise the fully built module is passed, so we just return it,
# and we make sure that the paramters passed are None to prevent confusion
def _parse_angular_terms(angular_terms, cutoff_fn, EtaA, Zeta, ShfA, ShfZ, Rca):

    if isinstance(angular_terms, torch.nn.Module):
        assert EtaA is None
        assert Zeta is None
        assert ShfA is None
        assert ShfZ is None
        assert Rca is None
        assert cutoff_fn is None
        return angular_terms
    else:
        assert isinstance(angular_terms, str)

    # currently only ANI-1 style angular terms or custom are supported
    if angular_terms == 'standard':
        angular_terms = StandardAngular(EtaA, Zeta, ShfA, ShfZ, Rca, cutoff_fn=cutoff_fn)
    elif angular_terms == 'ani1x':
        angular_terms = StandardAngular.like_1x()
    elif angular_terms == 'ani2x':
        angular_terms = StandardAngular.like_2x()
    elif angular_terms == 'ani1ccx':
        angular_terms = StandardAngular.like_1ccx()
        raise ValueError(f'Angular terms {angular_terms} are not implemented')
    return angular_terms


def _parse_radial_terms(radial_terms, cutoff_fn, EtaR, ShfR, Rcr):
    if isinstance(radial_terms, torch.nn.Module):
        assert EtaR is None
        assert ShfR is None
        assert Rcr is None
        assert cutoff_fn is None
        return radial_terms
    else:
        assert isinstance(radial_terms, str)

    if radial_terms == 'standard':
        radial_terms = StandardRadial(EtaR, ShfR, Rcr, cutoff_fn=cutoff_fn)
    elif radial_terms == 'ani1x':
        radial_terms = StandardRadial.like_1x()
    elif radial_terms == 'ani2x':
        radial_terms = StandardRadial.like_2x()
    elif radial_terms == 'ani1ccx':
        radial_terms = StandardRadial.like_1ccx()
        raise ValueError(f'Radial terms {radial_terms} are not implemented')
    return radial_terms








