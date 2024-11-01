r"""
Extending the local atomic features: AEVs with custom terms and cutoffs
=======================================================================

TorchANI allows for modification and customization of the AEV features
"""

# To begin with, let's first import the modules and setup devices we will use:
import math

import torch
from torch import Tensor

from torchani.cutoffs import Cutoff
from torchani.aev import ANIRadial, AEVComputer, AngularTerm
from torchani.utils import linspace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This example is meant to show how to take advantage of the modular AEV implementation
# We will use these coordinates and species:
coords = torch.tensor(
    [
        [
            [0.03192167, 0.00638559, 0.01301679],
            [-0.83140486, 0.39370209, -0.26395324],
            [-0.66518241, -0.84461308, 0.20759389],
            [0.45554739, 0.54289633, 0.81170881],
            [0.66091919, -0.16799635, -0.91037834],
        ]
    ],
    device=device,
)
species = torch.tensor([[1, 0, 0, 0, 0]], device=device)

# Suppose that we want to make an aev computer in the ANI 2x style:
aevcomp = AEVComputer.like_2x().to(device)
aevs = aevcomp(species, coords)
radial_len = aevcomp.radial_len
print("AEV computer similar to 2x")
print("for first atom, first 5 terms of radial:", aevs[0, 0, :5].tolist())
print(
    "for first atom, first 5 terms of angular:",
    aevs[0, 0, radial_len:radial_len + 5].tolist(),
)
print()

# suppose we want to make an AEV computer essentially like the 1x aev computer,
# but using a different cutoff function, such as a smooth cutoff
#
# WARNING: Be very careful, if a model has not been trained using this cutoff function
# then using this aev computer with it will give nonsensical results

aevcomp_smooth = AEVComputer.like_1x(cutoff_fn="smooth").to(device)
radial_len = aevcomp_smooth.radial_len
aevs = aevcomp_smooth(species, coords)
print("AEV computer similar to 1x, but with a smooth cutoff")
print("for first atom, first 5 terms of radial:", aevs[0, 0, :5].tolist())
print(
    "for first atom, first 5 terms of angular:",
    aevs[0, 0, radial_len:radial_len + 5].tolist(),
)
print()

# Lets say now we want to experiment with a different cutoff function, such as
# a biweight cutoff (WARNING: biweight does not have a continuous
# second derivative at the cutoff radius value, this is done just as an example)

# Since biweight is not coded in Torchani we can code it ourselves and pass it
# to the AEVComputer, as long as the forward method has this form, it will work!

# The same cutoff function will be used for both radial and angular terms


class CutoffBiweight(Cutoff):
    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return (cutoff**2 - distances**2) ** 2 / cutoff**4


cutoff_fn_biw = CutoffBiweight()
aevcomp_biw = AEVComputer.like_1x(cutoff_fn=cutoff_fn_biw).to(device)
radial_len = aevcomp_biw.radial_len
aevs = aevcomp_smooth(species, coords)
print("AEV computer similar to 1x, but with a custom cutoff function")
print("for first atom, first 5 terms of radial:", aevs[0, 0, :5].tolist())
print(
    "for first atom, first 5 terms of angular:",
    aevs[0, 0, radial_len:radial_len + 5].tolist(),
)
print()


# Lets try something a bit more complicated. Lets experiment with different angular
# terms that have a form of ``exp(-gamma * (cos(theta) - cos(theta0))**2)`` how can we
# do that?
#
# We can pass a custom module to the ``AEVComputer``. As long as it exposes the same API
# as ANIAngular (it has to have a *sublen*, a *cutoff*, a *cutoff_fn* and a *forward
# method* with the same signature)


class CosAngular(AngularTerm):
    def __init__(self, eta, shifts, gamma, sections, cutoff, cutoff_fn="cosine"):
        super().__init__(cutoff=cutoff, cutoff_fn=cutoff_fn)  # *Must* be called
        assert len(sections) == len(gamma)
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("eta", torch.tensor([eta]))
        self.register_buffer("shifts", torch.tensor(shifts))
        self.register_buffer("sections", torch.tensor(sections))
        self.sublen = len(shifts) * len(sections)  # *Must* have a sublen

    # The inputs are two tensors, of shapes (triples,) and (triples, 3)
    def forward(self, triple_distances: Tensor, triple_vectors: Tensor) -> Tensor:
        triple_vectors = triple_vectors.view(2, -1, 3, 1, 1)
        triple_distances = triple_distances.view(2, -1, 1, 1)
        cos_angles = triple_vectors.prod(0).sum(1) / torch.clamp(
            triple_distances.prod(0), min=1e-10
        )
        fcj12 = self.cutoff_fn(triple_distances, self.cutoff)
        term1 = triple_distances.sum(0) / 2 - self.shifts.view(-1, 1)
        term2 = cos_angles - torch.cos(self.sections.view(1, -1))
        exponent = self.eta * term1**2 + self.gamma.view(1, -1) * term2**2
        ret = 4 * torch.exp(-exponent) * (fcj12[0] * fcj12[1])
        return ret.view(-1, self.sublen)  # *Must* have shape (triples, sublen)


# Now lets initialize this function with some parameters
eta = 8.0
cutoff = 3.5
shifts = [0.9000, 1.5500, 2.2000, 2.8500]
sections = linspace(0.0, math.pi, 9)
gamma = [1023.0, 146.5, 36.0, 18.6, 15.5, 18.6, 36.0, 146.5, 1023.0]

# We will use standard radial terms in the ani-1x style but our custom angular
# terms, and we need to pass the same cutoff_fn to both
ani_radial = ANIRadial.like_1x(cutoff_fn="smooth")
cos_angular = CosAngular(eta, shifts, gamma, sections, cutoff, cutoff_fn="smooth")
aevcomp_cos = AEVComputer(radial=ani_radial, angular=cos_angular, num_species=4).to(
    device
)

radial_len = aevcomp_cos.radial_len
aevs = aevcomp_cos(species, coords)
print("AEV computer similar to 1x, but with custom angular terms")
print("for first atom, first 5 terms of radial:", aevs[0, 0, :5].tolist())
print(
    "for first atom, first 5 terms of angular:",
    aevs[0, 0, radial_len:radial_len + 5].tolist(),
)
print()
