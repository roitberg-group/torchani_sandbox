r"""Transforms to be applied to properties when training. The usage is the same
as transforms in torchvision.

Example:
Most TorchANI modules take atomic indices ("species") as input as a
default, so if you want to convert atomic numbers to indices and then subtract
the self atomic energies (SAE) when iterating from a batched dataset you can
call

from torchani.transforms import AtomicNumbersToIndices, SubtractSAE
from torchani.datasets import AniBatchedDataset

transform = torchani.transforms.Compose([AtomicNumbersToIndices(('H', 'C', 'N'), SubtractSAE([-0.57, -0.0045, -0.0035])])
training = AniBatchedDataset(path='/path/to/database/', transform=transform, split='training')
validation = AniBatchedDataset(path='/path/to/database/', transform=transform, split='validation')
"""
from typing import Dict, Sequence, Union, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from .utils import EnergyShifter, PERIODIC_TABLE
from .nn import SpeciesConverter


class SubtractRepulsion(torch.nn.Module):

    def __init__(self, elements: Union[Sequence[str], Sequence[int]]):
        super().__init__()

    def forward(self, properties: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError("Not yet implemented")
        return properties


class SubtractSAE(torch.nn.Module):

    def __init__(self, self_energies: Union[Tensor, Sequence[float]]):
        super().__init__()
        self.energy_shifter = EnergyShifter(torch.as_tensor(self_energies))

    def forward(self, properties: Dict[str, Tensor]) -> Dict[str, Tensor]:
        energies = properties['energies']
        energies = energies - self.energy_shifter((properties['species'], energies)).energies
        properties['energies'] = energies
        return properties


class AtomicNumbersToIndices(torch.nn.Module):

    def __init__(self, elements: Union[Sequence[str], Sequence[int]]):
        super().__init__()
        symbols = []

        if isinstance(elements[0], int):
            for e in elements:
                assert isinstance(e, int) and e > 0, f"Encountered an atomic number that is <= 0 {elements}"
                symbols.append(PERIODIC_TABLE[e])
        else:
            for e in elements:
                assert isinstance(e, str), "Input sequence must consist of chemical symbols or atomic numbers"
                symbols.append(e)

        self.converter = SpeciesConverter(elements)

    def forward(self, properties: Dict[str, Tensor]) -> Dict[str, Tensor]:
        species = self.converter((properties['species'], properties['coordinates'])).species
        properties['species'] = species
        return properties


# This code is copied from torchvision, but made JIT scriptable
class Compose(torch.nn.Module):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms: Sequence[torch.nn.Module]):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, properties: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for t in self.transforms:
            properties = t(properties)
        return properties

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def estimate_saes_sgd(dataset, elements=('H', 'C', 'N', 'O'), fit_intercept:
                      bool = False, max_epochs: int = 1, lr: float = 0.01,
                      device: str = 'cpu') -> Tuple[Tensor, Union[Tensor, None]]:
    # This only supports datasets were AtomicNumbersToIndices has not been run for now
    if isinstance(dataset, torch.utils.data.DataLoader):
        old_transform = dataset.dataset.transform
        dataset.dataset.transform = AtomicNumbersToIndices(elements)
    else:
        old_transform = dataset.transform
        dataset.transform = AtomicNumbersToIndices(elements)

    class LinearModel(torch.nn.Module):

        def __init__(self, num_species, fit_intercept: bool = False):
            super().__init__()
            self.register_parameter('m', torch.nn.Parameter(torch.ones(num_species, dtype=torch.float)))
            if fit_intercept:
                _b = torch.nn.Parameter(torch.zeros(num_species, dtype=torch.float))
            else:
                _b = None
            self.register_parameter('b', _b)

        def forward(self, x: Tensor) -> Tensor:
            x *= self.m
            if self.b is not None:
                x += self.b
            return x.sum(-1)

    num_species = len(elements)
    model = LinearModel(num_species, fit_intercept)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    print("Estimating SAE using stochastic gradient descent...")
    for _ in range(max_epochs):
        for properties in tqdm(dataset, total=len(dataset)):
            species = properties['species']
            input_ = torch.zeros((species.shape[0], num_species), dtype=torch.float)
            for n in range(num_species):
                input_[:, n] = (species == n).sum(-1).float()
            true_energies = properties['energies'].float()
            predicted_energies = model(input_)
            loss = (true_energies - predicted_energies).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    dataset.transform = old_transform
    if isinstance(dataset, torch.utils.data.DataLoader):
        dataset.dataset.transform = old_transform
    else:
        dataset.transform = old_transform
    model.m.requires_grad_(False)
    m_out = model.m.numpy().tolist()
    if model.b is not None:
        model.b.requires_grad_(False)
        b_out = model.b.numpy().tolist()
    else:
        b_out = None
    return m_out, b_out
