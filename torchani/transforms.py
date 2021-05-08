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
import math
import warnings

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


def calculate_saes_exact(dataset, elements=('H', 'C', 'N', 'O'),
                         fit_intercept: bool = False,
                         device: str = 'cpu', fraction: float = 1.0) -> Tuple[Tensor, Union[Tensor, None]]:
    if isinstance(dataset, torch.utils.data.DataLoader):
        old_transform = dataset.dataset.transform
        dataset.dataset.transform = AtomicNumbersToIndices(elements)
    else:
        old_transform = dataset.transform
        dataset.transform = AtomicNumbersToIndices(elements)

    num_species = len(elements)
    total_species_counts = []
    total_true_energies = []
    num_batches_to_use = math.ceil(len(dataset) * fraction)
    if num_batches_to_use == len(dataset):
        warnings.warn("Using all batches to estimate SAE, this may take up a lot of memory.")
    print(f'Using {num_batches_to_use} of a total of {len(dataset)} batches to estimate... SAE')
    for j, properties in enumerate(dataset):
        if j == num_batches_to_use:
            break
        species = properties['species'].to(device)
        true_energies = properties['energies'].float().to(device)
        species_counts = torch.zeros((species.shape[0], num_species), dtype=torch.float, device=device)
        for n in range(num_species):
            species_counts[:, n] = (species == n).sum(-1).float()
        total_species_counts.append(species_counts)
        total_true_energies.append(true_energies)
    if fit_intercept:
        total_species_counts.append(torch.ones(num_species, device=device, dtype=torch.float))
    total_true_energies = torch.cat(total_true_energies, dim=0)
    total_species_counts = torch.cat(total_species_counts, dim=0)

    # here total_true_energies is of shape m x 1 and total_species counts is m x n
    # n = num_species if we don't fit an intercept, and is equal to num_species + 1
    # if we fit an intercept. See the torch documentation for lstsq for more info
    x, _ = torch.lstsq(total_true_energies, total_species_counts)

    if isinstance(dataset, torch.utils.data.DataLoader):
        dataset.dataset.transform = old_transform
    else:
        dataset.transform = old_transform

    # solution to least squares problem is in the first n rows of x
    m_out = x[:num_species].T.squeeze()
    if fit_intercept:
        b_out = x[num_species]
    else:
        b_out = None
    return m_out, b_out


def estimate_saes_sgd(dataset, elements=('H', 'C', 'N', 'O'), fit_intercept:
                      bool = False, max_epochs: int = 1, lr: float = 0.01,
                      device: str = 'cpu') -> Tuple[Tensor, Union[Tensor, None]]:
    # This only supports datasets with no inplace calculations
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
                _b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float))
            else:
                _b = None
            self.register_parameter('b', _b)

        def forward(self, x: Tensor) -> Tensor:
            x *= self.m
            if self.b is not None:
                x += self.b
            return x.sum(-1)

    num_species = len(elements)
    model = LinearModel(num_species, fit_intercept).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    print("Estimating SAE using stochastic gradient descent...")
    for _ in range(max_epochs):
        for properties in tqdm(dataset, total=len(dataset)):
            species = properties['species'].to(device)
            species_counts = torch.zeros((species.shape[0], num_species), dtype=torch.float, device=device)
            for n in range(num_species):
                species_counts[:, n] = (species == n).sum(-1).float()
            true_energies = properties['energies'].float().to(device)
            predicted_energies = model(species_counts)
            loss = (true_energies - predicted_energies).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    if isinstance(dataset, torch.utils.data.DataLoader):
        dataset.dataset.transform = old_transform
    else:
        dataset.transform = old_transform
    model.m.requires_grad_(False)
    m_out = model.m.cpu()
    if model.b is not None:
        model.b.requires_grad_(False)
        b_out = model.b.cpu()
    else:
        b_out = None
    return m_out, b_out
