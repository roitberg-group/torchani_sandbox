r"""Transforms to be applied to properties when training. The usage is the same
as transforms in torchvision.

Example:
Most TorchANI modules take atomic indices ("species") as input as a
default, so if you want to convert atomic numbers to indices and then subtract
the self atomic energies (SAE) when iterating from a batched dataset you can
call

from torchani.transforms import AtomicNumbersToIndices, SubtractSAE, Compose
from torchani.datasets import AniBatchedDataset

transform = Compose([AtomicNumbersToIndices(('H', 'C', 'N'), SubtractSAE([-0.57, -0.0045, -0.0035])])
training = AniBatchedDataset('/path/to/database/', transform=transform, split='training')
validation = AniBatchedDataset('/path/to/database/', transform=transform, split='validation')
"""
from typing import Dict, Sequence, Union, Tuple, Optional, Any, List
import math
import warnings

import torch
from torch import Tensor

from .utils import EnergyShifter, PERIODIC_TABLE
from .nn import SpeciesConverter
from .datasets import AniBatchedDataset
from torch.utils.data import DataLoader


class SubtractRepulsion(torch.nn.Module):

    def __init__(self, elements: Union[Sequence[str], Sequence[int]]):
        super().__init__()

    def forward(self, properties: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError("Not yet implemented")
        return properties


class SubtractSAE(torch.nn.Module):

    def __init__(self, self_energies: Sequence[float]):
        super().__init__()
        self.energy_shifter = EnergyShifter(self_energies)

    def forward(self, properties: Dict[str, Tensor]) -> Dict[str, Tensor]:
        energies = properties['energies']
        energies = energies - self.energy_shifter((properties['species'], energies)).energies
        properties['energies'] = energies
        return properties


class AtomicNumbersToIndices(torch.nn.Module):

    def __init__(self, elements: Union[Sequence[str], Sequence[int]]):
        super().__init__()
        symbols: List[str] = []

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


class Compose(torch.nn.Module):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    # This code is copied from torchvision, but made JIT scriptable

    def __init__(self, transforms: Sequence[torch.nn.Module]):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, properties: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for t in self.transforms:
            properties = t(properties)
        return properties

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def calculate_saes(dataset: Union[DataLoader, AniBatchedDataset],
                         elements: Sequence[str],
                         mode: str = 'sgd',
                         fraction: float = 1.0,
                         **kwargs: Any) -> Tuple[Tensor, Optional[Tensor]]:
    if mode == 'exact':
        if 'lr' in kwargs.keys():
            raise ValueError("lr is only used with mode=sgd")
        if 'max_epochs' in kwargs.keys():
            raise ValueError("max_epochs is only used with mode=sgd")

    assert mode in ['sgd', 'exact']
    if isinstance(dataset, DataLoader):
        assert isinstance(dataset.dataset, AniBatchedDataset)
        old_transform = dataset.dataset.transform
        dataset.dataset.transform = AtomicNumbersToIndices(elements)
    else:
        assert isinstance(dataset, AniBatchedDataset)
        old_transform = dataset.transform
        dataset.transform = AtomicNumbersToIndices(elements)

    num_species = len(elements)
    num_batches_to_use = math.ceil(len(dataset) * fraction)
    print(f'Using {num_batches_to_use} of a total of {len(dataset)} batches to estimate SAE')

    if mode == 'exact':
        print('Calculating SAE using exact OLS method...')
        m_out, b_out = _calculate_saes_exact(dataset, num_species, num_batches_to_use, **kwargs)
    elif mode == 'sgd':
        print("Estimating SAE using stochastic gradient descent...")
        m_out, b_out = _calculate_saes_sgd(dataset, num_species, num_batches_to_use, **kwargs)

    if isinstance(dataset, DataLoader):
        assert isinstance(dataset.dataset, AniBatchedDataset)
        dataset.dataset.transform = old_transform
    else:
        assert isinstance(dataset, AniBatchedDataset)
        dataset.transform = old_transform
    return m_out, b_out


def _calculate_saes_sgd(dataset, num_species: int, num_batches_to_use: int,
                        device: str = 'cpu',
                        fit_intercept: bool = False,
                        max_epochs: int = 1,
                        lr: float = 0.01) -> Tuple[Tensor, Optional[Tensor]]:

    class LinearModel(torch.nn.Module):

        m: torch.nn.Parameter
        b: Optional[torch.nn.Parameter]

        def __init__(self, num_species: int, fit_intercept: bool = False):
            super().__init__()
            self.register_parameter('m', torch.nn.Parameter(torch.ones(num_species, dtype=torch.float)))
            if fit_intercept:
                self.register_parameter('b', torch.nn.Parameter(torch.zeros(1, dtype=torch.float)))
            else:
                self.register_parameter('b', None)

        def forward(self, x: Tensor) -> Tensor:
            x *= self.m
            if self.b is not None:
                x += self.b
            return x.sum(-1)

    model = LinearModel(num_species, fit_intercept).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(max_epochs):
        for j, properties in enumerate(dataset):
            if j == num_batches_to_use:
                break
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
    model.m.requires_grad_(False)
    m_out = model.m.data.cpu()

    b_out: Union[Tensor, None]
    if model.b is not None:
        model.b.requires_grad_(False)
        b_out = model.b.data.cpu()
    else:
        b_out = None
    return m_out, b_out


def _calculate_saes_exact(dataset, num_species: int, num_batches_to_use: int,
                          device: str = 'cpu',
                          fit_intercept: bool = False) -> Tuple[Tensor, Optional[Tensor]]:

    if num_batches_to_use == len(dataset):
        warnings.warn("Using all batches to estimate SAE, this may take up a lot of memory.")
    list_species_counts = []
    list_true_energies = []
    for j, properties in enumerate(dataset):
        if j == num_batches_to_use:
            break
        species = properties['species'].to(device)
        true_energies = properties['energies'].float().to(device)
        species_counts = torch.zeros((species.shape[0], num_species), dtype=torch.float, device=device)
        for n in range(num_species):
            species_counts[:, n] = (species == n).sum(-1).float()
        list_species_counts.append(species_counts)
        list_true_energies.append(true_energies)

    if fit_intercept:
        list_species_counts.append(torch.ones(num_species, device=device, dtype=torch.float))
    total_true_energies = torch.cat(list_true_energies, dim=0)
    total_species_counts = torch.cat(list_species_counts, dim=0)

    # here total_true_energies is of shape m x 1 and total_species counts is m x n
    # n = num_species if we don't fit an intercept, and is equal to num_species + 1
    # if we fit an intercept. See the torch documentation for lstsq for more info
    x, _ = torch.lstsq(total_true_energies, total_species_counts)
    # solution to least squares problem is in the first n rows of x
    m_out = x[:num_species].T.squeeze()

    b_out: Union[Tensor, None]
    if fit_intercept:
        b_out = x[num_species]
    else:
        b_out = None
    return m_out, b_out
