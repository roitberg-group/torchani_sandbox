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
from typing import Dict, Sequence, Union

import torch
from torch import Tensor

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
        all_ints = all([isinstance(e, int) for e in elements])
        all_strings = all([isinstance(e, str) for e in elements])
        assert all_ints or all_strings, "Input sequence must consist of chemical symbols or atomic numbers"
        if all_ints:
            assert [e > 0 for e in elements], f"Encountered an atomic number that is <= 0 {elements}"
            elements = [PERIODIC_TABLE[e] for e in elements]
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
