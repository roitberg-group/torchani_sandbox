r"""Transforms to be applied to properties when training. The usage is the same
as transforms in torchvision.

Most TorchANI modules take atomic indices ("species") as input as a
default, so if you want to convert atomic numbers to indices and then subtract
the self atomic energies (SAE) when iterating from a batched dataset you can
call::

    from torchani.transforms import AtomicNumbersToIndices, SubtractSAE, Compose
    from torchani.datasets import ANIBatchedDataset
    transform = Compose([AtomicNumbersToIndices(('H', 'C', 'N'), SubtractSAE([-0.57, -0.0045, -0.0035])])
    training = ANIBatchedDataset('/path/to/database/', transform=transform, split='training')
    validation = ANIBatchedDataset('/path/to/database/', transform=transform, split='validation')
"""
import typing as tp
import math
import warnings

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchani.utils import EnergyShifter, ATOMIC_NUMBERS
from torchani.nn import SpeciesConverter
from torchani.datasets import ANIBatchedDataset
from torchani.wrappers import Wrapper


class Transform(torch.nn.Module):
    atomic_numbers: tp.Optional[Tensor]
    r"""Base class for transformations that modify conformer properties on the fly"""
    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        raise NotImplementedError("Must be overriden by subclasses")


class SubtractEnergy(Transform):
    r"""Subtract the energy calculated from an arbitrary Wrapper module
    This can be coupled with, e.g., a pairwise potential in order to subtract
    analytic energies before training.
    """
    def __init__(self, wrapper: Wrapper):
        super().__init__()
        if not wrapper.periodic_table_index:
            raise ValueError("Wrapper module should have periodic_table_index=True")
        self.wrapper = wrapper

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        properties['energies'] -= self.wrapper((properties["species"], properties["coordinates"])).energies
        return properties


class SubtractSAE(Transform):
    def __init__(self, symbols: tp.Sequence[str], self_energies: tp.List[float], intercept: float = 0.0):
        super().__init__()
        atomic_numbers = [ATOMIC_NUMBERS[s] for s in symbols]
        if len(self_energies) != len(atomic_numbers):
            raise ValueError("There should be one self energy per element")
        self.register_buffer('atomic_numbers', torch.tensor(atomic_numbers, dtype=torch.long))
        if intercept != 0.0:
            self_energies.append(intercept)
            # for some reason energy_shifter is defaulted as double, so I make
            # it float here
            self.energy_shifter = EnergyShifter(self_energies, fit_intercept=True).float()
        else:
            self.energy_shifter = EnergyShifter(self_energies).float()

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        properties['energies'] -= self.energy_shifter.sae(properties['species'])
        return properties


class AtomicNumbersToIndices(Transform):
    r"""
    This class converts atomic numbers to arbitrary indices, it is very error
    prone to use it and not recommended, but it is provided for legacy support
    """
    def __init__(self, symbols: tp.Sequence[str]):
        super().__init__()
        atomic_numbers = [ATOMIC_NUMBERS[s] for s in symbols]
        warnings.warn(
            "It is not recommended convert atomic numbers to indices, this is "
            " very error prone and can generate multiple issues"
        )
        self.register_buffer('atomic_numbers', torch.tensor(atomic_numbers, dtype=torch.long))
        self.converter = SpeciesConverter(symbols)

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        species = self.converter((properties['species'], properties['coordinates'])).species
        properties['species'] = species
        return properties


class Compose(Transform):
    """Composes several transforms together.

    Args:
        transforms (list of `Transform` objects): list of transforms to compose.
    """
    # This code is mostly copied from torchvision, but made JIT scriptable
    def __init__(self, transforms: tp.Sequence[Transform]):
        super().__init__()

        # Validate that all transforms use the same atomic numbers
        atomic_numbers: tp.List[Tensor] = [
            t.atomic_numbers for t in transforms if t.atomic_numbers is not None
        ]
        if atomic_numbers:
            if not all(a == atomic_numbers[0] for a in atomic_numbers):
                raise ValueError("Two or more of your transforms use different atomic numbers, this is incorrect")
            self.register_buffer('atomic_numbers', torch.tensor(atomic_numbers[0], dtype=torch.long))
        else:
            self.register_buffer('atomic_numbers', None)

        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
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


def calculate_saes(dataset: tp.Union[DataLoader, ANIBatchedDataset],
                         elements: tp.Sequence[str],
                         mode: str = 'sgd',
                         fraction: float = 1.0,
                         fit_intercept: bool = False,
                         device: str = 'cpu',
                         max_epochs: int = 1,
                         lr: float = 0.01) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
    if mode == 'exact':
        if lr != 0.01:
            raise ValueError("lr is only used with mode=sgd")
        if max_epochs != 1:
            raise ValueError("max_epochs is only used with mode=sgd")

    assert mode in ['sgd', 'exact']
    if isinstance(dataset, DataLoader):
        assert isinstance(dataset.dataset, ANIBatchedDataset)
        old_transform = dataset.dataset.transform
        dataset.dataset.transform = AtomicNumbersToIndices(elements)
    else:
        assert isinstance(dataset, ANIBatchedDataset)
        old_transform = dataset.transform
        dataset.transform = AtomicNumbersToIndices(elements)

    num_species = len(elements)
    num_batches_to_use = math.ceil(len(dataset) * fraction)
    print(f'Using {num_batches_to_use} of a total of {len(dataset)} batches to estimate SAE')

    if mode == 'exact':
        print('Calculating SAE using exact OLS method...')
        m_out, b_out = _calculate_saes_exact(dataset, num_species, num_batches_to_use,
                                             device=device, fit_intercept=fit_intercept)
    elif mode == 'sgd':
        print("Estimating SAE using stochastic gradient descent...")
        m_out, b_out = _calculate_saes_sgd(dataset, num_species, num_batches_to_use,
                                           device=device,
                                           fit_intercept=fit_intercept,
                                           max_epochs=max_epochs, lr=lr)

    if isinstance(dataset, DataLoader):
        assert isinstance(dataset.dataset, ANIBatchedDataset)
        dataset.dataset.transform = old_transform
    else:
        assert isinstance(dataset, ANIBatchedDataset)
        dataset.transform = old_transform
    return m_out, b_out


def _calculate_saes_sgd(dataset, num_species: int, num_batches_to_use: int,
                        device: str = 'cpu',
                        fit_intercept: bool = False,
                        max_epochs: int = 1,
                        lr: float = 0.01) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:

    class LinearModel(torch.nn.Module):

        m: torch.nn.Parameter
        b: tp.Optional[torch.nn.Parameter]

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
            if j == num_batches_to_use - 1:
                break
    model.m.requires_grad_(False)
    m_out = model.m.data.cpu()

    b_out: tp.Optional[Tensor]
    if model.b is not None:
        model.b.requires_grad_(False)
        b_out = model.b.data.cpu()
    else:
        b_out = None
    return m_out, b_out


def _calculate_saes_exact(dataset, num_species: int, num_batches_to_use: int,
                          device: str = 'cpu',
                          fit_intercept: bool = False) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:

    if num_batches_to_use == len(dataset):
        warnings.warn("Using all batches to estimate SAE, this may take up a lot of memory.")
    list_species_counts = []
    list_true_energies = []
    for j, properties in enumerate(dataset):
        species = properties['species'].to(device)
        true_energies = properties['energies'].float().to(device)
        species_counts = torch.zeros((species.shape[0], num_species), dtype=torch.float, device=device)
        for n in range(num_species):
            species_counts[:, n] = (species == n).sum(-1).float()
        list_species_counts.append(species_counts)
        list_true_energies.append(true_energies)
        if j == num_batches_to_use - 1:
            break

    if fit_intercept:
        list_species_counts.append(torch.ones(num_species, device=device, dtype=torch.float))
    total_true_energies = torch.cat(list_true_energies, dim=0)
    total_species_counts = torch.cat(list_species_counts, dim=0)

    # here total_true_energies is of shape m x 1 and total_species counts is m x n
    # n = num_species if we don't fit an intercept, and is equal to num_species + 1
    # if we fit an intercept. See the torch documentation for linalg.lstsq for more info
    x = torch.linalg.lstsq(total_species_counts, total_true_energies.unsqueeze(-1), driver='gels').solution
    m_out = x.T.squeeze()

    b_out: tp.Optional[Tensor]
    if fit_intercept:
        b_out = x[num_species]
    else:
        b_out = None
    return m_out, b_out
