r"""Implementation of the SRB (short range basis) correction performed by the
B97-3c functional to account for systematically "too large" covalent bond lengths
(this is in general a problem for the HF-3c method). Equation and defaults for
scaling radius and charge taken from https://aip.scitation.org/doi/pdf/10.1063/1.5012601"""
import torch
from torch import Tensor
from . import units
from .dispersion import constants
from .utils import ATOMIC_NUMBERS
from .standalone import StandalonePairwiseWrapper
from .nn import SpeciesEnergies
from .aev.cutoffs import _parse_cutoff_fn
from typing import Union, Sequence, Tuple


class EnergySRB(torch.nn.Module):
    y_ab: Tensor
    sqrt_alpha_ab: Tensor
    k_rep_ab: Tensor

    def __init__(self,
                 elements: Sequence[str] = ('H', 'C', 'N', 'O'),
                 scaling_charge: float = 10.0,
                 scaling_radius: float = 0.08,
                 cutoff: float = 5.2,
                 cutoff_fn: Union[str, torch.nn.Module] = 'smooth'):
        super().__init__()
        supported_znumbers = torch.tensor([ATOMIC_NUMBERS[e] for e in elements], dtype=torch.long)
        # note that SRB uses the same covalent radii as D3
        covalent_radii = units.angstrom2bohr(constants.get_covalent_radii())[supported_znumbers]
        # We will actually need to multiply the distances by scaled covalent
        # radii, so we precalculate the factor here
        self.register_buffer('distances_factor', - scaling_radius / covalent_radii)
        # The exponential prefactor is - q/2 * sqrt(Za * Zb), which we also
        # precalculate here for efficiency
        _exp_prefactor = torch.outer(supported_znumbers, supported_znumbers)
        _exp_prefactor = -scaling_charge * torch.sqrt(_exp_prefactor) / 2
        self.register_buffer('exp_prefactor', _exp_prefactor)

        self.cutoff_function = _parse_cutoff_fn(cutoff_fn)
        self.cutoff = cutoff

    def _calculate_srb(self,
                       species_energies: Tuple[Tensor, Tensor],
                       atom_index12: Tensor,
                       distances: Tensor) -> Tuple[Tensor, Tensor]:
        # all internal calculations of this module are made with atomic units,
        # so distances are first converted to bohr
        distances = units.angstrom2bohr(distances)
        species, energies = species_energies
        assert distances.ndim == 1, "distances should be 1 dimensional"
        assert species.ndim == 2, "species should be 2 dimensional"
        assert atom_index12.ndim == 2, "atom_index12 should be 2 dimensional"
        assert len(distances) == atom_index12.shape[1]

        # Distances has all interaction pairs within a given cutoff, for a
        # molecule or set of molecules and atom_index12 holds all pairs of
        # indices species is of shape (C x Atoms)
        num_atoms = species.shape[1]
        species12 = species.flatten()[atom_index12]

        # find pre-computed constant multiplications for every species pair
        scaled_charge_prefactor = self.exp_prefactor[species12[0], species12[1]]
        distances_factor = self.distances_factor[species12[0], species12[1]]
        # note that negative signs are already included in prefactors
        srb_energies = scaled_charge_prefactor * torch.exp(distances_factor * distances)

        if self.cutoff_function is not None:
            cutoff = units.angstrom2bohr(self.cutoff)
            srb_energies *= self.cutoff_function(distances, cutoff)

        molecule_indices = torch.div(atom_index12[0], num_atoms, rounding_mode='floor')
        energies.index_add_(0, molecule_indices, srb_energies)
        return SpeciesEnergies(species, energies)

    def forward(self,
                species_energies: Tuple[Tensor, Tensor],
                atom_index12: Tensor,
                distances: Tensor) -> Tuple[Tensor, Tensor]:
        return self._calculate_srb(species_energies, atom_index12, distances)


class StandaloneEnergySRB(StandalonePairwiseWrapper, EnergySRB):
    def _perform_module_actions(self,
                                species_coordinates: Tuple[Tensor, Tensor],
                                atom_index12: Tensor,
                                distances: Tensor) -> Tuple[Tensor, Tensor]:
        species, _ = species_coordinates
        energies = torch.zeros(species.shape[0], dtype=distances.dtype, device=distances.device)
        species_energies = (species, energies)
        return self._calculate_srb(species_energies, atom_index12, distances)
