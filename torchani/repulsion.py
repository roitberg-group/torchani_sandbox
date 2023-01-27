from typing import Tuple, Sequence, Union, Optional

import torch
from torch import Tensor

from . import units
from .utils import ATOMIC_NUMBERS
from .nn import SpeciesEnergies
from .standalone import StandalonePairwiseWrapper
from .parse_repulsion_constants import alpha_constants, y_eff_constants
from .aev.cutoffs import _parse_cutoff_fn
from .compat import Final


class RepulsionXTB(torch.nn.Module):
    r"""Calculates the xTB repulsion energy terms for a given molecule as seen
    in work by Grimme: https://pubs.acs.org/doi/10.1021/acs.jctc.8b01176"""

    cutoff: Final[float]
    y_ab: Tensor
    sqrt_alpha_ab: Tensor
    k_rep_ab: Tensor

    def __init__(self,
                 cutoff: float = 5.2,
                 alpha: Sequence[float] = None,
                 y_eff: Sequence[float] = None,
                 k_rep_ab: torch.Tensor = None,
                 elements: Sequence[str] = ('H', 'C', 'N', 'O'),
                 cutoff_fn: Union[str, torch.nn.Module] = 'smooth'):
        super().__init__()
        supported_znumbers = torch.tensor([ATOMIC_NUMBERS[e] for e in elements], dtype=torch.long)

        # by default alpha, y_eff and krep parameters are taken from Grimme et. al.
        if alpha is None:
            _alpha = torch.tensor(alpha_constants)[supported_znumbers]
        if y_eff is None:
            _y_eff = torch.tensor(y_eff_constants)[supported_znumbers]
        if k_rep_ab is None:
            k_rep_ab = torch.full((len(ATOMIC_NUMBERS) + 1, len(ATOMIC_NUMBERS) + 1), 1.5)
            k_rep_ab[1, 1] = 1.0
            k_rep_ab = k_rep_ab[supported_znumbers, :][:, supported_znumbers]
        assert k_rep_ab.shape[0] == len(elements)
        assert k_rep_ab.shape[1] == len(elements)
        assert len(_y_eff) == len(elements)
        assert len(_alpha) == len(elements)

        self.cutoff_function = _parse_cutoff_fn(cutoff_fn)
        self.cutoff = cutoff
        # pre-calculate pairwise parameters for efficiency
        self.register_buffer('y_ab', torch.outer(_y_eff, _y_eff))
        self.register_buffer('sqrt_alpha_ab', torch.sqrt(torch.outer(_alpha, _alpha)))
        self.register_buffer('k_rep_ab', k_rep_ab)
        self.ANGSTROM_TO_BOHR = units.ANGSTROM_TO_BOHR

    def _calculate_repulsion(self,
                             species: Tensor,
                             atom_index12: Tensor,
                             distances: Tensor) -> Tensor:

        # all internal calculations of this module are made with atomic units,
        # so distances are first converted to bohr
        distances = distances * self.ANGSTROM_TO_BOHR

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
        y_ab = self.y_ab[species12[0], species12[1]]
        sqrt_alpha_ab = self.sqrt_alpha_ab[species12[0], species12[1]]
        k_rep_ab = self.k_rep_ab[species12[0], species12[1]]

        # calculates repulsion energies using distances and constants
        prefactor = (y_ab / distances)
        rep_energies = prefactor * torch.exp(-sqrt_alpha_ab * (distances ** k_rep_ab))

        if self.cutoff_function is not None:
            rep_energies *= self.cutoff_function(distances, self.cutoff * self.ANGSTROM_TO_BOHR)

        energies = torch.zeros(species.shape[0], dtype=rep_energies.dtype, device=rep_energies.device)
        molecule_indices = torch.div(atom_index12[0], num_atoms, rounding_mode='floor')
        energies.index_add_(0, molecule_indices, rep_energies)
        return energies

    def forward(self,
                species: Tensor,
                atom_index12: Tensor,
                distances: Tensor,
                diff_vector: Optional[Tensor] = None) -> Tensor:
        # diff_vector is unused in 2-body potentials
        return self._calculate_repulsion(species, atom_index12, distances)


# Wrapper to keep compatibility with old ANI code
class RepulsionCalculator(RepulsionXTB):
    def forward(self,  # type: ignore
                species_energies: Tuple[Tensor, Tensor],
                atom_index12: Tensor,
                distances: Tensor) -> Tuple[Tensor, Tensor]:
        species, energies = species_energies
        rep_energies = self._calculate_repulsion(species, atom_index12, distances)
        return SpeciesEnergies(species, energies + rep_energies)


class StandaloneRepulsionCalculator(StandalonePairwiseWrapper, RepulsionCalculator):  # type: ignore
    def _perform_module_actions(self,
                                species_coordinates: Tuple[Tensor, Tensor],
                                atom_index12: Tensor,
                                distances: Tensor) -> Tuple[Tensor, Tensor]:
        species, _ = species_coordinates
        energies = torch.zeros(species.shape[0], dtype=distances.dtype, device=distances.device)
        repulsion_energies = self._calculate_repulsion(species, atom_index12, distances)
        return SpeciesEnergies(species, energies + repulsion_energies)
