from typing import Tuple

import torch
from torch import Tensor

from . import units
from .nn import SpeciesEnergies
from .aev import CutoffSmooth


class RepulsionCalculator(torch.nn.Module):
    """Calculates the QMDFF repulsion energy terms for a given molecule
    as seen in work by Grimme: https://pubs.acs.org/doi/10.1021/acs.jctc.8b01176"""

    def __init__(self, Rcr, alpha_buf=None, yeff_buf=None, krep_buf=None, num_species=4, cutoff_function=CutoffSmooth):

        super().__init__()
        # cutoff distance in Bohr radii for consistency with repulsion parameters
        Rcr_bohr = Rcr * units.ANGSTROM_TO_BOHR
        self.register_buffer('Rcr_bohr', torch.tensor(Rcr_bohr))
        # parameter values for each kind of atom
        Vyeff = torch.tensor([1.105388, 4.231078, 5.242592, 5.784415])
        Valpha = torch.tensor([2.213717, 1.247655, 1.682689, 2.165712])
        # krep for each pair is k=1.5-kA*kB*0.5 (1.5 or 1)
        Vkrep = torch.tensor([1., 0., 0., 0.])
        # matrix of combinations of parameters (pre-calculates constants)
        y_ab = torch.outer(Vyeff, Vyeff)
        sqrt_alpha_ab = torch.sqrt(torch.outer(Valpha, Valpha))
        k_rep_ab = 1.5 - torch.outer(Vkrep, Vkrep) * 0.5

        self.register_buffer('y_ab', y_ab)
        self.register_buffer('sqrt_alpha_ab', sqrt_alpha_ab)
        self.register_buffer('k_rep_ab', k_rep_ab)

        self.cutoff_function = cutoff_function(Rcr_bohr) if cutoff_function is not None else None

    def forward(self, species_energies: Tensor, atom_index12: Tensor, distances: Tensor) -> Tuple[Tensor, Tensor]:

        species, energies = species_energies
        assert len(species) == 1, "Not implemented for batch calculations"
        assert distances.ndim == 1, "distances should be 1 dimensional"
        assert species.ndim == 2, "species_energies should be 2 dimensional"
        assert atom_index12.ndim == 2, "atom_index12 should be 2 dimensional"
        assert len(distances) == atom_index12.shape[1]

        # distances has all interaction pairs within a given cutoff,
        # for a molecule or set of molecules
        # and atom_index12 holds all pairs of indices
        # species is of shape (C x Atoms)
        species12 = species.flatten()[atom_index12]

        # distances need to be in Bohr radii units (cutoff distance too for coherence)
        distances_bohr = distances * units.ANGSTROM_TO_BOHR

        # find pre-computed constant multiplications for every species pair
        y_ab = self.y_ab[species12[0], species12[1]]
        sqrt_alpha_ab = self.sqrt_alpha_ab[species12[0], species12[1]]
        k_rep_ab = self.k_rep_ab[species12[0], species12[1]]

        # calculates repulsion energies using distances and constants
        # NOTE: for batch implementation we should add in only one dimension
        prefactor = (y_ab / distances_bohr)
        repulsion_energy = prefactor * torch.exp(-sqrt_alpha_ab * (distances_bohr ** k_rep_ab))

        if self.cutoff_function is not None:
            repulsion_energy *= self.cutoff_function(distances_bohr)

        repulsion_energy = repulsion_energy.sum()

        energies += repulsion_energy
        return SpeciesEnergies(species, energies)
