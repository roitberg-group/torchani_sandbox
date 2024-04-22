import typing as tp

from torch import Tensor

from torchani.models import BuiltinModel
from torchani.tuples import SpeciesEnergies


class ExCorrModel(BuiltinModel):
    #  It is necessary to pass "the coeff aev computer"  to init instead of the normal aev computer
    # typing may not like this, we can fix it later
    # typing also wont like overloading forward, again, fix it later
    def forward(  # type: ignore
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        coefficients: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        """Calculates predicted energies for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: tuple of tensors, species and energies for the given configurations
        """
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, coefficients=coefficients, cell=cell, pbc=pbc)
        species, energies = self.neural_networks(species_aevs)
        energies += self.energy_shifter(species)
        return SpeciesEnergies(species, energies)
