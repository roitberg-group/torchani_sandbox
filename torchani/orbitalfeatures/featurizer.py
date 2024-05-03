import typing as tp

import torch
from torch import Tensor

from torchani.tuples import SpeciesAEV
from torchani.aev import AEVComputer
from torchani.orbitalfeatures.simple_orbital_aev import SimpleOrbitalAEVComputer


class ExCorrAEVComputer(AEVComputer):
    def __init__(
        self,
        Rcr: tp.Optional[float] = None,
        Rca: tp.Optional[float] = None,
        EtaR: tp.Optional[Tensor] = None,
        ShfR: tp.Optional[Tensor] = None,
        EtaA: tp.Optional[Tensor] = None,
        Zeta: tp.Optional[Tensor] = None,
        ShfA: tp.Optional[Tensor] = None,
        ShfZ: tp.Optional[Tensor] = None,
        num_species: tp.Optional[int] = None,
        use_cuda_extension=False,
        use_cuaev_interface=False,
        cutoff_fn='cosine',
        neighborlist='full_pairwise',
        radial_terms='standard',
        angular_terms='standard',
        use_geometric_aev: bool = True,
    ) -> None:
        # forbid cuaev for now
        assert not self.use_cuda_extension
        assert not self.cudaev_is_initialized
        super().__init__(
            Rcr,
            Rca,
            EtaR,
            ShfR,
            EtaA,
            Zeta,
            ShfA,
            ShfZ,
            num_species,
            False,
            False,
            cutoff_fn,
            neighborlist,
            radial_terms,
            angular_terms,
        )
        self.use_geometric_aev = use_geometric_aev
        self.simple_orbital_aev_computer = SimpleOrbitalAEVComputer()

    def forward(  # type: ignore
        self,
        input_: tp.Tuple[Tensor, Tensor],
        coefficients: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None
    ) -> SpeciesAEV:
        species, coordinates = input_
        # check shapes for correctness
        assert species.dim() == 2
        assert coordinates.dim() == 3
        assert (species.shape == coordinates.shape[:2]) and (coordinates.shape[2] == 3)

        # validate cutoffs
        assert self.angular_terms.cutoff < self.radial_terms.cutoff

        # WARNING: The coordinates that are input into the neighborlist are **not** assumed to be
        # mapped into the central cell for pbc calculations,
        # and **in general are not**
        aev = self.simple_orbital_aev_computer(coefficients)

        if self.use_geometric_aev:
            neighbor_data = self.neighborlist(species, coordinates, self.radial_terms.cutoff, cell, pbc)
            geometric_aev = self._compute_aev(
                element_idxs=species,
                neighbor_idxs=neighbor_data.indices,
                distances=neighbor_data.distances,
                diff_vectors=neighbor_data.diff_vectors,
            )
            aev = torch.cat((geometric_aev, aev), dim=-1)
        return SpeciesAEV(species, aev)
