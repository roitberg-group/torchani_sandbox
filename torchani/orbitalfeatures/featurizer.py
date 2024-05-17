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
        use_simple_orbital_aev: bool = True,
        use_angular_info_in_simple_orbital_aev: bool = False,
        basis_functions='spd',
        use_geometric_aev: bool = True,
    ) -> None:
        # forbid cuaev for now
        assert not use_cuda_extension
        assert not use_cuaev_interface
        assert (basis_functions == 'spd' or basis_functions == 'sp' or basis_functions == 's')
        if ((basis_functions == 's') and use_simple_orbital_aev):
            assert (not use_angular_info_in_simple_orbital_aev)
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
        self.use_simple_orbital_aev = use_simple_orbital_aev
        self.use_angular_info_in_simple_orbital_aev = use_angular_info_in_simple_orbital_aev
        self.basis_functions = basis_functions
        self.use_geometric_aev = use_geometric_aev
        if use_simple_orbital_aev:
            if basis_functions == 'spd':
                orbital_aev_length = 21
            elif basis_functions == 'sp':
                orbital_aev_length = 13            
            elif basis_functions == 's':
                orbital_aev_length = 9
            if use_angular_info_in_simple_orbital_aev:
                #Only p and d AOVs have angular info associated
                #The number of s+d AOVs is simple_orbital_aev_length-9 (because we need to substract the 9 s AOVs)
                #If we calculate the number of angles as N(N-1)/2 (with N the number of s+d AOVs), we have:
                orbital_aev_length = orbital_aev_length + ((orbital_aev_length-9)*(orbital_aev_length-10)/2)
            self.orbital_aev_computer = SimpleOrbitalAEVComputer()   
        else:
            #To do -> Include an AEV-like expansion for the AOVs
            pass

        if use_geometric_aev:
            self.aev_length = self.radial_length + self.angular_length + orbital_aev_length
        else:
            self.aev_length = orbital_aev_length

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
        assert coefficients.dim() == 3
        assert coefficients.shape[-1] == 45
        assert (species.shape == coordinates.shape[:2]) and (coordinates.shape[2] == 3)

        # validate cutoffs
        assert self.angular_terms.cutoff < self.radial_terms.cutoff

        # WARNING: The coordinates that are input into the neighborlist are **not** assumed to be
        # mapped into the central cell for pbc calculations,
        # and **in general are not**

        if self.use_simple_orbital_aev:
            aev = self.orbital_aev_computer(coefficients,self.use_angular_info_in_simple_orbital_aev,self.basis_functions )
        else:
            #To do -> Same but for an AEV-like expansion of the AOVs
            aev = torch.tensor([])  # Placeholder until implemented

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
