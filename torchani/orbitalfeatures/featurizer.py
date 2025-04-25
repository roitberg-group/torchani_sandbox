import typing as tp
from pathlib import Path

import torch
from torch import Tensor

from torchani.tuples import SpeciesAEV
from torchani.aev import AEVComputer
from torchani.orbitalfeatures.orbital_aev import OrbitalAEVComputer


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
        use_angular_info: bool = False,
        use_angular_radial_coupling: bool = False,
        basis_functions = 'spd',
        normalization_library = None,
        use_geometric_aev: bool = True,
        NOShfS = 16,
        NOShfR = 16,
        NOShfA = 8,
        NOShfZ = 1,
        LowerOShfS = -2.00,
        UpperOShfS = 2.00,
        LowerOShfR = -2.00,
        UpperOShfR = 2.00,
        LowerOShfA = 0.00,
        UpperOShfA = 0.30,
        LowerOShfZ = 2.00,
        UpperOShfZ = 2.00,
        OEtaS = 20.0,
        OEtaR = 20.0,
        OEtaA = 12.0,
        OZeta = 14.0,
    ) -> None:
        # forbid cuaev for now
        assert not use_cuda_extension
        assert not use_cuaev_interface
        assert (basis_functions == 'spd' or basis_functions == 'sp' or basis_functions == 's')
        if ((basis_functions == 's') and use_simple_orbital_aev):
            assert (not use_angular_info)
        if (not use_angular_info):
            assert (not use_angular_radial_coupling)
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
        if (normalization_library not None):
            norm_file = Path(normalization_library)
            self.normalization_library = torch.load(norm_file)
        else:
            self.normalization_library = None
        
        self.use_simple_orbital_aev = use_simple_orbital_aev
        self.use_angular_info = use_angular_info
        self.use_angular_radial_coupling = use_angular_radial_coupling
        self.basis_functions = basis_functions
        self.use_geometric_aev = use_geometric_aev
        self.NOShfS = NOShfS
        self.NOShfR = NOShfR
        self.NOShfA = NOShfA
        self.NOShfZ = NOShfZ
        self.LowerOShfS = LowerOShfS
        self.UpperOShfS = UpperOShfS
        self.LowerOShfR = LowerOShfR
        self.UpperOShfR = UpperOShfR
        self.LowerOShfA = LowerOShfA
        self.UpperOShfA = UpperOShfA
        self.LowerOShfZ = LowerOShfZ
        self.UpperOShfZ = UpperOShfZ
        self.OEtaS = OEtaS
        self.OEtaR = OEtaR
        self.OEtaA = OEtaA
        self.OZeta = OZeta
        if use_simple_orbital_aev:
            if basis_functions == 'spd':
                orbital_aev_length = 21
            elif basis_functions == 'sp':
                orbital_aev_length = 13             
            elif basis_functions == 's':
                orbital_aev_length = 9
            if  use_angular_info:
                #Only p and d AOVs have angular info associated
                #The number of s+d AOVs is simple_orbital_aev_length-9 (because we need to substract the 9 s AOVs)
                #If we calculate the number of angles as N(N-1)/2 (with N the number of s+d AOVs), we have:
                nangles = int((orbital_aev_length-9)*(orbital_aev_length-10)/2)
                orbital_aev_length = orbital_aev_length + nangles
            self.orbital_aev_computer = OrbitalAEVComputer()   
        else:
            #To do -> Include an AEV-like expansion for the AOVs
            if basis_functions == 'spd':
                #To do RECALCULATE THIS AFTER DECIDING WHAT TO DO WITH THE D COEFFS<
                orbital_aev_length = 9*self.NOShfS+4*self.NOShfR+0 
            elif basis_functions == 'sp':
                orbital_aev_length = 9*self.NOShfS+4*self.NOShfR
            elif basis_functions == 's':
                orbital_aev_length = 9*self.NOShfS
            if  use_angular_info:
                nangles = int((orbital_aev_length-9)*(orbital_aev_length-10)/2)
                angular_orbital_aev_length = nangles*self.NOShfZ
                if use_angular_radial_coupling:
                    angular_orbital_aev_length = angular_orbital_aev_length + 9*NOShfA
                orbital_aev_length = orbital_aev_length + angular_orbital_aev_length
            self.orbital_aev_computer = OrbitalAEVComputer()   
        #To do: Do we actually need to define an orbital_aev_length if we are not using the geometric aevs?
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

        aev = self.orbital_aev_computer(
                coefficients=coefficients,
                species=species,
                use_simple_orbital_aev = self.use_simple_orbital_aev,
                use_angular_info = self.use_angular_info,
                use_angular_radial_coupling = self.use_angular_radial_coupling,
                basis_functions = self.basis_functions,
                normalization_library = self.normalization_library,
                NOShfR = self.NOShfR,
                NOShfA = self.NOShfA,
                NOShfZ = self.NOShfZ,
                LowerOShfR = self.LowerOShfR,
                UpperOShfR = self.UpperOShfR,
                LowerOShfA = self.LowerOShfA,
                UpperOShfA = self.UpperOShfA,
                LowerOShfZ = self.LowerOShfZ,
                UpperOShfZ = self.UpperOShfZ,
                OEtaS = self.OEtaS
                OEtaR = self.OEtaR,
                OEtaA = self.OEtaA,
                OZeta = self.OZeta,
            )

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
