import typing as tp
import torch
from torch import Tensor

#from torchani.aev import AEVComputer
#from torchani.aev.aev_terms import StandardAngular, StandardRadial
#from torchani.orbitalfeatures.orb_utils import AtomicNumsToInts

class ExCorrAEVComputerVariation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        species_coords: tp.Tuple[Tensor, Tensor],
        coefficients: Tensor,
    ) -> tp.Tuple[Tensor, Tensor]:
        
        return species, torch.tensor(0)

    def _reshape_coefficients(
        self,
        species: Tensor,
        coefficients: Tensor,
        combine: bool,
    ) -> tp.Tuple[Tensor, Tensor]: 
        """ Output: A tuple containing 2 tensors: one with the s-type coefficients,
        and another with the sp, p and d-type coefficients. The sp-type coefficients 
        are calculated combining the s and p-type coefficiens that share exponents in
        their associated functions. This is done by default to reduce the number of
        "orbital types" in next steps, unless combine is set to False.
        """
        nconformers, natoms, _ = coefficients.shape

        s_core_coeffs = coefficients[:,:,:5]   # Shape: (nconformers, natoms, 4)
        s_valence_coeffs = coefficients[:,:,5:9] # Shape (nconformers, natoms, 5)
        p_coeffs = coefficients[:,:,9:21] # Shape: (nconformers, natoms, 12)
        d_coeffs = coefficients[:,:,21:]  # Shape: (nconformers, natoms, 24)

        # Reshape p_coeffs to make it easier to handle individual components
        p_coeffs_reshaped = p_coeffs.view(nconformers, natoms, 4, 3)  # Shape: (nconformers, natoms, 4, 3)

        # Reshape p_coeffs to make it easier to handle individual components
        d_coeffs_reshaped = d_coeffs.view(nconformers, natoms, 8, 3)  # Shape: (nconformers, natoms, 8, 3)
        
        if combine:
            # Expand s_to_combine to make it compatible for element-wise addition
            s_valence_coeffs_expanded = s_valence_coeffs.unsqueeze(-1).expand_as(p_coeffs_reshaped)  # Shape: (nconformers, natoms, 4, 3)
            # Add (1/3) of the corresponding s term to each p coefficient
            p_coeffs_reshaped = p_coeffs_reshaped + (1/3) * s_valence_coeffs_expanded
        else: 
            # If we are not combinng s and p coefficients, all s cofficients are consided "core"
            s_core_coeffs = torch.cat((s_core_coeffs, s_valence_coeffs), dim=2) 
        
        # Correcting the reordering of d_coeffs
        # Transformation [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz] -> [Dxx, Dyy, Dzz, Dyz, Dxz, Dxy]
        # which maps to indices [0, 3, 5, 4, 2, 1] respectively
        d_coeffs_reshaped_reordered = d_coeffs_reshaped[:, :, [0, 3, 5, 4, 2, 1]]

        # Splitting into two groups: diagonal [Dxx, Dyy, Dzz] and off-diagonal [Dyz, Dxz, Dxy]
        d_diagonal = d_coeffs_reshaped_reordered[:, :, :3]
        d_off_diagonal = d_coeffs_reshaped_reordered[:, :, 3:]

        # Concatenate modified p and d coefficients to form the desired "matrix"
        orbital_matrix = torch.cat([p_coeffs_reshaped, d_coeffs_reshaped], dim=2)  # Shape (nconformers, natoms, 12, 3)

        return s_core_coeffs, orbital_matrix

