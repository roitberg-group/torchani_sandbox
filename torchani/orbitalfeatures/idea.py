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
        species: Tensor,
        coefficients: Tensor,
    ) -> tp.Tuple[Tensor, Tensor, Tensor, Tensor]:
        s_coeffs, orbital_matrix = self._reshape_coefficients(coefficients)

        neighbor_idxs, distances = self._get_orbital_dists_and_idx(orbital_matrix)

        return s_coeffs, orbital_matrix, neighbor_idxs, distances

    def _reshape_coefficients(
        self,
        coefficients: Tensor,
    ) -> tp.Tuple[Tensor, Tensor]: 
        """ Output: A tuple containing 2 tensors: one with the s-type coefficients,
        and another with the p and d-type coefficients in the form of a matrix. 

        The obtained orbital matrix will look like:

        [p0x  p0y  p0z]
        ...
        [p3x  p3y  p3z]
        [d0xx d0yy d0zz]
        [d0zy d0zx d0xy]
        ...
        [d3xx d3yy d3zz]
        [d3zy d3zx d3xy]

        Where each row of this matrix is an Atomic Orbital Vector (AOV)
        """
        nconformers, natoms, _ = coefficients.shape

        s_coeffs = coefficients[:,:,:9]   # Shape: (nconformers, natoms, 9)
        p_coeffs = coefficients[:,:,9:21] # Shape: (nconformers, natoms, 12)
        d_coeffs = coefficients[:,:,21:]  # Shape: (nconformers, natoms, 24)

        # Reshape p_coeffs to make it easier to handle individual components
        p_coeffs_reshaped = p_coeffs.view(nconformers, natoms, 4, 3)  # Shape: (nconformers, natoms, 4, 3)

        # Reshape d_coeffs to make it easier to handle individual components
        # Correcting the reordering of d_coeffs
        # Transformation [Dxx, Dxy, Dyy, Dzx, Dzy, Dzz] -> [Dxx, Dyy, Dzz, Dzy, Dzx, Dxy]
        # which maps to indices [0, 2, 5, 4, 3, 1] respectively
        d_coeffs_reshaped = d_coeffs.view(nconformers, natoms, 4, 6)  # Shape: (nconformers, natoms, 4, 6) 
        d_coeffs_reshaped_reordered = d_coeffs_reshaped[:, :, :, [0, 2, 5, 4, 3, 1]]
        d_coeffs_reshaped_reordered = d_coeffs_reshaped_reordered.view(nconformers, natoms, 8, 3)  # Shape: (nconformers, natoms, 8, 3)        

        # Splitting into two groups: diagonal [Dxx, Dyy, Dzz] and off-diagonal [Dzy, Dzx, Dxy]
        d_diagonal = d_coeffs_reshaped_reordered[:, :, :3]
        d_off_diagonal = d_coeffs_reshaped_reordered[:, :, 3:]

        # Concatenate modified p and d coefficients to form the desired "matrix"
        orbital_matrix = torch.cat([p_coeffs_reshaped, d_diagonal, d_off_diagonal], dim=2)  # Shape (nconformers, natoms, 12, 3)

        return s_coeffs, orbital_matrix

    def _get_orbital_dists_and_idx(
        self,
        orbital_matrix: Tensor,
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        """ Output: A tuple containing 3 tensors resembling the neighbor_idxs,
        and distances tensors from the geometric AEVs.The neighor_idxs tensor,
        for each atom of each conformer, looks like:
         
          [ 0  0  0  ... 0 ]
          [ 1  2  3  ... 12]
          
        Note the number of pairs is 12 just like the number of AOVs forming the orbital matrix
        Here 0 represents the actual atom, and the numbers from 1 to 12 represent the
        surroinding AOVs. Of course that the AOVs could be zeros depending on the atom species,
        but that will be taken care off later.

        The distances tensor simply involves evaluating the module of the AOVs. In other words,
        the only distances we are worried about are the distances between the actual atom and a
        12 "fake" atoms in the coordinates of the AOVs.
          
        """
        nconformers, natoms, _ , _ = orbital_matrix.shape

        # Create a new tensor of zeros with the desired shape (nconformers, natoms, 2, 12)
        neighbor_idxs = torch.zeros((nconformers, natoms, 2, 12), dtype=torch.int)
        # Assigning the sequence to each position in the third dimension slice 2
        neighbor_idxs[:, :, 1, :] = torch.arange(1, 13, dtype=torch.int)
        # Calculate modules of the AOVs
        distances = torch.sqrt((orbital_matrix ** 2).sum(dim=-1))       

        return neighbor_idxs, distances


    def _get_orbital_dists_and_idx(
        self,
        orbital_matrix: Tensor,
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:


