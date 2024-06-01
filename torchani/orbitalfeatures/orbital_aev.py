import typing as tp
import torch
from torch import Tensor


# This calculates ONLY the coefficients part of the AEV
class SimpleOrbitalAEVComputer(torch.nn.Module):
    def forward(
        self,
        coefficients: Tensor,
        basis_functions: str,
        use_angular_info: bool,
    ) -> Tensor:
        # ) -> tp.Tuple[Tensor, Tensor, Tensor, Tensor]:
        # We first need to reshape the coefficients for their manipulation
        # The s-type coefficients are processed with a custom radial AEV computer
        # The p and d-type coefficientes form an orbital matrix that basically
        # Describes the coordinates in the "space of coefficients" of fake atoms
        # Around each actual atom.

        if basis_functions == 's':
            return self._reshape_coefficients(coefficients,basis_functions) # shape (nconf, natoms, simple_orbital_aevs_length)
        
        s_coeffs, orbital_matrix = self._reshape_coefficients(coefficients,basis_functions)

        # In order to use the AEV computer function from the geometric AEVs, we need
        # to generate arrays that can serve as input for such function
        distances = torch.linalg.norm(orbital_matrix, dim=-1)
        simple_orbital_aevs = torch.cat((s_coeffs, distances), dim=-1)
        if use_angular_info:
            angles = self._get_angles_from_orbital_matrix(orbital_matrix,distances)
            simple_orbital_aevs = torch.cat((simple_orbital_aevs, angles), dim=-1)
        return simple_orbital_aevs  # shape (nconf, natoms, simple_orbital_aevs_length)

    def _reshape_coefficients(
        self,
        coefficients: Tensor,
        basis_functions: str,
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

        Where each row of this matrix is an Atomic Orbital Vector (AOV). This resembles the
        diff_vec tensor (for a single atom) fron the geometric AEVs.
        """
        nconformers, natoms, _ = coefficients.shape

        s_coeffs = coefficients[:, :, :9]    # Shape: (nconformers, natoms, 9)
        p_coeffs = coefficients[:, :, 9:21]  # Shape: (nconformers, natoms, 12)
        d_coeffs = coefficients[:, :, 21:]   # Shape: (nconformers, natoms, 24)

        if basis_functions == 's':
            return s_coeffs
        
        # Reshape p_coeffs to make it easier to handle individual components
        p_coeffs_reshaped = p_coeffs.view(nconformers, natoms, 4, 3)  # Shape: (nconformers, natoms, 4, 3)

        if basis_functions == 'sp':
            return s_coeffs, p_coeffs_reshaped #In this case the orbital_matrix only have AOVs from p-type coeffients
        
        # If we are in the 'spd' case, the orbital_matrix includes also AOVs from d-type coefficients

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

    def _get_angles_from_orbital_matrix(
        self,
        orbital_matrix: Tensor,
        distances: Tensor
    ) -> Tensor:
        nconformers, natoms, naovs = distances.shape
        # Normalize the vectors using the provided distances

        # Create a mask for the zero vectors in the orbital matrix
        zero_mask = (orbital_matrix.abs() < 1e-12).all(dim=-1)

        # Perform the normalization, avoid division by zero by using where
        orbital_matrix_normalized = torch.where(
        zero_mask.unsqueeze(-1),
        torch.zeros_like(orbital_matrix),
        orbital_matrix / distances.unsqueeze(-1)        
        )
        # orbital_matrix_normalized = orbital_matrix / distances.unsqueeze(-1)
           
        # Calculate angles between each vector and the following vectors
        nangles = int((naovs-1)*naovs/2)
        angles = torch.zeros((nconformers, natoms,nangles))
        k = 0
        for i in range(naovs):
            for j in range(i+1, naovs):
                cos_angles = torch.einsum('ijk,ijk->ij', orbital_matrix_normalized[:, :, i, :], orbital_matrix_normalized[:, :, j, :])
                angles[:, :, k] = torch.acos(0.95 * cos_angles) # 0.95 is multiplied to the cos values to prevent acos from returning NaN.     
                k = k + 1
        return angles