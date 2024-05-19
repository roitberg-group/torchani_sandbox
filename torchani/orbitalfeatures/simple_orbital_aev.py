import typing as tp
import torch
from torch import Tensor


# This calculates ONLY the coefficients part of the AEV
class SimpleOrbitalAEVComputer(torch.nn.Module):
    def forward(
        self,
        coefficients: Tensor,
        basis_functions: str,
        use_angular_info: Bool,
    ) -> Tensor:
        # ) -> tp.Tuple[Tensor, Tensor, Tensor, Tensor]:
        # We first need to reshape the coefficients for their manipulation
        # The s-type coefficients are processed with a custom radial AEV computer
        # The p and d-type coefficientes form an orbital matrix that basically
        # Describes the coordinates in the "space of coefficients" of fake atoms
        # Around each actual atom.
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
            return s_coeffs, torch.tensor([]) #In this case the orbital_matrix is empty
        
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
        #To do 
        #If basis_functions is 'spd', orbital_matrix.shape is (nconformers, natoms, 12, 3) and distances shape is (nconformers, natoms, 12)
        #If basis_functions is 'sp',  orbital_matrix.shape is (nconformers, natoms,  4, 3) and distances shape is (nconformers, natoms, 4)

        nconformers, natoms, naovs = distances.shape
         
        # Normalize the vectors using the provided distances
        orbital_matrix_normalized = orbital_matrix / distances.unsqueeze(-1)

        # Calculate angles between each vector and the following vectors
        angles = torch.zeros((nconformers, natoms, naovs, naovs))
        for i in range(naovs):
            for j in range(i+1, naovs):
                cos_angles = torch.einsum('ijkl,ijml->ijkm', orbital_matrix_normalized[:, :, i, :], orbital_matrix_normalized[:, :, j, :])                    
                angles[:, :, i, j] = torch.acos(0.95 * cos_angles) # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
     
        return angles
 

    def _get_aev_inputs(
        self,
        orbital_matrix: Tensor,
    ) -> tp.Tuple[Tensor, Tensor]:
        """ Output: A tuple containing 3 tensors resembling the neighbor_idxs,
        distances and diff_vectors tensors from the geometric AEVs.

        To construct the neighbor_idxs tensor, a set of 12 (4 p + 8 d) fake atoms
        is "attached" to each conformer. The coordinates of these fake atoms are
        given in the "coefficients" space, and are those of the corresponding AOVs.

        This way, if a given confomer has natoms the indexes 0:natoms belong to
        actual atoms, while the indexes natoms:natoms+12 represent the fake atoms.

        For example, let's consider the case in which we only have two corformers:
        A water molecule (H2O, H H O) and carbon monoxide (C O)
        This way, the neighbor_idxs tensor would look like:

        [ 2  2  2  ... 2   0  0  0 ... 0   1  1  1 ... 1  ]
        [ 3  4  5  ... 14  3  4  5 ... 14  3  4  5 ... 14 ]

        Here the first row correspond to actual atoms indexes, while the second one
        corresponds to fake atoms. Note that natoms is 3, that is the size of the
        larger molecule.

        The distances tensor simply involves evaluating the module of the AOVs.
        In other words, the only distances we are worried about are the distances
        between the actual atom and a 12 "fake" atoms in the coordinates of the AOVs.

        In the example from above, the distances tensor would look like:
        [ d2-3  d2-4  d2-5  ... d2-14  d0-3  d0-4  d0-5 ... d0-14  d1-3  d1-4  d1-5 ... d1-14 ]

        The diff_vectors tesor follows the same idea.
        """
        # orbital_matrix = orbital_matrix.view(-1, 12, 3)
        # First dimension is natoms * nconformers * 12
        orbital_matrix = orbital_matrix.view(-1, 12, 3)
        distances = torch.linalg.norm(orbital_matrix, dim=-1)
        ntotal = orbital_matrix.shape[0] * orbital_matrix.shape[1]
        # nconformers, natoms, _, _ = orbital_matrix.shape

        # TODO: This may not be needed, it is kind of strange that we need to do it
        # ntotal = nconformers * natoms
        top = torch.repeat_interleave(
            torch.arange(ntotal, dtype=torch.long, device=orbital_matrix.device),
            12,
        )
        fake_atom_idxs = torch.arange(ntotal, ntotal + 12, dtype=torch.long, device=orbital_matrix.device)
        bottom = torch.tile(fake_atom_idxs, (ntotal,))
        neighbor_idxs = torch.cat((top.unsqueeze(0), bottom.unsqueeze(0)), dim=0)

        # neighbor_idxs[:, :, 0, :] = torch.arange(0, natoms, dtype=torch.long)
        # neighbor_idxs[:, :, 1, :] =
        # Calculate modules of the AOVs
        # distances = torch.sqrt((orbital_matrix ** 2).sum(dim=-1))
        # distances = torch.linalg.norm(orbital_matrix)
        # Permute the tensor to rearrange the dimensions to (2, nconformers, natoms, 12)
        # distances = distances.permute(2, 0, 1, 3)
        # # Flatten all dimensions except the first into one dimension
        # distances = distances.reshape(2, -1)  # Now shape (2, npairs), where npairs = 12*natoms*nconformers
        # For now it only works
        return neighbor_idxs, distances
