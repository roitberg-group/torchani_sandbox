import typing as tp
import torch
from torch import Tensor
import numpy as np

# This calculates ONLY the coefficients part of the AEV
class OrbitalAEVComputer(torch.nn.Module):
    def forward(
        self,
        coefficients: Tensor,
        species: Tensor,
        basis_functions: str,
        use_simple_orbital_aev: bool,
        use_angular_info: bool,
        NOShfS = int,
        NOShfR = int,
        NOShfA = int,
        NOShfZ = int,
        LowerOShfS = float,
        UpperOShfS = float,
        LowerOShfR = float,
        UpperOShfR = float,
        LowerOShfA = float,
        UpperOShfA = float,
        LowerOShfZ = float,
        UpperOShfZ = float,
        OEtaS = float,
        OEtaR = float,
        OEtaA = float,
        OZeta = float,
    ) -> Tensor:
        # We first need to reshape the coefficients for their manipulation
        # The s-type coefficients are processed with a custom radial AEV computer
        # The p and d-type coefficientes form an orbital matrix that basically
        # Describes the coordinates in the "space of coefficients" of fake atoms
        # Around each actual atom.
        
        s_coeffs, orbital_matrix = self._reshape_coefficients(coefficients,basis_functions)

        # Return "simple_orbital_aevs" if corresponding
        if use_simple_orbital_aev:
            if basis_functions == 's':
                return self._reshape_coefficients(coefficients,basis_functions) # shape (nconf, natoms, simple_orbital_aevs_length)                       
            distances = torch.linalg.norm(orbital_matrix, dim=-1)
            simple_orbital_aevs = torch.cat((s_coeffs, distances), dim=-1)
            if use_angular_info:
                angles = self._get_angles_from_orbital_matrix(orbital_matrix,distances,True)
                simple_orbital_aevs = torch.cat((simple_orbital_aevs, angles), dim=-1)
                # TEST
                # simple_orbital_aevs = angles
            return simple_orbital_aevs  # shape (nconf, natoms, simple_orbital_aevs_length)
        
        else: # Return actual orbital_aevs
            nconf, natoms, nscoeffs = s_coeffs.shape
            # Define s shifts and reshape for broadcasting
            OShfS = torch.linspace(LowerOShfS, UpperOShfS, NOShfS)
            OShfS = OShfS.view(1, 1, 1, NOShfS)
            # Normalize s_coeffs based on predefined mu and sigma values on a given dataset
            s_coeffs = self._normalice_s_coeffs(s_coeffs, species)
            #To test normalization
            # if basis_functions == 's':
            #     return s_coeffs            
            # Ensure the tensor is correctly shaped for broadcasting
            s_coeffs = s_coeffs.view(nconf, natoms, nscoeffs, 1)            
            # Compute the s component of the orbital AEV
            s_orbital_aev = torch.exp(-OEtaS * ((s_coeffs - OShfS) ** 2))
                       
            if basis_functions == 's':
                return s_orbital_aev

            distances = torch.linalg.norm(orbital_matrix, dim=-1)
            _, _, ndistances = distances.shape                  
            # Define r shifts and reshape for broadcasting
            OShfR = torch.linspace(LowerOShfR, UpperOShfR, NOShfR)
            OShfR = OShfR.view(1, 1, 1, NOShfR)
            # Ensure the tensor is correctly shaped for broadcasting
            distances = distances.view(nconf, natoms, ndistances, 1)
            # Compute the squared differences
            radial_orbital_aev = torch.exp(-OEtaR * ((distances - OShfR) ** 2))
            # Concatenate the s and radial contributions to the orbital aevs
            orbital_aev = torch.cat((s_orbital_aev, radial_orbital_aev), dim=-1)            
            if use_angular_info:
                angles, avdistperangle = self._get_angles_from_orbital_matrix(orbital_matrix,distances,False)
                _, _, nangles = angles.shape
                # Define r shifts for the angular component of the AEV and reshape for broadcasting
                OShfA = torch.linspace(LowerOShfA, UpperOShfA, NOShfA)                
                # Define angle sections and reshape for broadcasting
                OShfZ = torch.linspace(LowerOShfZ, UpperOShfZ, NOShfZ)                
                # Expand angles for ShfZ
                expanded_angles = angles.unsqueeze(-1)  # Adding an extra dimension for broadcasting ShfZ
                expanded_angles = expanded_angles.expand(-1, -1, -1, NOShfZ)  # Explicitly expand to match ShfZ

                # Expand ShfZ to match angles
                OShfZ = OShfZ.view(1, 1, 1, NOShfZ)  # Reshape for broadcasting
                OShfZ = OShfZ.expand(nconf, natoms, nangles, NOShfZ)  # Match dimensions
 
                # Calculate factor1
                factor1 = ((1 + torch.cos(expanded_angles - OShfZ)) / 2)**OZeta

                # Expand avdistperangle for ShfA
                expanded_avdistperangle = avdistperangle.unsqueeze(-1)  # Adding an extra dimension for ShfA
                expanded_avdistperangle = expanded_avdistperangle.expand(-1, -1, -1, NOShfA)  # Match dimensions of ShfA

                # Expand ShfA to match avdistperangle
                OShfA = OShfA.view(1, 1, 1, NOShfA)  # Reshape for broadcasting
                OShfA = OShfA.expand(nconf, natoms, nangles, NOShfA)  # Match dimensions

                # Calculate factor2
                factor2 = torch.exp(-OEtaA * (expanded_avdistperangle - OShfA)**2)

                # Combine factors
                angular_orbital_aev = 2 * factor1.unsqueeze(-1) * factor2.unsqueeze(3)

                # Reshape to the final desired shape
                angular_orbital_aev = angular_orbital_aev.reshape(nconf, natoms, nangles * NOShfA * NOShfZ)
               
                orbital_aev = torch.cat((orbital_aev, angular_orbital_aev), dim=-1) 

        return orbital_aev
        
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
            return s_coeffs, []
        
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

        # Splitting into two groups: diagonal [Dxx, Dyy, Dzz] and off-diagonal [Dzy, Dzx, Dxy]
        d_diagonal = d_coeffs_reshaped_reordered[:, :, :, :3]
        d_off_diagonal = d_coeffs_reshaped_reordered[:, :, :, 3:]

        # Concatenate modified p and d coefficients to form the desired "matrix"                
        orbital_matrix = torch.cat([p_coeffs_reshaped, d_diagonal, d_off_diagonal], dim=2)  # Shape (nconformers, natoms, 12, 3)

        return s_coeffs, orbital_matrix

    def _normalice_s_coeffs(
        self,
        s_coeffs: Tensor,
        species: Tensor,
    ) -> Tensor:
        
        #For now this is written for H and O, with values extracted from an only-water dataset
        #Define a Tensor with the values required for normalization:
        s_coeffs_H_mu = [-0.0013891633279463555,
                          0.052817133273316774,
                          0.139812833597163,
                          0.04197082575153015]
        s_coeffs_O_mu = [0.3869724094534165,
                         1.1195726605859562,
                         4.597656411500856,
                         4.18116756869605,
                        -0.777753667038269,
                         0.8991094564592526,
                         0.25287442792706855]
        raw_data = [
           s_coeffs_H_mu,  # H = Z=1
           s_coeffs_O_mu   # O = Z=8
        ]
        # Pad each row to length 9
        padded_data = []
        for row in raw_data:
            row_padded = row + [0.0] * (9 - len(row))
            padded_data.append(row_padded)

        # Convert to a torch tensor of shape (2, 9)
        s_coeffs_mus = torch.tensor(padded_data, dtype=torch.float)

        #Does the same for the sigmas
        s_coeffs_O_sigma = [0.01648580936203453,
                            0.04360467004470585,
                            0.10939958699162289,
                            0.1452881891452822,
                            0.24744978852384295,
                            0.6757096584771374,
                            0.15629951845443388]      
        s_coeffs_H_sigma = [0.014984132924379046,
                            0.06597323056314336,
                            0.05229082868259066,
                            0.10317217531939492]
        raw_data = [
           s_coeffs_H_sigma,  # H = Z=1
           s_coeffs_O_sigma   # O = Z=8
        ]
        # Pad each row to length 9 (we padd sigma with 1 to avoid dividing by zero in the next step)
        padded_data = []
        for row in raw_data:
            row_padded = row + [1.0] * (9 - len(row))
            padded_data.append(row_padded)

        # Convert to a torch tensor of shape (2, 9), here 2 is the number of atomic species
        s_coeffs_sigmas = torch.tensor(padded_data, dtype=torch.float)        
         
        #####################################################################
        # Now we normalize
        # Example shapes:
        # s_coeffs:        (nconformers, natoms, 9)
        # species:         (nconformers, natoms) with entries in {1, 8}
        # s_coeffs_mus:    (2, 9) -- first row for H, second row for O
        # s_coeffs_sigmas: (2, 9) -- first row for H, second row for O

        nconformers, natoms, _ = s_coeffs.shape # s_coeffs shape: (nconformers, natoms, 9)

        # 1) Convert atomic numbers to 0/1 indices
        #    0 => hydrogen (Z=1), 1 => oxygen (Z=8).
        species_idx = (species == 8).long()  # shape (nconformers, natoms)

        # 2) Use advanced indexing so that for each (i,j), we pick row 0 or 1
        #    from s_coeffs_mus and s_coeffs_sigmas.
        #    The result has shape (nconformers, natoms, 9).
        atom_mus = s_coeffs_mus[species_idx, :]
        atom_sigmas = s_coeffs_sigmas[species_idx, :]

        # 3) Perform the normalization per atom
        #    shape (nconformers, natoms, 9)
        s_coeffs_normalized = (s_coeffs - atom_mus) / atom_sigmas
        return s_coeffs_normalized


    def _normalice_p_distances(
        self,
        distances: Tensor,
        species: Tensor,
    ) -> Tensor:
        
        #For now this is written for H and O, with values extracted from an only-water dataset
        #Define a Tensor with the values required for normalization:

        p_distances_O_mu = [0.04944367235325901,
                         0.044594583975525556,
                         0.04214927903340951]

        raw_data = [
           p_distances_O_mu   # O = Z=8
        ]

        # Pad to length 4
        padded_data = []
        for row in raw_data:
            row_padded = row + [0.0] * (4 - len(row))
            padded_data.append(row_padded)

        # Convert to a torch tensor of shape (2, 9)
        p_coeffs_mus = torch.tensor(padded_data, dtype=torch.float)

        #Does the same for the sigmas
        p_coeffs_O_sigma = [0.023578402218927527,
                        0.06970818495769757,
                        0.05985785104138217]    

        raw_data = [
           p_coeffs_O_sigma   # O = Z=8
        ]
        # Pad each row to length 9
        padded_data = []
        for row in raw_data:
            row_padded = row + [0.0] * (9 - len(row))
            padded_data.append(row_padded)

        # Convert to a torch tensor of shape (2, 9)
        s_coeffs_sigmas = torch.tensor(padded_data, dtype=torch.float)        
         
        #####################################################################
        # Now we normalize
        # Example shapes:
        # s_coeffs:        (nconformers, natoms, 9)
        # species:         (nconformers, natoms) with entries in {1, 8}
        # s_coeffs_mus:    (2, 9) -- first row for H, second row for O
        # s_coeffs_sigmas: (2, 9) -- first row for H, second row for O

        nconformers, natoms, _ = s_coeffs.shape # s_coeffs shape: (nconformers, natoms, 9)

        # 1) Convert atomic numbers to 0/1 indices
        #    0 => hydrogen (Z=1), 1 => oxygen (Z=8).
        species_idx = (species == 8).long()  # shape (nconformers, natoms)

        # 2) Use advanced indexing so that for each (i,j), we pick row 0 or 1
        #    from s_coeffs_mus and s_coeffs_sigmas.
        #    The result has shape (nconformers, natoms, 9).
        atom_mus = s_coeffs_mus[species_idx, :]
        atom_sigmas = s_coeffs_sigmas[species_idx, :]

        # 3) Perform the normalization per atom
        #    shape (nconformers, natoms, 9)
        s_coeffs_normalized = (s_coeffs - atom_mus) / atom_sigmas
        return s_coeffs_normalized

    # def _normalize_distances(
    #     self,
    #     distances: Tensor,
    #     species: Tensor,
    # ) -> Tensor:
        
    #     0.04944367235325901 0.023578402218927527
    #     0.044594583975525556 0.06970818495769757
    #     0.04214927903340951 0.05985785104138217
    #     nconformers, natoms, naovs = distances.shape


    def _get_angles_from_orbital_matrix(
        self,
        orbital_matrix: Tensor,
        distances: Tensor,
        use_simple_orbital_aev: bool,
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
           
        # Calculate angles between each vector and the following vectors
        nangles = int((naovs-1)*naovs/2)
        angles = torch.zeros((nconformers, natoms,nangles))
        k = 0
        if use_simple_orbital_aev:
            for i in range(naovs):
                for j in range(i+1, naovs):
                    cos_angles = torch.einsum('ijk,ijk->ij', orbital_matrix_normalized[:, :, i, :], orbital_matrix_normalized[:, :, j, :])
                    angles[:, :, k] = torch.acos(0.9999 * cos_angles) # 0.95 is multiplied to the cos values to prevent acos from returning NaN.     
                    k = k + 1
            return angles
        else:
            avdistperangle = torch.zeros((nconformers, natoms, nangles))        
            for i in range(naovs):
                for j in range(i+1, naovs):
                    cos_angles = torch.einsum('ijk,ijk->ij', orbital_matrix_normalized[:, :, i, :], orbital_matrix_normalized[:, :, j, :])
                    angles[:, :, k] = torch.acos(0.9999 * cos_angles) # 0.95 is multiplied to the cos values to prevent acos from returning NaN.     
                    avdistperangle[:, :, k] = (distances[:, :, i]+distances[:, :, j])/2.0
                    k = k + 1
            return angles,avdistperangle
