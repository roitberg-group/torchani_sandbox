import typing as tp
import torch
from torch import Tensor
import numpy as np

# This calculates ONLY the coefficients part of the AEV
class OrbitalAEVComputer(torch.nn.Module):
    def forward(
        self,
        coefficients: Tensor,
        normalization_library: Tensor,
        species: Tensor,
        basis_functions: str,
        use_simple_orbital_aev: bool,
        use_angular_info: bool,
        use_angular_radial_coupling: bool,
        NOShfS: int,
        NOShfR: int,
        NOShfA: int,
        NOShfTheta: int,
        LowerOShfS: float,
        UpperOShfS: float,
        LowerOShfR: float,
        UpperOShfR: float,
        LowerOShfA: float,
        UpperOShfA: float,
        LowerOShfTheta: float,
        UpperOShfTheta: float,
        OEtaS: float,
        OEtaR: float,
        OEtaA: float,
        OZeta: float,
    ) -> Tensor:    
        # We first need to reshape the coefficients for their manipulation
        # The s-type coefficients are processed with a custom radial AEV computer
        # The p and d-type coefficientes form an orbital matrix that basically
        # Describes the coordinates in the "space of coefficients" of fake atoms
        # Around each actual atom.
        
        s_coeffs, p_matrix, D_blocks = self._reshape_coefficients(coefficients,basis_functions)
        Z_to_idx = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}

        #Normalizes s_coeffs
        # Convert species -> row indices
        species_idx = self._map_species_to_idx(species, Z_to_idx)  # shape (nconf, natoms)

        # Unpack normalization library
        # s_coeffs_mus, s_coeffs_sigmas, p_norms_mus, p_norms_sigmas = normalization_library   # unpack
        
        # # Trims p normalization values to make them match other stuff
        # p_norms_mus=p_norms_mus[:,:4]
        # p_norms_sigmas = p_norms_sigmas[:,:4]
        # s_coeffs = self._normalize(s_coeffs,species_idx,s_coeffs_mus,s_coeffs_sigmas) #TODO: normalization of D norms?

        # Return "simple_orbital_aevs" if corresponding
        if use_simple_orbital_aev:
            # Case s
            if basis_functions == 's':
                return s_coeffs
            # Case sp or spd           

            p_norms = torch.linalg.norm(p_matrix, dim=-1)

            # p_norms = self._normalize(p_norms, species_idx, p_norms_mus, p_norms_sigmas) #TODO: normalization of D norms?

            if basis_functions == 'sp':
                simple_orbital_aevs = torch.cat((s_coeffs, p_norms), dim=-1)
                if use_angular_info:
                    angles = self._get_angles_from_p_matrix(p_matrix,p_norms,True)
                    simple_orbital_aevs = torch.cat((simple_orbital_aevs, angles), dim=-1)
                return simple_orbital_aevs  # shape (nconf, natoms, simple_orbital_aevs_length)
            else: # Case spd   
                #D blocks: (B, N, 4, 3, 3)
                D_norms = torch.linalg.norm(D_blocks, dim=(3,4)) # Frobenius norm of each D block -> (B,N,4)
                # D_norms = self._normalize(D_norms, species_idx, d_norms_mus, d_norms_sigmas) #TODO: normalization of D norms?
                #-----------------------
                # TEST
                # B, N, _ = coefficients.shape
                # d_raw = coefficients[:, :, 21:].reshape(B, N, 24)  # (B,N,24)
                # return d_raw
                # END TEST
                #-----------------------
                simple_orbital_aevs = torch.cat((s_coeffs, p_norms, D_norms), dim=-1)
                if use_angular_info:
                    angles_pp = self._get_angles_from_p_matrix(p_matrix,p_norms,True)
                    angles_pq = self._get_angles_p_d_blocks(p_matrix,D_blocks,p_norms,D_norms)
                    angles_qq = self._get_angles_d_blocks(D_blocks,D_norms)
                    angles = torch.cat((angles_pp,angles_pq,angles_qq),dim=-1)
                    simple_orbital_aevs = torch.cat((simple_orbital_aevs, angles), dim=-1)
                    #TESTING
                    # simple_orbital_aevs = torch.cat((simple_orbital_aevs, angles_pp), dim=-1)
                # return  torch.cat((D_norms,angles_pq,angles_qq), dim = -1)  # shape (nconf, natoms, simple_orbital_aevs_length)         
                # print(f"Simple orbital AEVs shape: {simple_orbital_aevs.shape}")
                return simple_orbital_aevs  # shape (nconf, natoms, simple_orbital_aevs_length)

        else: # Return actual orbital_aevs
            # In this case will need to know the device and dtype of the input tensors
            device, dtype = coefficients.device, coefficients.dtype
            nconf, natoms, nscoeffs = s_coeffs.shape
            # Define s shifts and reshape for broadcasting
            OShfS = torch.linspace(LowerOShfS, UpperOShfS, NOShfS, device=device, dtype=dtype)
            OShfS = OShfS.view(1, 1, 1, NOShfS)

            # s_coeffs are already normalized here, so we only prepare the tensor for broadcasting
            s_coeffs = s_coeffs.view(nconf, natoms, nscoeffs, 1)      

            # Compute the s component of the orbital AEV
            s_orbital_aev = torch.exp(-OEtaS * ((s_coeffs - OShfS) ** 2))
            
            # Reshapes
            s_orbital_aev = s_orbital_aev.view(nconf,natoms,nscoeffs*NOShfS)

            # s_orbital_aev = s_terms.sum(dim=2)

            if basis_functions == 's':
                return s_orbital_aev

            p_norms = torch.linalg.norm(p_matrix, dim=-1)
            
            # p_norms = self._normalize(p_norms, species_idx, p_norms_mus, p_norms_sigmas)

            _, _, np_norms = p_norms.shape                  
            # Define r shifts and reshape for broadcasting
            OShfR = torch.linspace(LowerOShfR, UpperOShfR, NOShfR, device=device, dtype=dtype)
            OShfR = OShfR.view(1, 1, 1, NOShfR)
            # Ensure the tensor is correctly shaped for broadcasting
            p_norms_exp = p_norms[..., None]          # add dim, no data copy
            # p_norms = p_norms.view(nconf, natoms, np_norms, 1)
            # Compute the squared differences
            radial_orbital_aev = torch.exp(-OEtaR * ((p_norms_exp - OShfR) ** 2))
            # Concatenate the s and radial contributions to the orbital aevs

            # Reshapes
            radial_orbital_aev = radial_orbital_aev.view(nconf,natoms,np_norms*NOShfR)

            orbital_aev = torch.cat((s_orbital_aev, radial_orbital_aev), dim=-1)            
            if use_angular_info:
                angles, avdistperangle = self._get_angles_from_p_matrix(p_matrix,p_norms,False)
                _, _, nangles = angles.shape

                # Define angle sections and reshape for broadcasting
                OShfTheta = torch.linspace(LowerOShfTheta, UpperOShfTheta, NOShfTheta, device=device, dtype=dtype)

                # Expand angles for ShfZ
                expanded_angles = angles.unsqueeze(-1)  # Adding an extra dimension for broadcasting ShfZ
                expanded_angles = expanded_angles.expand(-1, -1, -1, NOShfTheta)  # Explicitly expand to match ShfZ

                # Expand ShfZ to match angles
                OShfTheta = OShfTheta.view(1, 1, 1, NOShfTheta)  # Reshape for broadcasting
                OShfTheta = OShfTheta.expand(nconf, natoms, nangles, NOShfTheta)  # Match dimensions
 
                # Calculate factor1
                angular_orbital_aev = ((1 + torch.cos(expanded_angles - OShfTheta)) / 2)**OZeta
                angular_orbital_aev = 2**(1-OZeta)*angular_orbital_aev.unsqueeze(-1)

                if (use_angular_radial_coupling):
                    # Define r shifts for the angular component of the AEV and reshape for broadcasting
                    OShfA = torch.linspace(LowerOShfA, UpperOShfA, NOShfA, device=device, dtype=dtype)  
                    # Expand avdistperangle for ShfA
                    expanded_avdistperangle = avdistperangle.unsqueeze(-1)  # Adding an extra dimension for ShfA
                    expanded_avdistperangle = expanded_avdistperangle.expand(-1, -1, -1, NOShfA)  # Match dimensions of ShfA

                    # Expand ShfA to match avdistperangle
                    OShfA = OShfA.view(1, 1, 1, NOShfA)  # Reshape for broadcasting
                    OShfA = OShfA.expand(nconf, natoms, nangles, NOShfA)  # Match dimensions

                    # Calculate factor2
                    factor2 = torch.exp(-OEtaA * (expanded_avdistperangle - OShfA)**2)

                    # Combine factors
                    # angular_orbital_aev = angular_orbital_aev * factor2.unsqueeze(-1) #unsqueeze(3)?
                    angular_orbital_aev = angular_orbital_aev.unsqueeze(-1) * factor2.unsqueeze(-2)  #TODO check this (B,N,K,NOShfTheta,NOShfA)

                    # Reshape to the final desired shape
                    print(angular_orbital_aev.shape)
                    print(nconf, natoms, nangles * NOShfA * NOShfTheta)
                    angular_orbital_aev = angular_orbital_aev.reshape(nconf, natoms, nangles * NOShfA * NOShfTheta)
                else:
                    angular_orbital_aev = angular_orbital_aev.reshape(nconf, natoms, nangles *NOShfTheta)

                orbital_aev = torch.cat((orbital_aev, angular_orbital_aev), dim=-1) 

        return orbital_aev
        
    def _reshape_coefficients(
        self,
        coefficients: Tensor,
        basis_functions: str,
    ) -> tp.Tuple[Tensor, Tensor]:
        """ Output: A tuple containing 3 tensors: one with the s-type coefficients,
        another with the p-type coefficients in the form of a matrix, and another
        with invariants from the d-type coefficients.

        The obtained p matrix will look like:

        [p0x  p0y  p0z]
        ...
        [p3x  p3y  p3z]

        Where each row of this matrix is an Atomic Orbital Vector (AOV). This resembles the
        diff_vec tensor (for a single atom) fron the geometric AEVs. 
        
        And the d coefficients will be first organized as tensors Q
        [
        [d0xx d0xy d0xz],
        [d0yx d0yy d0yz],
        [d0zx d0zy d0zz],

        ...
        [d3xx d3xy d3xz],
        [d3yx d3yy d3yz],
        [d3zx d3zy d3zz]
        ]

        """
        nconformers, natoms, _ = coefficients.shape

        s_coeffs = coefficients[:, :, :9]    # Shape: (nconformers, natoms, 9)
        p_coeffs = coefficients[:, :, 9:21]  # Shape: (nconformers, natoms, 12)
        d_coeffs = coefficients[:, :, 21:]   # Shape: (nconformers, natoms, 24)

        if basis_functions == 's':
            return s_coeffs, torch.tensor([]), torch.tensor([])
        
        # Reshape p_coeffs to make it easier to handle individual components
        p_coeffs_reshaped = p_coeffs.view(nconformers, natoms, 4, 3)  # Shape: (nconformers, natoms, 4, 3)

        if basis_functions == 'sp':
            return s_coeffs, p_coeffs_reshaped, torch.tensor([]) #In this case the p_matrix only have AOVs from p-type coeffients
        
        # If we are in the 'spd' case, the p_matrix includes also AOVs from d-type coefficients

        # Reshape d_coeffs to make it easier to handle individual components
        # Correcting the reordering of d_coeffs
        # Transformation [Dxx, Dxy, Dyy, Dzx, Dzy, Dzz] -> [Dxx, Dyy, Dzz, Dzy, Dzx, Dxy]
        # which maps to indices [0, 2, 5, 4, 3, 1] respectively
        d_coeffs_reshaped = d_coeffs.view(nconformers, natoms, 4, 6)  # Shape: (nconformers, natoms, 4, 6)
        d_coeffs_reshaped = d_coeffs.view(nconformers, natoms, 4, 6)  # (B, N, 4, 6)

        # Unpack Cartesian components (xx, xy, yy, zx, zy, zz)
        Dxx = d_coeffs_reshaped[..., 0]  # (B,N,4)
        Dxy = d_coeffs_reshaped[..., 1]
        Dyy = d_coeffs_reshaped[..., 2]
        Dzx = d_coeffs_reshaped[..., 3]  # xz
        Dzy = d_coeffs_reshaped[..., 4]  # yz
        Dzz = d_coeffs_reshaped[..., 5]

        # Build D blocks: (B, N, 4, 3, 3)
        D_blocks = torch.zeros(
            (nconformers, natoms, 4, 3, 3), dtype=coefficients.dtype, device=coefficients.device
            )
        # Row x: [xx, xy, xz]
        D_blocks[:, :, :, 0, 0] = Dxx
        D_blocks[:, :, :, 0, 1] = Dxy
        D_blocks[:, :, :, 0, 2] = Dzx
        # Row y: [yx, yy, yz]
        D_blocks[:, :, :, 1, 0] = Dxy
        D_blocks[:, :, :, 1, 1] = Dyy
        D_blocks[:, :, :, 1, 2] = Dzy
        # Row z: [zx, zy, zz]
        D_blocks[:, :, :, 2, 0] = Dzx
        D_blocks[:, :, :, 2, 1] = Dzy
        D_blocks[:, :, :, 2, 2] = Dzz

        # --- Make traceless: Q = D - (Tr D / 3) I ---
        tr_D = D_blocks[..., 0, 0] + D_blocks[..., 1, 1] + D_blocks[..., 2, 2]     # (B,N,4)
        I3 = torch.eye(3, dtype=D_blocks.dtype, device=D_blocks.device).view(1,1,1,3,3)
        D_blocks = D_blocks - (tr_D[..., None, None] / 3.0) * I3                   # (B,N,4,3,3)

        # Flatten for downstream use
        # q_matrix = D_blocks.view(nconformers, natoms, 12, 3)                        # (B,N,12,3)

        return s_coeffs, p_coeffs_reshaped, D_blocks


    def _normalize(
        self,
        coeffs: Tensor,
        species_idx: Tensor,
        mus: Tensor,
        sigmas: Tensor
    ) -> Tensor:
        
        # Advanced indexing to fetch the right μ and σ for *each* atom
        atom_mus = mus[species_idx, :]
        atom_sigmas = sigmas[species_idx, :]
        print("DEBUGGING")
        print(atom_mus.shape,atom_sigmas.shape,coeffs.shape)
        coeffs_normalized = (coeffs - atom_mus) / atom_sigmas
        return coeffs_normalized

    def _get_angles_from_p_matrix(
        self,
        p_matrix: Tensor,
        distances: Tensor,
        use_simple_orbital_aev: bool,
    ) -> Tensor:
        nconformers, natoms, naovs = distances.shape
        # Normalize the vectors using the provided distances

        # Create a mask for the zero vectors in the orbital matrix
        zero_mask = (p_matrix.abs() < 1e-12).all(dim=-1)

        # Perform the normalization, avoid division by zero by using where
        p_matrix_normalized = torch.where(
        zero_mask.unsqueeze(-1),
        torch.zeros_like(p_matrix),
        p_matrix / distances.unsqueeze(-1)        
        )
           
        # Calculate angles between each vector and the following vectors
        nangles = int((naovs-1)*naovs/2)
        angles = torch.zeros((nconformers, natoms,nangles), device=p_matrix.device, dtype=p_matrix.dtype)
        k = 0
        if use_simple_orbital_aev:
            for i in range(naovs):
                for j in range(i+1, naovs):
                    cos_angles = torch.einsum('ijk,ijk->ij', p_matrix_normalized[:, :, i, :], p_matrix_normalized[:, :, j, :])
                    cos_angles = torch.clamp(cos_angles, -0.9999, 0.9999)
                    angles[:, :, k] = torch.acos(cos_angles)
                    k = k + 1
            return angles
        else:
            avdistperangle = torch.zeros((nconformers, natoms, nangles))        
            for i in range(naovs):
                for j in range(i+1, naovs):
                    cos_angles = torch.einsum('ijk,ijk->ij', p_matrix_normalized[:, :, i, :], p_matrix_normalized[:, :, j, :])
                    cos_angles = torch.clamp(cos_angles, -0.9999, 0.9999)
                    angles[:, :, k] = torch.acos(cos_angles)
                    avdistperangle[:, :, k] = (distances[:, :, i]+distances[:, :, j])/2.0
                    k = k + 1
            return angles,avdistperangle

    def _get_angles_p_d_blocks(self,
                               p_matrix: Tensor,
                               D_blocks: Tensor,
                               p_norms: Tensor,
                               D_norms: Tensor):
        nconformers, natoms, naovs = p_norms.shape
        _, _, ndblocks, _, _ = D_blocks.shape
        nangles = naovs*ndblocks
        angles = torch.zeros((nconformers, natoms,nangles), device=p_matrix.device, dtype=p_matrix.dtype)
        k = 0
        for i in range(naovs):
            for j in range(ndblocks):
                angles[:, :, k] = self._Q_vs_p_angle(D_blocks[:,:,j,:,:],p_matrix[:,:,i,:])
                k = k + 1
        return angles
    
    def _get_angles_d_blocks(self,
                             D_blocks: Tensor,
                             D_norms: Tensor):
        nconformers, natoms, ndblocks, _, _ = D_blocks.shape
        nangles = int((ndblocks-1)*ndblocks/2)
        angles = torch.zeros((nconformers, natoms,nangles), device=D_blocks.device, dtype=D_blocks.dtype)
        k = 0
        for i in range(ndblocks):
            for j in range(i+1, ndblocks):
                angles[:, :, k] = self._tensor_angle(D_blocks[:,:,i,:,:],D_blocks[:,:,j,:,:])
                k = k + 1
        return angles

    def _map_species_to_idx(
            self,
            species: torch.Tensor, 
            Z_to_idx: dict,
    )-> Tensor:
        """
        species: shape (nconformers, natoms)
        Returns a new tensor of same shape with the row index in [0..N-1].
        """
        # Initialize an integer tensor
        species_idx = torch.zeros_like(species, dtype=torch.long)

        # For each known Z in Z_to_idx, fill in the appropriate row index
        for z, row in Z_to_idx.items():
            mask = (species == z)
            species_idx[mask] = row
        return species_idx

    def _frob_norm(self, M: Tensor, eps: float = 1e-12) -> Tensor:
        # Frobenius norm per (batch, atom, block) over the last two dims
        # e.g. Q.shape == (B, N, 3, 3) -> returns (B, N)
        return torch.linalg.norm(M, dim=(-2, -1)).clamp_min(eps)

    def _tensor_angle(self, A: Tensor, B: Tensor, eps: float = 1e-12) -> Tensor:
        # A, B: (..., 3, 3) -> returns (...,) angle in radians
        num = torch.einsum('...ij,...ij->...', A, B)
        den = self._frob_norm(A, eps) * self._frob_norm(B, eps)
        c = torch.clamp(num / den, -0.999999, 0.999999)
        return torch.acos(c)

    def _Q_vs_p_angle(self, Q: Tensor, p: Tensor, eps: float = 1e-12) -> Tensor:
        # Q: (..., 3, 3), p: (..., 3) -> returns (...,) angle in radians
        p_hat = p / p.norm(dim=-1, keepdim=True).clamp_min(eps)
        num = torch.einsum('...i,...ij,...j->...', p_hat, Q, p_hat)
        den = self._frob_norm(Q, eps)
        c = torch.clamp(num / den, -0.999999, 0.999999)
        return torch.acos(c)
