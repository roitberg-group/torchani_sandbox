import torch
from torch import Tensor
from pathlib import Path
import numpy as np

# ------------------------------------------------------------------
# 0)  Load the μ and σ tables you saved earlier
#     (shape == (2, 7, 9):   axis‑0 = {mus, sigmas}, axis‑1 = species)
# ------------------------------------------------------------------
norm_file = Path("normalization_library.pt")
s_coeffs_mus, s_coeffs_sigmas = torch.load(norm_file)   # unpack

# sanity check
print("μ table shape:", s_coeffs_mus.shape)       # (7, 9)
print("σ table shape:", s_coeffs_sigmas.shape)    # (7, 9)

# ------------------------------------------------------------------
# 1)  Build a tiny synthetic batch
# ------------------------------------------------------------------
batch_size, natoms = 2, 6
supported_species = torch.tensor([1, 6, 7, 8, 9, 16, 17])          # H, C, N, O, F, S, Cl

# random species for each atom
species = supported_species[torch.randint(0, len(supported_species),
                                          (batch_size, natoms))]

species = torch.tensor([[8,1,1,8,1,1],[8,1,1,8,1,1]])

Z_to_idx = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}


def map_species_to_idx(species: torch.Tensor, Z_to_idx: dict) -> torch.Tensor:
    """
    species: shape (nconformers, natoms)
    Returns a new tensor of same shape with the row index in [0..N-1].
    """
    # Initialize an integer tensor
    species_idx = torch.empty_like(species, dtype=torch.long)

    # For each known Z in Z_to_idx, fill in the appropriate row index
    # E.g. if species[i,j] == 8, row=3
    for z, row in Z_to_idx.items():
        mask = (species == z)
        species_idx[mask] = row

    return species_idx

# 2) Convert species -> row indices
species_idx = map_species_to_idx(species, Z_to_idx)  # shape (nconf, natoms)

print(species_idx)
# random s‑coefficients (normally you’d read these from your dataset)
s_coeffs = torch.randn(batch_size, natoms, 9)

print("\nRandom species tensor:\n", species)
print("\nRaw s‑coeffs (conformer‑0, atom‑0):\n", s_coeffs[0, 0])

# ------------------------------------------------------------------
# 3)  Advanced indexing to fetch the right μ and σ for *each* atom
# ------------------------------------------------------------------
atom_mus    = s_coeffs_mus[species_idx,:]      # shape (batch, natoms, 9)
atom_sigmas = s_coeffs_sigmas[species_idx,:]   # shape (batch, natoms, 9)

print(atom_mus)
# ------------------------------------------------------------------
# 4)  Normalise
# ------------------------------------------------------------------
s_coeffs_norm = (s_coeffs - atom_mus) / atom_sigmas

print("\nNormalised s‑coeffs (conformer‑0, atom‑0):\n", s_coeffs_norm[0, 0])
