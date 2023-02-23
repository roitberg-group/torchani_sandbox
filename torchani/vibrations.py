from typing import Optional, NamedTuple
import math

import torch
from torch import Tensor

from .geometry import internals_matrix_transform
from .units import sqrt_mhessian2invcm, sqrt_mhessian2milliev, mhessian2fconst
from .utils import _maybe_infer_masses, PADDING


class VibAnalysis(NamedTuple):
    freqs: Tensor
    modes: Tensor
    fconstants: Tensor
    rmasses: Tensor


def vibrational_analysis(
    hessian: Tensor,
    atomic_numbers: Optional[Tensor] = None,
    masses: Optional[Tensor] = None,
    coordinates: Optional[Tensor] = None,
    mode_type: str = 'MDU',
    freq_unit: str = 'cm^-1',
    project_to_internals: bool = False,
    seed: Optional[int] = None,
    atomic_numbers_padding: int = int(PADDING["species"]),
):
    """Computing the vibrational wavenumbers from hessian.

    Note that normal modes in many popular software packages such as
    Gaussian and ORCA are output as mass deweighted normalized (MDN).
    Normal modes in ASE are output as mass deweighted unnormalized (MDU).
    Some packages such as Psi4 let you choose different normalizations.

    Force constants and reduced masses are calculated as in Gaussian.

    mode_type should be one of:
    - MWN (mass weighted normalized)
    - MDU (mass deweighted unnormalized)
    - MDN (mass deweighted normalized)

    MDU modes are not orthogonal, not normalized.
    MDN modes are not orthogonal, normalized.
    MDN and MDU are linear combinations of cartesian coordinates.

    MWN modes both are orthogonal and normalized (othonormal
    MWN are linear combinations of mass weighted cartesian coordinates (x' = sqrt(m)x).

    Imaginary frequencies are output as negative numbers.

    Very small negative or positive frequencies may correspond to
    translational, and rotational modes.
    """
    if freq_unit == 'meV':
        freq_unit_converter = sqrt_mhessian2milliev
    elif freq_unit == 'cm^-1':
        freq_unit_converter = sqrt_mhessian2invcm
    else:
        raise ValueError('Only meV and cm^-1 are supported right now')

    masses = _maybe_infer_masses(
        atomic_numbers,
        masses,
        hessian.dtype,
        atomic_numbers_padding
    )
    assert masses.shape[0] == 1

    if project_to_internals:
        assert coordinates is not None, "Coordinates are needed to project to internals"
        assert coordinates.shape[0] == 1, 'Currently only supporting computing one molecule a time'
    else:
        assert seed is None, "Seed is not used if not projecting to internals"
        assert coordinates is None, "Coordinates is unused if not projecting to internals"

    assert hessian.shape[0] == 1, 'Currently only supporting computing one molecule a time'
    # Solving the eigenvalue problem: Hq = w^2 * T q
    # where H is the Hessian matrix, q is the normal coordinates,
    # T = diag(m1, m1, m1, m2, m2, m2, ....) is the mass
    # We solve this eigenvalue problem through Lowdin diagnolization:
    # Hq = w^2 * Tq ==> Hq = w^2 * T^(1/2) T^(1/2) q
    # Letting q' = T^(1/2) q, we then have
    # T^(-1/2) H T^(-1/2) q' = w^2 * q'
    num_coords = hessian.shape[1]
    sqrt_mass = masses.sqrt().repeat_interleave(3, dim=1)  # shape (molecule, 3 * atoms)
    inv_sqrt_mass = 1 / sqrt_mass
    mass_scaled_hessian = hessian * inv_sqrt_mass.unsqueeze(1) * inv_sqrt_mass.unsqueeze(2)
    if mass_scaled_hessian.shape[0] != 1:
        raise ValueError('The input should contain only one molecule')
    mass_scaled_hessian = mass_scaled_hessian.squeeze(0)
    eigenvalues, eigenvectors = torch.linalg.eigh(mass_scaled_hessian)

    if project_to_internals:
        transform = internals_matrix_transform(
            coordinates,
            masses=masses,
            seed=seed,
        )
        # disallow batching
        transform = transform.squeeze(0)
        transformed_hessian = (transform.T @ mass_scaled_hessian) @ transform
        print(transformed_hessian)

    # eigenvalues and eigenvectors are (M, 3N) and (M, 3N, 3N) respectively
    signs = torch.sign(eigenvalues)
    angular_frequencies = eigenvalues.abs().sqrt()
    frequencies = angular_frequencies / (2 * math.pi)
    frequencies = frequencies * signs
    # converting from sqrt(hartree / (amu * angstrom^2)) to cm^-1 or meV
    wavenumbers = freq_unit_converter(frequencies)

    # Note that the normal modes are the COLUMNS of the eigenvectors matrix
    mass_weighted_normalized = eigenvectors.t()
    mass_deweighted_unnormalized = mass_weighted_normalized * inv_sqrt_mass
    norm_factors = 1 / torch.linalg.norm(mass_deweighted_unnormalized, dim=1)  # units are sqrt(AMU)
    rmasses = norm_factors**2  # units are AMU
    # The conversion factor for Ha/(AMU*A^2) to mDyne/(A*AMU) is about 4.3597482
    fconstants = mhessian2fconst(eigenvalues) * rmasses  # units are mDyne/A

    if mode_type == "MDU":
        modes = (mass_deweighted_unnormalized).reshape(num_coords, -1, 3)
    elif mode_type == 'MWN':
        modes = (mass_weighted_normalized).reshape(num_coords, -1, 3)
    else:
        assert mode_type == "MDU"
        mass_deweighted_normalized = mass_deweighted_unnormalized * norm_factors.unsqueeze(1)
        modes = (mass_deweighted_normalized).reshape(num_coords, -1, 3)

    return VibAnalysis(wavenumbers, modes, fconstants, rmasses)
