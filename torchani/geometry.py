r""" Utilities to manipulate (weighted) clouds of points in 3D and generate
some specific geometries"""
from typing import Optional, Tuple

import torch
from torch import Tensor

from .structs import EvaluesEvectors
from .utils import (
    _maybe_infer_masses,
    _maybe_infer_dummy_atoms,
    _NUMBERS_PADDING,
    gram_schmidt,
)


def displace_to_com_frame(
    coordinates: Tensor,  # (M, A, 3)
    atomic_numbers: Optional[Tensor] = None,  # (M, A)
    masses: Optional[Tensor] = None,  # (M, A)
    is_dummy_atom: Optional[Tensor] = None,
    atomic_numbers_padding: int = _NUMBERS_PADDING,
) -> Tensor:
    r"""Displace coordinates to the center-of-mass frame, padding atoms can be
    included.

    Returns the displaced coordinates

    If only masses are passed then dummy atoms have to be specifically flagged,
    otherwise dummy coordinates are also displaced.
    """
    is_dummy_atom = _maybe_infer_dummy_atoms(
        atomic_numbers,
        is_dummy_atom,
        atomic_numbers_padding
    )
    masses = _maybe_infer_masses(
        atomic_numbers,
        masses,
        atomic_numbers_padding=atomic_numbers_padding,
        dtype=coordinates.dtype
    )
    mass_sum = masses.unsqueeze(-1).sum(dim=1, keepdim=True)
    com_coordinates = coordinates * masses.unsqueeze(-1) / mass_sum
    com_coordinates = com_coordinates.sum(dim=1, keepdim=True)
    centered_coordinates = coordinates - com_coordinates
    if is_dummy_atom is not None:
        # copy the dummy coordinates
        centered_coordinates[is_dummy_atom, :] = coordinates[is_dummy_atom, :]
    return centered_coordinates


def inertia_tensor(
    coordinates: Tensor,  # (M, A, 3)
    atomic_numbers: Optional[Tensor] = None,  # (M, A)
    masses: Optional[Tensor] = None,  # (M, A)
    atomic_numbers_padding: int = _NUMBERS_PADDING,
    from_center_of_mass: bool = True,
) -> Tensor:  # (M, 3, 3)
    # supports batching, masses of dummy atoms must be zero
    masses = _maybe_infer_masses(
        atomic_numbers,
        masses,
        atomic_numbers_padding=atomic_numbers_padding,
        dtype=coordinates.dtype
    )
    if from_center_of_mass:
        # dummy atoms will get weighted as zero due to mass so it is
        # not necessary to flag them here
        coordinates = displace_to_com_frame(
            coordinates,
            masses=masses,
        )  # coordinates padding is not important here
    scaled_coordinates = torch.sqrt(masses.unsqueeze(-1)) * coordinates
    cov = torch.matmul(scaled_coordinates.transpose(-1, -2), scaled_coordinates)
    batched_trace = torch.diagonal(cov, dim1=1, dim2=2).sum(-1)
    inertia_tensor = batched_trace.view(-1, 1, 1) * torch.eye(3).view(1, 3, 3) - cov
    return inertia_tensor


def principal_inertia_axes(
    coordinates: Tensor,  # (M, A, 3)
    atomic_numbers: Optional[Tensor] = None,  # (M, A)
    masses: Optional[Tensor] = None,  # (M, A)
    from_center_of_mass: bool = True,
) -> EvaluesEvectors:  # (M, 3), (M, 3, 3)
    # supports batching, masses of dummy atoms must be zero
    #
    # torch eigh returns Q where A = QDQ', so AQ = QD
    # so the eigenvectors are in the columns, this seems to be the same
    # convention that gaussian uses for diagonalizing the inertia matrix,
    # so the output eigenvectors matrix is gaussian's X matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(
        inertia_tensor(
            coordinates,
            masses=masses,
            atomic_numbers=atomic_numbers,
            from_center_of_mass=from_center_of_mass,
        )
    )
    return EvaluesEvectors(eigenvalues, eigenvectors)


def internals_matrix_transform(
    coordinates: Tensor,
    atomic_numbers: Optional[Tensor] = None,
    masses: Optional[Tensor] = None,
    atomic_numbers_padding: int = _NUMBERS_PADDING,
    threshold: float = 1e-8,
    geometry_check: bool = True,
    orthonormality_check: bool = False,
    seed: Optional[int] = None,
) -> Tensor:
    # returns the D matrix, that converts cartesian coordinates
    # into internal coordinates which have the rotation and translation
    # parts factored out in a submatrix
    #
    # I'm honestly not sure if it is possible to do this with dummy atoms, so for now
    # I will not allow them, the issue is that I don't know what would happen
    # with the rotation, translation, etc, vectors.
    # also matrix multiplication, inversion, etc, is significantly slowed down
    # with dummy atoms.
    #
    # The sanity check may be slow for large molecules, and I believe it is
    # largely unneeded, but gaussian performs it
    #
    # The orthonormality check for the matrix is very expensive for large
    # molecules (it is cuadratic on the number of coordinates)
    assert coordinates.shape[0] == 1
    if atomic_numbers is not None:
        assert atomic_numbers.shape[0] == 1

    num_molecules = coordinates.shape[0]
    num_atoms = coordinates.shape[1]
    masses = _maybe_infer_masses(
        atomic_numbers,
        masses,
        atomic_numbers_padding=atomic_numbers_padding,
        dtype=coordinates.dtype
    )
    assert masses.shape[0] == 1
    # generate D1 ... D3 translation vectors
    translations = torch.eye(
        3,
        dtype=coordinates.dtype
    ).unsqueeze(0).repeat(num_molecules, num_atoms, 1)
    # translations is shape (M, 3A, 3)

    # generate D4, ... D6 rotation vectors
    # no need to flag dummy atoms here, since they will get weighted with 0.0
    # anyways? not sure if dummy atoms work correctly with this function so
    # I'm disallowing them anyways for now
    centered_coordinates = displace_to_com_frame(
        coordinates,
        masses=masses,
    )
    # this is the gaussian X matrix, 3x3
    eigenvectors = principal_inertia_axes(
        centered_coordinates,
        masses=masses,
        from_center_of_mass=False,
    ).evectors
    # these matrices are (M, 3, 3), and the centered coordinates are (M, A, 3)
    # so in order to perform a batched matrix product I repeat along the atoms
    # dimension
    repeated_evectors = eigenvectors.unsqueeze(1).repeat(1, num_atoms, 1, 1)
    p_vectors = torch.matmul(
        repeated_evectors,
        centered_coordinates.unsqueeze(-1)
    ).squeeze(-1)
    # shape of p_vectors is (M, A, 3), same as centered coordinates,

    # there are three rotation vectors, the output
    _rot_vectors = [
        torch.cross(
            p_vectors,
            row.view(1, 1, -1),
            dim=-1,
        )
        for row in eigenvectors.unbind(1)
    ]
    rotations = torch.zeros_like(translations)
    rotations[:, 0:-2:3] = _rot_vectors[0]
    rotations[:, 1:-1:3] = _rot_vectors[1]
    rotations[:, 2::3] = _rot_vectors[2]
    # the way the D4, D5, D6 vectors work, the _rot_vectos have to be stacked in an
    # interleaved fashion, The first row vector goes in the first coord, the second
    # in the second coord, the third in the third coord, and they go in steps of 3
    # gram schmidt goes here
    eckart_vectors = torch.cat([translations, rotations], dim=2)

    sqrt_mass = masses.sqrt().repeat_interleave(3, dim=1).unsqueeze(-1)
    # sqrt_mass is shape (M, 3A, 1)
    eckart_vectors *= sqrt_mass

    # filter out one or three here if the molecule is linear or point-like
    norms = torch.linalg.norm(eckart_vectors, dim=1)
    is_discarded = (norms < threshold)  # shape (M, 6)

    if geometry_check:
        discarded_num = is_discarded.sum(dim=1)  # shape (M,)
        is_monoatomic = structure_is_monoatomic(coordinates, atomic_numbers == -1)
        is_linear = structure_is_linear(coordinates, atomic_numbers == -1)
        is_polyatomic_nonlinear = (~is_linear) & (~is_monoatomic)
        # for linear molecules 1 is discarded, for atoms 3 are discarded
        # for polyatomic nonlinear none are discarded
        assert (discarded_num[is_monoatomic] == 3).all()
        assert (discarded_num[is_linear] == 1).all()
        assert (discarded_num[is_polyatomic_nonlinear] == 0).all()
        # for batching it may be a good idea to group together linear,
        # nonlinear and atoms, and then perform each one in turn.

    # here I assume non batching
    is_discarded = is_discarded.squeeze(0)
    eckart_vectors = torch.stack(
        [eckart_vectors[:, :, j] for j, d in enumerate(is_discarded) if not d],
        dim=-1
    )
    d_matrix = gram_schmidt(
        eckart_vectors,
        assume_initial_orthogonal=True,
        orthonormality_check=orthonormality_check,
        seed=seed,
    )
    return d_matrix


def batched_vector_angle(
    vectors1: Tensor,
    vectors2: Tensor,
    return_cos: bool = False,
    vector1_is_dummy: Optional[Tensor] = None,
    vector2_is_dummy: Optional[Tensor] = None,
    dummy_cos: float = 1.0,
) -> Tensor:
    # vectors must be (M, A, F) (M, A, F) and nonbatch dimensions should be
    # broadcastable
    out = (vectors1 * vectors2).sum(-1)
    out = out / torch.linalg.norm(vectors1, dim=-1)
    out = out / torch.linalg.norm(vectors2, dim=-1)
    out = torch.clip(out, -1, 1)

    if vector1_is_dummy is not None:
        out[vector1_is_dummy] = dummy_cos
    if vector2_is_dummy is not None:
        out[vector2_is_dummy] = dummy_cos

    if return_cos:
        return out
    return torch.arccos(out)


def inertia_is_monoatomic(
    inertia_eigenvalues: Tensor,
    threshold: float = 1e-6,
) -> Tensor:
    return (inertia_eigenvalues.abs() < threshold).all(dim=-1)


def inertia_is_linear(
    inertia_eigenvalues: Tensor,
    threshold: float = 1e-6,
) -> Tensor:
    return (inertia_eigenvalues.abs() < threshold).sum(dim=-1) == 1


def inertia_is_polyatomic_nonlinear(
    inertia_eigenvalues: Tensor,
    threshold: float = 1e-6,
) -> Tensor:
    return ~(inertia_eigenvalues.abs() < threshold).any(dim=-1)


def structure_is_monoatomic(
    coordinates: Tensor,
    atom_is_dummy: Optional[Tensor] = None,
) -> Tensor:
    # supports batching
    if atom_is_dummy is None:
        return torch.tensor([coordinates.shape[1] == 1],
                device=coordinates.device).repeat(coordinates.shape[0], 1)
    physical_atoms = (~atom_is_dummy).sum(dim=-1)
    return physical_atoms == 1


# WARNING: function fails in the degenarate case of superimposed atoms
def structure_is_linear(
    coordinates: Tensor,
    atom_is_dummy: Optional[Tensor] = None,
    threshold: float = 1e-6
) -> Tensor:
    # supports batching
    if atom_is_dummy is None:
        # this is not optimized for this case
        atom_is_dummy = torch.zeros(
            coordinates.shape[:-1],
            dtype=torch.bool,
            device=coordinates.device
        )
    physical_atoms = (~atom_is_dummy).sum(dim=-1)
    monoatomic = (physical_atoms == 1)
    diatomic = (physical_atoms == 2)
    potentially_nondiatomic_linear = (~diatomic) & (~monoatomic)
    if potentially_nondiatomic_linear.any():
        coordinates = coordinates[potentially_nondiatomic_linear]
        atom_is_dummy = atom_is_dummy[potentially_nondiatomic_linear]
        nondiatomic_linear = _nondiatomic_linear(
            coordinates,
            atom_is_dummy,
            threshold,
        )
        linear = diatomic.clone()
        linear[potentially_nondiatomic_linear] |= nondiatomic_linear
        return linear
    return diatomic


def _nondiatomic_linear(
    coordinates: Tensor,
    atom_is_dummy: Tensor,
    threshold: float = 1e-6
) -> Tensor:
    # supports batching
    # for each structure I pick a physical atom
    argsort = torch.argsort(atom_is_dummy, dim=1)
    # all of these have 3 or more physical atoms so this can't fail
    any_physical_idx1 = argsort[:, 0].view(-1, 1, 1).repeat(1, 1, 3)
    any_physical_idx2 = argsort[:, 1].view(-1, 1, 1).repeat(1, 1, 3)
    any_physical_coord1 = torch.gather(coordinates, 1, any_physical_idx1)
    # displace all coordinates to measure from a physical coord
    displaced_coords = coordinates - any_physical_coord1
    # pick one nonzero displaced coordinate for each structure
    any_physical_coord2 = torch.gather(displaced_coords, 1, any_physical_idx2)
    # set the physical coordinates to be dummy, so that cosine with them is
    # automatically set to 1.0
    atom_is_dummy[:, any_physical_idx1] = True

    # if the structure is linear the cos(angle) between any atom and the first atom
    # should be 1 or -1.
    cos_angles = batched_vector_angle(
        any_physical_coord2,
        displaced_coords,
        return_cos=True,
        vector2_is_dummy=atom_is_dummy,
        dummy_cos=1.0,
    )
    return ((cos_angles.abs() - 1).abs() <= threshold).all(dim=-1)


# WARNING: function fails in the degenarate case of superimposed atoms
def structure_is_polyatomic_nonlinear(
    coordinates: Tensor,
    atom_is_dummy: Optional[Tensor] = None,
) -> Tensor:
    # supports batching
    not_monoatomic = ~structure_is_monoatomic(coordinates, atom_is_dummy)
    not_linear = ~structure_is_linear(coordinates, atom_is_dummy)
    return not_monoatomic & not_linear


def tile_into_tight_cell(
    species: Tensor,
    coordinates: Tensor,
    repeats: Tuple[int, int, int] = (3, 3, 3),
    noise: Optional[float] = None,
    delta: float = 1.0,
    density: Optional[float] = None,
    fixed_displacement_size: Optional[float] = None,
    make_coordinates_positive: bool = True
):
    r""" Tile
    Arguments:
        repeats: Integer or tuple of integers (larger than zero), how many
            repeats in each direction, to expand the given species_coordinates.
            tiling can be into a square or rectangular cell.
        noise: uniform noise in the range -noise, +noise is added to the
            coordinates to prevent exact repetition if given.
        If density is given (units of molecule / A^3), the box length is scaled
        to produce the desired molecular density. For water, density = 0.0923
        at 300 K approximately.
    """
    device = coordinates.device

    coordinates = coordinates.squeeze()
    if isinstance(repeats, int):
        repeats = torch.tensor([repeats, repeats, repeats],
                               dtype=torch.long,
                               device=device)
    else:
        assert len(repeats) == 3
        repeats = torch.tensor(repeats, dtype=torch.long, device=device)
        assert (repeats >= 1).all(), 'At least one molecule should be present'
    assert coordinates.dim() == 2

    # displace coordinates so that they are all positive
    eps = torch.tensor(1e-10, device=device, dtype=coordinates.dtype)
    neg_coords = torch.where(coordinates < 0, coordinates, -eps)
    min_x = neg_coords[:, 0].min()
    min_y = neg_coords[:, 1].min()
    min_z = neg_coords[:, 2].min()
    displace_r = torch.tensor([min_x, min_y, min_z], device=device, dtype=coordinates.dtype)
    assert (displace_r <= 0).all()
    coordinates_positive = coordinates - displace_r
    coordinates_positive = coordinates + eps
    assert (coordinates_positive > 0).all()

    # get the maximum position vector in the set of coordinates ("diameter" of molecule)
    coordinates_positive = coordinates_positive.unsqueeze(0)
    max_dist_to_origin = coordinates_positive.norm(2, -1).max()
    if make_coordinates_positive:
        coordinates = coordinates_positive
    else:
        coordinates = coordinates.unsqueeze(0)

    x_disp = torch.arange(0, repeats[0], device=device)
    y_disp = torch.arange(0, repeats[1], device=device)
    z_disp = torch.arange(0, repeats[2], device=device)

    displacements = torch.cartesian_prod(x_disp, y_disp, z_disp).to(coordinates.dtype)
    num_displacements = len(displacements)
    # Calculate what the displacement size should be to match a specific density
    # If delta is given instead, then the displacement size will be the molecule
    # diameter plus that fixed delta
    # If fixed_displacement_size is given instead, then that will be the displacement
    # size, without taking into account any other factors
    msg = "delta, density and fixed_displacement_size are mutually exclusive"
    if density is not None:
        assert delta == 1.0, msg
        assert fixed_displacement_size is None, msg
        delta = (1 / repeats) * ((num_displacements / density)**(1 / 3) - max_dist_to_origin)
        box_length = max_dist_to_origin + delta
    elif fixed_displacement_size is not None:
        assert delta == 1.0, msg
        assert density is None, msg
        box_length = fixed_displacement_size
    else:
        assert density is None, msg
        assert fixed_displacement_size is None, msg
        box_length = max_dist_to_origin + delta

    displacements *= box_length
    species = species.repeat(1, num_displacements)
    coordinates = torch.cat([coordinates + d for d in displacements], dim=1)
    if noise is not None:
        coordinates += torch.empty(coordinates.shape, device=device).uniform_(-noise, noise)
    cell_length = box_length * repeats
    cell = torch.diag(torch.tensor(cell_length.cpu().numpy().tolist(), device=device, dtype=coordinates.dtype))
    return species, coordinates, cell
