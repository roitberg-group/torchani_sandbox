import typing as tp

from torch import Tensor
from torchani.grad import forces, energies_and_forces
from torchani.models import BuiltinModel
from torchani.potentials import PotentialWrapper

Model = tp.Union[BuiltinModel, PotentialWrapper]


def energies_forces_and_hessian(
    model: Model,
    species: Tensor,
    coordinates: Tensor,
    retain_graph: tp.Optional[bool] = None,
    create_graph: bool = False,
    keep_requires_grad: bool = False,
) -> tp.Dict[str, Tensor]:
    output = energies_and_forces(
        model,
        species,
        coordinates,
        retain_graph=True,
        create_graph=True,
        keep_requires_grad=True,
    )
    result = hessian(
        output["forces"],
        coordinates,
        retain_graph=retain_graph,
        create_graph=create_graph,
        keep_requires_grad=keep_requires_grad,
    )
    output.update(result)
    return output


def forces_and_hessian(
    energies: Tensor,
    coordinates: Tensor,
    retain_graph: tp.Optional[bool] = None,
    create_graph: bool = False,
    keep_requires_grad: bool = False,
) -> tp.Dict[str, Tensor]:
    output = forces(
        energies,
        coordinates,
        retain_graph=True,
        create_graph=True,
        keep_requires_grad=True,
    )
    result = hessian(
        output["forces"],
        coordinates,
        retain_graph=retain_graph,
        create_graph=create_graph,
        keep_requires_grad=keep_requires_grad,
    )
    output.update(result)
    return output


def hessian(
    energies: Tensor,
    coordinates: Tensor,
    retain_graph: tp.Optional[bool] = None,
    create_graph: bool = False,
    keep_requires_grad: bool = False,
) -> tp.Dict[str, Tensor]:
    if not coordinates.requires_grad:
        raise ValueError("Coordinates input to this function must require grad")
    forces = -torch.autograd.grad(
        energies.sum(),
        coordinates,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )[0]
    if not keep_requires_grad:
        coordinates.requires_grad_(False)
    return {"hessian": hessian}


def hessian(
    forces: Tensor,
    coordinates: Tensor,
) -> Tensor:
    """Compute analytical hessian from the energy graph or force graph.

    Arguments:
        coordinates (:class:`torch.Tensor`): Tensor of shape `(molecules, atoms, 3)`
        energies (:class:`torch.Tensor`): Tensor of shape `(molecules,)`, if specified,
            then `forces` must be `None`. This energies must be computed from
            `coordinates` in a graph.
        forces (:class:`torch.Tensor`): Tensor of shape `(molecules, atoms,
            3)`, if specified,
            then `energies` must be `None`. This forces must be computed from
            `coordinates` in a graph.

    Returns:
        :class:`torch.Tensor`: Tensor of shape `(molecules, 3A, 3A)` where A is
        the number of atoms in each molecule
    """
    flattened_force = forces.flatten(start_dim=1)
    force_components = flattened_force.unbind(dim=1)
    return -torch.stack(
        [
            _get_derivatives_not_none(coordinates, f, retain_graph=True).flatten(
                start_dim=1
            )
            for f in force_components
        ],
        dim=1,
    )


def vibrational_analysis(masses, hessian, mode_type="MDU", unit="cm^-1"):
    """Computing the vibrational wavenumbers from hessian.

    Note that normal modes in many popular software packages such as
    Gaussian and ORCA are output as mass deweighted normalized (MDN).
    Normal modes in ASE are output as mass deweighted unnormalized (MDU).
    Some packages such as Psi4 let ychoose different normalizations.
    Force constants and reduced masses are calculated as in Gaussian.

    mode_type should be one of:
    - MWN (mass weighted normalized)
    - MDU (mass deweighted unnormalized)
    - MDN (mass deweighted normalized)

    MDU modes are not orthogonal, and not normalized,
    MDN modes are not orthogonal, and normalized.
    MWN modes are orthonormal, but they correspond
    to mass weighted cartesian coordinates (x' = sqrt(m)x).

    Imaginary frequencies are output as negative numbers.
    Very small negative or positive frequencies may correspond to
    translational, and rotational modes.
    """
    if unit == "meV":
        unit_converter = sqrt_mhessian2milliev
    elif unit == "cm^-1":
        unit_converter = sqrt_mhessian2invcm
    else:
        raise ValueError("Only meV and cm^-1 are supported right now")

    assert (
        hessian.shape[0] == 1
    ), "Currently only supporting computing one molecule a time"
    # Solving the eigenvalue problem: Hq = w^2 * T q
    # where H is the Hessian matrix, q is the normal coordinates,
    # T = diag(m1, m1, m1, m2, m2, m2, ....) is the mass
    # We solve this eigenvalue problem through Lowdin diagnolization:
    # Hq = w^2 * Tq ==> Hq = w^2 * T^(1/2) T^(1/2) q
    # Letting q' = T^(1/2) q, we then have
    # T^(-1/2) H T^(-1/2) q' = w^2 * q'
    inv_sqrt_mass = (1 / masses.sqrt()).repeat_interleave(
        3, dim=1
    )  # shape (molecule, 3 * atoms)
    mass_scaled_hessian = (
        hessian * inv_sqrt_mass.unsqueeze(1) * inv_sqrt_mass.unsqueeze(2)
    )
    if mass_scaled_hessian.shape[0] != 1:
        raise ValueError("The input should contain only one molecule")
    mass_scaled_hessian = mass_scaled_hessian.squeeze(0)
    eigenvalues, eigenvectors = torch.linalg.eigh(mass_scaled_hessian)
    signs = torch.sign(eigenvalues)
    angular_frequencies = eigenvalues.abs().sqrt()
    frequencies = angular_frequencies / (2 * math.pi)
    frequencies = frequencies * signs
    # converting from sqrt(hartree / (amu * angstrom^2)) to cm^-1 or meV
    wavenumbers = unit_converter(frequencies)

    # Note that the normal modes are the COLUMNS of the eigenvectors matrix
    mw_normalized = eigenvectors.t()
    md_unnormalized = mw_normalized * inv_sqrt_mass
    norm_factors = 1 / torch.linalg.norm(md_unnormalized, dim=1)  # units are sqrt(AMU)
    md_normalized = md_unnormalized * norm_factors.unsqueeze(1)

    rmasses = norm_factors**2  # units are AMU
    # The conversion factor for Ha/(AMU*A^2) to mDyne/(A*AMU) is about 4.3597482
    fconstants = mhessian2fconst(eigenvalues) * rmasses  # units are mDyne/A

    if mode_type == "MDN":
        modes = (md_normalized).reshape(frequencies.numel(), -1, 3)
    elif mode_type == "MDU":
        modes = (md_unnormalized).reshape(frequencies.numel(), -1, 3)
    elif mode_type == "MWN":
        modes = (mw_normalized).reshape(frequencies.numel(), -1, 3)

    return VibAnalysis(wavenumbers, modes, fconstants, rmasses)
