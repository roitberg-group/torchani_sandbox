from typing import Optional
import torch
from torch import Tensor
from .geometry import displace_to_com_frame


def compute_dipole(
    coordinates: Tensor,
    charges: Tensor,
    atomic_numbers: Optional[Tensor] = None,
    masses: Optional[Tensor] = None,
    center_of_mass: bool = True
) -> Tensor:
    """
    Compute dipoles in eA

    Arguments:
        species (torch.Tensor): (M, N), species must be atomic numbers.
        coordinates (torch.Tensor): (M, N, 3), unit should be Angstrom.
        charges (torch.Tensor): (M, N), unit should be e.
        center_of_mass (bool): When calculating dipole for charged molecule,
            it is necessary to displace the coordinates to the center-of-mass frame.
    Returns:
        dipoles (torch.Tensor): (M, 3)
    """
    assert charges.shape == coordinates.shape[:-1]
    if atomic_numbers is not None:
        assert charges.shape == atomic_numbers.shape
    if masses is not None:
        assert charges.shape == masses.shape
    charges = charges.unsqueeze(-1)
    if center_of_mass:
        coordinates = displace_to_com_frame(
            coordinates,
            atomic_numbers=atomic_numbers,
            masses=masses,
        )
    dipoles = torch.sum(charges * coordinates, dim=1)
    return dipoles
