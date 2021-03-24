import torch
""" Utilities to generate some specific geometries"""


def tile_into_cube(species_coordinates, box_length=3.5, repeats=3, noise=None):
    # convenience function that takes a molecule and tiles it
    # into a periodic square crystal cell
    species, coordinates = species_coordinates
    device = coordinates.device

    x_disp, y_disp, z_disp = (torch.eye(3, device=device)
                              * box_length).unbind(0)
    x_disp = torch.arange(0, repeats, device=device) * box_length
    displacements = torch.cartesian_prod(x_disp, x_disp, x_disp)
    num_displacements = len(displacements)
    species = species.repeat(1, num_displacements)
    coordinates = torch.cat([coordinates + d for d in displacements], dim=1)
    if noise is not None:
        coordinates += torch.empty(coordinates.shape,
                                   device=device).uniform_(-noise, noise)
    return species, coordinates
