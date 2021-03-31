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

def tile_into_tight_cell(species_coordinates, repeats=(3, 3, 3), noise=None, delta=1.0, density=None):
    # density is in molecules per angstrom^3
    # convenience function that takes a molecule and tiles it
    # into a periodic square or rectangular crystal cell
    # for water density = 0.0923 approx at 300 K
    species, coordinates = species_coordinates
    device = coordinates.device

    coordinates = coordinates.squeeze()
    if isinstance(repeats, int):
        repeats = torch.tensor([repeats, repeats, repeats], dtype=torch.long, device=device)
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
    coordinates = coordinates - displace_r 
    coordinates = coordinates + eps
    assert (coordinates > 0).all()

    coordinates = coordinates.unsqueeze(0)
    max_dist_to_origin = coordinates.norm(2, -1).max()

    x_disp = torch.arange(0, repeats[0], device=device) 
    y_disp = torch.arange(0, repeats[1], device=device) 
    z_disp = torch.arange(0, repeats[2], device=device) 

    displacements = torch.cartesian_prod(x_disp, y_disp, z_disp).double()
    num_displacements = len(displacements)
    if density is not None:
        delta = (1/repeats)*((num_displacements/density)**(1/3) - max_dist_to_origin)
    box_length = max_dist_to_origin + delta
    displacements *= box_length
    species = species.repeat(1, num_displacements)
    coordinates = torch.cat([coordinates + d for d in displacements], dim=1)
    if noise is not None:
        coordinates += torch.empty(coordinates.shape, device=device).uniform_(-noise, noise)
    cell_length = box_length * repeats
    cell = torch.diag(torch.tensor(cell_length.cpu().numpy().tolist(), device=device, dtype=coordinates.dtype))
    return species, coordinates, cell
