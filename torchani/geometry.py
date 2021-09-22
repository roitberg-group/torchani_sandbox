""" Utilities to generate some specific geometries"""
import torch


def _rotate(vector, axis, angle):
    # rotate a vector using an axis and an angle, a batch of vectors can be
    # passed (shape (N, 3)) and all will be rotated
    axis_x_vector = torch.cross(axis, vector, dim=-1)
    term1 = axis * (axis * angle).sum(-1)
    term2 = torch.cos(angle) * torch.cross(axis_x_vector, axis, dim=-1)
    term3 = torch.sin(angle) * axis_x_vector
    return term1 + term2 + term3


def align_geometry_to_bond(coordinates, atom1, atom2):
    # atom1 and atom2 are two atoms that define a bond
    assert coordinates.shape[0] == 1
    coordinates = coordinates.view(-1, 3)
    # first atom1 is translated to  the origin
    coordinates -= coordinates[atom1]
    # now the bond vector goes from the origin to atom2
    bond_vector = coordinates[atom2]
    # second, coordinate vectors should be rotated so that the Z axis lies in
    # the direction of atom1 to rotate we need an axis and an angle, we will
    # rotate using an axis perpendicular to the plane made by the bond and the
    # z axis
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=bond_vector.dtype, device=bond_vector.device)
    rot_axis = torch.cross(bond_vector, z_axis)
    rot_axis = rot_axis / rot_axis.norm()
    # the angle to rotate will be the angle between the bond vector and the z axis
    cosine = torch.nn.CosineSimilarity()
    angle = torch.acos(cosine(rot_axis, z_axis))
    rotated_coordinates = _rotate(coordinates, rot_axis, angle)
    rotated_coordinates.unsqueeze(0)
    return rotated_coordinates


def displace_dimer_along_bond(coordinates, atom1, atom2, distance, start_overlapped=True):
    # the dimer is assumed to be composed of an even number of atoms, the first
    # and second A/2 atoms correspond to both molecules in the dimer
    # respectively.
    assert coordinates.shape[0] == 1
    assert coordinates.shape[1] % 2 == 0
    molecule_size = coordinates.shape[1] / 2
    assert molecule_size.is_integer()
    molecule_size = int(molecule_size)
    coordinates = coordinates.view(-1, 3)
    coordinates_a = coordinates[:molecule_size]
    assert len(coordinates_a) == molecule_size
    coordinates_b = coordinates[molecule_size:]
    assert len(coordinates_b) == molecule_size
    diff_vector = coordinates_a[atom1] - coordinates_b[atom2]

    if start_overlapped:
        assert diff_vector.norm() < distance, f"The distance is less than the bond distance {diff_vector.norm()} > {distance}"

    coordinates_a += (diff_vector / diff_vector.norm()) * distance
    coordinates = torch.cat((coordinates_a, coordinates_b), dim=0)
    return coordinates.unsqueeze(0)


def tile_into_tight_cell(species_coordinates, repeats=(3, 3, 3), noise=None, delta=1.0,
                         density=None, fixed_displacement_size=None, make_coordinates_positive: bool = True):
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
    species, coordinates = species_coordinates
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
