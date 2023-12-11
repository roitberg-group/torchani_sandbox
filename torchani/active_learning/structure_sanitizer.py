import numpy as np
from scipy.spatial import distance_matrix

def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[2:]  # Skip the first two lines

    atoms = []
    coordinates = []

    for line in lines:
        parts = line.split()
        atoms.append(parts[0])
        coordinates.append([float(x) for x in parts[1:4]])

    return atoms, np.array(coordinates)

def compute_connectivity(coordinates, threshold=1.8):
    dist_matrix = distance_matrix(coordinates, coordinates)
    connectivity = dist_matrix < threshold
    np.fill_diagonal(connectivity, False)  # Exclude self-connectivity
    return connectivity

def find_largest_connected_component(connectivity):
    num_atoms = len(connectivity)
    visited = set()
    largest_component = set()

    def dfs(current):
        if current in visited:
            return set()
        visited.add(current)
        component = {current}
        for neighbor in range(num_atoms):
            if connectivity[current, neighbor]:
                component.update(dfs(neighbor))
        return component

    for atom in range(num_atoms):
        component = dfs(atom)
        if len(component) > len(largest_component):
            largest_component = component

    return largest_component

def write_xyz(atoms, coordinates, file_path):
    with open(file_path, 'w') as file:
        file.write(f"{len(atoms)}\n\n")
        for atom, coord in zip(atoms, coordinates):
            file.write(f"{atom} {' '.join(map(str, coord))}\n")

# Example usage
input_xyz = "DEC7_capped_C4H7N3O5_1_C3H4N3O3_atom3.xyz"  # Replace with your file path
output_xyz = "sanitized_C4H7N3O5_1_C3H4N3O3_atom3.xyz"

atoms, coordinates = read_xyz(input_xyz)
connectivity = compute_connectivity(coordinates)
largest_component = find_largest_connected_component(connectivity)

# Filter atoms and coordinates to keep only those in the largest connected component
filtered_atoms = [atom for i, atom in enumerate(atoms) if i in largest_component]
filtered_coords = coordinates[list(largest_component)]

write_xyz(filtered_atoms, filtered_coords, output_xyz)
