# This script shoudl take an input from the Isolator module and output gaussian input files and slurm scripts

import torch
import torchani
from torchani.active_learning import isolator
import hashlib
import os
import numpy as np

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

def hash_xyz_coordinates(filepath):
    """Generate MD5 hash for the coordinates in an XYZ file."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()[2:-1]  # Skip the first two and the last line -- fix depending on how xyz files are written
            for line in lines:
                hasher.update(line.encode('utf-8'))
        return hasher.hexdigest()
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def remove_duplicate_xyz_files(directory):
    seen_hashes = set()
    duplicate_count = 0

    for filename in os.listdir(directory):
        if filename.endswith(".xyz"):
            filepath = os.path.join(directory, filename)
            file_hash = hash_xyz_coordinates(filepath)
            if file_hash is None:
                continue

            if file_hash in seen_hashes:
                os.remove(filepath)
                duplicate_count += 1
                print(f"Deleted duplicate file: {filename}")
            else:
                seen_hashes.add(file_hash)

    print(f"Total duplicate files deleted: {duplicate_count}")

def write_gaussian_input(symbols, coordinates, file_name, theory='B3LYP', basis_set='6-31G(d)'):
    # TO DO:
        # Input should be species, convert to symbols
        # ???
    header = f"%chk={file_name}.chk\n# {theory}/{basis_set} SP\n\nTitle Card Required\n\n0 1\n"
    molecule_data = "\n".join([f"{symbol} {' '.join(map(str, coord))}" for symbol, coord in zip(symbols, coordinates)])
    footer = "\n\n"

    with open(f"{file_name}.com", "w") as f:
        f.write(header + molecule_data + footer)

# Example usage
# create_gaussian_com_file(['H', 'O', 'H'], [[0, 0, 0], [0, 0, 1], [0, 1, 0]], "water_molecule")

def write_slurm(com_file, script_name):
    script_content = f"""
    #!/bin/bash
    #SBATCH --job-name={com_file}
    #SBATCH --output={com_file}.out
    #SBATCH --error={com_file}.err
    #SBATCH --time=01:00:00
    #SBATCH --partition=your_partition
    #SBATCH --mem=4GB

    module load gaussian
    g09 < {com_file}.com > {com_file}.log
    """
    with open(f"{script_name}.sh", "w") as f:
        f.write(script_content)

# Example usage
# create_slurm_script("water_molecule", "run_water_molecule")


if __name__ == "__main__":
    directory = input("Enter the directory path containing XYZ files: ").strip()
    remove_duplicate_xyz_files(directory)