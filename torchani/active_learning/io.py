# This script shoudl take an input from the Isolator module and output gaussian input files and slurm scripts

import torch
import torchani
from torchani.AL_protocol import isolator
import hashlib
import os

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

if __name__ == "__main__":
    directory = input("Enter the directory path containing XYZ files: ").strip()
    remove_duplicate_xyz_files(directory)


def write_gaussian_input(
    input_name,
    output
):
    pass

def write_slurm(
    gaussian_input,
    output_file_prefix
):
    pass