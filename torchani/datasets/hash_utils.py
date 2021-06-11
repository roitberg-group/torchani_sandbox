from hashlib import md5
import time
import torch
from torch import Tensor
from typing import Tuple
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import numpy as np
import h5py


class ConformerHasher:

    def __init__(self, decimals=6):
        self._decimals = decimals
        self._similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-10)
        # eigenvectors is lazily calculated by get_conformation_hash if needed
        self._eigenvectors = None

    def get_invariant_hash(self, species_coordinates: Tuple[Tensor, Tensor]) -> str:
        # This is an experimental function that generates a unique translationally
        # and rotationally invariant hash value for a given species_coordinates
        # pair, the hash value is computed using md5

        # The main objective here is to obtain an invariant ordering of the atoms,
        # to do this we look for invariant properties and keep breaking ties. Each
        # time we order we need to round to some low number of decimals, otherwise
        # we risk ordering differently if the invariants change slightly due to
        # floating point precision, which can happen due to translations or
        # rotations being applied previous to the hash calculation.

        # sanity check, this must be reset each time the function runs
        assert self._eigenvectors is None
        species, coordinates = species_coordinates

        # Manipulations are in double precision to prevent errors from building up,
        # the original coordinates are always used for the calculation of the
        # distance matrix for the same reason
        coordinates = coordinates.to(torch.double)

        # First we sort using atomic numbers as the order, note that torch sorting
        # is not necessarily stable, but we don't mind.
        first_sort_idxs = torch.argsort(species)
        coordinates = coordinates[first_sort_idxs]
        species = species[first_sort_idxs]

        # If all atomic numbers are unique we are done and we return early
        counts_species = torch.unique_consecutive(species, return_counts=True)[1]
        if (counts_species == 1).all():
            return self._hash_from_ordered_atoms((species, coordinates))

        # To break ties we first use vector magnitudes from the origin after
        # displacing to the center of geometry to ensure translational invariance.
        centered_coordinates = coordinates - coordinates.mean(dim=0)

        # It is technically not necessary to calculate magnitudes for all atomic
        # numbers but it is simpler and very fast
        magnitudes = centered_coordinates.norm(dim=1, p=2)
        coordinates, counts_mags = self._use_invariant_to_break_ties(coordinates, magnitudes, counts_species)
        if (counts_mags == 1).all():
            return self._hash_from_ordered_atoms((species, coordinates))

        cosines = self._get_cosines_with_eigenspace(centered_coordinates, 0)
        coordinates, counts_cosines = self._use_invariant_to_break_ties(coordinates, cosines, counts_mags)
        if (counts_cosines == 1).all():
            return self._hash_from_ordered_atoms((species, coordinates))

        # It is extremely unlikely that we hit this function
        cosines = self._get_cosines_with_eigenspace(centered_coordinates, 1)
        coordinates, counts_cosines = self._use_invariant_to_break_ties(coordinates, cosines, counts_cosines)
        if (counts_cosines == 1).all():
            return self._hash_from_ordered_atoms((species, coordinates))

        # If we fell true we fail because currently the algorithm has no way of
        # breaking ties here I believe this is very difficult, unless the
        # molecule has so much symmetry that you can actually permute two or
        # more atoms and have the exact same conformation.
        raise RuntimeError("Ties could not be broken with cosines, "
                           "due to a limitation of the hashing "
                           "algorithm this molecule can't be hashed")

    def _use_invariant_to_break_ties(self, coordinates, invariant, counts_previous_invariant):
        invariant = around(invariant, decimals=self._decimals)
        split_invariant = torch.split(invariant, counts_previous_invariant.tolist())
        sorted_split_invariant_and_idxs = [torch.sort(s) for s in split_invariant]

        # We reorder the coordinates using these indices
        coordinates = self._reorder_coordinates(coordinates, sorted_split_invariant_and_idxs)
        counts_invariant = torch.cat([torch.unique_consecutive(invariant_idxs.values, return_counts=True)[1]
                                     for invariant_idxs in sorted_split_invariant_and_idxs])
        return coordinates, counts_invariant

    def _get_cosines_with_eigenspace(self, coordinates, eigenspace_idx=0):

        # lazy initialization of eigenvectors
        if self._eigenvectors is None:
            # The moment of inertia is just the covariance matrix if the center of
            # mass has been subtracted, analogously for the moment of geometry and
            # center of geometry (these don't take mass into account)
            moment_of_geometry = cov(coordinates.T, bias=True)

            # By default these eigenvalues are in ascending order
            eigenvalues, eigenvectors = torch.linalg.eigh(moment_of_geometry)

            eigenvalues = around(eigenvalues, self._decimals)
            if eigenvalues[eigenspace_idx] == eigenvalues[eigenspace_idx + 1]:
                raise RuntimeError("The next eigenspace must be distinct, "
                                   "otherwise we can't perform the projection uniquely")
            self._eigenvectors = eigenvectors

        # eigenvectors are columns of the output matrix
        vector = self._eigenvectors[:, eigenspace_idx].unsqueeze(0)
        # We use abs cos since sometimes linalg.eigh can compute the negative
        # of the requested eigenvector
        return self._similarity(vector.repeat(coordinates.shape[0], 1), coordinates).abs()

    def _hash_from_ordered_atoms(self, species_coordinates):
        self._eigenvectors = None
        species, coordinates = species_coordinates
        # As a final step we calculate the distance matrix between all the coordinates
        diff_vector = coordinates.view(1, -1, 3) - coordinates.view(-1, 1, 3)
        distance_matrix = diff_vector.norm(dim=-1, p=2)
        triu_idxs = torch.triu_indices(distance_matrix.shape[0], distance_matrix.shape[1], offset=1)
        distances = distance_matrix[triu_idxs[0], triu_idxs[1]]

        # We round the distance matrix to 0.2 angstroms to account for potential
        # differences, between machines and to have some threshold under which we
        # consider molecules to be equal. This rounding operation is important to
        # ensure that for example, the hash will be robust to a rotation or a
        # translation. In practice the resistance is not perfect and sometimes
        # fails, especially for translations.
        numbers_str = ''.join(map(str, species.view(-1).tolist()))
        distances = (around(distances, 1) * 10).to(torch.long)
        coordinates_str = ''.join(map(str, distances.view(-1).tolist()))
        merged_str = f'{numbers_str}{coordinates_str}'

        hasher = md5()  # outputs a length 32 str
        hasher.update(bytes(merged_str, 'ascii'))
        return hasher.hexdigest()

    @staticmethod
    def _reorder_coordinates(coordinates, sorted_vals_and_idxs):
        split_coordinates = torch.split(coordinates, [len(s.indices) for s in sorted_vals_and_idxs])
        split_coordinates = [coords[vals_idxs.indices]
                             for coords, vals_idxs in zip(split_coordinates, sorted_vals_and_idxs)]
        return torch.cat(split_coordinates)


def cov(x, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)
    https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    x = x if rowvar else x.transpose(-1, -2)
    x = x - x.mean(dim=-1, keepdim=True)
    factor = 1 / (x.shape[-1] - int(not bool(bias)))
    return factor * x @ x.transpose(-1, -2).conj()


def around(x, decimals=0):
    """from
    https://discuss.pytorch.org/t/round-tensor-to-x-decimal-places/25832"""
    factor = 10 ** decimals
    return torch.round(x * factor) / factor


def hash_all_conformations(dataset, **kwargs):
    hasher = ConformerHasher(**kwargs)
    for k in dataset.keys():
        g = dataset._get_group(k, non_element_keys=('coordinates',), element_keys=('species',))
        hashes = []
        print(g['invariant_hash'])
        exit()
        for s, c, h in zip(g['species'], g['coordinates']):
            hash_ = hasher.get_invariant_hash((s, c))
            hashes.append(hash_)
        hashes = np.asarray(hashes, dtype=bytes)
        with h5py.File(ds._store_file, 'r+') as f:
            f[k].create_dataset('invariant_hash', data=hashes)
        print(hashes)


# 3 attempts and both True for benchmark
false_positive_attempts_per_conformation = 0
false_positive_attempts = 0
false_positives = 0
random_translate = False
random_rotate = False
max_translation = 10

false_negative_attempts_per_conformation = 0
false_negative_attempts = 0
false_negatives = 0
negative_threshold = 0.15

from torchani.datasets import AniH5Dataset  # noqa
ds = AniH5Dataset('/media/samsung1TBssd/comp6_experimental/COMP6-v1-wB97X-631Gd/S66x8-v1-wB97X-631Gd.h5')
hash_all_conformations(ds)

hasher = ConformerHasher(decimals=6)

if false_positive_attempts_per_conformation > 0:
    assert random_translate or random_rotate
start = time.time()
hashes = []
for c in tqdm(ds.iter_conformers(include_properties=('coordinates', 'species')), total=ds.num_conformers):
    coordinates = c['coordinates']
    species = c['species']
    hash_ = hasher.get_invariant_hash((species, coordinates))
    hashes.append(hash_)

    for _ in range(false_positive_attempts_per_conformation):
        coordinates_dummy = coordinates.clone()
        false_positive_attempts += 1
        if random_translate:
            coordinates_dummy += torch.rand((1, 3), dtype=torch.float) * max_translation
        if random_rotate:
            random_rotation = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float)
            coordinates_dummy = coordinates_dummy @ random_rotation.T
        hash_2 = hasher.get_invariant_hash((species, coordinates_dummy))
        if hash_ != hash_2:
            false_positives += 1
            print('false_positive, cumcount: ', false_positives)

    for _ in range(false_negative_attempts_per_conformation):
        coordinates_dummy = coordinates.clone()
        false_negative_attempts += 1
        random_rotation = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float)
        random_displacement = random_rotation @ torch.tensor([negative_threshold, 0, 0])
        random_idx = torch.randint(low=0, high=coordinates.shape[0], size=(1,)).item()
        coordinates_dummy[random_idx, :] += random_displacement
        hash_2 = hasher.get_invariant_hash((species, coordinates_dummy))
        if hash_ == hash_2:
            false_negatives += 1
            print('false_negative, cumcount: ', false_negatives)

end = time.time()

start_dummy = time.time()
for c in tqdm(ds.iter_conformers(include_properties=('coordinates', 'atomic_numbers')), total=ds.num_conformers):
    pass
end_dummy = time.time()

total_ = end_dummy - start_dummy
try:
    print('false negative percentage = ', false_negatives * 100 / false_negative_attempts)
    print('false positive percentage = ', false_positives * 100 / false_negative_attempts)
except Exception:
    pass
print(((end - start) - (end_dummy - start_dummy)) / ds.num_conformers)
exit()
