r"""Utilities for working with ANI Datasets"""
import torch
import h5py
from pathlib import Path
from torch import Tensor
from typing import List, Tuple, Optional, Sequence

from ..units import hartree2kcalmol
from ..models import BuiltinModel
from ..utils import pad_atomic_properties, tqdm
from ..nn import Ensemble
from ._annotations import KeyIdx, Properties, PathLike
from .datasets import (AniH5Dataset,
                       AniH5DatasetList,
                       DatasetWithFlag,
                       _create_numpy_properties_handle_str,
                       _may_need_cache_update)


def copy_linked_data(source: AniH5Dataset, dest: PathLike, verbose: bool = True) -> AniH5Dataset:
    # When a dataset is unlinked from the HDF5 file the underlying buffer
    # is still part of the file, so the file does not reduce its size.
    # After deleting conformations or properties the dataset can be rebuilt
    # on a different location. Since only accessible data is copied the
    # size is reduced.
    num_groups: int = source.num_conformer_groups
    dest = Path(dest).resolve()
    with h5py.File(dest, 'x') as f:
        # copy all groups
        for key, properties in tqdm(source.numpy_items(repeat_nonbatch_keys=False),
                                    desc='Copying linked data',
                                    total=num_groups,
                                    disable=not verbose):
            f.create_group(key)
            group = f[key]
            _create_numpy_properties_handle_str(group, properties)
        # copy metadata
        with h5py.File(source._store_file, 'r') as self_file:
            for k, v in self_file.attrs.items():
                f.attrs.create(k, data=v)
    # The copied dataset is re initialized to ensure it has been copied
    # properly
    return AniH5Dataset(dest,
                        flag_property=source._flag_property,
                        validate_metadata=True,
                        verbose=verbose)


@_may_need_cache_update
def concatenate_datasets(dest: PathLike,
                         datasets: Sequence[PathLike],
                         dest_name: str = '',
                         verbose: bool = True) -> DatasetWithFlag:

    ds_list = AniH5DatasetList(datasets)
    dest_ds = AniH5Dataset(Path(dest).resolve(),
                           create=True,
                           supported_properties=ds_list[0].supported_properties)

    with tqdm(desc='Concatenating datasets',
              total=ds_list.num_conformer_groups,
              disable=not verbose) as pbar:

        for j, ds in enumerate(ds_list):
            for k, properties in ds.numpy_items(repeat_nonbatch_keys=False):
                # mypy does not know that @wrap'ed functions have this attribute
                # and this is ugly to fix
                needs_update = dest_ds.append_numpy_conformers.__wrapped__(dest_ds, properties, k)[1]  # type: ignore
                pbar.update()
    return dest_ds, needs_update


def filter_by_high_force(dataset: AniH5Dataset,
                         threshold: int = 2,
                         device: str = 'cpu',
                         max_split: int = 2560,
                         delete_inplace: bool = False,
                         verbose: bool = True) -> Optional[Tuple[Tensor, Tensor, List[KeyIdx]]]:
    # Threshold is 2 Ha / Angstrom
    bad_conformations: List[Properties] = []
    bad_keys_and_idxs: List[KeyIdx] = []
    with torch.no_grad():
        for key, g in tqdm(dataset.items(),
                           total=dataset.num_conformer_groups,
                           desc=f"Filtering where any force component > {threshold} Ha / Angstrom"):
            species, coordinates, forces = _fetch_splitted_properties(g, ('species', 'coordinates', 'forces'), max_split)
            for split_idx, (s, c, f) in enumerate(zip(species, coordinates, forces)):
                s, c, f = s.to(device), c.to(device), f.to(device)
                # any over atoms and over x y z
                bad_idxs = (f.abs() > threshold).any(dim=-1).any(dim=-1).nonzero().squeeze()
                if bad_idxs.numel() > 0:
                    _append_bad_keys_and_idxs(bad_idxs, bad_keys_and_idxs, key, split_idx, max_split)
                    _append_bad_conformations(bad_idxs, bad_conformations, s, c)
                del s, c, f
    if delete_inplace:
        _delete_bad_conformations(dataset, bad_keys_and_idxs, verbose)
    return _return_padded_conformations_or_none(bad_conformations, bad_keys_and_idxs, device)


def filter_by_high_energy_error(dataset: AniH5Dataset,
                                model: BuiltinModel,
                                threshold: int = 100,
                                device: str = 'cpu',
                                max_split: int = 2560,
                                delete_inplace: bool = False,
                                verbose: bool = True) -> Optional[Tuple[Tensor, Tensor, List[KeyIdx]]]:
    bad_conformations: List[Properties] = []
    bad_keys_and_idxs: List[KeyIdx] = []
    model = model.to(device)
    assert model.periodic_table_index, "Periodic table index must be True to filter high energy error"
    is_ensemble = isinstance(model.neural_networks, Ensemble)
    with torch.no_grad():
        for key, g in tqdm(dataset.items(),
                           total=dataset.num_conformer_groups,
                           desc=f"Filtering where any |energy error| > {threshold} kcal / mol"):
            # properties are split into pieces of up to max_split to avoid
            # loading large groups into GPU memory at the same time and
            # calculating over them
            species, coordinates, target_energies = _fetch_splitted_properties(g, ('species', 'coordinates', 'energies'), max_split)
            for split_idx, (s, c, ta) in enumerate(zip(species, coordinates, target_energies)):
                s, c, ta = s.to(device), c.to(device), ta.to(device)
                if is_ensemble:
                    member_energies = model.members_energies((s, c)).energies
                else:
                    member_energies = model((s, c)).energies.unsqueeze(0)
                errors = hartree2kcalmol((member_energies - ta).abs())
                # any over individual models of the ensemble
                bad_idxs = (errors > threshold).any(dim=0).nonzero().squeeze()
                if bad_idxs.numel() > 0:
                    _append_bad_keys_and_idxs(bad_idxs, bad_keys_and_idxs, key, split_idx, max_split)
                    _append_bad_conformations(bad_idxs, bad_conformations, s, c)
                del s, c, ta
    if delete_inplace:
        _delete_bad_conformations(dataset, bad_keys_and_idxs, verbose)
    # None is returned if no bad_idxs are found (bad_conformations is an empty
    # list in this case)
    return _return_padded_conformations_or_none(bad_conformations, bad_keys_and_idxs, device)


def _delete_bad_conformations(dataset: AniH5Dataset, bad_keys_and_idxs: List[KeyIdx], verbose: bool) -> None:
    if bad_keys_and_idxs:
        total_filtered = sum([v.numel() for (k, v) in bad_keys_and_idxs])
        for (key, idx) in bad_keys_and_idxs:
            dataset.delete_conformers(key, idx)
    else:
        total_filtered = 0
    if verbose:
        print(f"Deleted {total_filtered} conformations")


def _return_padded_conformations_or_none(bad_conformations: List[Properties],
                                         bad_keys_and_idxs: List[KeyIdx],
                                         device: str) -> Optional[Tuple[Tensor, Tensor, List[KeyIdx]]]:
    if bad_conformations:
        properties = pad_atomic_properties(bad_conformations)
        return properties['species'].to(device), properties['coordinates'].to(device), bad_keys_and_idxs
    else:
        return None


def _fetch_splitted_properties(properties: Properties, keys_to_split: Sequence[str], max_split: int) -> Tuple[Tuple[Tensor, ...], ...]:
    # NOTE: len of output tuple is the same as len of input keys_to_split
    return tuple(torch.split(properties[k], max_split) for k in keys_to_split)


def _append_bad_keys_and_idxs(bad_idxs: Tensor,
                              bad_keys_and_idxs: List[KeyIdx],
                              key: str,
                              split_idx: int,
                              max_split: int) -> None:
    bad_idxs_in_group = bad_idxs + split_idx * max_split
    bad_keys_and_idxs.append((key, bad_idxs_in_group))


def _append_bad_conformations(bad_idxs: Tensor,
                              bad_conformations: List[Properties],
                              s: Tensor,
                              c: Tensor) -> None:
    bad_species_of_split = s[bad_idxs].cpu().clone()
    bad_coordinates_of_split = c[bad_idxs].cpu().clone()
    if not bad_species_of_split.dim() == 2:
        bad_species_of_split = bad_species_of_split.unsqueeze(0)
        bad_coordinates_of_split = bad_coordinates_of_split.unsqueeze(0)
    assert bad_species_of_split.dim() == 2
    assert bad_coordinates_of_split.dim() == 3
    bad_conformations.append({'species': bad_species_of_split, 'coordinates': bad_coordinates_of_split})
