r"""Utilities for working with ANI Datasets"""
from pathlib import Path
from torch import Tensor
from typing import List, Tuple, Optional, Sequence

import torch

from ..units import hartree2kcalmol
from ..models import BuiltinModel
from ..utils import pad_atomic_properties, tqdm
from ..nn import Ensemble
from ._annotations import KeyIdx, Conformers, PathLike
from .datasets import ANIDataset


def concatenate(source: ANIDataset,
                       dest_path: PathLike, *,
                       verbose: bool = True,
                       delete_originals: bool = False) -> ANIDataset:
    if source.grouping not in ['by_formula', 'by_num_atoms']:
        raise ValueError("Please regroup your dataset before concatenating")
    source_paths = [sub_ds._store_location for sub_ds in source._datasets.values()]
    dest_path = Path(dest_path).resolve()

    if dest_path.exists():
        raise ValueError('Destination path must be a new file name')
    if not source._num_subds > 1:
        raise ValueError("Need more than one subdataset to concatenate")

    dest = ANIDataset(dest_path,
                      create=True,
                      grouping=source.grouping,
                      property_aliases=source._first_subds._storename_to_alias,
                      verbose=False)

    for k, v in tqdm(source.numpy_items(),
                  desc='Concatenating datasets',
                  total=source.num_conformer_groups,
                  disable=not verbose):
        dest.append_numpy_conformers(k.split('/')[-1], v)

    if delete_originals:
        for p in tqdm(source_paths,
                      desc='Deleting original store',
                      total=len(source_paths),
                      disable=not verbose):
            p.unlink()
    return dest


def filter_by_high_force(dataset: ANIDataset, *,
                         threshold: float = 2.0,
                         criteria: str = 'components',
                         device: str = 'cpu',
                         max_split: int = 2560,
                         delete_inplace: bool = False,
                         verbose: bool = True) -> Optional[Tuple[Tensor, Tensor, List[KeyIdx]]]:
    if criteria == 'magnitude':
        desc = f"Filtering where force magnitude > {threshold} Ha / Angstrom"
    elif criteria == 'components':
        desc = f"Filtering where any force component > {threshold} Ha / Angstrom"
    else:
        raise ValueError('Criteria must be one of "magnitude" or "components"')

    # Threshold is by default 2 Ha / Angstrom
    bad_conformations: List[Conformers] = []
    bad_keys_and_idxs: List[KeyIdx] = []
    with torch.no_grad():
        for key, g in tqdm(dataset.items(),
                           total=dataset.num_conformer_groups,
                           desc=desc,
                           disable=not verbose):
            # conformers are split into pieces of up to max_split to avoid
            # loading large groups into GPU memory at the same time and
            # calculating over them
            species, coordinates, forces = _fetch_splitted(g, ('species', 'coordinates', 'forces'), max_split)
            for split_idx, (s, c, f) in enumerate(zip(species, coordinates, forces)):
                s, c, f = s.to(device), c.to(device), f.to(device)
                if criteria == 'components':
                    # any over atoms and over x y z
                    bad_idxs = (f.abs() > threshold).any(dim=-1).any(dim=-1).nonzero().squeeze()
                elif criteria == 'magnitude':
                    # any over atoms
                    bad_idxs = (f.norm(-1) > threshold).any(dim=-1).nonzero().squeeze()
                if bad_idxs.numel() > 0:
                    _append_bad_keys_and_idxs(bad_idxs, bad_keys_and_idxs, key, split_idx, max_split)
                    _append_bad_conformations(bad_idxs, bad_conformations, s, c)
                del s, c, f
    if delete_inplace:
        _delete_bad_conformations(dataset, bad_keys_and_idxs, verbose)
    # None is returned if no bad_idxs are found (bad_conformations is an empty
    # list in this case)
    return _return_padded_conformations_or_none(bad_conformations, bad_keys_and_idxs, device)


def filter_by_high_energy_error(dataset: ANIDataset,
                                model: BuiltinModel, *,
                                threshold: int = 100,
                                device: str = 'cpu',
                                max_split: int = 2560,
                                delete_inplace: bool = False,
                                verbose: bool = True) -> Optional[Tuple[Tensor, Tensor, List[KeyIdx]]]:
    bad_conformations: List[Conformers] = []
    bad_keys_and_idxs: List[KeyIdx] = []
    model = model.to(device)
    if not model.periodic_table_index:
        raise ValueError("Periodic table index must be True to filter high energy error")
    is_ensemble = isinstance(model.neural_networks, Ensemble)
    with torch.no_grad():
        for key, g in tqdm(dataset.items(),
                           total=dataset.num_conformer_groups,
                           desc=f"Filtering where any |energy error| > {threshold} kcal / mol"):
            species, coordinates, target_energies = _fetch_splitted(g, ('species', 'coordinates', 'energies'), max_split)
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
    return _return_padded_conformations_or_none(bad_conformations, bad_keys_and_idxs, device)


def _delete_bad_conformations(dataset: ANIDataset, bad_keys_and_idxs: List[KeyIdx], verbose: bool) -> None:
    if bad_keys_and_idxs:
        total_filtered = sum([v.numel() for (k, v) in bad_keys_and_idxs])
        for (key, idx) in tqdm(bad_keys_and_idxs,
                               total=len(bad_keys_and_idxs),
                               desc='Deleting filtered conformers',
                               disable=not verbose):
            dataset.delete_conformers(key, idx)
    else:
        total_filtered = 0
    if verbose:
        print(f"Deleted {total_filtered} conformations")


def _return_padded_conformations_or_none(bad_conformations: List[Conformers],
                                         bad_keys_and_idxs: List[KeyIdx],
                                         device: str) -> Optional[Tuple[Tensor, Tensor, List[KeyIdx]]]:
    if bad_conformations:
        conformers = pad_atomic_properties(bad_conformations)
        return conformers['species'].to(device), conformers['coordinates'].to(device), bad_keys_and_idxs
    else:
        return None


def _fetch_splitted(conformers: Conformers, keys_to_split: Sequence[str], max_split: int) -> Tuple[Tuple[Tensor, ...], ...]:
    # NOTE: len of output tuple is the same as len of input keys_to_split
    return tuple(torch.split(conformers[k], max_split) for k in keys_to_split)


def _append_bad_keys_and_idxs(bad_idxs: Tensor,
                              bad_keys_and_idxs: List[KeyIdx],
                              key: str,
                              split_idx: int,
                              max_split: int) -> None:
    bad_idxs_in_group = bad_idxs + split_idx * max_split
    bad_keys_and_idxs.append((key, bad_idxs_in_group))


def _append_bad_conformations(bad_idxs: Tensor,
                              bad_conformations: List[Conformers],
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
