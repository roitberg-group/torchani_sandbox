r"""Utilities for creating batched datasets"""
import importlib
import warnings
import math
import json
import pickle
import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional, Sequence, List
from collections import OrderedDict

import h5py
import torch
from torch import Tensor
import numpy as np

from ..utils import pad_atomic_properties, cumsum_from_zero, PADDING
from .datasets import AniH5DatasetList
from ._annotations import Properties, PathLike, Transform

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None  # type: ignore
if PKBAR_INSTALLED:
    import pkbar


def _save_batch(path: Path, idx: int, batch: Properties, file_format: str) -> None:
    # We use pickle, numpy or hdf5 since saving in
    # pytorch format is extremely slow
    batch = {k: v.numpy() for k, v in batch.items()}
    if file_format == 'pickle':
        with open(path.joinpath(f'batch{idx}.pkl'), 'wb') as batch_file:
            pickle.dump(batch, batch_file)
    elif file_format == 'numpy':
        np.savez(path.joinpath(f'batch{idx}'), **batch)
    elif file_format == 'hdf5':
        with h5py.File(path.joinpath(f'batch{idx}.h5'), 'w-') as f:
            for k, v in batch.items():
                f.create_dataset(k, data=v)
    elif file_format == 'single_hdf5':
        with h5py.File(path.joinpath(f'{path.name}_single.h5'), 'a') as f:
            f.create_group(f'batch{idx}')
            g = f[f'batch{idx}']
            for k, v in batch.items():
                g.create_dataset(k, data=v)


def create_batched_dataset(h5_path: PathLike,
                           dest_path: Optional[PathLike] = None,
                           shuffle: bool = True,
                           shuffle_seed: Optional[int] = None,
                           file_format: str = 'hdf5',
                           include_properties: Optional[Sequence[str]] = ('species', 'coordinates', 'energies'),
                           batch_size: int = 2560,
                           max_batches_per_packet: int = 350,
                           padding: Optional[Dict[str, float]] = None,
                           splits: Optional[Dict[str, float]] = None,
                           folds: Optional[int] = None,
                           inplace_transform: Optional[Transform] = None,
                           verbose: bool = True) -> None:

    if folds is not None and splits is not None:
        raise ValueError('Only one of ["folds", "splits"] should be specified')

    # NOTE: All the tensor manipulation in this function is handled in CPU
    if file_format == 'single_hdf5':
        warnings.warn('Depending on the implementation, a single HDF5 file may'
                      'not support parallel reads, so using num_workers > 1 may'
                      'have a detrimental effect on performance, its probably better'
                      'to save in many hdf5 files with file_format=hdf5')
    if dest_path is None:
        dest_path = Path(f'./batched_dataset_{file_format}').resolve()
    dest_path = Path(dest_path).resolve()

    h5_path = Path(h5_path).resolve()
    if h5_path.is_dir():
        # sort paths according to file names for reproducibility
        paths_list = [p for p in h5_path.iterdir() if p.suffix == '.h5']
        filenames_list = [p.name for p in paths_list]
        sorted_paths = [p for _, p in sorted(zip(filenames_list, paths_list))]
        h5_datasets = AniH5DatasetList(sorted_paths)
    elif h5_path.is_file():
        h5_datasets = AniH5DatasetList([h5_path])

    # (1) Get all indices and shuffle them if needed
    #
    # These are pairs of indices that index first the group and then the
    # specific conformer, it is possible to just use one index for
    # everything but this is simpler at the cost of slightly more memory.
    # First we get all group sizes for all datasets concatenated in a tensor, in the same
    # order as h5_datasets
    group_sizes_values = torch.cat([torch.tensor(list(h5ds.group_sizes.values()), dtype=torch.long) for h5ds in h5_datasets])
    conformer_indices = torch.cat([torch.stack((torch.full(size=(s.item(),), fill_value=j, dtype=torch.long),
                                     (torch.arange(0, s.item(), dtype=torch.long))), dim=-1)
                                     for j, s in enumerate(group_sizes_values)])

    rng = _get_random_generator(shuffle, shuffle_seed)

    conformer_indices = _maybe_shuffle_indices(conformer_indices, rng)

    # (2) Split shuffled indices according to requested dataset splits or folds
    # by defaults we use splits, if folds or splits is specified we
    # do the specified operation
    if folds is not None:
        conformer_splits, split_paths = _divide_into_folds(conformer_indices, dest_path, folds, rng)
    else:
        if splits is None:
            splits = {'training': 0.8, 'validation': 0.2}

        if not math.isclose(sum(list(splits.values())), 1.0):
            raise ValueError("The sum of the split fractions has to add up to one")

        conformer_splits, split_paths = _divide_into_splits(conformer_indices, dest_path, splits)

    # (3) Compute the batch indices for each split and save the conformers to disk
    _save_splits_into_batches(split_paths,
                              conformer_splits,
                              inplace_transform,
                              file_format,
                              include_properties,
                              h5_datasets,
                              padding,
                              batch_size,
                              max_batches_per_packet,
                              verbose)

    # log creation data
    creation_log = {'datetime_created': str(datetime.datetime.now()),
                    'source_path': h5_path.as_posix(),
                    'splits': splits,
                    'folds': folds,
                    'padding': PADDING if padding is None else padding,
                    'is_inplace_transformed': inplace_transform is not None,
                    'shuffle': shuffle,
                    'shuffle_seed': shuffle_seed,
                    'include_properties': include_properties if include_properties is not None else 'all',
                    'batch_size': batch_size,
                    'total_num_conformers': len(conformer_indices),
                    'total_conformer_groups': len(group_sizes_values)}

    with open(dest_path.joinpath('creation_log.json'), 'w') as logfile:
        json.dump(creation_log, logfile, indent=1)


def _get_random_generator(shuffle: bool = False, shuffle_seed: Optional[int] = None) -> Optional[torch.Generator]:

    if shuffle_seed is not None:
        assert shuffle
        seed = shuffle_seed
    else:
        # non deterministic seed
        seed = torch.random.seed()

    rng: Optional[torch.Generator]

    if shuffle:
        rng = torch.random.manual_seed(seed)
    else:
        rng = None

    return rng


def _maybe_shuffle_indices(conformer_indices: Tensor,
                           rng: Optional[torch.Generator] = None) -> Tensor:
    total_num_conformers = len(conformer_indices)
    if rng is not None:
        shuffle_indices = torch.randperm(total_num_conformers, generator=rng)
        conformer_indices = conformer_indices[shuffle_indices]
    else:
        warnings.warn("Dataset will not be shuffled, this should only be used for debugging")
    return conformer_indices


def _divide_into_folds(conformer_indices: Tensor,
                        dest_path: Path,
                        folds: int,
                        rng: Optional[torch.Generator] = None) -> Tuple[Tuple[Tensor, ...], 'OrderedDict[str, Path]']:

    # the idea here is to work with "blocks" of size num_conformers / folds
    # cast to list for mypy
    conformer_blocks = list(torch.chunk(conformer_indices, folds))
    conformer_splits: List[Tensor] = []
    split_paths_list: List[Tuple[str, Path]] = []

    print(f"Generating {folds} folds for cross validation or ensemble training")
    for f in range(folds):
        # the first shuffle is necessary so that validation splits are shuffled
        validation_split = conformer_blocks[f]

        training_split = torch.cat(conformer_blocks[:f] + conformer_blocks[f + 1:])
        # afterwards all training folds are reshuffled to get different
        # batching for different models in the ensemble / cross validation
        # process (it is technically redundant to reshuffle the first one but
        # it is done for simplicity)
        training_split = _maybe_shuffle_indices(training_split, rng)
        conformer_splits.extend([training_split, validation_split])
        split_paths_list.extend([(f'training{f}', dest_path.joinpath(f'training{f}')),
                                 (f'validation{f}', dest_path.joinpath(f'validation{f}'))])
    split_paths = OrderedDict(split_paths_list)

    _create_split_paths(split_paths)

    return tuple(conformer_splits), split_paths


def _divide_into_splits(conformer_indices: Tensor,
                        dest_path: Path,
                        splits: Dict[str, float]) -> Tuple[Tuple[Tensor, ...], 'OrderedDict[str, Path]']:
    total_num_conformers = len(conformer_indices)
    split_sizes = OrderedDict([(k, int(total_num_conformers * v)) for k, v in splits.items()])
    split_paths = OrderedDict([(k, dest_path.joinpath(k)) for k in split_sizes.keys()])

    _create_split_paths(split_paths)

    leftover = total_num_conformers - sum(split_sizes.values())
    if leftover != 0:
        # We slightly modify a random section if the fractions don't split
        # the dataset perfectly. This also automatically takes care of the
        # cases leftover > 0 and leftover < 0
        any_key = list(split_sizes.keys())[0]
        split_sizes[any_key] += leftover
        assert sum(split_sizes.values()) == total_num_conformers
    conformer_splits = torch.split(conformer_indices, list(split_sizes.values()))
    assert len(conformer_splits) == len(split_sizes.values())
    print(f'Splits have number of conformers: {dict(split_sizes)}.'
          f' The requested percentages were: {splits}')
    return conformer_splits, split_paths


def _create_split_paths(split_paths: 'OrderedDict[str, Path]') -> None:
    for p in split_paths.values():
        if p.is_dir():
            subdirs = [d for d in p.iterdir()]
            if subdirs:
                raise ValueError('The dest_path provided already has files'
                                 ' or directories, please provide'
                                 ' a different path')
        else:
            if p.is_file():
                raise ValueError('The dest_path is a file, it should be a directory')
            p.mkdir(parents=True)


def _save_splits_into_batches(split_paths: 'OrderedDict[str, Path]',
                              conformer_splits: Tuple[Tensor, ...],
                              inplace_transform: Optional[Transform],
                              file_format: str,
                              include_properties: Optional[Sequence[str]],
                              h5_datasets: AniH5DatasetList,
                              padding: Optional[Dict[str, float]],
                              batch_size: int,
                              max_batches_per_packet: int,
                              verbose: bool) -> None:
    # NOTE: Explanation for following logic, please read
    #
    # This sets up a given number of batches (packet) to keep in memory and
    # then scans the dataset and find the conformers needed for the packet. It
    # then saves the batches and fetches the next packet.
    #
    # A "packet" is a list that has tensors, each of which
    # has batch indices, for instance [tensor([[0, 0, 1, 1, 2], [1, 2, 3, 5]]),
    #                                  tensor([[3, 5, 5, 5], [1, 2, 3, 3]])]
    # would be a "packet" of 2 batch_indices, each of which has in the first
    # row the index for the group, and in the second row the index for the
    # conformer
    #
    # It is important to do this with a packet and not only 1 batch.  The
    # number of reads to the h5 file is batches x conformer_groups x 3 for 1x
    # (factor of 3 from energies, species, coordinates), which means ~ 2000 x
    # 3000 x 3 = 9M reads, this is a bad bottleneck and very slow, even if we
    # fetch all necessary molecules from each conformer group simultaneously.
    #
    # Doing it for all batches at the same time is (reasonably) fast, ~ 9000
    # reads, but in this case it means we will have to put all, or almost all
    # the dataset into memory at some point, which is not feasible for larger
    # datasets.
    if inplace_transform is None:
        inplace_transform = lambda x: x  # noqa: E731

    # get all group keys concatenated in a list, with the associated file indexes
    file_idxs_and_group_keys = list(h5_datasets.iter_file_key())

    use_pbar = PKBAR_INSTALLED and verbose
    for split_path, indices_of_split in zip(split_paths.values(), conformer_splits):
        all_batch_indices = torch.split(indices_of_split, batch_size)

        all_batch_indices_packets = [all_batch_indices[j:j + max_batches_per_packet]
                                    for j in range(0, len(all_batch_indices), max_batches_per_packet)]
        num_batch_indices_packets = len(all_batch_indices_packets)

        overall_batch_idx = 0
        for j, batch_indices_packet in enumerate(all_batch_indices_packets):
            num_batches_in_packet = len(batch_indices_packet)
            # Now first we cat and sort according to the first index in order to
            # fetch all conformers of the same group simultaneously
            batch_indices_cat = torch.cat(batch_indices_packet, 0)
            indices_to_sort_batch_indices_cat = torch.argsort(batch_indices_cat[:, 0])
            sorted_batch_indices_cat = batch_indices_cat[indices_to_sort_batch_indices_cat]
            uniqued_idxs_cat, counts_cat = torch.unique_consecutive(sorted_batch_indices_cat[:, 0],
                                                                    return_counts=True)
            cumcounts_cat = cumsum_from_zero(counts_cat)

            # batch_sizes and indices_to_unsort are needed for the
            # reverse operation once the conformers have been
            # extracted
            batch_sizes = [len(batch_indices) for batch_indices in batch_indices_packet]
            indices_to_unsort_batch_cat = torch.argsort(indices_to_sort_batch_indices_cat)
            assert len(batch_sizes) <= max_batches_per_packet

            if use_pbar:
                pbar = pkbar.Pbar(f'=> Saving batch packet {j + 1} of {num_batch_indices_packets}'
                                  f' of split {split_path.name},'
                                  f' in format {file_format}', len(counts_cat))

            all_conformers: List[Properties] = []
            for step, (group_idx, count, start_index) in enumerate(zip(uniqued_idxs_cat, counts_cat, cumcounts_cat)):
                # select the specific group from the whole list of files
                file_idx, group_key = file_idxs_and_group_keys[group_idx.item()]

                # get a slice with the indices to extract the necessary
                # conformers from the group for all batches in pack.
                end_index = start_index + count
                selected_indices = sorted_batch_indices_cat[start_index:end_index, 1]

                # Important: to prevent possible bugs / errors, that may happen
                # due to incorrect conversion to indices, species is **always*
                # converted to atomic numbers when saving the batched dataset.
                conformers = h5_datasets.get_conformers(file_idx, group_key,
                                                                  selected_indices,
                                                                  include_properties)
                all_conformers.append(conformers)
                if use_pbar:
                    pbar.update(step)
            batches_cat = pad_atomic_properties(all_conformers, padding)
            # Now we need to reassign the conformers to the specified
            # batches. Since to get here we cat'ed and sorted, to
            # reassign we need to unsort and split.
            # The format of this is {'species': (batch1, batch2, ...), 'coordinates': (batch1, batch2, ...)}
            batch_packet_dict = {k: torch.split(t[indices_to_unsort_batch_cat], batch_sizes)
                                 for k, t in batches_cat.items()}

            for packet_batch_idx in range(num_batches_in_packet):
                batch = {k: v[packet_batch_idx] for k, v in batch_packet_dict.items()}
                batch = inplace_transform(batch)
                _save_batch(split_path, overall_batch_idx, batch, file_format)
                overall_batch_idx += 1
