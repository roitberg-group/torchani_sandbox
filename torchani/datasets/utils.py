"""Utilities for working with ANI Datasets"""
from ..models import BuiltinModel
from .datasets import AniH5Dataset
from ..units import hartree2kcalmol
from ..utils import pad_atomic_properties
import torch
from typing import List, Tuple, Dict, Optional, Sequence
from torch import Tensor
from tqdm import tqdm


def filter_by_high_force(dataset: AniH5Dataset, threshold: 2, device: str = 'cpu', max_split: int = 2560) -> Optional[Tuple[Tensor, Tensor]]:
    bad_conformations: List[Dict[str, Tensor]] = []
    with torch.no_grad():
        for key, g in tqdm(dataset.items(), total=dataset.num_conformer_groups):
            species, coordinates, forces = _fetch_splitted_properties(g, ('species', 'coordinates', 'forces'), max_split)
            for s, c, f in zip(species, coordinates, forces):
                s, c, f = s.to(device), c.to(device), f.to(device)
                # any over atoms and over x y z
                bad_idxs = (f.abs() > threshold).any(dim=-1).any(dim=-1).nonzero().squeeze()
                _maybe_append_bad_conformations(bad_idxs, bad_conformations, s, c)
                del s, c, f
    return _return_padded_conformations_or_none(bad_conformations, device)


def filter_by_high_energy_error(dataset: AniH5Dataset, model: BuiltinModel, threshold: 100, device: str = 'cpu', max_split: int = 2560) -> Optional[Tuple[Tensor, Tensor]]:
    bad_conformations: List[Dict[str, Tensor]] = []
    model = model.to(device)
    assert model.periodic_table_index, "Periodic table index must be True to filter high energy error"
    with torch.no_grad():
        for key, g in tqdm(dataset.items(), total=dataset.num_conformer_groups):
            # properties are split into pieces of up to max_split to avoid
            # loading large groups into GPU memory at the same time and
            # calculating over them
            species, coordinates, target_energies = _fetch_splitted_properties(g, ('species', 'coordinates', 'energies'), max_split)
            for s, c, ta in zip(species, coordinates, target_energies):
                s, c, ta = s.to(device), c.to(device), ta.to(device)
                member_energies = model.members_energies((s, c)).energies
                errors = hartree2kcalmol((member_energies - ta).abs())
                # any over individual models of the ensemble
                bad_idxs = (errors > threshold).any(dim=0).nonzero().squeeze()
                _maybe_append_bad_conformations(bad_idxs, bad_conformations, s, c)
                del s, c, ta
    # None is returned if no bad_idxs are found (bad_conformations is an empty
    # list in this case)
    return _return_padded_conformations_or_none(bad_conformations, device)


def _return_padded_conformations_or_none(bad_conformations: List[Dict[str, Tensor]], device: str) -> Optional[Tuple[Tensor, Tensor]]:
    if bad_conformations:
        properties = pad_atomic_properties(bad_conformations)
        return properties['species'].to(device), properties['coordinates'].to(device)
    else:
        return None


def _fetch_splitted_properties(properties: Dict[str, Tensor], keys_to_split: Sequence[str], max_split: int):
    return tuple(torch.split(properties[k], max_split) for k in keys_to_split)


def _maybe_append_bad_conformations(bad_idxs: Tensor, bad_conformations: List[Dict[str, Tensor]], s: Tensor, c: Tensor) -> None:
    if bad_idxs.numel() > 0:
        bad_species_of_split = s[bad_idxs].cpu().clone()
        bad_coordinates_of_split = c[bad_idxs].cpu().clone()
        if not bad_species_of_split.dim() == 2:
            bad_species_of_split = bad_species_of_split.unsqueeze(0)
            bad_coordinates_of_split = bad_coordinates_of_split.unsqueeze(0)
        assert bad_species_of_split.dim() == 2
        assert bad_coordinates_of_split.dim() == 3
        bad_conformations.append({'species': bad_species_of_split, 'coordinates': bad_coordinates_of_split})
