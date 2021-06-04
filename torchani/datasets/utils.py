"""Utilities for working with ANI Datasets"""
from ..models import BuiltinModel
from .datasets import AniH5Dataset
from ..units import hartree2kcalmol
import torch
from tqdm import tqdm

def filter_by_high_energy_error(dataset: AniH5Dataset, model: BuiltinModel, threshold: 100, device: str = 'cpu', max_split_size=5000) -> 'AniH5Dataset':
    model = model.to(device)
    bad_conformations = []
    count = 0
    with torch.no_grad():
        for key, g in tqdm(dataset.items(), total=dataset.num_conformer_groups):
            species = torch.split(g['species'], max_split_size)
            coordinates = torch.split(g['coordinates'], max_split_size)
            target_energies = torch.split(g['energies'], max_split_size)
            #member_energies = model.members_energies((species.unsqueeze(0), coordinates.unsqueeze(0))).energies
            for s, c, ta in zip(species, coordinates, target_energies):
                s = s.to(device)
                c = c.to(device)
                ta = ta.to(device)
                member_energies = model.members_energies((s, c)).energies
                errors = hartree2kcalmol((member_energies - ta).abs())
                bad_idxs = (errors > threshold).any(dim=0).nonzero().squeeze()
                if bad_idxs.numel() > 0:
                    count += bad_idxs.numel()
                del s
                del c
                del ta
        print(count)
