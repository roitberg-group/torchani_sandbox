from pathlib import Path
import matplotlib.pyplot as plt
import torch
import math
from torchani import geometry
import molecule_utils as mu
from torchani.short_range_basis import StandaloneEnergySRB
from torchani.aev.cutoffs import CutoffDummy

if __name__ == '__main__':
    makers = {'water': mu.make_water,
            'methane': mu.make_methane,
            'ammonia': mu.make_ammonia,
            'carbon_monoxide': mu.make_carbon_monoxide}
    energies = {k: [] for k in makers.keys()}
    srb = StandaloneEnergySRB(cutoff_fn=CutoffDummy(), neighborlist_cutoff=math.inf, elements=('H', 'C', 'N', 'O'), periodic_table_index=True)
    displacements = torch.linspace(0.1, 2.0, 200)
    # there doesn't seem to be an appreciable effect for things further away than 2 angstroms
    for molecule, maker in makers.items():
        path = Path(f'./bond_geometries/{molecule}').resolve()
        path.mkdir(parents=True, exist_ok=True)
        for d in displacements:
            species, coordinates = geometry.displace_along_bond(maker('cpu'), 0, 1, d)
            energies[molecule].append(srb((species, coordinates)).energies.item())
    for k, v in energies.items():
        fig, ax = plt.subplots()
        ax.scatter(displacements, v)
        ax.set_title(k)
        plt.show()
