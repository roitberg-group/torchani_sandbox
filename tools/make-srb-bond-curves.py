from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torchani import geometry
from torchani.dispersion import constants
import molecule_utils as mu
from torchani.short_range_basis import StandaloneEnergySRB
import numpy as np


def fn(x):
    return - 5.0 * np.sqrt(6 * 8) * np.exp(-0.08 * x / 1.18)


if __name__ == '__main__':
    x = np.linspace(0, 8, 100)
    plt.plot(x, fn(x))
    plt.show()
    exit()


    makers = {'water': mu.make_water,
            'methane': mu.make_methane,
            'ammonia': mu.make_ammonia,
            'carbon_monoxide': mu.make_carbon_monoxide}
    energies = {k: [] for k in makers.keys()}
    srb = StandaloneEnergySRB(cutoff_fn='smooth', neighborlist_cutoff=5.2, elements=('H', 'C', 'N', 'O'), periodic_table_index=True)
    displacements = torch.linspace(-1.18, 6, 100)
    for molecule, maker in makers.items():
        path = Path(f'./bond_geometries/{molecule}').resolve()
        path.mkdir(parents=True, exist_ok=True)
        for d in displacements:
            species, coordinates = geometry.displace_along_bond(maker('cpu'), 0, 1, d)
            energies[molecule].append(srb((species, coordinates)).energies.item())
            # tensor_to_xyz(path / f'{d:.3f}.xyz', (species, coordinates))
    for k, v in energies.items():
        fig, ax = plt.subplots()
        ax.scatter(displacements, v)
        ax.set_title(k)
        plt.show()
