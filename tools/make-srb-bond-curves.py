from pathlib import Path
import torchani
import matplotlib.pyplot as plt
import torch
import math
from torchani import geometry
from torchani import molecule_utils
import matplotlib as mpl
from torchani.short_range_basis import StandaloneEnergySRB
from torchani.aev.cutoffs import CutoffDummy

if __name__ == '__main__':
    mpl.rc('font', size=22)
    makers = {'water': molecule_utils.make_water,
              'methane': molecule_utils.make_methane,
              'ammonia': molecule_utils.make_ammonia,
              'carbon_monoxide': molecule_utils.make_carbon_monoxide}
    srb = StandaloneEnergySRB(cutoff_fn=CutoffDummy(), neighborlist_cutoff=math.inf, elements=('H', 'C', 'N', 'O'), periodic_table_index=True)
    models = {'ANID': torchani.models.ANID(periodic_table_index=True).double(), 'ANI2x': torchani.models.ANI2x(periodic_table_index=True).double()}
    energies = {}
    for name in makers.keys():
        energies[name] = {k: [] for k in models.keys()}
    displacements = torch.linspace(-0.9, 5.0, 200)
    # There doesn't seem to be an appreciable effect for things further away than 2 angstroms
    for molecule, maker in makers.items():
        path = Path(f'./bond_geometries/{molecule}').resolve()
        path.mkdir(parents=True, exist_ok=True)
        for d in displacements:
            species, coordinates = geometry.displace_along_bond(maker('cpu'), 0, 1, d)
            for name, m in models.items():
                energies[molecule][name].append(m((species, coordinates)).energies.item())
    for k, mol_energies in energies.items():
        fig, ax = plt.subplots()
        for name, v in mol_energies.items():
            ax.scatter(displacements, v, label=name)
        ax.set_title(k)
        ax.set_xlabel(f'Distance ($A$)')
        ax.set_ylabel(f'Energy ($Ha$)')
        plt.legend()
        plt.show()
