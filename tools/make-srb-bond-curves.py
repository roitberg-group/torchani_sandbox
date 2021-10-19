from pathlib import Path
import matplotlib.pyplot as plt
import torch
import math
from functools import partial
import tempfile
import subprocess
import re
from torchani import geometry
from torchani import molecule_utils
import matplotlib as mpl
from torchani.utils import PERIODIC_TABLE, tqdm
from torchani.short_range_basis import StandaloneEnergySRB
from torchani.aev.cutoffs import CutoffDummy


class DummyInner:
    def __init__(self, e):
        self._e = e

    def item(self):
        return self._e


class DummyEnergies:
    def __init__(self, e):
        self.energies = DummyInner(e)


def orca_calculator(species_coordinates, periodic_table_index=True, functional='B973c', basis_set='def2-mTZVP', srb=False):
    with tempfile.TemporaryDirectory() as td:
        species, coordinates = species_coordinates
        species = species.squeeze(0)
        coordinates = coordinates.squeeze(0)
        tmp_orca_input = Path(td).resolve().joinpath('orca.in')
        with open(tmp_orca_input, 'w') as f:
            f.write(f'! {functional} {basis_set} tightscf scfconvforced\n')
            f.write('\n')
            f.write('* xyz 0 1\n')
            for s, c in zip(species, coordinates):
                f.write(f'{PERIODIC_TABLE[s]} {c[0]} {c[1]} {c[2]}\n')
            f.write('*\n')
        assert tmp_orca_input.is_file()
        p = subprocess.run(f'orca {tmp_orca_input.as_posix()}'.split(), capture_output=True, cwd=td)
        out_string = p.stdout.decode('utf-8')
        if srb:
            pattern = r'SRB correction(.*)\n'
        else:
            pattern = r'FINAL SINGLE POINT ENERGY(.*)\n'
        match = re.findall(pattern, out_string)
        if len(match) > 1:
            raise RuntimeError(f"more than 1 match for time in {match}")
        elif not match:
            return DummyEnergies(math.nan)
        tmp_orca_input.unlink()
    return DummyEnergies(float(match[0].strip()))


if __name__ == '__main__':
    mpl.rc('font', size=22)
    makers = {'water': molecule_utils.make_water,
            'methane': molecule_utils.make_methane,
            'ammonia': molecule_utils.make_ammonia,
            'carbon_monoxide': molecule_utils.make_carbon_monoxide}
    srb = StandaloneEnergySRB(cutoff_fn=CutoffDummy(), neighborlist_cutoff=math.inf, elements=('H', 'C', 'N', 'O'), periodic_table_index=True)
    orca = partial(orca_calculator, srb=True)
    calculators = {'srb': srb, 'orca': orca}
    energies = {n: {k: [] for k in makers.keys()} for n in calculators.keys()}
    displacements = torch.linspace(0.1, 2.0, 10)
    # there doesn't seem to be an appreciable effect for things further away than 2 angstroms
    for molecule, maker in makers.items():
        path = Path(f'./bond_geometries/{molecule}').resolve()
        path.mkdir(parents=True, exist_ok=True)
        for d in tqdm(displacements, total=len(displacements)):
            species, coordinates = geometry.displace_along_bond(maker('cpu'), 0, 1, d)
            for n, calc in calculators.items():
                energies[n][molecule].append(calc((species, coordinates)).energies.item())
    for k in makers.keys():
        fig, ax = plt.subplots()
        for n, e in zip(calculators.keys(), energies.values()):
            ax.scatter(displacements, e[k], label=n)
        ax.set_title(k)
        ax.set_xlabel(f'Distance ($A$)')
        ax.set_ylabel(f'Energy ($Ha$)')
        plt.legend()
        plt.show()
