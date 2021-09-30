from torchani.geometry import displace_dimer_along_bond, displace_dimer_along_plane
import pickle
import re
import subprocess
from functools import partial
from torchani.dispersion import StandaloneDispersionD3
import tempfile
# import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import math
from torchani.models import ANI1x, ANI1ccx, ANI2x, ANID
import molecule_utils as mu
from torchani.utils import PERIODIC_TABLE
from torchani.aev.cutoffs import CutoffSmooth, CutoffDummy
import numpy as np
from tqdm import tqdm


def ANInoD(**kwargs):
    model = ANID(**kwargs)
    model.pre_aev_potentials = torch.nn.ModuleList([model.pre_aev_potentials[-1]])
    return model


def save_xyz_geometries(species_coordinates):
    root = Path(__file__).resolve().parent.joinpath(f'dftd3_geometries/{molecule}/')
    if not root.is_dir():
        root.mkdir(parents=True)
    xyz_path = root.joinpath(f'{d:.3f}.xyz')
    if xyz_path.exists():
        print("Not saving coordinates since path already exists")
    mu.tensor_to_xyz(xyz_path, species_coordinates)


class DummyEnergies:
    def __init__(self, e):
        self.energies = DummyInner(e)


class DummyInner:
    def __init__(self, e):
        self._e = e

    def item(self):
        return self._e


def orca_calculator(species_coordinates, periodic_table_index=True, functional='B973c', basis_set='def2-mTZVP'):
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
        match = re.findall(r'FINAL SINGLE POINT ENERGY(.*)\n', out_string)
        if len(match) > 1:
            raise RuntimeError(f"more than 1 match for time in {match}")
        elif not match:
            match = [math.nan]
        tmp_orca_input.unlink()
    return DummyEnergies(float(match[0].strip()))


def dftd3_calculator(species_coordinates, periodic_table_index=True):
    tmp_df_file = Path(__file__).resolve().parent.joinpath('.dftd3par.local')
    assert not tmp_df_file.exists()
    with open(tmp_df_file, 'w') as f:
        f.write('1.000 0.3700 1.5000 4.1000 14 4')
    # Periodic table index is a dummy variable
    tmpfile = Path(__file__).parent.resolve().joinpath('tmp')
    assert not tmpfile.exists()
    mu.tensor_to_xyz(tmpfile, species_coordinates)

    p = subprocess.run(f'time dftd3 {tmpfile.as_posix()}'.split(), capture_output=True)
    out_string = p.stdout.decode('ascii')
    match = re.findall(r'time elapsed:(.*?)\n', out_string, re.MULTILINE)
    assert len(match) == 1, f"more than 1 match for time in {match}"

    match = re.findall(r'^ Edisp /kcal,au.*?\n', out_string, re.MULTILINE)
    if not match:
        print('match not found')
        print('out string:', out_string)
        with open(tmpfile, 'r') as f:
            xyz = f.read()
        print('xyz file:', xyz)
        exit()
    assert len(match) == 1, f"more than 1 match for dftd3 dispersion energy in {match}"
    tmpfile.unlink()
    tmp_df_file.unlink()

    dftd3_energy = float(match[0].split()[-1])

    return DummyEnergies(dftd3_energy)


if __name__ == '__main__':
    makers = {'water': mu.make_water,
            'methane': mu.make_methane,
            'ammonia': mu.make_ammonia}
    displace_to_limit = True
    # I want to plot the following:
    # D3 only (with different cutoffs_fn and/or cutoffs) and ANI + D3
    ani_models = True
    pure_d3 = False
    dft_energies = False
    displace_plane = False
    ani_plus_d3 = False
    save_geometries = False
    plot = True
    if plot:
        for molecule in makers.keys():
            with open(f'{molecule}_d3_curves.pkl', 'rb') as f:
                plot_data = pickle.load(f)
                fig, ax = plt.subplots()
                colors = ['g', 'darkred', 'darkblue', 'purple', 'orange']
                for c, (k, v) in zip(colors, plot_data['energies'].items()):
                    if not pure_d3 and 'D3-' in k:
                        continue
                    if not ani_models and 'D3-' not in k:
                        continue
                    if ani_plus_d3 and 'D3-' not in k:
                        ax.plot(plot_data['displacements'], np.asarray(v) + np.asarray(plot_data['energies']['D3-fortran']), label=k + '+D3')
                    k = k.replace('ORCA', 'B97-3c')
                    ax.plot(plot_data['displacements'], np.asarray(v) - v[-1], label=k, linewidth=2.0, color=c)
                    ax.set_title(molecule.capitalize())
                    ax.set_xlabel(r'Intermolecular distance ($\AA$)')
                    ax.set_ylabel('Energy (Ha)')
                plt.legend()
                plt.show()
        exit()
    cutoff = 8.5
    start_distance = 0.1  # or 0.1?
    end_distance = 4.0
    orders = [4]
    # models = {'ANI-1x': ANI1x, 'ANI-2x': ANI2x, 'ANI-1ccx': ANI1ccx, 'ANI-D': ANID, 'ANInoD': ANInoD}
    models = {'ANI-D3': ANID, 'ANI (No D3)': ANInoD, 'ANI-2x': ANI2x}
    energy_calculators = {}
    if pure_d3:
        for order in orders:
            energy_calculators.update({f'D3-smooth-cut-order{order}': StandaloneDispersionD3(cutoff_fn=CutoffSmooth(order=order),
                                                                                             neighborlist_cutoff=cutoff,
                                                                                             periodic_table_index=True, functional='B97-3c')})
        energy_calculators.update({'D3-smooth-no-cut': StandaloneDispersionD3(cutoff_fn=CutoffDummy(),
                                                                              neighborlist_cutoff=math.inf,
                                                                              periodic_table_index=True, functional='B97-3c')})
        energy_calculators.update({'D3-fortran': dftd3_calculator})
    if ani_models:
        for name, model in models.items():
            energy_calculators.update({f'{name}': model(periodic_table_index=True).float()})
    if dft_energies:
        energy_calculators.update({'B973c': partial(orca_calculator, functional='B973c')})
        energy_calculators.update({'wB97X': partial(orca_calculator, functional='wB97X', basis_set='6-31G(d)')})

    for j, (molecule, maker) in enumerate(makers.items()):
        energies = {k: [] for k in energy_calculators.keys()}
        species1, coordinates1 = maker('cpu')
        species2, coordinates2 = maker('cpu')
        atom1 = -2
        atom2 = -1
        # we are grabbing two different atoms
        assert species1[0, atom1] != species1[0, atom2]
        species = torch.cat((species1, species2), dim=1)
        coordinates_orig = torch.cat((coordinates1, coordinates2), dim=1)
        r1 = coordinates_orig[0, atom1]
        r2 = coordinates_orig[0, atom2]
        bond_distance = (r2 - r1).norm()
        start_displace = bond_distance + start_distance
        end_displace = bond_distance + end_distance
        displacements = torch.linspace(start_displace, end_displace, 20)
        for d in tqdm(displacements):
            if molecule == 'water' and displace_plane:
                coordinates = displace_dimer_along_plane(coordinates_orig.clone(), atom1, atom2, 0, d)
            else:
                coordinates = displace_dimer_along_bond(coordinates_orig.clone(), atom1, atom2, d)
            if save_geometries:
                save_xyz_geometries((species, coordinates))

            for name, calc in energy_calculators.items():
                energy = calc((species, coordinates.float())).energies.item()
                energies[name].append(energy)
        with open(f'{molecule}_d3_curves.pkl', 'wb') as f:
            pickle.dump({'energies': energies,
                         'displacements': displacements - bond_distance}, f)
