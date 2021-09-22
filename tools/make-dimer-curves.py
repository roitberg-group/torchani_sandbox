from torchani.geometry import displace_dimer_along_bond
import pickle
from torchani.dispersion import StandaloneDispersionD3
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from torchani.models import ANI1x, ANI1ccx, ANI2x
from molecule_utils import make_water, make_methane, make_ammonia, tensor_to_xyz
from torchani.aev.cutoffs import CutoffSmooth, CutoffDummy
from torchani.aev.neighbors import FullPairwise


def save_xyz_geometries(species, coordinates):
    root = Path(__file__).resolve().parent.joinpath(f'dftd3_geometries/{molecule}/')
    if not root.is_dir():
        root.mkdir(parents=True)
    xyz_path = root.joinpath(f'{d:.3f}.xyz')
    if xyz_path.exists():
        print("Not saving coordinates since path already exists")
    tensor_to_xyz(xyz_path, (species, coordinates))


if __name__ == '__main__':
    makers = {'water': make_water,
            'methane': make_methane,
            'ammonia': make_ammonia}
    models = {'1x': ANI1x, '2x': ANI2x, '1ccx': ANI1ccx}
    displace_to_limit = True
    use_ani = False
    save_geometries = True
    plot = False
    plot_cutoffs = True
    plot_dft_energies = False

    if use_ani:
        models_to_use = models.keys()
        suffix = '_ani'
    else:
        models_to_use = ['']
        suffix = ''

    for model_str in models_to_use:
        if plot:
            fig, ax = plt.subplots(1, len(makers), sharex=True, sharey=False, figsize=(4 * len(makers), 4), dpi=400)
            for j, molecule in enumerate(makers.keys()):
                with open(f'{molecule}_d3_curves{suffix}{model_str}.pkl', 'rb') as f:
                    data = pickle.load(f)
                    energies = data['energies']
                    displacements = data['displacements']

                if plot_dft_energies:
                    with open(f'{molecule}_orca_energies_d3bj.pkl', 'rb') as f:
                        data = pickle.load(f)
                        idx = np.argsort(data['distances'])
                        df_energies = data['df_energies'][idx]
                        dispersion_energies = data['dispersion_energies'][idx]
                        df_plus_disp = (df_energies + dispersion_energies).tolist()
                    energies.update({'wB97M': df_energies.tolist(), 'wB97MD3BJ': df_plus_disp})
                    with open(f'{molecule}_orca_energies.pkl', 'rb') as f:
                        data = pickle.load(f)
                        idx = np.argsort(data['distances'])
                        df_energies = data['total_energies'][idx]
                    energies.update({'wB97M': df_energies.tolist()})

                colors = ['r', 'g', 'purple', 'b', 'orange', 'k', 'pink', 'darkred', 'fuchsia']
                assert len(colors) >= len(energies.keys()), "not enough colors"
                colors = colors[:len(energies.keys())]
                for i, (k, ens) in enumerate(energies.items()):
                    if not plot_cutoffs:
                        if 'd3_cut' in k:
                            continue
                    if 'd3_only' in k:
                        continue
                    ens = torch.tensor(ens)
                    color = colors[i]
                    widths = 0.5
                    if displace_to_limit:
                        ax[j].scatter(displacements, ens
                                - ens[-1], s=1.0, color=color)
                        ax[j].plot(displacements, ens
                                - ens[-1], linewidth=widths, label=k, color=color)
                    else:
                        ax[j].scatter(displacements, ens, s=1.0, color=color)
                        ax[j].plot(displacements, ens, linewidth=widths, label=k, color=color)
                        ax[j].hlines(y=ens[-1], xmin=displacements[0],
                                xmax=displacements[-1], linestyles='dashed',
                                linewidths=widths, colors=color)
                ax[j].set_title(f'{molecule}')
                ax[j].set_ylabel(r'Energy (Ha)')
                ax[j].set_xlabel(r'Distance ($\AA$)')
                ax[j].set_xlim(displacements[0], displacements[-1])
                # ax[j].set_ylim(-0.025, 0.015)
                ax[j].legend()
            plt.savefig(f'dimer_curves{suffix}{model_str}.png')
        else:
            cutoff = 8.0
            start_distance = 2.0  # or 0.1?
            end_distance = 8.5
            orders = [2, 4, 6]
            if not use_ani:
                disp_calcs = {or_: StandaloneDispersionD3(cutoff_fn=CutoffSmooth(order=or_), neighborlist_cutoff=cutoff, periodic_table_index=True) for or_ in orders}
                disp_calcs.update({0: StandaloneDispersionD3(cutoff_fn=CutoffDummy(), neighborlist_cutoff=math.inf, periodic_table_index=True)})
            else:
                d3_only = StandaloneDispersionD3(cutoff_fn=CutoffDummy(), neighborlist_cutoff=math.inf, periodic_table_index=True)
                ani_only = models[model_str](periodic_table_index=True).double()
                disp_calcs = {or_: models[model_str](dispersion=True,
                    periodic_table_index=True,
                    dispersion_cutoff_function=CutoffSmooth(order=or_)).double() for or_ in orders}
                disp_calcs.update({0: models[model_str](dispersion=True, periodic_table_index=True).double()})
                for or_ in orders:
                    disp_calcs[or_].aev_computer.neighborlist = FullPairwise(cutoff)
                disp_calcs[0].aev_computer.neighborlist = FullPairwise(50.0)  # effectively no cutoff

            for j, (molecule, maker) in enumerate(makers.items()):
                species1, coordinates1 = maker('cpu')
                species2, coordinates2 = maker('cpu')
                energies = {f'd3_cut{or_}': [] for or_ in orders}
                energies.update({'d3_no_cut': []})
                if use_ani:
                    energies.update({'no_d3': []})
                    energies.update({'d3_only': []})

                atom1 = -2
                atom2 = -1
                # we are grabbing a hydrogen and another atom
                assert species1[0, atom1] != species1[0, atom2]
                species = torch.cat((species1, species2), dim=1)
                coordinates_orig = torch.cat((coordinates1, coordinates2), dim=1)
                r1 = coordinates_orig[0, atom1]
                r2 = coordinates_orig[0, atom2]
                bond_distance = (r2 - r1).norm()
                start_displace = bond_distance + start_distance
                end_displace = bond_distance + end_distance
                displacements = torch.linspace(start_displace, end_displace, 100)

                for d in displacements:
                    coords = coordinates_orig.clone()
                    coordinates = displace_dimer_along_bond(coords, atom1, atom2, d)

                    if save_geometries:
                        save_xyz_geometries(species, coordinates)
                    d3_no_cut_energy = disp_calcs[0]((species, coordinates)).energies.item()
                    energies['d3_no_cut'].append(d3_no_cut_energy)
                    if use_ani:
                        d3_only_energy = d3_only((species, coordinates)).energies.item()
                        energies['d3_only'].append(d3_only_energy)
                        ani_only_energy = ani_only((species, coordinates)).energies.item()
                        energies['no_d3'].append(ani_only_energy)
                    for or_ in orders:
                        energy = disp_calcs[or_]((species, coordinates)).energies.item()
                        energies[f'd3_cut{or_}'].append(energy)

                with open(f'{molecule}_d3_curves{suffix}{model_str}.pkl', 'wb') as f:
                    pickle.dump({'energies': energies,
                                 'displacements': displacements - bond_distance}, f)
