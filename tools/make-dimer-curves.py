from torchani.geometry import displace_dimer_along_bond
from torchani.dispersion import StandaloneDispersionD3, DispersionD3
import torchani
import matplotlib.pyplot as plt
import torch
from molecule_utils import make_water, make_methane, make_ammonia
from torchani.aev.cutoffs import CutoffSmooth

# this comparison test assumes dftd3 is installed and in PATH, so that it can
# be called directly with subprocess

if __name__ == '__main__':
    makers = [make_water, make_methane, make_ammonia]
    molecules = ['water', 'methane', 'ammonia']
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(12, 4), dpi=400)
    cutoff = 8.0
    no_ani = False
    if no_ani:
        starting_distance = 0.1
        disp = StandaloneDispersionD3()
        disp_cut = StandaloneDispersionD3(cutoff_function=CutoffSmooth(cutoff), neighborlist_cutoff=cutoff)
        disp_cut4 = StandaloneDispersionD3(cutoff_function=CutoffSmooth(cutoff, order=4), neighborlist_cutoff=cutoff)
        disp_cut6 = StandaloneDispersionD3(cutoff_function=CutoffSmooth(cutoff, order=6), neighborlist_cutoff=cutoff)
    else:
        starting_distance = 2.0
        disp = torchani.models.ANI2x(periodic_table_index=True).double()
        disp_cut = torchani.models.ANI2x(dispersion=True, periodic_table_index=True).double()
        disp_cut4 = torchani.models.ANI2x(dispersion=True, periodic_table_index=True).double()
        disp_cut6 = torchani.models.ANI2x(dispersion=True, periodic_table_index=True).double()

        disp_cut.dispersion_calculator = DispersionD3(cutoff_function=CutoffSmooth(cutoff))
        disp_cut4.dispersion_calculator = DispersionD3(cutoff_function=CutoffSmooth(cutoff, order=4))
        disp_cut6.dispersion_calculator = DispersionD3(cutoff_function=CutoffSmooth(cutoff, order=6))

    for j, (maker, molecule) in enumerate(zip(makers, molecules)):
        species1, coordinates1 = maker('cpu')
        species2, coordinates2 = maker('cpu')
        limit_energy = disp((species1, coordinates1)).energies.item() * 2
        limit_energy_cut = disp_cut((species1, coordinates1)).energies.item() * 2
        limit_energy_cut4 = disp_cut4((species1, coordinates1)).energies.item() * 2
        limit_energy_cut6 = disp_cut6((species1, coordinates1)).energies.item() * 2
        atom1 = -2
        atom2 = -1
        # we are grabbing a hydrogen and another atom
        assert species1[0, atom1] != species1[0, atom2]
        species = torch.cat((species1, species2), dim=1)
        coordinates_orig = torch.cat((coordinates1, coordinates2), dim=1)
        bond_distance = (coordinates_orig[0, atom1] - coordinates_orig[0, atom2]).norm()
        displacements = torch.linspace(bond_distance + starting_distance, bond_distance + cutoff, 100)
        energies = []
        energies_cut = []
        energies_cut4 = []
        energies_cut6 = []
        energies_ani = []
        for d in displacements:
            coordinates = displace_dimer_along_bond(coordinates_orig.clone(), atom1, atom2, d)
            energies.append(disp((species, coordinates)).energies.item())
            energies_cut.append(disp_cut((species, coordinates)).energies.item())
            energies_cut4.append(disp_cut4((species, coordinates)).energies.item())
            energies_cut6.append(disp_cut6((species, coordinates)).energies.item())
        colors = ['r', 'g', 'purple', 'b']
        if not no_ani:
            labels = ['smooth ^2', 'exp smooth ^4', 'exp smooth ^6', 'no D3']
        else:
            labels = ['smooth ^2', 'exp smooth ^4', 'exp smooth ^6', 'no cut']
        limits = [limit_energy_cut, limit_energy_cut4, limit_energy_cut6, limit_energy]
        all_energies = [energies_cut, energies_cut4, energies_cut6, energies]
        for ens, c, l, lim in zip(all_energies, colors, labels, limits):
            ax[j].scatter(displacements - bond_distance, ens, s=1.0, color=c)
            ax[j].plot(displacements - bond_distance, ens, linewidth=1.0, label=l, color=c)
            ax[j].hlines(y=lim, xmin=0, xmax=cutoff, linestyles='dashed', linewidths=1.0, colors=c)
        ax[j].set_title(f'{molecule}')
        ax[j].set_ylabel(r'Energy (Ha)')
        ax[j].set_xlabel(r'Distance ($\AA$)')
        ax[j].set_xlim(starting_distance, cutoff)
        ax[j].legend()
    if no_ani:
        plt.savefig('dimer_curves.png')
    else:
        plt.savefig('dimer_curves_ani.png')
