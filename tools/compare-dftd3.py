import torchani
import pickle
from pathlib import Path
import time
from torchani.dispersion import StandaloneDispersionD3
import matplotlib.pyplot as plt
import torch
import re
import subprocess
import argparse
from molecule_utils import tensor_to_xyz
from torchani.aev.cutoffs import CutoffSmooth

# this comparison test assumes dftd3 is installed and in PATH, so that it can
# be called directly with subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--device',
        help='Device of modules and tensors',
        default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--cutoff-smooth',
                        action='store_true', default=False,
                        help='Use a smooth cutoff for torchani')
    parser.add_argument('--cutoff-smooth4',
                        action='store_true', default=False,
                        help='Use a smooth cutoff with order 4 for torchani')
    parser.add_argument('--all-molecules',
                        action='store_true', default=False,
                        help='Compare all molecules')
    parser.add_argument('--num-molecules',
                        default=500,
                        help='Number of molecules to compare')
    parser.add_argument('--dataset-path',
                        default='../dataset/ani1-up_to_gdb4/ani_gdb_s03.h5',
                        help='Path of the dataset, can a hdf5 file \
                            or a directory containing hdf5 files')
    parser.add_argument('--plot', help='Path of file to plot')
    args = parser.parse_args()

    if args.plot is not None:
        if "_cut." in args.plot:
            extra_string = "_cut"
        elif "_cut4." in args.plot:
            extra_string = "_cut4"
        else:
            extra_string = ""

        with open(args.plot, 'rb') as f:
            data = pickle.load(f)
            ani_energies = data['ani']
            dftd3_energies = data['dftd3']
            atoms = data['atoms']
        fig, ax = plt.subplots()
        ax.scatter(dftd3_energies, ani_energies, s=0.5)
        ax.set_xlabel('DFTD3 energies (Ha)')
        ax.set_ylabel('ANI D3 energies (Ha)')
        ax.set_xlim(min(dftd3_energies), max(dftd3_energies))
        ax.set_ylim(min(dftd3_energies), max(dftd3_energies))

        plt.savefig(f'dftd3{extra_string}.png')

        atoms = torch.tensor(atoms)
        ani_energies = torch.tensor(ani_energies)
        dftd3_energies = torch.tensor(dftd3_energies)
        mae = torch.abs(ani_energies - dftd3_energies).mean()
        rmse = torch.sqrt((ani_energies - dftd3_energies).pow(2).mean())
        relative_error = torch.abs((ani_energies - dftd3_energies) / dftd3_energies) * 100

        fig, ax = plt.subplots()
        ax.scatter(dftd3_energies, relative_error, s=0.5)
        ax.set_xlabel('DFTD3 energies (Ha)')
        ax.set_ylabel(r'Relative error  $(E_{ani-d3} - E_{dftd3})/E_{dftd3}$ (%)')
        ax.set_xlim(min(dftd3_energies), max(dftd3_energies))
        ax.set_ylim(min(relative_error), max(relative_error))
        plt.savefig(f'dftd3_error{extra_string}.png')

        fig, ax = plt.subplots()
        ax.scatter(atoms, relative_error, s=0.5)
        ax.set_xlabel('Num. atoms')
        ax.set_ylabel(r'Relative error  $(E_{ani-d3} - E_{dftd3})/E_{dftd3}$ (%)')
        ax.set_xlim(min(atoms) - 1, max(atoms) + 1)
        ax.set_ylim(min(relative_error), max(relative_error))
        plt.savefig(f'dftd3_error_size{extra_string}.png')

        print('MAE', mae)
        print('RMSE', rmse)
        print('mean relative error', relative_error.mean(), ' %')
        exit()

    dataset = torchani.data.load(args.dataset_path).species_to_indices(
        "periodic_table").shuffle().collate(1).cache()

    pickle_name = 'dispersion_energies'
    if args.cutoff_smooth:
        pickle_name += '_cut'
        cutoff_function = CutoffSmooth(8.0)
    elif args.cutoff_smooth4:
        pickle_name += '_cut4'
        cutoff_function = CutoffSmooth(8.0, order=4)
    else:
        cutoff_function = None
    pickle_name += '.pkl'
    disp = StandaloneDispersionD3(neighborlist_cutoff=8.0, cutoff_function=cutoff_function).to(args.device)
    try:
        tmp_df_file = Path(__file__).resolve().parent.joinpath('.dftd3par.local')
        assert not tmp_df_file.exists()
        with open(tmp_df_file, 'w') as f:
            f.write('1.000 0.0000 0.2641 5.4959 14 4')
        ani_energies = []
        dftd3_energies = []
        atoms = []
        for i, properties in enumerate(dataset):
            if not args.all_molecules:
                if i == args.num_molecules:
                    break
            species = properties['species'].to(args.device)
            coordinates = properties['coordinates'].to(args.device).float()
            assert coordinates.shape[0] == 1
            assert species.shape[0] == 1
            num_atoms = species.shape[1]
            species_coordinates = (species, coordinates)
            start = time.time()
            ani_dftd3_energy = disp(species_coordinates).energies.item()
            end = time.time()
            time_for_ani = end - start
            tmpfile = Path(__file__).parent.resolve().joinpath('tmp')
            assert not tmpfile.exists()
            tensor_to_xyz(tmpfile, species_coordinates)

            p = subprocess.run(f'time dftd3 {tmpfile.as_posix()}'.split(), capture_output=True)
            out_string = p.stdout.decode('ascii')
            match = re.findall(r'time elapsed:(.*?)\n', out_string, re.MULTILINE)
            assert len(match) == 1, f"more than 1 match for time in {match}"
            time_for_dftd3 = float(match[0])

            match = re.findall(r'^ Edisp /kcal,au.*?\n', out_string, re.MULTILINE)
            if not match:
                print('match not found')
                print('out string:', out_string)
                with open(tmpfile, 'r') as f:
                    xyz = f.read()
                print('xyz file:', xyz)
                exit()
            assert len(match) == 1, f"more than 1 match for dftd3 dispersion energy in {match}"

            dftd3_energy = match[0].split()[-1]
            dftd3_energy = float(dftd3_energy)
            atoms.append(num_atoms)
            ani_energies.append(ani_dftd3_energy)
            dftd3_energies.append(dftd3_energy)
            print(ani_dftd3_energy, dftd3_energy, time_for_ani, time_for_dftd3)
            tmpfile.unlink()
        with open(pickle_name, 'wb') as f:
            pickle.dump({'ani': ani_energies, 'dftd3': dftd3_energies, 'atoms': atoms}, f)
    except Exception as ex:
        raise ex
    finally:
        if tmp_df_file.exists():
            tmp_df_file.unlink()
        if tmpfile.exists():
            tmpfile.unlink()
