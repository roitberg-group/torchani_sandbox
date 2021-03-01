import ase
import ase.io
import ase.md
import argparse
import torchani
import autonvtx
import torch

parser = argparse.ArgumentParser()
parser.add_argument('filename', help="file for the molecule")
args = parser.parse_args()

molecule = ase.io.read(args.filename)
model = torchani.models.ANI1x()[0].cuda()
calculator = model.ase()
molecule.set_calculator(calculator)
dyn = ase.md.verlet.VelocityVerlet(molecule, timestep=1 * ase.units.fs)

dyn.run(1000)  # warm up


def time_functions_in_model(model, function_names_list):
    # Wrap all the functions from "function_names_list" from the model
    # "model" with a timer
    for n in function_names_list:
        setattr(model, n, time_func(n, getattr(model, n)))


def time_func(key, func):

    def wrapper(*args, **kwargs):
        torch.cuda.nvtx.range_push(key)
        ret = func(*args, **kwargs)
        torch.cuda.nvtx.range_pop()
        return ret

    return wrapper


# enable timers
functions_to_time_aev = ['_compute_radial_aev', '_compute_angular_aev', '_compute_difference_vector',
                         '_compute_aev', '_triple_by_molecule']
functions_to_time_neighborlist = ['_full_pairwise', '_full_pairwise_pbc']

timers = {k: 0.0 for k in functions_to_time_aev + functions_to_time_neighborlist}

aev_computer = model.aev_computer
time_functions_in_model(aev_computer, functions_to_time_aev)
time_functions_in_model(aev_computer.neighborlist, functions_to_time_neighborlist)

torch.cuda.cudart().cudaProfilerStart()
autonvtx(model)
with torch.autograd.profiler.emit_nvtx(record_shapes=True):
    dyn.run(10)
