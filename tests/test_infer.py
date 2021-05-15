import sys
import os
import torch
import torchani
from ase.io import read
import pytest


path = os.path.dirname(os.path.realpath(__file__))
# Disable Tensorfloat, errors between two run of same model for large system could reach 1e-3.
# However note that this error for large system is not that big actually.
torch.backends.cuda.matmul.allow_tf32 = False
use_mnps = [True, False] if torchani.infer.mnp_is_installed else [False]


@pytest.fixture(scope="module")
def ani2x():
    return torchani.models.ANI2x(periodic_table_index=True, model_index=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Infer model needs cuda is available")
@pytest.mark.parametrize("device", ['cuda', 'cpu'])
@pytest.mark.parametrize("use_mnp", use_mnps)
def test_bmm_ensemble(device, use_mnp, ani2x):
    model_iterator = ani2x.neural_networks
    aev_computer = torchani.AEVComputer.like_2x(use_cuda_extension=(device == 'cuda'))
    ensemble = torchani.nn.Sequential(aev_computer, model_iterator).to(device)
    bmm_ensemble = torchani.nn.Sequential(aev_computer, ani2x.neural_networks.to_infer_modle(use_mnp=use_mnp)).to(device)
    files = ['small.pdb', '1hz5.pdb', '6W8H.pdb']
    for file in files:
        filepath = os.path.join(path, f'../dataset/pdb/{file}')
        mol = read(filepath)
        species = torch.tensor([mol.get_atomic_numbers()], device=device)
        positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=device)
        speciesPositions = ani2x.species_converter((species, positions))
        species, coordinates = speciesPositions
        coordinates.requires_grad_(True)

        _, energy1 = ensemble((species, coordinates))
        force1 = torch.autograd.grad(energy1.sum(), coordinates)[0]
        _, energy2 = bmm_ensemble((species, coordinates))
        force2 = torch.autograd.grad(energy2.sum(), coordinates)[0]

        torch.testing.assert_allclose(energy1, energy2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_allclose(force1, force2, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Infer model needs cuda is available")
@pytest.mark.parametrize("device", ['cuda', 'cpu'])
@pytest.mark.parametrize("use_mnp", use_mnps)
def test_ani_infer_model(device, use_mnp, ani2x):
    model_iterator = ani2x.neural_networks
    aev_computer = torchani.AEVComputer.like_2x(use_cuda_extension=(device == 'cuda'))
    model_ref = torchani.nn.Sequential(aev_computer, model_iterator[0]).to(device)
    model_infer = torchani.nn.Sequential(aev_computer, model_iterator[0].to_infer_modle(use_mnp=use_mnp)).to(device)
    files = ['small.pdb', '1hz5.pdb', '6W8H.pdb']
    for file in files:
        filepath = os.path.join(path, f'../dataset/pdb/{file}')
        mol = read(filepath)
        species = torch.tensor([mol.get_atomic_numbers()], device=device)
        positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=device)
        speciesPositions = ani2x.species_converter((species, positions))
        species, coordinates = speciesPositions
        coordinates.requires_grad_(True)

        _, energy1 = model_ref((species, coordinates))
        force1 = torch.autograd.grad(energy1.sum(), coordinates)[0]
        _, energy2 = model_infer((species, coordinates))
        force2 = torch.autograd.grad(energy2.sum(), coordinates)[0]

        torch.testing.assert_allclose(energy1, energy2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_allclose(force1, force2, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main(sys.argv)
