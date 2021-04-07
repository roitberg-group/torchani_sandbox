import torch
import torchani
from torchani.repulsion import RepulsionCalculator, StandaloneRepulsionCalculator

# This is an example of how to use the repulsion interactions coded in torchani
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# by default the dispersion interactions have no cutoff
rep = StandaloneRepulsionCalculator(periodic_table_index=True)

coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True, device=device)

species = torch.tensor([[6, 1, 1, 1, 1]], device=device)
rep_energy = rep((species, coordinates)).energies
derivative = torch.autograd.grad(rep_energy.sum(), coordinates)[0]
force = -derivative
print('Dispersion Energy:', rep_energy.item())
print('Force:', force.squeeze())
