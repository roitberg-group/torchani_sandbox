import torch
from torchani.md import TorchLangevin
import torchani

# THis class runs dynamics fully on the GPU, without needing to send tensors to the
# CPU every step, like when using the ASE interface

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchani.models.ANI2x(periodic_table_index=True).to(device).double()
coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True, device=device).double()
species = torch.tensor([[6, 1, 1, 1, 1]], device=device)
coordinates = coordinates.repeat(2, 1, 1)
species = species.repeat(2, 1)

# T in K, gamma in 1/fs, timestep in fs
dyn = TorchLangevin(model, species, coordinates, 300, 0.02, 0.1)

dyn.run(10000, print_every=10)
