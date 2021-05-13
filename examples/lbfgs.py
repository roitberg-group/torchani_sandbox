import torch
import torchani

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchani.models.ANI2x(periodic_table_index=True).to(device)

for p in model.parameters():
    p.requires_grad_(False)

coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True, device=device)
species = torch.tensor([[6, 1, 1, 1, 1]], device=device)
opt = torch.optim.LBFGS((coordinates,))


def closure():
    opt.zero_grad()
    energy = model((species, coordinates)).energies
    energy.backward()
    return energy


opt.step(closure)

print(coordinates)
