import torch
import torchani

# save whole model and the individual ensemble members
model = torchani.models.ANI2x(periodic_table_index=True, use_neurochem_source=True)
torch.save(model.state_dict(), './ani2x_state_dict.pt')
torch.save(model.aev_computer.state_dict(), './aev_2x_state_dict.pt')
for j in range(8):
    m = model[j]
    torch.save(m.state_dict(), f'./ani2x_{j}_state_dict.pt')

model = torchani.models.ANI1x(periodic_table_index=True, use_neurochem_source=True)
torch.save(model.state_dict(), './ani1x_state_dict.pt')
torch.save(model.aev_computer.state_dict(), './aev_1x_state_dict.pt')
for j in range(8):
    m = model[j]
    torch.save(m.state_dict(), f'./ani1x_{j}_state_dict.pt')

model = torchani.models.ANI1ccx(periodic_table_index=True, use_neurochem_source=True)
torch.save(model.state_dict(), './ani1ccx_state_dict.pt')
torch.save(model.aev_computer.state_dict(), './aev_1ccx_state_dict.pt')
for j in range(8):
    m = model[j]
    torch.save(m.state_dict(), f'./ani1ccx_{j}_state_dict.pt')
