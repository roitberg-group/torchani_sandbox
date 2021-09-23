import torch
import torchani
from pathlib import Path


def save_model_and_parts(model, name):
    suffix = name[3:]
    path = Path(f'./state_dicts/{name}').resolve()
    path.mkdir(parents=True)
    torch.save(model.state_dict(), path.joinpath(f'{name}_state_dict.pt'))
    torch.save(model.aev_computer.angular_terms.state_dict(), path.joinpath(f'angular_{suffix}_state_dict.pt'))
    torch.save(model.aev_computer.radial_terms.state_dict(), path.joinpath(f'radial_{suffix}_state_dict.pt'))
    for j in range(len(model)):
        m = model[j]
        torch.save(m.state_dict(), path.joinpath(f'{name}_{j}_state_dict.pt'))


if __name__ == '__main__':
    # save whole model and the individual ensemble members
    save_model_and_parts(torchani.models.ANI1x(periodic_table_index=True, use_neurochem_source=True), 'ani1x')
    save_model_and_parts(torchani.models.ANI2x(periodic_table_index=True, use_neurochem_source=True), 'ani2x')
    save_model_and_parts(torchani.models.ANI1ccx(periodic_table_index=True, use_neurochem_source=True), 'ani1ccx')
