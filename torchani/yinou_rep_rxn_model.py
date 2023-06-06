from pathlib import Path
from collections import OrderedDict

import torch
import torchani
from torchani import atomics
from torchani.utils import EnergyShifter, sorted_gsaes
from torchani.units import HARTREE_TO_KCALMOL

def ANIRepulsionEnsemble(path_for_state_dicts: str, functional: str, basis_set: str, num_models: int, device):
    # Load ANI ensemble with dispersion (use dispersion_calculation branch)
    # path_for_state_dicts: path to store best*.pt files
    # functional: "wb97x", "b973c", case insensitive
    # basis_set: "631gd", "def2mtzvp", case insensitive
    def atomic_maker(atom: str = "H"):
        dims_for_atoms = {
            "H": (384, 160, 128, 96),
            "C": (384, 144, 112, 96),
            "N": (384, 128, 112, 96),
            "O": (384, 128, 112, 96),
        }
        return atomics.standard(
            dims_for_atoms[atom], activation=torch.nn.GELU(), bias=False
        )

    elements = ("H", "C", "N", "O")
    model = torchani.models.ANI1x(
        pretrained=False,
        cutoff_fn="smooth",
        atomic_maker=atomic_maker,
        ensemble_size=num_models,
        repulsion=True,
        repulsion_kwargs={"symbols": elements, "cutoff": 5.2},
        #periodic_table_index=True,
    ).to(device)
    path = Path(path_for_state_dicts).resolve()
    for f in sorted(path.iterdir()):
        if (not f.suffix == ".pt") or (not f.name.startswith("best")):
            continue
        index = int(f.stem.split("_")[-1])
        old_state_dict = torch.load(f, map_location=torch.device("cpu"))

        # This snippet of code creates a new state dict with the correct keys
        # instead of 0.0.weight -> H.0.weight for example
        state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            new_key = k.split(".")
            new_key[0] = elements[int(new_key[0])]
            new_key = ".".join(new_key)
            state_dict[new_key] = v

        model.neural_networks[index].load_state_dict(state_dict)


    model.energy_shifter = EnergyShifter(sorted_gsaes(elements, functional, basis_set))

    return model


def ANIRepulsionModel(state_dict_file: str, functional: str, basis_set: str, device):

    def atomic_maker(atom: str = "H"):
        dims_for_atoms = {
            "H": (384, 160, 128, 96),
            "C": (384, 144, 112, 96),
            "N": (384, 128, 112, 96),
            "O": (384, 128, 112, 96),
        }
        return atomics.standard(
            dims_for_atoms[atom], activation=torch.nn.GELU(), bias=False
        )

    elements = ("H", "C", "N", "O")
    model = torchani.models.ANI1x(
        pretrained=False,
        cutoff_fn="smooth",
        atomic_maker=atomic_maker,
        ensemble_size=1,
        repulsion=True,
        repulsion_kwargs={"symbols": elements, "cutoff": 5.2},
        #periodic_table_index=True,
    ).to(device)

    state_dict = torch.load(state_dict_file)
    new_state_dict = OrderedDict()
    elements = ("H", "C", "N", "O")
    for k, v in state_dict.items():
        key_tokens = k.split(".")
        new_key = f"0.{elements[int(key_tokens[0])]}.{'.'.join(key_tokens[1:])}"
        new_state_dict[new_key] = v
    model.neural_networks.load_state_dict(new_state_dict)

    model.energy_shifter = EnergyShifter(sorted_gsaes(elements, functional, basis_set))

    return model
