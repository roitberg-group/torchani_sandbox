import torch
import torchani

def ANI2x_Repulsion_Model():
    elements = ("H", "C", "N", "O", "S", "F", "Cl")

    def dispersion_atomics(atom: str = "H"):
        dims_for_atoms = {
            "H": (1008, 256, 192, 160),
            "C": (1008, 256, 192, 160),
            "N": (1008, 192, 160, 128),
            "O": (1008, 192, 160, 128),
            "S": (1008, 160, 128, 96),
            "F": (1008, 160, 128, 96),
            "Cl": (1008, 160, 128, 96),
        }
        return torchani.atomics.standard(
            dims_for_atoms[atom], activation=torch.nn.GELU(), bias=False
        )

    model = torchani.models.ANI2x(
        pretrained=False,
        cutoff_fn="smooth",
        atomic_maker=dispersion_atomics,
        ensemble_size=7,
        repulsion=True,
        repulsion_kwargs={
            "symbols": elements,
            "cutoff": 5.1,
            "cutoff_fn": torchani.aev.cutoffs.CutoffSmooth(order=2),
        },
        periodic_table_index=True,
        model_index=None,
        cell_list=False,
        use_cuda_extension=False,
    )
    state_dict = torchani.models._fetch_state_dict(
        "anid_state_dict_mod.pt", private=True
    )
    for key in state_dict.copy().keys():
        if key.startswith("potentials.0"):
            state_dict.pop(key)
    for key in state_dict.copy().keys():
        if key.startswith("potentials.1"):
            new_key = key.replace("potentials.1", "potentials.0")
            state_dict[new_key] = state_dict[key]
            state_dict.pop(key)
    for key in state_dict.copy().keys():
        if key.startswith("potentials.2"):
            new_key = key.replace("potentials.2", "potentials.1")
            state_dict[new_key] = state_dict[key]
            state_dict.pop(key)

    model.load_state_dict(state_dict, strict=True)
    # setup repulsion calculator
    model.rep_calc = model.potentials[0]

    return model
