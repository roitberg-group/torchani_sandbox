import torch
import typing as tp

from torch import Tensor

from torchani.models import BuiltinModel
from torchani.potentials import PotentialWrapper

Model = tp.Union[BuiltinModel, PotentialWrapper]


def energies_and_forces(
    model: Model,
    species: Tensor,
    coordinates: Tensor,
    cell: tp.Optional[Tensor] = None,
    pbc: tp.Optional[Tensor] = None,
    retain_graph: tp.Optional[bool] = None,
    create_graph: bool = False,
    keep_requires_grad: bool = False,
) -> tp.Dict[str, Tensor]:
    coordinates.requires_grad_(True)
    energies = model((species, coordinates), cell=cell, pbc=pbc).energies
    output = forces(
        energies,
        coordinates,
        retain_graph=retain_graph,
        create_graph=create_graph,
        keep_requires_grad=keep_requires_grad,
    )
    output.update({"energies": energies})
    return output


def forces(
    energies: Tensor,
    coordinates: Tensor,
    retain_graph: tp.Optional[bool] = None,
    create_graph: bool = False,
    keep_requires_grad: bool = False,
) -> tp.Dict[str, Tensor]:
    if not coordinates.requires_grad:
        raise ValueError("Coordinates input to this function must require grad")
    result = -torch.autograd.grad(
        energies.sum(),
        coordinates,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )[0]
    if not keep_requires_grad:
        coordinates.requires_grad_(False)
    return {"forces": result}
