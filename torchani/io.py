import torch
import typing as tp
from pathlib import Path

from torch import Tensor

from torchani.utils import ATOMIC_NUMBERS, pad_atomic_properties


def read_xyz(
    path: tp.Union[str, Path],
    coordinates_dtype: torch.dtype = torch.float,
    device: tp.Union[torch.device, tp.Literal["cpu", "cuda"]] = "cpu",
) -> tp.Tuple[Tensor, Tensor]:
    r"""
    Read an xyz file with possibly many coordinates and species and return a
    (species, coordinates) tuple of tensors. The shapes of the tensors are (C,
    A) and (C, A, 3) respectively, where C is the number of conformations, A
    the maximum number of atoms (conformations with less atoms are padded with
    species=-1 and coordinates=0.0).
    """
    path = Path(path).resolve()
    properties: tp.List[tp.Dict[str, Tensor]] = []
    with open(path, mode="rt", encoding="utf-8") as f:
        lines = iter(f)
        while True:
            species = []
            coordinates = []
            try:
                num = int(next(lines))
            except StopIteration:
                break
            next(lines)  # xyz comment
            for _ in range(num):
                line = next(lines)
                s, x, y, z = line.split()
                if s in ATOMIC_NUMBERS:
                    species.append(ATOMIC_NUMBERS[s])
                else:
                    species.append(int(s))
                coordinates.append([float(x), float(y), float(z)])
            properties.append(
                {
                    "coordinates": torch.tensor(
                        [coordinates],
                        dtype=coordinates_dtype,
                        device=device,
                    ),
                    "species": torch.tensor(
                        [species],
                        dtype=torch.long,
                        device=device,
                    ),
                }
            )
    pad_properties = pad_atomic_properties(properties)
    return pad_properties["species"], pad_properties["coordinates"]
