r"""Mypy type aliases"""
from typing import Tuple, Dict, Union, Callable
from torch import Tensor
from pathlib import Path
from numpy import ndarray

KeyIdx = Tuple[str, Tensor]
Properties = Dict[str, Tensor]
PathLike = Union[str, Path]
Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]
NdarrayProperties = Dict[str, ndarray]
