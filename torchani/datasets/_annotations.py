r"""Mypy type aliases"""
import sys
from typing import Tuple, Dict, Union, Callable, TypeVar
from torch import Tensor
from pathlib import Path
from numpy import ndarray, dtype
from collections import OrderedDict

# This is needed for compatibility with python 3.6, where numpy typing doesn't
# work correctly
if sys.version_info[:2] < (3, 7):
    # This doesn't really matter anyways since it is only for mypy
    DTypeLike = dtype
else:
    from numpy import typing as numpy_typing
    DTypeLike = numpy_typing.DTypeLike

KeyIdx = Tuple[str, Tensor]
Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]

Conformers = Dict[str, Tensor]
NumpyConformers = Dict[str, ndarray]
MaybeNumpyConformers = TypeVar('MaybeNumpyConformers', NumpyConformers, Conformers)

PathLike = Union[str, Path]
PathLikeODict = Union['OrderedDict[str, str]', 'OrderedDict[str, Path]']
