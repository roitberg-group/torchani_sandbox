r"""Mypy type aliases"""
from typing import Tuple, Dict, Union, Callable, TypeVar, Any
from torch import Tensor
from pathlib import Path
from numpy import ndarray, dtype
from collections import OrderedDict
import sys

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

Properties = Dict[str, Tensor]
NumpyProperties = Dict[str, 'ndarray[Any, Any]']
MaybeNumpyProperties = TypeVar('MaybeNumpyProperties', NumpyProperties, Properties)

PathLike = Union[str, Path]
PathLikeODict = Union['OrderedDict[str, str]', 'OrderedDict[str, Path]']
