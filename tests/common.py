import typing as tp
import unittest
from itertools import product

import torch
from torch import Tensor
from parameterized import parameterized_class

from torchani.testing import TestCase


def jit(model: torch.nn.Module) -> torch.nn.Module:
    # isinstance(torch.nn.Module(), torch.jit.ScriptModule) == True
    return torch.jit.script(model)


def get_cls_name(cls: type, idx: int, params: tp.Dict[str, tp.Any]) -> str:
    return f"{cls.__name__}_{params['device']}{'_jit' if params['jit'] else ''}"


@parameterized_class(
    ("device", "jit"),
    product(("cpu", "cuda"), (False, True)),
    class_name_func=get_cls_name,
)
class TorchaniTest(TestCase):
    device: str
    jit: bool

    def _setup(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.jit:
            return torch.jit.script(model)
        return model

    def testEqual(self):
        tensor = torch.tensor(1.0)

        class CustomModule(torch.nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x

        m = self._setup(CustomModule())
        self.assertEqual(1.0, m(tensor).item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
