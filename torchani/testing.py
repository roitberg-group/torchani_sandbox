import typing as tp
from itertools import product

import torch
from parameterized import parameterized_class
from torch.testing._internal.common_utils import TestCase, make_tensor  # noqa: F401


def _get_cls_name(cls: type, idx: int, params: tp.Dict[str, tp.Any]) -> str:
    return f"{cls.__name__}_{params['device']}{'_jit' if params['jit'] else ''}"


def expand(device: tp.Union[tp.Literal["cpu"], tp.Literal["cuda"], None] = None, jit: tp.Optional[bool] = None):
    if device not in (None, "cpu", "cuda"):
        raise ValueError("Device must be None or one of 'cpu', 'cuda'")
    _device = ("cpu", "cuda") if device is None else (device,)
    _jit = (True, False) if jit is None else (jit,)
    decorator = parameterized_class(
        ("device", "jit"),
        product(_device, _jit),
        class_name_func=_get_cls_name,
    )
    return decorator


@expand()
class TorchaniTest(TestCase):
    device: str
    jit: bool

    def _setup(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.jit:
            return torch.jit.script(model)
        return model

    def testName(self) -> None:
        pass


__all__ = ["make_tensor", "TestCase", "TorchaniTest", "expand"]
