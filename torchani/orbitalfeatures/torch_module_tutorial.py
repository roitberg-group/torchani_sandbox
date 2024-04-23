import typing as tp

import torch
from torch import Tensor

# Using classes as functions
class FunctionClass:
    def __call__(self, x):
        return x ** 2

    def another_method(self, x):
        return x ** 3

# Disadvantage of just having a function is you have no state

my_function = FunctionClass()


out = my_function(3)
print(out)

out = my_function.another_method(4)
print(out)

# Using classes as functions with state, so you keep the parameters inside
class PowerTaker:
    def __init__(self, a: int = 2) -> None:
        self.a = a

    def __call__(self, x):
        return x ** self.a

pow_taker = PowerTaker(4)

out = pow_taker(3)
print(out)

pow_taker.a = 2


out = pow_taker(3)
print(out)


# In torch you need this
class CustomModule(torch.nn.Module):  # This has everything that torch.nn.Module has
    # You NEED to call init
    def __init__(self) -> None:
        # torch.nn.Module needs to initialize some stuff, so you need this
        super().__init__()  # This calls the init of the superclass

    # This is the equivalent of "__call__"
    def forward(self, x: Tensor) -> Tensor:
        return x ** 2


mod = CustomModule()


input_ = torch.arange(4)

output = mod(input_)

print(output)


# What you need to do:
class JonyCustomModule(torch.nn.Module):
    def __init__(self, factor) -> None:
        super().__init__()
        # Initialize all params you need here
        self._factor = factor

    def forward(self, x: Tensor) -> Tensor:  # Inputs and outputs that the "function" computes
        # The actual calculation
        out = self._factor * x ** 2
        # Inside here you can call the aux functions

        out = self._calculate_aux_tensors(out)
        return out

    # Auxiliary functions to perform the calculation
    # Just to organize code
    def _calculate_aux_tensors(self, x: Tensor) -> Tensor:
        return x ** 3 + torch.sqrt(x)
