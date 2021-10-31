from torchani import atomics
from collections import OrderedDict
import torch
from torch import Tensor


class StaticAtomicModule(torch.nn.Module):
    def __init__(self, modules, species):
        super().__init__()
        # currently species has to be given as 0 1 2 3, this is a static module
        # that can be cast as cuda graph
        self.species = species.flatten()
        self.register_buffer('H_idx', (self.species == 0).nonzero().view(-1))
        self.register_buffer('C_idx', (self.species == 1).nonzero().view(-1))
        self.register_buffer('N_idx', (self.species == 2).nonzero().view(-1))
        self.register_buffer('O_idx', (self.species == 3).nonzero().view(-1))
        self.register_buffer('output', torch.zeros(self.species.shape))
        self._networks = torch.nn.ModuleDict(modules)

    def forward(self, aev: Tensor) -> Tensor:
        output = self.output.clone()
        output.index_add_(0, self.H_idx, self._networks['H'](aev.index_select(0, self.H_idx)).view(-1))
        output.index_add_(0, self.C_idx, self._networks['C'](aev.index_select(0, self.C_idx)).view(-1))
        output.index_add_(0, self.N_idx, self._networks['N'](aev.index_select(0, self.N_idx)).view(-1))
        output.index_add_(0, self.O_idx, self._networks['O'](aev.index_select(0, self.O_idx)).view(-1))
        return output.unsqueeze(0).sum(dim=1)


atomic_maker = atomics.like_1x
elements = ['H', 'C', 'N', 'O']
atomic_networks = OrderedDict([(e, atomic_maker(e)) for e in elements])

species = torch.tensor([[0, 0, 0, 0, 1]])
m = StaticAtomicModule(atomic_networks, species)
aev = torch.randn((5, 384))
aev.requires_grad_(True)
energy = m(aev).sum()
energy.backward()
print(aev.grad)

m = torch.cuda.make_graphed_callables(m, (aev,))
aev = aev.detach().clone()
aev.requires_grad_(True)
energy = m(aev).sum()
energy.backward()
print(aev.grad)
