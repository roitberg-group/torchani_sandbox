r"""Adaptor that outputs only charges

Adaptor has a dual output layer, needed to
use the two-output state dicts.
"""
from collections import OrderedDict

import torch
from torch import Tensor


class ChargeNetworkAdaptor(torch.nn.ModuleDict):
    @staticmethod
    def ensureOrderedDict(modules):
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od

    def __init__(self, modules):
        super().__init__(self.ensureOrderedDict(modules))

    def forward(
        self,
        element_idxs: Tensor,
        aevs: Tensor
    ) -> Tensor:

        element_idxs_ = element_idxs.flatten()
        aevs = aevs.flatten(0, 1)

        output = aevs.new_zeros(element_idxs_.shape)

        for i, module in enumerate(self.values()):
            selected_idx = (element_idxs_ == i).nonzero().view(-1)
            if selected_idx.shape[0] > 0:
                input_ = aevs.index_select(0, selected_idx)
                output.index_add_(0, selected_idx, module(input_)[:, 1].view(-1))
        atomic_charges = output.view_as(element_idxs)
        return atomic_charges
