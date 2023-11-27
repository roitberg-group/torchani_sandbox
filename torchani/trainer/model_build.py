import torch
import torchani
from torchani import atomics
from copy import deepcopy

class ModelBuilder:
    def __init__(self, config, device):
        self.config = config
        self.device = device
    

    def setup_architecture(self, aevsize)
"""
# Usage
builder = ModelBuilder(config, device)
aev_computer = builder.AEV_Computer()
energy_shifter = builder.Energy_Shifter()
model, nn, modules = builder.model_creator(aev_computer)
"""
