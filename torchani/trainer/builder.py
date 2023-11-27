import torchani
import config

class ModelBuilder:

    def __init__(self, train_config):
        self.inputs = train_config.inputs
        self.aev_computer = torchani.AEVComputer(**torchani.neurochem.Constants(self.inputs['constants']))
        
    def forward(self):
        if self.inputs['netlike1x']:
            print('yes')
            modules = [torchani.atomics.like_1x(a) for a in self.inputs['elements']]
            #modules = [torchani.atomics.like_1x(a, aev_dim=self.aev_computer.aev_length) for a in self.inputs['elements']] ##updating atomics to look like this, currently only default
        print(modules)
