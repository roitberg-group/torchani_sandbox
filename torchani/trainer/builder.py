import torchani
import config

class ModelBuilder:

    def __init__(self, train_config, device):
        self.device = device
        self.inputs = train_config.inputs
        self.aev_computer = torchani.AEVComputer(**torchani.neurochem.Constants(self.inputs['constants']))
        self.modules= self.architecture()

    def architecture(self):
        # Method flawed, as either netlike1x or netlike2x must be true. however if user wants to use specific params that are not defaulted in neurochem, they would label both as False. This should be modified or custom params is not allowed in trainer, and further modification by user must be done
        if self.inputs['netlike1x']:
            modules = [torchani.atomics.like_1x(a) for a in self.inputs['elements']]
            #modules = [torchani.atomics.like_1x(a, aev_dim=self.aev_computer.aev_length) for a in self.inputs['elements']] ##updating atomics to look like this, currently only default
        if self.inputs['netlike2x']:
            modules = [torchani.atomics.like_2x(a) for a in self.inputs['elements']]
            #modules = [torchani.atomics.like_2x(a, aev_dim=self.aev_computer.aev_length) for a in self.inputs['elements']] ##updating atomics to look like this, currently only default
        return modules

    def init_params(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, a=1.0)
            if self.inputs['bias'] == False:
                None
            else: 
                torch.nn.init.zeros_(m.bias)
    
    def standard_model(self):
        nn = torchani.ANIModel(self.modules)
        nn.apply(self.init_params)
        model = torchani.nn.Sequential(self.aev_computer, nn).to(device)
        return nn, model

