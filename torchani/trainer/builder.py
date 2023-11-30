import torch
from torch.utils import tensorboard
import torchani
import config
import datetime
import os
import shutil


class Builder:

    def __init__(self, train_config, device):
        self.device = device
        self.inputs = train_config.inputs
        self.path = train_config.path
        self.time = datetime.datetime.now()
        self.aev_computer = torchani.AEVComputer(**torchani.neurochem.Constants(self.inputs['constants']))
        self.modules= self.architecture()
        self.nn, self.model = self.standard_model()
        self.board, self.latest, self.best = self.log_setup()


    def architecture(self):
        # Method flawed, as either netlike1x or netlike2x must be true. however if user wants to use specific params that are not defaulted in neurochem, they would label both as False. This should be modified or custom params is not allowed in trainer, and further modification by user must be done
        if self.inputs['netlike1x']:
            modules = [torchani.atomics.like_1x(a, bias=self.inputs['bias']) for a in self.inputs['elements']]
            #modules = [torchani.atomics.like_1x(a, aev_dim=self.aev_computer.aev_length) for a in self.inputs['elements']] ##updating atomics to look like this, currently only default
        if self.inputs['netlike2x']:
            modules = [torchani.atomics.like_2x(a, bias=self.inputs['bias']) for a in self.inputs['elements']]
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
        model = torchani.nn.Sequential(self.aev_computer, nn).to(self.device)
        return nn, model

    def log_setup(self): 
        # Return at end to insure copying of all necessary input. Need a clever way of saving the ini file. Maybe it must be called editor? Can save ini path and return to this class as a self variable
        log = '{}{}_{}'.format(self.inputs['logdir'], self.time.strftime("%Y%m%d_%H%M"), self.inputs['projectlabel'])
        print(log)
        assert os.path.isdir(log)==False, "Oops! This project sub-directory already exists."
        if not os.path.isdir(log):
            print('Creating your log sub-directory.')
            os.makedirs(log)
        training_writer = tensorboard.SummaryWriter(log_dir='{}/train'.format(log))
        latest_checkpoint = '{}/latest.pt'.format(log)
        best_checkpoint = '{}/best.pt'.format(log)
        shutil.copyfile(self.path, '{}/config.ini'.format(log))
        return training_writer, latest_checkpoint, best_checkpoint

    

