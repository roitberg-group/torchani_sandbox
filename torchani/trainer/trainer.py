import torch
import torchani
from torchani.transforms import AtomicNumbersToIndices, SubtractSAE
from pathlib import Path
class Data:
#I hate the function provided below. it is too messy. and FILLED with bugs. I think it has a bug with making the batched directory to None, which is insane. Stefan ran into this issue. This is a place holder for now as I think there is a better way to lay this out. Want groups opinion
# Also I do not like that the h5 has to be from dir and would like to change this
    def __init__(self, architecture):
        self.build = architecture
        #energy shifter; crrently only allowing to access the sorted gsae function in torchani utils. user can modify this dictionary rather than allowing a separate data file access?
        # also should we allow for training to sae? to reproduce published networks. seems unnecessary to me as the publsihed networks are outdated and readily available for use
        self.energy_shifter = torchani.utils.sorted_gsaes(self.build.inputs['elements'], self.build.inputs['functional'], self.build.inputs['basis_set'])
        self.training, self.validation = self.sample_data_loader()

    def sample_data_loader(self):
        # ds_path can either be a path or None
        # if it is a path, it can either exist or not
        # if it is None -> In memory
        # if it is an existing path -> use it
        # if it is a nonoe existing path -> create it, and then use it
        in_memory = self.build.inputs['batch_path'] is None
        transform = torchani.transforms.Compose([AtomicNumbersToIndices(self.build.inputs['elements']), SubtractSAE(self.build.inputs['elements'], self.energy_shifter)])
        if in_memory:
            learning_sets = torchani.datasets.create_batched_dataset(self.build.inputs['h5_path'],
                                        include_properties=self.build.inputs['include_properties'],
                                        batch_size=self.build.inputs['batch_size'],
                                        inplace_transform=transform,
                                        shuffle_seed=123456789,
                                        splits=self.build.inputs['data_split'], direct_cache=True)
            training = torch.utils.data.DataLoader(learning_sets['training'],
                                               shuffle=True,
                                               num_workers=1,
                                               prefetch_factor=2,
                                               pin_memory=True,
                                               batch_size=None)
            validation= torch.utils.data.DataLoader(learning_sets['validation'],
                                                 shuffle=False,
                                                 num_workers=1,
                                                 prefetch_factor=2, pin_memory=True, batch_size=None)
        else:
            if not Path(self.build.inputs['batch_path']).resolve().is_dir():
                h5 = torchani.datasets.ANIDataset.from_dir(self.build.inputs['h5_path'])
                torchani.datasets.create_batched_dataset(h5,
                                                 dest_path=self.build.inputs['batch_path'],
                                                 batch_size=self.build.inputs['batch_size'],
                                                 include_properties=self.build.inputs['include_properties'],
                                                 splits = self.build.inputs['data_split'])
            # This below loads the data if dspath exists
            training = torchani.datasets.ANIBatchedDataset(self.build.inputs['batch_path'], transform=transform, split='training')
            validation = torchani.datasets.ANIBatchedDataset(self.build.inputs['batch_path'], transform=transform, split='validation')
            training = torch.utils.data.DataLoader(training,
                                           shuffle=True,
                                           num_workers=1,
                                           prefetch_factor=2,
                                           pin_memory=True,
                                           batch_size=None)
            validation = torch.utils.data.DataLoader(validation,
                                             shuffle=False,
                                             num_workers=1,
                                             prefetch_factor=2,
                                             pin_memory=True,
                                             batch_size=None)
        return training, validation

class Trainer:

    def __init__(self, architecture, data):
        self.build = architecture
        self.training = data.training
        self.validation = data.validation
    
    def optimizer(self):
        params = []
        for mod in self.build.modules:
            for i in range(4): #Hard coding 4 in that default architecture is 4 linear layers. Can change to a more clever way for more variable architecture. 
                if 'weight_decay' in  self.build.inputs.keys():
                    if self.build.inputs['weight_decay'][i]:
                        params.append({'params': [mod[i+i].weight], 'weight_decay': self.build.inputs['weight_decay'][i]})
                    else:
                        params.append({'params': [mod[i+i].weight]})
                else:
                    params.append({'params': [mod[i+i].weight]})
        AdamW = torch.optim.AdamW(params)
        return AdamW

    def learning_rate_scheduler(self):
        if 'plateau_scheduler_params' in  self.build.inputs.keys():
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer(), **self.build.inputs['plateau_scheduler_params'])
            print(scheduler)
