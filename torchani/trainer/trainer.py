import torch
import torchani
from torchani.transforms import AtomicNumbersToIndices, SubtractSAE
from pathlib import Path
import os
import math
import tqdm 

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
        self.energy_shifter = data.energy_shifter
        self.training = data.training
        self.validation = data.validation
        self.mse_sum = torch.nn.MSELoss(reduction='sum')
        self.mse = torch.nn.MSELoss(reduction='none')
    
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
        else: 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer())
        return scheduler

    def save_model(self, checkpoint):
        torch.save({
            'model': self.build.nn.state_dict(), 
            'AdamW': self.optimizer().state_dict(), 
            'self_energies': self.energy_shifter, 
            'AdamW_scheduler': self.learning_rate_scheduler()
            }, checkpoint)

    
    def _energy_mse(self, true, predicted):
        return self.mse_sum(predicted, true).item()

    def _force_mse(self, true, predicted, num_atoms):
        return (self.mse(true, predicted).sum(dim=(1,2))/(3*num_atoms)).sum()

    def _rmse_(self, total_mse, count): 
        return torchani.units.hartree2kcalmol(math.sqrt(total_mse/count))
    
    def validate(self):
        val_dict = {}
        forces  = self.build.inputs.get('forces') if 'forces' in self.build.inputs.keys() and self.build.inputs['forces'] else None
        
        total_energy_mse = 0.0
        total_force_mse = 0.0 if forces else None
        count = 0
        for i, properties in tqdm.tqdm(
                    enumerate(self.validation),
                    total=len(self.validation),
                    desc="Validating"):
            species = properties['species'].to(self.build.device)
            coordinates = properties['coordinates'].to(self.build.device).float().requires_grad_(True)
            true_energies = properties['energies'].to(self.build.device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            
            _, predicted_energies = self.build.model((species, coordinates))
            
            count += true_energies.shape[0]
            total_energy_mse += self._energy_mse(true_energies, predicted_energies)

            if forces:
                true_forces = properties['forces'].to(self.build.device).float()
                predicted_forces = -torch.autograd.grad(predicted_energies.sum(), coordinates)[0]
                total_force_mse += self._force_mse(true_forces, predicted_forces, num_atoms)

        
        val_dict['energy_rmse'] = self._rmse_(total_energy_mse, count)
        val_dict['forces_rmse'] = self._rmse_(total_force_mse, count) if forces else None
        return val_dict
    
    def _energy_loss(self, true, predicted, num_atoms):
        return (self.mse(predicted, true)/num_atoms.sqrt()).mean()

    def _force_loss(self, true, predicted, num_atoms):
        return (self.mse(true, predicted).sum(dim=(1,2)) / (3.0*num_atoms)).mean()

    def train(self):
        ## Provide option for MTL? what do we call it? Logorithmic loss training? 
        optimizer = self.optimizer()
        scheduler = self.learning_rate_scheduler()
        max_epochs = self.build.inputs['max_num_epochs'] if 'max_num_epochs' in self.build.inputs.keys() else 2000
        early_LR = self.build.inputs['early_stop_learning_rate'] if 'early_stop_learning_rate' in self.build.inputs.keys() else 1.0e-7
        writer = self.build.board
        forces  = self.build.inputs.get('forces') if 'forces' in self.build.inputs.keys() and self.build.inputs['forces'] else None

        print("Training starting from epoch {}.".format(scheduler.last_epoch))

        for _ in range(scheduler.last_epoch+1, max_epochs):
            validation_metrics = self.validate()
            for k,v in validation_metrics.items():
                if v:
                    writer.add_scalar(k, v, scheduler.last_epoch)
            
            learning_rate = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', learning_rate, scheduler.last_epoch)
            if learning_rate < early_LR:
                print("Training ending. Minimum learning rate reached.")
                break

            criteria = validation_metrics['energy_rmse']
            if scheduler.is_better(criteria, scheduler.best):
                print ("Saving the model, epoch={}, RMSE={}.".format(scheduler.last_epoch, criteria))
                self.save_model(self.build.best)
                for k,v in validation_metrics.items():
                    if v:
                        writer.add_scalar('best_{}'.format(k), v, scheduler.last_epoch)
            scheduler.step(criteria)

            for i, properties in tqdm.tqdm(
                    enumerate(self.training), 
                    total=len(self.training), 
                    desc="epoch {}".format(scheduler.last_epoch)):
                
                species = properties['species'].to(self.build.device)
                coordinates = properties['coordinates'].to(self.build.device).float().requires_grad_(True)
                true_energies = properties['energies'].to(self.build.device).float()
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)

                _, predicted_energies = self.build.model((species, coordinates))
                loss = self._energy_loss(true_energies, predicted_energies, num_atoms)

                if forces:
                    true_forces = properties['forces'].to(self.build.device).float()
                    predicted_forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                    loss += self._force_loss(true_forces, predicted_forces, num_atoms)

            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('batch_loss', loss, i)

            self.save_model(self.build.latest)
            
        print("Training ending. Maximum epochs reached.")





            
            

