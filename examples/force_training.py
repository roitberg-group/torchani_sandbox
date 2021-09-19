from pathlib import Path
from typing import Tuple, Dict

import torch
from torch import Tensor
import torch.utils.tensorboard
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchani
from torchani import datasets, transforms, training


class ForceRunner(training.Runner):
    def inner_loop(self, batch, metrics: Dict[str, Tensor]) -> Tuple[Tensor, int]:
        species = batch['species'].long()
        target_energies = batch['energies'].float()
        target_forces = batch['forces'].float()
        num_atoms = (species >= 0).sum(dim=1).float()
        with torch.enable_grad():
            with torch.autograd.detect_anomaly():
                coordinates = batch['coordinates'].float().requires_grad_(True)
                predicted_energies = self._model((species, coordinates)).energies
                predicted_forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
        force_coefficient = 0.1
        squared_energy_error = self._squared_error(predicted_energies, target_energies)
        energy_loss = squared_energy_error / num_atoms.sqrt()
        squared_force_error = self._squared_error(predicted_forces, target_forces).sum(dim=(1, 2))
        force_loss = squared_force_error / num_atoms
        batch_loss = energy_loss + force_coefficient * force_loss
        metrics['energy_rmse_hartree'] += squared_energy_error.detach().sum().item()
        metrics['energy_mae_hartree'] += torch.abs(predicted_energies.detach() - target_energies.detach()).sum().item()

        metrics['force_rmse_hartree_per_angstrom'] += force_loss.detach().sum().item()
        metrics['force_mae_hartree_per_angstrom'] += (torch.sqrt(squared_force_error.detach()) / num_atoms).sum().item()
        return batch_loss, species.shape[0]

    def set_extra_metrics(self):
        return {'energy_rmse_hartree': 0.0, 'energy_mae_hartree': 0.0, 'force_rmse_hartree_per_angstrom': 0.0, 'force_mae_hartree_per_angstrom': 0.0}


if __name__ == '__main__':
    ROOT_DATASET_PATH = Path('/media/samsung1TBssd/Git-Repos/Datasets')
    BATCH_SIZE = 2560
    SPLITS = {'training': 0.8, 'validation': 0.2}
    FOLDS = None
    DATASET_CLASS = datasets.ANI2x
    FUNCTIONAL = 'B973c'
    BASIS_SET = 'def2mTZVP'
    NUM_WORKERS = 2
    PREFETCH_FACTOR = 2
    SELECTED_PROPERTIES = {'energies', 'species', 'coordinates', 'forces'}
    DATASET_NAME = f"2x-{FUNCTIONAL}-{BASIS_SET}"
    VALIDATION_SPLIT = 'validation'
    TRAINING_SPLIT = 'training'
    BATCH_ALL_PROPERTIES = True
    training_set, validation_set = training.prepare_learning_sets(DATASET_CLASS, ROOT_DATASET_PATH, DATASET_NAME,
                                                                  BATCH_SIZE, NUM_WORKERS, PREFETCH_FACTOR,
                                                                  VALIDATION_SPLIT, TRAINING_SPLIT, FUNCTIONAL, BASIS_SET,
                                                                  SELECTED_PROPERTIES, SPLITS, FOLDS, BATCH_ALL_PROPERTIES)
    DEVICE = 'cuda'
    USE_CUAEV = True
    LOG_TENSORBOARD = True
    LOG_CSV = True
    RUNS_ROOT_DIR = '/media/samsung1TBssd/Git-Repos/torchani-runs'
    SET_NAME = f'trials-{DATASET_NAME}'
    SPECIFIC_RUN_NAME = 'cuaev-dispersion-b973c-actual-anomaly-2'
    # Model
    model = torchani.models.ANI2x(pretrained=False, model_index=0, use_cuda_extension=USE_CUAEV, periodic_table_index=True, repulsion=False, dispersion=True)
    # GSAEs
    #model.energy_shifter.self_energies = torch.tensor([-0.499321200000, -37.83383340000, -54.57328250000, -75.04245190000], dtype=torch.float)
    model.energy_shifter.self_energies = torch.tensor([-0.499321200000, -37.83383340000, -54.57328250000, -75.04245190000, -398.1577125334925, -99.80348506781634, -460.168193942], dtype=torch.float)
    # Transforms
    elements = model.get_chemical_symbols()
    # Optimizer
    INITIAL_LR = 1e-3
    WEIGHT_DECAY = 1e-7
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=WEIGHT_DECAY, lr=INITIAL_LR)
    # Lr scheduler
    MAX_EPOCHS = 1001  # exclusive
    EARLY_STOPPING_LR = 1.0e-6
    TRACK_METRIC = 'force_rmse_kcalpermol_per_angstrom'
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=50, threshold=1e-4)
    # Training / Validation runner, the runner tracks the best metric
    runner = ForceRunner(model, optimizer, transforms.Identity(), device=torch.device(DEVICE))
    # Checkpoint paths
    run_output_path = Path(f'{RUNS_ROOT_DIR}/{SET_NAME}/{SPECIFIC_RUN_NAME}/').resolve()
    persistent_objects = {'optimizer': optimizer, 'scheduler': scheduler, 'model': model, 'runner': runner}
    training.execute_training(persistent_objects,
                              scheduler,
                              runner,
                              optimizer,
                              training_set,
                              validation_set,
                              run_output_path,
                              TRACK_METRIC,
                              INITIAL_LR,
                              MAX_EPOCHS,
                              EARLY_STOPPING_LR,
                              LOG_TENSORBOARD,
                              LOG_CSV,
                              __file__)
