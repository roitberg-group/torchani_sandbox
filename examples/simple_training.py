from pathlib import Path
from typing import Tuple, Dict

import torch
from torch import Tensor
import torch.utils.tensorboard
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchani
from torchani import datasets, transforms, training


class EnergyRunner(training.Runner):
    def inner_loop(self, batch, metrics: Dict[str, Tensor], train: bool = False) -> Tuple[Tensor, int]:
        species = batch['species'].long()
        coordinates = batch['coordinates'].float()
        target_energies = batch['energies'].float()
        num_atoms = (species >= 0).sum(dim=1).float()

        predicted_energies = self._model((species, coordinates)).energies

        squared_energy_error = self._squared_error(predicted_energies, target_energies)
        batch_loss = squared_energy_error / num_atoms.sqrt()

        metrics['energy_rmse_hartree'] += squared_energy_error.detach().sum().item()
        metrics['energy_mae_hartree'] += torch.abs(predicted_energies.detach() - target_energies.detach()).sum().item()
        return batch_loss, species.shape[0]

    def set_extra_metrics(self):
        return {'energy_rmse_hartree': 0.0, 'energy_mae_hartree': 0.0}


if __name__ == '__main__':
    ROOT_DATASET_PATH = Path('/media/samsung1TBssd/Git-Repos/Datasets')
    DATASET_NAME = '1x-plain'
    BATCH_SIZE = 2560
    SPLITS = {'training': 0.8, 'validation': 0.2}
    FOLDS = None
    DATASET_CLASS = datasets.ANI1x
    NUM_WORKERS = 2
    PREFETCH_FACTOR = 2
    SELECTED_PROPERTIES = {'energies', 'species', 'coordinates'}
    VALIDATION_SPLIT = 'validation'
    TRAINING_SPLIT = 'training'
    training_set, validation_set = training.prepare_learning_sets(DATASET_CLASS, ROOT_DATASET_PATH, DATASET_NAME,
                                                                  BATCH_SIZE, NUM_WORKERS, PREFETCH_FACTOR,
                                                                  VALIDATION_SPLIT, TRAINING_SPLIT, SELECTED_PROPERTIES, SPLITS, FOLDS)
    USE_CUAEV = True
    RUNS_ROOT_DIR = '/media/samsung1TBssd/Git-Repos/torchani-runs'
    SET_NAME = 'trials-2'
    SPECIFIC_RUN_NAME = '1x-cuaev'
    # Model
    model = torchani.models.ANI1x(pretrained=False, model_index=0, use_cuda_extension=USE_CUAEV, periodic_table_index=True)
    # GSAEs
    model.energy_shifter.self_energies = torch.tensor([-0.499321200000, -37.83383340000, -54.57328250000, -75.04245190000], dtype=torch.float)
    # Transforms
    elements = model.get_chemical_symbols()
    # Optimizer
    INITIAL_LR = 1e-3
    WEIGHT_DECAY = 1e-7
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=WEIGHT_DECAY, lr=INITIAL_LR)
    # Lr scheduler
    MAX_EPOCHS = 100  # exclusive
    EARLY_STOPPING_LR = 1.0e-6
    TRACK_METRIC = 'energy_rmse_kcalpermol'
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=50, threshold=1e-4)
    # Training / Validation runner, the runner tracks the best metric
    runner = EnergyRunner(model, optimizer, transforms.Identity(), device=torch.device('cuda'))
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
                              EARLY_STOPPING_LR)
