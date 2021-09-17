import time
import math
from pathlib import Path
from pprint import pprint
from typing import Optional, Union, Dict, Callable, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.utils.tensorboard
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from tqdm import tqdm

import torchani
from torchani import datasets, transforms, units

# Mypy
DatasetType = Union[torch.utils.data.DataLoader, datasets.ANIBatchedDataset]
DeviceType = Union[int, str, torch.device]
Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]
PathLike = Union[str, Path]
ScalarMetrics = Dict[str, float]
LRScheduler = Union[ReduceLROnPlateau, _LRScheduler]
Stateful = Union[torch.nn.Module, torchani.models.BuiltinModel, LRScheduler, Optimizer]


# Validation / Training logic, including loss
class Runner:
    def __init__(self, model: torchani.models.BuiltinModel,
                       optimizer: Optimizer,
                       transform: Optional[Transform] = None,
                       device: Optional[DeviceType] = None,
                       best_metric: float = math.inf):
        self._transform = transform
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self._model = model
        self._optimizer = optimizer
        self._squared_error = torch.nn.MSELoss(reduction='none')
        # metric to track to check if it improves
        self.best_metric = best_metric
        self.best_metric_improved_last_run = False

    def inner_loop(self, batch, metrics: Dict[str, Tensor], train: bool = False) -> Tuple[Tensor, int]:
        # This method is the one that should be overriden, must return the batch loss
        # (not averaged) and the number of conformations in the batch (shape of species)
        species = batch['species'].long()
        coordinates = batch['coordinates'].float()
        target_energies = batch['energies'].float()

        predicted_energies = self._model((species, coordinates)).energies

        batch_loss = self._squared_error(predicted_energies, target_energies)
        return batch_loss, species.shape[0]

    def set_extra_metrics(self):
        # Must return metric_name : initial value dict, metrics that end with
        # "hartree" or "hartree_per_angstrom" are treated specially (other
        # metrics with different units are added to them)
        return {}

    def _run(self, dataset: DatasetType,
                   epoch: Optional[int] = None,
                   train: bool = False,
                   use_tqdm: bool = True,
                   verbose: bool = True) -> ScalarMetrics:
        metrics = {'loss': 0.0, 'count': 0}
        metrics.update(self.set_extra_metrics())
        for batch in tqdm(dataset, total=len(dataset),
                          desc=f"epoch {epoch}" if epoch is not None else None,
                          disable=not use_tqdm):
            batch = self._transform({k: v.to(self._device, non_blocking=True)
                                     for k, v in batch.items()})
            batch_loss, count = self._inner_loop(batch, metrics)
            if train:
                self._run_backwards(batch_loss.mean())
            metrics['loss'] += batch_loss.detach().sum().cpu().item()
            metrics['count'] += count
        metrics = self._average_metrics(metrics)
        metrics = self._add_kcalpermol_metrics(metrics)
        if verbose:
            self._print_metrics(dataset, metrics)
        return metrics

    def _add_kcalpermol_metrics(self, metrics):
        for k in metrics.keys():
            if k.endswith('_hartree'):
                metrics[k.replace('_hartree', '_kcalpermol')] = units.hartree2kcalmol(metrics[k])
            elif k.endswith('_hartree_per_angstrom'):
                metrics[k.replace('_hartree_per_angstrom', '_kcalpermol_per_Angstrom')] = units.hartree2kcalmol(metrics[k])
        return metrics

    def _average_metrics(self, metrics):
        count = metrics.pop('count')
        for k in metrics.keys():
            metrics[k] = (metrics[k] / count).item()
        return metrics

    def _run_backwards(self, batch_loss):
        self._optimizer.zero_grad()
        batch_loss.backward()
        self._optimizer.step()

    def _print_metrics(self, dataset, metrics):
        if isinstance(dataset, datasets.ANIBatchedDataset):
            split = dataset.split
        else:
            assert isinstance(dataset.dataset, datasets.ANIBatchedDataset)
            split = dataset.dataset.split
        print(f'{split} metrics:')
        pprint(metrics)

    def train(self, dataset: DatasetType, epoch: int, **kwargs: bool) -> ScalarMetrics:
        self._model.train()
        metrics = self._run(dataset, epoch, train=True, **kwargs)
        return metrics

    def eval(self, dataset: DatasetType, epoch: int, track_metric: Optional[str] = None, **kwargs: bool) -> ScalarMetrics:
        self._model.eval()
        with torch.no_grad():
            metrics = self._run(dataset, epoch, train=False, **kwargs)
        if track_metric is not None:
            if metrics[track_metric] < self.best_metric:
                self.best_metric = metrics[track_metric]
                self.best_metric_improved_last_run = True
            metrics.update({f'best_{track_metric}': self.best_metric})
        return metrics

    def to(self, device: DeviceType) -> 'Runner':
        self._device = device
        return self

    def load_state_dict(self, state_dict: ScalarMetrics) -> None:
        self.__dict__.update(state_dict)

    def state_dict(self) -> ScalarMetrics:
        return {'best_metric': self.best_metric}


class EnergyRunner(Runner):
    def inner_loop(self, batch, metrics: Dict[str, Tensor], train: bool = False) -> Tuple[Tensor, int]:
        species = batch['species'].long()
        coordinates = batch['coordinates'].float()
        target_energies = batch['energies'].float()
        num_atoms = (species >= 0).sum(dim=1).float()

        predicted_energies = self._model((species, coordinates)).energies

        squared_energy_error = self._squared_error(predicted_energies, target_energies)
        batch_loss = squared_energy_error / num_atoms.sqrt()

        metrics['energy_rmse_hartree'] += squared_energy_error.detach().sum().cpu()
        return batch_loss, species.shape[0]

    def set_extra_metrics(self):
        return {'energy_rmse_hartree': 0.0}


class Logger:
    def __init__(self, path: Optional[PathLike] = None, is_restart=False):
        path = Path(path).resolve() if path is not None else Path('./runs/default_set/default_run').resolve()
        if not is_restart:
            path.mkdir(parents=True, exist_ok=False)
        assert path.is_dir()
        self._writer = torch.utils.tensorboard.SummaryWriter(path)

    def log_scalars(self, step: int,
                    train_metrics: Optional[ScalarMetrics] = None,
                    validate_metrics: Optional[ScalarMetrics] = None,
                    other: Optional[ScalarMetrics] = None) -> None:
        if train_metrics is not None:
            for k, v in train_metrics.items():
                self._writer.add_scalar(f'{k}/train', v, step)
        if validate_metrics is not None:
            for k, v in validate_metrics.items():
                self._writer.add_scalar(f'{k}/validate', v, step)
        if other is not None:
            for k, v in other.items():
                self._writer.add_scalar(k, v, step)


# Checkpointing
def _ensure_state_dicts(objects):
    for v in objects.values():
        assert hasattr(v, 'state_dict')
        assert hasattr(v, 'load_state_dict')


def save_checkpoint(path: PathLike, objects: Dict[str, Stateful]) -> None:
    _ensure_state_dicts(objects)
    torch.save({k: v.state_dict() for k, v in objects.items()}, Path(path).resolve())


def load_checkpoint(path: PathLike, objects: Dict[str, Stateful]) -> None:
    _ensure_state_dicts(objects)
    checkpoint = torch.load(Path(path).resolve())
    for k in objects.keys():
        objects[k].load_state_dict(checkpoint[k])


if __name__ == '__main__':
    # transform, model and runner have to be sent to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training and validation sets
    batched_dataset_path = Path('./batched-1x-plain').resolve()
    if not batched_dataset_path.is_dir():
        ds = datasets.ANI1x('./1x-plain', download=True)
        datasets.create_batched_dataset(ds,
                                        dest_path=batched_dataset_path,
                                        batch_size=2560,
                                        splits={'training': 0.8, 'validation': 0.2})

    training = torch.utils.data.DataLoader(datasets.ANIBatchedDataset(batched_dataset_path, split='training'),
                                           shuffle=True,
                                           num_workers=2,
                                           prefetch_factor=2,
                                           pin_memory=True,
                                           batch_size=None)

    validation = torch.utils.data.DataLoader(datasets.ANIBatchedDataset(batched_dataset_path, split='validation'),
                                             shuffle=False,
                                             num_workers=2,
                                             prefetch_factor=2,
                                             pin_memory=True,
                                             batch_size=None)
    USE_CUAEV = True
    RUN_NAME = '1x-cuaev'
    SET_NAME = 'trials-2'

    # Model
    model = torchani.models.ANI1x(pretrained=False, model_index=0, use_cuda_extension=USE_CUAEV, periodic_table_index=True).to(device)
    # GSAEs
    model.energy_shifter.self_energies = torch.tensor([-0.499321200000, -37.83383340000, -54.57328250000, -75.04245190000],
                                                       dtype=torch.float, device=device)
    # Transforms
    elements = model.get_chemical_symbols()
    transform = transforms.Compose([transforms.Identity()]).to(device)

    # Optimizer and lr scheduler
    INITIAL_LR = 1e-3
    WEIGHT_DECAY = 1e-7
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=WEIGHT_DECAY, lr=INITIAL_LR)

    # Lr scheduler
    MAX_EPOCHS = 100  # exclusive
    EARLY_STOPPING_LR = 1.0e-6
    track_metric = 'energy_rmse_kcalpermol'
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=50, threshold=1e-4)

    # Training / Validation runner, the runner tracks the best metric
    runner = Runner(model, optimizer, transform).to(device)

    # Checkpoint paths
    latest_checkpoint = Path(f'./checkpoints/{SET_NAME}/{RUN_NAME}/latest.pt').resolve()
    best_checkpoint = Path(f'./checkpoints/{SET_NAME}/{RUN_NAME}/best.pt').resolve()
    persistent_objects = {'optimizer': optimizer, 'scheduler': scheduler, 'model': model, 'runner': runner}

    # Load latest checkpoint if it exists
    if latest_checkpoint.is_file():
        is_restart = True
        load_checkpoint(latest_checkpoint, persistent_objects)
    else:
        is_restart = False
        latest_checkpoint.parent.mkdir(parents=True, exist_ok=False)

    # Logging
    logger = Logger(f'./runs/{SET_NAME}/{RUN_NAME}', is_restart)

    # Main Training loop
    initial_epoch = scheduler.last_epoch  # type: ignore
    print("Training starting from epoch", initial_epoch)
    if not is_restart and initial_epoch == 0:  # Zeroth epoch is just validating
        validate_metrics = runner.eval(validation, initial_epoch, track_metric=track_metric)
        logger.log_scalars(initial_epoch, validate_metrics=validate_metrics, other={'learning_rate': INITIAL_LR})
        save_checkpoint(latest_checkpoint, persistent_objects)

    for epoch in range(initial_epoch + 1, MAX_EPOCHS):
        start = time.time()
        # Run training and validation
        train_metrics = runner.train(training, epoch)
        validate_metrics = runner.eval(validation, epoch, track_metric=track_metric)
        # LR Scheduler update
        metric = (validate_metrics[track_metric],) if isinstance(scheduler, ReduceLROnPlateau) else tuple()
        scheduler.step(*metric)
        # Checkpoint
        if runner.best_metric_improved_last_run:
            runner.best_metric_improved_last_run = False
            save_checkpoint(best_checkpoint, persistent_objects)
        save_checkpoint(latest_checkpoint, persistent_objects)
        # Logging
        learning_rate = optimizer.param_groups[0]['lr']
        logger.log_scalars(scheduler.last_epoch,  # type: ignore
                           train_metrics=train_metrics,
                           validate_metrics=validate_metrics,
                           other={'learning_rate': learning_rate,
                                  'epoch_time_seconds': time.time() - start})
        # Early stopping
        if learning_rate < EARLY_STOPPING_LR:
            break
