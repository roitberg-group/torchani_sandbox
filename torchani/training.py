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
from torchani import datasets, units

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
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self._transform = transform.to(device)
        self._model = model.to(device)
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
            batch_loss, count = self.inner_loop(batch, metrics)
            if train:
                self._run_backwards(batch_loss.mean())
            metrics['loss'] += batch_loss.detach().sum().item()
            metrics['count'] += count
        metrics = self._average_metrics(metrics)
        metrics = self._add_kcalpermol_metrics(metrics)
        if verbose:
            self._print_metrics(dataset, metrics)
        return metrics

    def _add_kcalpermol_metrics(self, metrics):
        for k in metrics.copy().keys():
            if k.endswith('_hartree'):
                metrics[k.replace('_hartree', '_kcalpermol')] = units.hartree2kcalmol(metrics[k])
            elif k.endswith('_hartree_per_angstrom'):
                metrics[k.replace('_hartree_per_angstrom', '_kcalpermol_per_Angstrom')] = units.hartree2kcalmol(metrics[k])
        return metrics

    def _average_metrics(self, metrics):
        count = metrics.pop('count')
        for k in metrics.copy().keys():
            metrics[k] = (metrics[k] / count)
            if 'rmse' in k:
                metrics[k] = math.sqrt(metrics[k])
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

    def load_state_dict(self, state_dict: ScalarMetrics) -> None:
        self.__dict__.update(state_dict)

    def state_dict(self) -> ScalarMetrics:
        return {'best_metric': self.best_metric}


class Logger:
    def __init__(self, path: Optional[PathLike] = None):
        path = Path(path).resolve() if path is not None else Path('./runs/default_set/default_run').resolve()
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


def _save_checkpoint(path: PathLike, objects: Dict[str, Stateful], kind: str = 'default') -> None:
    _ensure_state_dicts(objects)
    for k, v in objects.items():
        torch.save(v.state_dict(), Path(path).resolve() / f'{k}_{kind}.pt')


def _load_checkpoint(path: PathLike, objects: Dict[str, Stateful], kind: str = 'default') -> None:
    _ensure_state_dicts(objects)
    for k in objects.keys():
        objects[k].load_state_dict(torch.load(path / f'{k}_{kind}.pt'))


def prepare_learning_sets(DatasetClass, root_dataset_path, dataset_name, batch_size, num_workers, prefetch_factor, validation_split, training_split, selected_properties=None, splits=None, folds=None):
    assert (splits is None or folds is None) and (splits is not folds)
    batched_dataset_path = (root_dataset_path / dataset_name) / '-batched'
    if not batched_dataset_path.is_dir():
        if type(DatasetClass) == datasets.ANIDataset:
            ds = DatasetClass(root_dataset_path / dataset_name)
        else:
            ds = DatasetClass(root_dataset_path / dataset_name, download=True)
        datasets.create_batched_dataset(ds,
                                        dest_path=batched_dataset_path,
                                        batch_size=batch_size,
                                        shuffle_seed=123456789,
                                        splits=splits, folds=folds)

    training_set = torch.utils.data.DataLoader(datasets.ANIBatchedDataset(batched_dataset_path, split='training', properties=selected_properties),
                                               shuffle=True,
                                               num_workers=num_workers,
                                               prefetch_factor=prefetch_factor,
                                               pin_memory=True,
                                               batch_size=None)

    validation_set = torch.utils.data.DataLoader(datasets.ANIBatchedDataset(batched_dataset_path, split='validation', properties=selected_properties),
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 prefetch_factor=prefetch_factor,
                                                 pin_memory=True,
                                                 batch_size=None)
    return training_set, validation_set


def execute_training(persistent_objects, scheduler, runner, optimizer, training_set, validation_set, run_output_path, TRACK_METRIC, INITIAL_LR, MAX_EPOCHS, EARLY_STOPPING_LR):
    # Load latest checkpoint if it exists
    if run_output_path.is_dir() and any(run_output_path.iterdir()):
        is_restart = True
        _load_checkpoint(run_output_path, persistent_objects, kind='latest')
    else:
        is_restart = False
        run_output_path.mkdir(parents=True, exist_ok=True)

    # Logging
    logger = Logger(run_output_path)

    # Main Training loop
    initial_epoch = scheduler.last_epoch  # type: ignore
    print("Training starting from epoch", initial_epoch)
    if not is_restart and initial_epoch == 0:  # Zeroth epoch is just validating
        validate_metrics = runner.eval(validation_set, initial_epoch, track_metric=TRACK_METRIC)
        logger.log_scalars(initial_epoch, validate_metrics=validate_metrics, other={'learning_rate': INITIAL_LR})
        _save_checkpoint(run_output_path, persistent_objects, kind='latest')

    for epoch in range(initial_epoch + 1, MAX_EPOCHS):
        start = time.time()
        # Run training and validation
        train_metrics = runner.train(training_set, epoch)
        validate_metrics = runner.eval(validation_set, epoch, track_metric=TRACK_METRIC)
        # LR Scheduler update
        metric = (validate_metrics[TRACK_METRIC],) if isinstance(scheduler, ReduceLROnPlateau) else tuple()
        scheduler.step(*metric)
        # Checkpoint
        if runner.best_metric_improved_last_run:
            runner.best_metric_improved_last_run = False
            _save_checkpoint(run_output_path, persistent_objects, kind='best')
        _save_checkpoint(run_output_path, persistent_objects, kind='latest')
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
