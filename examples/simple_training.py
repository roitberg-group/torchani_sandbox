import time
import math
from pathlib import Path
from pprint import pprint
from typing import Optional, Union, Dict, Callable, Iterator

import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.utils.tensorboard
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from tqdm import tqdm

import torchani
from torchani import datasets, transforms, units
# Simple training script, bare bones, which uses mypy for extra safety

# Mypy
DatasetType = Union[torch.utils.data.DataLoader, datasets.AniBatchedDataset]
DeviceType = Union[int, str, torch.device]
Properties = Dict[str, Tensor]
Transform = Callable[[Properties], Properties]
PathLike = Union[str, Path]
ScalarMetrics = Dict[str, float]
LRScheduler = Union[ReduceLROnPlateau, _LRScheduler]
Stateful = Union[torch.nn.Module, torchani.models.BuiltinModel, LRScheduler, Optimizer]


# Validation / Training logic, including loss
class Runner:
    # Runner contains all the validation / training logic During the main loop
    # it calculates some metrics over a given dataset if training is requested
    # then it also performs backpropagation

    def __init__(self, model: torchani.models.BuiltinModel,
                       optimizer: Optimizer,
                       transform: Optional[Transform] = None,
                       best_metric: float = math.inf):
        self.model = model
        self.optimizer = optimizer
        self.mse = torch.nn.MSELoss(reduction='none')

        # metric to track to check if it improves
        self.best_metric = best_metric
        self.best_metric_improved_last_run = False

    def _run(self, dataset: DatasetType,
                   epoch: Optional[int] = None,
                   train: bool = False,
                   use_tqdm: bool = True,
                   verbose: bool = True) -> ScalarMetrics:
        # outputs for logging
        mean_epoch_loss = 0.0
        mean_epoch_rmse = 0.0
        count = 0

        self.model.train(train)
        for properties in self._maybe_wrap_with_tqdm(dataset, use_tqdm, epoch):
            properties = transform(properties)
            species, coordinates = properties['species'], properties['coordinates']
            target_energies = properties['energies']
            predicted_energies = self.model((species, coordinates)).energies
            num_atoms = (species >= 0).sum(dim=1, dtype=coordinates.dtype)
            mse = self.mse(predicted_energies, target_energies)
            scaled_mse = (mse / num_atoms.sqrt())

            # update for logging
            mean_epoch_loss += scaled_mse.detach().sum()
            mean_epoch_rmse += mse.detach().sum()
            count += species.shape[0]

            if train:
                loss = scaled_mse.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.model.train(not train)

        metrics = {'loss_hartree_squared': (mean_epoch_loss / count).item(),
                   'rmse_kcalpermol': units.hartree2kcalmol(torch.sqrt(mean_epoch_rmse / count)).item()}
        self._maybe_print_metrics(dataset, metrics, verbose)
        return metrics

    @staticmethod
    def _maybe_print_metrics(dataset: DatasetType,
                             metrics: ScalarMetrics,
                             verbose: bool = True) -> None:
        if verbose:
            if isinstance(dataset, datasets.AniBatchedDataset):
                split = dataset.split
            else:
                assert isinstance(dataset.dataset, datasets.AniBatchedDataset)
                split = dataset.dataset.split
            print(f'{split} metrics:')
            pprint(metrics)

    @staticmethod
    def _maybe_wrap_with_tqdm(dataset: DatasetType,
                              use_tqdm: bool = True,
                              epoch: Optional[int] = None) -> Iterator[Properties]:
        if use_tqdm:
            desc = f"epoch {epoch}" if epoch is not None else None
            batches = tqdm(dataset, total=len(dataset), desc=desc)
        else:
            batches = dataset
        return batches

    def train(self,
              dataset: DatasetType,
              epoch: int, **kwargs: bool) -> ScalarMetrics:
        metrics = self._run(dataset, epoch, train=True, **kwargs)
        return metrics

    def eval(self, dataset: DatasetType,
                   epoch: int,
                   track_metric: Optional[str] = None, **kwargs: bool) -> ScalarMetrics:
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


# Logging logic
class Logger:

    def __init__(self, path: Optional[PathLike] = None):
        path = path if path is not None else Path('./runs').resolve()
        path = Path(path).resolve()
        self.writer = torch.utils.tensorboard.SummaryWriter(path)

    def log_scalars(self, step: int,
                    train_metrics: Optional[ScalarMetrics] = None,
                    validate_metrics: Optional[ScalarMetrics] = None,
                    other: Optional[ScalarMetrics] = None) -> None:
        # log into tensorboard every epoch
        if train_metrics is not None:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f'{k}/train', v, step)

        if validate_metrics is not None:
            for k, v in validate_metrics.items():
                self.writer.add_scalar(f'{k}/validate', v, step)

        if other is not None:
            for k, v in other.items():
                self.writer.add_scalar(k, v, step)


# Checkpointing
def save_checkpoint(path: PathLike, objects: Dict[str, Stateful]) -> None:
    path = Path(path).resolve()
    # objects must have a state dict
    for v in objects.values():
        assert hasattr(v, 'state_dict')
        assert hasattr(v, 'load_state_dict')
    torch.save({k: v.state_dict() for k, v in objects.items()}, path)


def load_checkpoint(path: PathLike, objects: Dict[str, Stateful]) -> None:
    path = Path(path).resolve()
    # Objects must have a state dict
    for v in objects.values():
        assert hasattr(v, 'state_dict')
        assert hasattr(v, 'load_state_dict')

    checkpoint = torch.load(latest_checkpoint)
    for k in objects.keys():
        objects[k].load_state_dict(checkpoint[k])


if __name__ == '__main__':

    # Training and validation sets
    batch_size = 2560
    h5_path = Path('/home/ignacio/Datasets/ani1x_release_wb97x_dz.h5').resolve()
    batched_dataset_path = Path('./batched_dataset_1x').resolve()
    if not batched_dataset_path.is_dir():
        datasets.create_batched_dataset(h5_path,
                                        dest_path=batched_dataset_path,
                                        batch_size=batch_size,
                                        splits={'training': 0.8, 'validation': 0.2})

    training = torch.utils.data.DataLoader(datasets.AniBatchedDataset(batched_dataset_path, split='training'),
                                           shuffle=True,
                                           num_workers=2,
                                           prefetch_factor=2,
                                           pin_memory=True,
                                           batch_size=None)

    validation = torch.utils.data.DataLoader(datasets.AniBatchedDataset(batched_dataset_path, split='validation'),
                                             shuffle=False,
                                             num_workers=2,
                                             prefetch_factor=2,
                                             pin_memory=True,
                                             batch_size=None)

    # Model
    model = torchani.models.ANI1x(pretrained=False, periodic_table_index=True, model_index=0, bias=False, activation=torchani.nn.FittedSoftplus()).float()
    elements = model.get_chemical_symbols()
    model.energy_shifter = torchani.utils.EnergyShifter(torchani.utils.ground_atomic_energies(elements, 'RwB97X/6-31G(d)')).float()

    # Transforms
    transform = transforms.Compose([transforms.ToDevice(non_blocking=True),
                                    transforms.Cast(float_keys=('energies', 'coordinates'), long_keys=('species',))])

    # transform and model have to be sent to a device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    transform = transform.to(device)

    # Optimizer and lr scheduler
    initial_lr = 1e-3
    weight_decay = 1e-7
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=initial_lr)

    # Lr scheduler
    max_epochs = 2000
    early_stopping_learning_rate = 1.0e-6
    track_metric = 'rmse_kcalpermol'
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=50, threshold=1e-4)

    # Training / Validation runner, the runner also tracks the best metric
    runner = Runner(model, optimizer, transform)

    # Logging
    logger = Logger()

    # Checkpoint paths
    latest_checkpoint = Path('./latest.pt').resolve()
    best_checkpoint = Path('./best.pt').resolve()
    persistent_objects = {'optimizer': optimizer, 'scheduler': scheduler, 'model': model, 'runner': runner}

    # Load latest checkpoint if it exists
    if latest_checkpoint.is_file():
        load_checkpoint(latest_checkpoint, persistent_objects)

    # Main Training loop
    initial_epoch = scheduler.last_epoch  # type: ignore
    print("Training starting from epoch", initial_epoch)
    if initial_epoch == 0:
        validate_metrics = runner.eval(validation, initial_epoch, track_metric=track_metric)
        logger.log_scalars(initial_epoch, validate_metrics=validate_metrics, other={'learning_rate': initial_lr})

    for epoch in range(initial_epoch + 1, max_epochs + 1):
        start = time.time()
        # Run training and validation
        train_metrics = runner.train(training, epoch)
        validate_metrics = runner.eval(validation, epoch, track_metric=track_metric)

        # LR Scheduler update
        if isinstance(scheduler, ReduceLROnPlateau):  # This scheduler has a different step function
            scheduler.step(validate_metrics[track_metric])
        else:
            scheduler.step()

        # Checkpoint
        if runner.best_metric_improved_last_run:
            runner.best_metric_improved_last_run = False
            save_checkpoint(best_checkpoint, persistent_objects)
        save_checkpoint(latest_checkpoint, persistent_objects)

        # Logging
        last_epoch = scheduler.last_epoch  # type: ignore
        learning_rate = optimizer.param_groups[0]['lr']
        logger.log_scalars(last_epoch,
                           train_metrics=train_metrics,
                           validate_metrics=validate_metrics,
                           other={'learning_rate': learning_rate,
                                  'epoch_time_seconds': time.time() - start})
        # Early stopping
        if learning_rate < early_stopping_learning_rate:
            break
