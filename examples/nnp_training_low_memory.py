# type: ignore

import torch
import torchani
import os
import math
from pathlib import Path

import torch.utils.tensorboard
import tqdm
import pkbar  # noqa

from torchani.datasets import AniBatchedDataset, create_batched_dataset
from torchani.transforms import AtomicNumbersToIndices, SubtractSAE
from torchani.units import hartree2kcalmol

# Explanation of the Batched Dataset API for ANI, which is a dataset that
# consumes minimal memory since it lives on disk, and batches are fetched on
# the fly
# This example is meant for internal use of Roitberg's Group

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --Starting here this is different from the usual nnp_training.py--
h5_path = '/home/ignacio/Datasets/ani1x_release_wb97x_dz.h5'
batched_dataset_path = './batched_dataset_1x'


# We prebatch the dataset to train with memory efficiency, and comparable
# performance.
if not Path(batched_dataset_path).resolve().is_dir():
    create_batched_dataset(h5_path,
                           dest_path=batched_dataset_path,
                           file_format='numpy',
                           batch_size=2560,
                           splits={'training': 0.8, 'validation': 0.2})

# We pass a transform to the dataset to perform transformations on the fly, the
# API for transforms is very similar to torchvision https://pytorch.org/vision/stable/transforms.html
# with the difference that the transforms are applied to both target and inputs in all cases
elements = ('H', 'C', 'N', 'O')
self_energies = [-0.57, -0.0045, -0.0035, -0.008]
transform = torchani.transforms.Compose([AtomicNumbersToIndices(elements), SubtractSAE(self_energies)])

training = AniBatchedDataset(batched_dataset_path, transform=transform, split='training')
validation = AniBatchedDataset(batched_dataset_path, transform=transform, split='validation')

# Alternatively a transform can be passed to
# create_batched_dataset using the argument "inplace_transform", but this is
# only really recommended if your transforms takes a lot of time, since this
# will modify the dataset and may introduce hard to track discrepancies and
# reproducibility issues


# This batched dataset can be directly iterated upon, but it may be more practical
# to wrap it with a torch DataLoader
cache = False
if not cache:
    # If we decide not to cache the dataset it is a good idea to use
    # multiprocessing. Here we use some default useful arguments for
    # num_workers (extra cores for training) and prefetch_factor (data units
    # each worker buffers), but you should probably experiment depending on
    # your batch size and system to get the best performance. Performance can
    # be made in general almost the same as what you get caching the dataset
    # for pure python, but it is a bit slower than cacheing if using cuaev
    # (this is because cuaev is very fast).
    #
    # We also use shuffle=True, to shuffle batches every epoch (takes no time at all)
    # and pin_memory=True, to speed up transfer of memory to the GPU.
    #
    # If you can afford it in terms of memory you can sometimes get a bit of a
    # speedup by cacheing the validation set and setting persistent_workers = True
    # for the training set.
    #
    # NOTE: it is very important here to pass batch_size = None since the dataset is
    # already batched!
    #
    # NOTE: for more info about the DataLoader and multiprocessing read
    # https://pytorch.org/docs/stable/data.html
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
                                             batch_size=None)
elif cache:
    # If need some extra speedup you can cache the dataset before passing it to
    # the DataLoader or iterating on it, but this may occupy a lot of memory,
    # so be careful!!!
    #
    # Note: it is very important to **not** pass pin_memory=True here, since
    # cacheing automatically pins the memory of the whole dataset
    training = torch.utils.data.DataLoader(training.cache(),
                                           shuffle=True,
                                           batch_size=None)

    validation = torch.utils.data.DataLoader(validation.cache(),
                                             shuffle=False,
                                             batch_size=None)
# --Differences end here--
###############################################################################
# First lets define an aev computer like the one in the 1x model
aev_computer = torchani.AEVComputer.like_1x(use_cuda_extension=True)
# Now let's define atomic neural networks.
aev_dim = aev_computer.aev_length

H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 144),
    torch.nn.CELU(0.1),
    torch.nn.Linear(144, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
print(nn)

###############################################################################
# Initialize the weights and biases.
#
# .. note::
#   Pytorch default initialization for the weights and biases in linear layers
#   is Kaiming uniform. See: `TORCH.NN.MODULES.LINEAR`_
#   We initialize the weights similarly but from the normal distribution.
#   The biases were initialized to zero.
#
# .. _TORCH.NN.MODULES.LINEAR:
#   https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)


nn.apply(init_params)

###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
model = torchani.nn.Sequential(aev_computer, nn).to(device)

###############################################################################
# Now let's setup the optimizers. NeuroChem uses Adam with decoupled weight decay
# to updates the weights and Stochastic Gradient Descent (SGD) to update the biases.
# Moreover, we need to specify different weight decay rate for different layes.
#
# .. note::
#
#   The weight decay in `inputtrain.ipt`_ is named "l2", but it is actually not
#   L2 regularization. The confusion between L2 and weight decay is a common
#   mistake in deep learning.  See: `Decoupled Weight Decay Regularization`_
#   Also note that the weight decay only applies to weight in the training
#   of ANI models, not bias.
#
# .. _Decoupled Weight Decay Regularization:
#   https://arxiv.org/abs/1711.05101

AdamW = torch.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[2].weight], 'weight_decay': 0.00001},
    {'params': [H_network[4].weight], 'weight_decay': 0.000001},
    {'params': [H_network[6].weight]},
    # C networks
    {'params': [C_network[0].weight]},
    {'params': [C_network[2].weight], 'weight_decay': 0.00001},
    {'params': [C_network[4].weight], 'weight_decay': 0.000001},
    {'params': [C_network[6].weight]},
    # N networks
    {'params': [N_network[0].weight]},
    {'params': [N_network[2].weight], 'weight_decay': 0.00001},
    {'params': [N_network[4].weight], 'weight_decay': 0.000001},
    {'params': [N_network[6].weight]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[2].weight], 'weight_decay': 0.00001},
    {'params': [O_network[4].weight], 'weight_decay': 0.000001},
    {'params': [O_network[6].weight]},
])

SGD = torch.optim.SGD([
    # H networks
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].bias]},
    {'params': [H_network[6].bias]},
    # C networks
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].bias]},
    {'params': [C_network[4].bias]},
    {'params': [C_network[6].bias]},
    # N networks
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].bias]},
    {'params': [N_network[4].bias]},
    {'params': [N_network[6].bias]},
    # O networks
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].bias]},
    {'params': [O_network[6].bias]},
], lr=1e-3)

###############################################################################
# Setting up a learning rate scheduler to do learning rate decay
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

###############################################################################
# Train the model by minimizing the MSE loss, until validation RMSE no longer
# improves during a certain number of steps, decay the learning rate and repeat
# the same process, stop until the learning rate is smaller than a threshold.
#
# We first read the checkpoint files to restart training. We use `latest.pt`
# to store current training state.
latest_checkpoint = 'latest.pt'

###############################################################################
# Resume training from previously saved checkpoints:
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])

###############################################################################
# During training, we need to validate on validation set and if validation error
# is better than the best, then save the new best model to a checkpoint


def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    model.train(False)
    with torch.no_grad():
        for properties in validation:
            species = properties['species'].to(device, non_blocking=True)
            coordinates = properties['coordinates'].to(device, non_blocking=True).float()
            true_energies = properties['energies'].to(device, non_blocking=True).float()
            _, predicted_energies = model((species, coordinates))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]
    model.train(True)
    return hartree2kcalmol(math.sqrt(total_mse / count))


###############################################################################
# We will also use TensorBoard to visualize our training process
tensorboard = torch.utils.tensorboard.SummaryWriter()

###############################################################################
# Finally, we come to the training loop.
#
# In this tutorial, we are setting the maximum epoch to a very small number,
# only to make this demo terminate fast. For serious training, this should be
# set to a much larger value
mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 100
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'best.pt'

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    print('RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)

    learning_rate = AdamW.param_groups[0]['lr']

    if learning_rate < early_stopping_learning_rate:
        break

    # checkpoint
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse)
    SGD_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):
        species = properties['species'].to(device, non_blocking=True)
        coordinates = properties['coordinates'].to(device, non_blocking=True).float()
        true_energies = properties['energies'].to(device, non_blocking=True).float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates))

        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)
