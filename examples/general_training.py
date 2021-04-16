# type: ignore
from pathlib import Path
import os
import math
import pickle

import torch
import torchani
import torch.utils.tensorboard
import tqdm

from torchani.units import hartree2kcalmol
from torchani import atomics

# This example is meant to be a general training example for internal use by Roitberg's Group

# device to run the training should always be a GPU unless you plan to train for 100 years
assert torch.cuda.is_available()
device = torch.device('cuda')

# First we will construct our model, our model will support some elements:
elements = ('H', 'C', 'N', 'O')
# We make an aev computer exactly like the 1x aev computer
aev_computer = torchani.AEVComputer.like_1x()
# We also make an energy shifter that uses GSAEs
atomic_modules = [(e, atomics.make_like_1x(e)) for e in elements]

neural_networks = torchani.ANIModel(atomic_modules)

# Let's now create our model. If we pass a level of theory to the energy
# shifter argument, the model will automatically add the GSAE's associated with
# that level of theory with our output, otherwise we can pass a prebuilt energy
# shifter that has some energies in it already
model = torchani.models.BuiltinModel(aev_computer,
                                     neural_networks,
                                     energy_shifter='RwB97X',
                                     elements=elements,
                                     periodic_table_index=True).to(device)
# note that BuiltinModel is fairly minimalistic, so if you want to do some
# modifications, subclassing it should be very straightforward

# We now initialize the weights and biases using kaiming normal


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)


model.apply(init_params)

# our model should be ready for training now! we will first print it to make
# sure we built it correctly
print(model)

# now we set our dataset path, this assumes we are running under torchani/examples
dspath = Path(__file__).joinpath('../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5').resolve()
batch_size = 2560


# The first time we run training the training and validation sets will be
# pickled to ensure reproducibility on restarts
# note that there really is no need to subtract self energies from the dataset,
# since they are extremely fast to calculate, but you can do that if you want.
# if you decide to subtract self energies it
pickled_dataset_path = 'dataset.pkl'
if os.path.isfile(pickled_dataset_path):
    print(f'Unpickling preprocessed dataset found in {pickled_dataset_path}')
    with open(pickled_dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    training = dataset['training'].collate(batch_size).cache()
    validation = dataset['validation'].collate(batch_size).cache()
    energy_shifter.self_energies = dataset['self_energies'].to(device)
else:
    print(f'Processing dataset in {dspath}')
    training, validation = torchani.data.load(dspath)\
                                        .subtract_self_energies(energy_shifter)\
                                        .species_to_indices("periodic_table")\
                                        .shuffle()\
                                        .split(0.8, None)
    with open(pickled_dataset_path, 'wb') as f:
        pickle.dump({'training': training,
                     'validation': validation,
                     'self_energies': energy_shifter.self_energies.cpu()}, f)
    training = training.collate(batch_size).cache()
    validation = validation.collate(batch_size).cache()
print('Self atomic energies: ', energy_shifter.self_energies)

###############################################################################
# When iterating the dataset, we will get a dict of name->property mapping
#

# we will train with the AdamW optimizer, with some sensible defaults
AdamW = torch.optim.AdamW(model.parameters(), weight_decay=1e-7, lr=0.5e-4)
# A learning rate scheduler allows for lr decay
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)

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
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])

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
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
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
mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 1500
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

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates))

        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

        AdamW.zero_grad()
        loss.backward()
        AdamW.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
    }, latest_checkpoint)
