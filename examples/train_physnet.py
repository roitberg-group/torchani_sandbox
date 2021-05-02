# type: ignore
from time import time
import torch
import torchani
import os
import math
import torch.utils.tensorboard
from tqdm import tqdm
import pickle
from pathlib import Path
from torchani.units import hartree2kcalmol
from torchani.physnet import HierarchicalModel, HierarchicalLoss


def _copy(self, target):
    import shutil  # noqa
    assert self.is_file()
    shutil.copy(str(self), str(target))  # str() only there for Python < (3, 6)


Path.copy = _copy

# PhysNet trains with a very small batch size of 32 molecules
hparams = {'batch_size': 32}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ## setup all paths #####
runname = 'train_physnet'
dspath = '/home/ignacio/Datasets/ani1x_release_wb97x_dz.h5'

root = Path('/media/samsung1TBssd/Git-Repos/torchani_sandbox/examples')
rundir = root.joinpath(f'runs/{runname}')
detaildir = root.joinpath(f'detail/{runname}')

if not rundir.is_dir():
    rundir.mkdir()

if not detaildir.is_dir():
    detaildir.mkdir()

latest_checkpoint = rundir.joinpath('latest.pt').as_posix()
best_model_checkpoint = rundir.joinpath('best.pt').as_posix()
tensorboard_path = rundir.as_posix()
tensorboard_detail_path = detaildir.as_posix()
pickled_dataset_path = root.joinpath('dataset_full_batched_1280.pkl').resolve().as_posix()

# copy this script to 'runs' to keep track of exactly what was used
current_file = Path(__file__).resolve()
dest = rundir.joinpath(f'{current_file.name}').resolve()
if current_file.as_posix() != dest.as_posix():
    assert current_file.is_file()
    current_file.copy(dest)

# set tensorboard outs
tb = torch.utils.tensorboard.SummaryWriter(tensorboard_path)
tb_detail = torch.utils.tensorboard.SummaryWriter(tensorboard_detail_path)
##########################

# ### Load dataset ####
batch_size = hparams['batch_size']
if os.path.isfile(pickled_dataset_path):
    print(f'Unpickling preprocessed dataset found in {pickled_dataset_path}')
    with open(pickled_dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    batched = dataset.get('batched', False)
    if batched:
        print(f'The dataset is batched with batch size: {batched}')
        training = dataset['training'].pin_memory()
        validation = dataset['validation'].pin_memory()
    else:
        training = dataset['training'].collate(batch_size).cache().pin_memory()
        validation = dataset['validation'].collate(batch_size).cache().pin_memory()
else:
    print(f'Processing dataset in {dspath}')
    training, validation = torchani.data.load(dspath)\
                                        .species_to_indices()\
                                        .shuffle()\
                                        .split(0.8, None)
    training = training.collate(batch_size).cache()
    validation = validation.collate(batch_size).cache()
    with open(pickled_dataset_path, 'wb') as f:
        pickle.dump({'training': training,
                     'validation': validation,
                     'batched': batch_size}, f)
    training = training.pin_memory()
    validation = validation.pin_memory()
###########################

# build model ##
model = HierarchicalModel().float().to(device)
print(model)
################

# In the PhysNet paper they always train with forces ON
train_forces = False
#############################################

max_epochs = 1000
# Although in the paper they don't say so, they seem to clip gradients if norm
# > 1000.0 in their training code (by default)
clip_norm = True
max_norm_for_clipping = 1000.0

# They don't seem to use L2 regularization but they have the option, and they
# have the option to use dropout but by default with keep_prob = 1.0 (this
# means a rate of 1 - 1 = 0, so no dropout),

# at least for the SN2 reactions dataset they also seem to use learning
# rate decay, with 10 000 000 steps and a decay rate of 0.1 (exponential)
# their parameters for force, dipole and charge in the code are very different
# from the ones in the paper
# they are:
# --force_weight=52.91772105638412
# --charge_weight=14.399645351950548
# --dipole_weight=27.211386024367243
# they also train with D3, and use fixed D3 parameters (not learned) by
# default, but they have the possibility to learn the parameters

# physnet trains with AMSGrad (in my tests this performs significantly worse
# for ANI)
opt = torch.optim.Adam(model.parameters(), amsgrad=True)
# scheduler is only here to keep track of epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max_epochs, factor=0.5)

mae = torch.nn.L1Loss(reduction='none')  # this is MAE
hloss = HierarchicalLoss()


def validate(model, validation, epoch):
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
            # Physnet also outputs hierarchical energies, but these are
            # not used for validation
            _, predicted_energies, _ = model((species, coordinates))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]
    model.train(True)
    rmse = hartree2kcalmol(math.sqrt(total_mse / count))
    print(f'Validation RMSE: {rmse} at epoch {epoch}', flush=True)
    return rmse


best_validation_rmse = math.inf
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model'])
    opt.load_state_dict(checkpoint['opt'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_validation_rmse = checkpoint['best_validation_rmse']

print('Num model parameters:', sum([p.numel() for p in model.parameters()]))
print("Training starting from epoch:", scheduler.last_epoch)

if scheduler.last_epoch <= 0:
    rmse = validate(model, validation, scheduler.last_epoch)
    tb.add_scalar('validation_rmse_kcalmol', rmse, scheduler.last_epoch)
    tb.add_scalar('best_validation_rmse_kcalmol', best_validation_rmse, scheduler.last_epoch)
    tb.add_scalar('learning_rate', opt.param_groups[0]['lr'], scheduler.last_epoch)

for _ in range(max_epochs):
    start = time()
    cached_params = {}
    for i, properties in tqdm(enumerate(training), total=len(training)):
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        if train_forces:
            coordinates.requires_grad_(True)

        opt.zero_grad()
        true_energies = properties['energies'].to(device).float()
        _, predicted_energies, atomic_hierarchical_energies = model((species, coordinates))
        # check for NaN
        assert (predicted_energies == predicted_energies).all()
        assert (atomic_hierarchical_energies == atomic_hierarchical_energies).all()

        if train_forces:
            true_forces = properties['forces'].to(device).float()
            predicted_forces = -torch.autograd.grad(predicted_energies.sum(),
                    coordinates, create_graph=True, retain_graph=True)[0]

        non_dummy_atoms = (species != -1).sum(dim=1, dtype=true_energies.dtype)

        # average over batch for energy
        loss = mae(predicted_energies, true_energies).mean()
        # average over batch for hierarchy
        loss += 1e-2 * hloss(species, atomic_hierarchical_energies).mean()
        if train_forces:
            # divide by total components, then average over batch
            loss += 100 * (mae(predicted_forces, true_forces) / (3 * non_dummy_atoms)).sum(1, 2).mean()

        loss.backward()
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm_for_clipping)
        opt.step()

    training = training.shuffle()
    rmse = validate(model, validation, scheduler.last_epoch)

    # best checkpoint
    if rmse < best_validation_rmse:
        best_validation_rmse = rmse
        torch.save(model.state_dict(), best_model_checkpoint)

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(rmse)
    else:
        scheduler.step()

    tb.add_scalar('validation_rmse_kcalmol', rmse, scheduler.last_epoch)
    tb.add_scalar('best_validation_rmse_kcalmol', best_validation_rmse, scheduler.last_epoch)
    tb.add_scalar('learning_rate', opt.param_groups[0]['lr'], scheduler.last_epoch)
    tb.add_scalar('training_loss', loss.item(), scheduler.last_epoch)

    # epoch checkpoint
    torch.save({
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_validation_rmse': best_validation_rmse,
    }, latest_checkpoint)

    # log epoch time
    end = time()
    tb.add_scalar('epoch_time_s', end - start, scheduler.last_epoch)
    print("Epoch time: ", end - start, flush=True)
