import typing as tp
import time
import argparse

import torch
from tqdm import tqdm

from torchani import datasets
from torchani.datasets import create_batched_dataset
from torchani.models import ANI1x
from torchani.units import hartree2kcalpermol
from tool_utils import time_functions_in_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        help="Name of builtin dataset to train on",
        nargs="?",
        default="TestData",
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device of modules and tensors",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Number of conformations of each batch",
        default=2560,
        type=int,
    )
    parser.add_argument(
        "--no-synchronize",
        action="store_true",
        help="Whether to wrap functions with torch.cuda.synchronize()",
    )
    parser.add_argument(
        "-n",
        "--num_epochs",
        help="epochs",
        default=5,
        type=int,
    )
    args = parser.parse_args()

    if args.no_synchronize:
        synchronize = False
        print(
            "WARNING: Synchronization creates some small overhead but if CUDA"
            " streams are not synchronized the timings before and after a"
            " function do not reflect the actual calculation load that"
            " function is performing. Only run this benchmark without"
            " synchronization if you know very well what you are doing"
        )
    else:
        synchronize = True

    model = ANI1x(model_index=0).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    mse = torch.nn.MSELoss(reduction="none")

    # enable timers
    timers: tp.Dict[str, int] = dict()

    # time these functions
    fn_to_time_aev = [
        "_compute_radial_aev",
        "_compute_angular_aev",
        "_compute_aev",
        "_triple_by_molecule",
        "forward",
    ]
    time_functions_in_model(
        model.aev_computer.neighborlist,
        ["forward"],
        timers,
        synchronize,
    )
    time_functions_in_model(
        model.aev_computer.angular_terms,
        ["forward"],
        timers,
        synchronize,
    )
    time_functions_in_model(
        model.aev_computer.radial_terms,
        ["forward"],
        timers,
        synchronize,
    )
    time_functions_in_model(
        model.aev_computer,
        fn_to_time_aev,
        timers,
        synchronize,
    )
    time_functions_in_model(
        model.neural_networks,
        ["forward"],
        timers,
        synchronize,
    )
    time_functions_in_model(
        model.energy_shifter,
        ["forward"],
        timers,
        synchronize,
    )
    print("=> loading dataset")
    try:
        ds = getattr(datasets, args.dataset)(verbose=False)
    except AttributeError:
        raise RuntimeError(f"Dataset {args.dataset} could not be found")
    splits = create_batched_dataset(
        ds,
        splits={"training": 1.0},
        direct_cache=True,
        batch_size=args.batch_size,
        verbose=False,
    )
    train = torch.utils.data.DataLoader(
        splits["training"], batch_size=None, shuffle=True
    )

    print("=> starting training")
    start = time.perf_counter()
    for epoch in range(0, args.num_epochs):
        for properties in tqdm(
            train,
            total=len(train),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            leave=False,
        ):
            species = properties["species"].to(device=args.device)
            coordinates = properties["coordinates"].to(
                device=args.device, dtype=torch.float
            )
            true_energies = properties["energies"].to(
                device=args.device, dtype=torch.float
            )
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            predicted_energies = model((species, coordinates)).energies
            loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            rmse = (
                hartree2kcalpermol((mse(predicted_energies, true_energies)).mean())
                .detach()
                .cpu()
                .numpy()
            )
            loss.backward()
            optimizer.step()
    if synchronize:
        torch.cuda.synchronize()
    stop = time.perf_counter()

    for k in timers.keys():
        timers[k] = timers[k] / args.num_epochs
    total = (stop - start) / args.num_epochs

    for k in timers:
        print(f"{k} - {timers[k]:.4f}s")
    print(f"Total epoch time - {total:.4f}s")
