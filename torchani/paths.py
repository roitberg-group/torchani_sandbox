r"""
Default location for various TorchANI resources
"""

import typing as tp
import os
from pathlib import Path
from torchani.annotations import StrPath

DATA_DIR = Path(Path.home(), ".local", "share", "torchani")

STATE_DICTS = DATA_DIR / "StateDicts"
DATASETS = DATA_DIR / "Datasets"
NEUROCHEM = DATA_DIR / "Neurochem"

RESOURCES = Path(__file__).resolve().parent / "resources"


def set_data_dir(data_dir: tp.Optional[StrPath] = None) -> None:
    global STATE_DICTS, DATASETS, NEUROCHEM
    if data_dir is None:
        ENV_DATA_DIR = os.getenv("TORCHANI_DATA_DIR")
        if ENV_DATA_DIR:
            DATA_DIR = Path(ENV_DATA_DIR)
        else:
            DATA_DIR = Path(Path.home(), ".local", "share", "torchani")
    else:
        DATA_DIR = Path(data_dir)

    STATE_DICTS = DATA_DIR / "StateDicts"
    DATASETS = DATA_DIR / "Datasets"
    NEUROCHEM = DATA_DIR / "Neurochem"
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    STATE_DICTS.mkdir(exist_ok=True, parents=True)
    DATASETS.mkdir(exist_ok=True, parents=True)
    NEUROCHEM.mkdir(exist_ok=True, parents=True)


set_data_dir()
