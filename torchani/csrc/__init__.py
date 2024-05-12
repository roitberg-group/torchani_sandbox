import os
import importlib.metadata

CUAEV_IS_INSTALLED = "torchani.cuaev" in importlib.metadata.metadata(
    __package__.split(".")[0]
).get_all("Provides", [])

MNP_IS_INSTALLED = "torchani.mnp" in importlib.metadata.metadata(
    __package__.split(".")[0]
).get_all("Provides", [])

# This env var is meant to be used by developers to manually disable extensions
# for testing purposes
if "TORCHANI_DISABLE_EXTENSIONS" not in os.environ:
    CUAEV_IS_INSTALLED = False
    MNP_IS_INSTALLED = False


__all__ = ["CUAEV_IS_INSTALLED", "MNP_IS_INSTALLED"]
