import importlib.metadata

CUAEV_IS_INSTALLED = "torchani.cuaev" in importlib.metadata.metadata(
    __package__.split(".")[0]
).get_all("Provides", [])

MNP_IS_INSTALLED = "torchani.mnp" in importlib.metadata.metadata(
    __package__.split(".")[0]
).get_all("Provides", [])


__all__ = ["CUAEV_IS_INSTALLED", "MNP_IS_INSTALLED"]
