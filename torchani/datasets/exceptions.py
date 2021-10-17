from typing import Mapping, Any


class IncompatibleDummyProperty(ValueError):
    def __init__(self, incompatibles: Mapping[str, Any]):
        self.incompatibles = incompatibles
