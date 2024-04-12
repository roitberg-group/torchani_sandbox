r"""
The assembler's responsibility is to build an ANI-style model from the different
necessary parts, in such a way that all the parts of the model interact in the correct
way and there are no compatibility issues among them.

An energy-predicting ANI-style model consists of:

- Featurizer (typically an AEVComputer, or subclass)
- Container for atomic networks (typically ANIModel)
- Atomic Networks Dict {"H": torch.nn.Module(), "C": torch.nn.Module, ...}
- Self Energies Dict (In Ha) {"H": -12.0, "C": -75.0, ...}
- Shifter (typically EnergyShifter, or subclass)

One or more Potentials (Typically RepulsionXTB, TwoBodyDispersionD3 or Coulombic potentials)
(TBA, VDW potential)

Each of the potentials will have their own cutoff, and the Featurizer has two
cutoffs, an angular and a radial cutoff (the radial cutoff must be larger than
the angular cutoff, and it is recommended that the angular cutoff is kept
small, roughly 3.5 Ang or less).

These pieces are assembled into a Model, which is a subclass of BuiltinModel
"""
import math
from dataclasses import dataclass
from collections import OrderedDict
import typing as tp

import torch

from torchani.models import BuiltinModel, BuiltinModelPairInteractions
from torchani.neighbors import BaseNeighborlist, _parse_neighborlist
from torchani.cutoffs import _parse_cutoff_fn, Cutoff
from torchani.potentials import PairwisePotential
from torchani.aev import AEVComputer
from torchani.nn import ANIModel, Ensemble
from torchani.utils import EnergyShifter, GSAES

ModelType = tp.Type[BuiltinModel]
NeighborlistType = tp.Type[BaseNeighborlist]
FeaturizerType = tp.Type[AEVComputer]
PairwisePotentialType = tp.Type[PairwisePotential]
AtomicContainerType = tp.Type[ANIModel]
ShifterType = tp.Type[EnergyShifter]


# None cutoff means the global cutoff_fn will be used
# Otherwise, a specific cutoff fn can be specified for the wrapper or the
# Featurizer
class FeaturizerWrapper:
    def __init__(
        self,
        cls: FeaturizerType,
        cutoff_fn: tp.Union[Cutoff, str] = "global",
        radial_cutoff: float = 5.2,
        angular_cutoff: float = 3.5,
        lot: str = "",
    ) -> None:
        if radial_cutoff < 0 or angular_cutoff < 0:
            raise ValueError("Both the radial and angular cutoffs shold be positive or 0")
        if radial_cutoff < angular_cutoff:
            raise ValueError("Radial cutoff must be larger than the angular cutoff")
        self.cls = cls
        self.cutoff_fn = cutoff_fn
        self.radial_cutoff = radial_cutoff
        self.angular_cutoff = angular_cutoff

    @property
    def cutoff(self) -> float:
        return self.radial_cutoff


@dataclass
class PotentialWrapper:
    cls: PairwisePotential
    cutoff_fn: tp.Union[Cutoff, str] = "global"
    cutoff: float = math.inf
    lot: str = ""


class Assembler:
    def __init__(self) -> None:

        self._neighborlist_type: tp.Union[NeighborlistType, str]
        self._featurizer: FeaturizerWrapper = FeaturizerWrapper(AEVComputer)
        self._pairwise_potentials: tp.List[PotentialWrapper] = []

        # This part of the assembler organizes the self-energies, the
        # symbols and the atomic networks
        self._self_energies: tp.Dict[str, float] = {}
        self._atomic_networks: tp.Dict[str, torch.nn.Module] = {}
        self._shifter_type: ShifterType = EnergyShifter
        self._atomic_container_type: AtomicContainerType = ANIModel
        self._symbols: tp.Tuple[str, ...] = ()
        self._ensemble_size: int = 1

        # This is the general container for all the parts of the model
        self._model_type: ModelType = BuiltinModel

        # This is a deprecated feature, it should probably not be used
        self.periodic_table_index = True

    @property
    def ensemble_size(self) -> int:
        return self._ensemble_size

    @ensemble_size.setter
    def ensemble_size(self, value: int) -> None:
        if value < 0:
            raise ValueError("Ensemble size must be positive")
        self._ensemble_size = value

    @property
    def elements_num(self) -> int:
        return len(self._symbols)

    @property
    def symbols(self) -> tp.Tuple[str, ...]:
        return self._symbols

    @symbols.setter
    def symbols(self, symbols: tp.Sequence[str]) -> None:
        self._symbols = tuple(symbols)

    @property
    def atomic_networks(self) -> tp.OrderedDict[str, torch.nn.Module]:
        odict = OrderedDict()
        for k in self.symbols:
            odict[k] = self._atomic_networks[k]
        return odict

    @atomic_networks.setter
    def atomic_networks(self, value: tp.Mapping[str, torch.nn.Module]) -> None:
        if not self.symbols:
            # TODO: sort symbols correctly
            self.symbols = tuple(sorted(value.keys()))
        elif set(self.symbols) != set(value.keys()):
            raise ValueError(
                f"Atomic networks don't match supported elements {self._symbols}"
            )
        self._atomic_networks = {k: v for k, v in value.items()}

    @property
    def self_energies(self) -> tp.OrderedDict[str, float]:
        odict = OrderedDict()
        for k in self.symbols:
            odict[k] = self._self_energies[k]
        return odict

    @self_energies.setter
    def self_energies(self, value: tp.Mapping[str, float]) -> None:
        if not self.symbols:
            # TODO: sort symbols correctly
            self.symbols = tuple(sorted(value.keys()))
        elif set(self.symbols) != set(value.keys()):
            raise ValueError(
                f"Self energies don't match supported elements {self._symbols}"
            )
        self._self_energies = {k: v for k, v in value.items()}

    def set_gsaes_as_self_energies(self, functional: str = "", basis_set: str = "", lot: str = "", symbols: tp.Iterable[str] = ()) -> None:
        if (functional and basis_set) and not lot:
            lot = f'{functional}-{basis_set}'
        elif not (functional or basis_set) and lot:
            pass
        else:
            raise ValueError("Incorrect specification, either specify only lot, or both functional and basis set")

        if not symbols:
            symbols = self.symbols
        gsaes = GSAES[lot.lower()]
        self.self_energies = OrderedDict([(s, gsaes[s]) for s in symbols])

    def set_shifter(self, shifter_type: ShifterType) -> None:
        self._shifter_type = shifter_type

    def set_atomic_container(self, atomic_container_type: AtomicContainerType) -> None:
        self._atomic_container_type = atomic_container_type

    def set_featurizer(
        self,
        featurizer_type: FeaturizerType,
        cutoff_fn: tp.Union[Cutoff, str] = "global",
        angular_cutoff: float = 3.5,
        radial_cutoff: float = 5.1,
    ) -> None:
        self._featurizer = FeaturizerWrapper(
            featurizer_type,
            cutoff_fn=cutoff_fn,
            angular_cutoff=angular_cutoff,
            radial_cutoff=radial_cutoff,
        )

    def set_neighborlist(self, neighborlist_type: tp.Union[NeighborlistType, str]) -> None:
        if isinstance(neighborlist_type, str) and neighborlist_type not in ["full_pairwise", "cell_list"]:
            raise ValueError("Unsupported neighborlist")
        self._neighborlist_type = neighborlist_type

    def set_global_cutoff_fn(self, cutoff_fn: tp.Union[Cutoff, str]) -> None:
        self._general_cutoff_fn = _parse_cutoff_fn(cutoff_fn)

    def add_pairwise_potential(self, pair_type: PairwisePotentialType, cutoff: float = math.inf, cutoff_fn: tp.Union[Cutoff, str] = "global") -> None:
        if not issubclass(self._model_type, BuiltinModelPairInteractions):
            # Override the model if it is exactly equal to this class
            if self._model_type == BuiltinModel:
                self._model_type = BuiltinModelPairInteractions
            else:
                raise ValueError("The model class must support pairwise potentials in order to add potentials")
        self._pairwise_potentials.append(pair_type)

    def assemble(self) -> BuiltinModel:
        # Here it is necessary to get the largest cutoff to attach to the neighborlist, right?
        max_cutoff = 0.0
        neighborlist = _parse_neighborlist(self.neighborlist_type, max_cutoff)

        featurizer = self._featurizer.cls(
            cutoff_fn=self._featurizer.cutoff_fn if self._featurizer,
            neighborlist=neighborlist,
            angular_terms=self.angular,
            radial_terms=self.radial,
            num_species=self.elements_num,
        )
        neural_networks: tp.Union[ANIModel, Ensemble]
        if self.ensemble_size > 1:
            containers = []
            for j in range(self.ensemble_size):
                containers.append(self._atomic_container_type(self.atomic_networks))
            neural_networks = Ensemble(containers)
        else:
            neural_networks = self._atomic_container_type(self.atomic_networks)
        shifter = self._shifter_type(tuple(self.self_energies.values()))

        potentials = self.pairwise_potential_dict
        if potentials:
            pairwise_potentials = []
            for name, Cls in potentials.items():
                pairwise_potentials.append(
                    Cls(
                        symbols=self.symbols,
                        cutoff=self.cutoffs[name],
                        cutoff_fn=self.cutoff_fns[name],
                    )
                )
            kwargs = {"pairwise_potentials": pairwise_potentials}
        else:
            kwargs = {}
        return self._model_type(
            aev_computer=featurizer,
            energy_shifter=shifter,
            elements=self.symbols,
            neural_networks=neural_networks,
            periodic_table_index=self.periodic_table_index,
            **kwargs,
        )
