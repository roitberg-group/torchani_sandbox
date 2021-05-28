# -*- coding: utf-8 -*-
"""The ANI model zoo that stores public ANI models.

Currently the model zoo has three models: ANI-1x, ANI-1ccx, and ANI-2x.
The parameters of these models are stored in `ani-model-zoo`_ repository and
will be automatically downloaded the first time any of these models are
instantiated. The classes of these models are :class:`ANI1x`, :class:`ANI1ccx`,
and :class:`ANI2x` these are subclasses of :class:`torch.nn.Module`.
To use the models just instantiate them and either
directly calculate energies or get an ASE calculator. For example:

.. _ani-model-zoo:
    https://github.com/aiqm/ani-model-zoo

.. code-block:: python

    ani1x = torchani.models.ANI1x()
    # compute energy using ANI-1x model ensemble
    _, energies = ani1x((species, coordinates))
    ani1x.ase()  # get ASE Calculator using this ensemble
    # convert atom species from string to long tensor
    ani1x.species_to_tensor(['C', 'H', 'H', 'H', 'H'])

    model0 = ani1x[0]  # get the first model in the ensemble
    # compute energy using the first model in the ANI-1x model ensemble
    _, energies = model0((species, coordinates))
    model0.ase()  # get ASE Calculator using this model
    # convert atom species from string to long tensor
    model0.species_to_tensor(['C', 'H', 'H', 'H', 'H'])
"""
import os
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
import torch
from torch import Tensor
from typing import Tuple, Optional, NamedTuple, Sequence, Union, Type
from .nn import SpeciesConverter, SpeciesEnergies, Ensemble, ANIModel
from .utils import ChemicalSymbolsToInts, PERIODIC_TABLE, EnergyShifter, path_is_writable
from .aev import AEVComputer
from .compat import Final
from . import atomics


NN = Union[ANIModel, Ensemble]


class SpeciesEnergiesQBC(NamedTuple):
    species: Tensor
    energies: Tensor
    qbcs: Tensor


class BuiltinModel(torch.nn.Module):
    r"""Private template for the builtin ANI models """

    atomic_numbers: Tensor
    periodic_table_index: Final[bool]

    def __init__(self,
                 aev_computer: AEVComputer,
                 neural_networks: NN,
                 energy_shifter,
                 elements: Sequence[str],
                 periodic_table_index: bool = False):

        super().__init__()

        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter
        self.species_to_tensor = ChemicalSymbolsToInts(elements)
        self.species_converter = SpeciesConverter(elements)

        self.periodic_table_index = periodic_table_index
        numbers = torch.tensor([PERIODIC_TABLE.index(e) for e in elements], dtype=torch.long)
        self.register_buffer('atomic_numbers', numbers)

        # checks are performed to make sure all modules passed support the
        # correct number of species
        if energy_shifter.fit_intercept:
            assert len(energy_shifter.self_energies) == len(self.atomic_numbers) + 1
        else:
            assert len(energy_shifter.self_energies) == len(self.atomic_numbers)

        assert len(self.atomic_numbers) == self.aev_computer.num_species

        if isinstance(self.neural_networks, Ensemble):
            for nnp in self.neural_networks:
                assert len(nnp) == len(self.atomic_numbers)
        else:
            assert len(self.neural_networks) == len(self.atomic_numbers)

    @torch.jit.unused
    def get_chemical_symbols(self) -> Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted properties for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: energies for the given configurations
        """
        in_species, species_idx, coordinates = self._get_species_and_indices(species_coordinates)
        aevs = self.aev_computer((species_idx, coordinates), cell=cell, pbc=pbc).aevs
        energies = self.neural_networks((species_idx, aevs)).energies
        energies = self.energy_shifter((species_idx, energies)).energies
        return SpeciesEnergies(in_species, energies)

    @torch.jit.export
    def _get_species_and_indices(self, species_coordinates: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        in_species, coordinates = species_coordinates

        if self.periodic_table_index:
            species_idx = self.species_converter(species_coordinates).species
        else:
            species_idx = in_species.clone()

        if (species_idx >= self.aev_computer.num_species).any():
            raise ValueError(f'Unknown species found in {species_coordinates[0]}')

        return in_species, species_idx, coordinates

    def atomic_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        pbc: Optional[Tensor] = None, average: bool = True) -> SpeciesEnergies:
        """Calculates predicted atomic energies of all atoms in a molecule

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled
            average: If True (the default) it returns the average over all models
                in the ensemble, should there be more than one (shape (C, A)),
                otherwise it returns one atomic energy per model (shape (M, C, A)).

        Returns:
            species_atomic_energies: species and energies for the given configurations
                note that the shape of species is (C, A), where C is
                the number of configurations and A the number of atoms, and
                the shape of energies is (C, A) for a BuiltinModel.
        """
        in_species, species_idx, coordinates = self._get_species_and_indices(species_coordinates)
        aevs = self.aev_computer((species_idx, coordinates), cell=cell, pbc=pbc).aevs
        atomic_energies = self.neural_networks._atomic_energies((species_idx, aevs))
        atomic_energies += self.energy_shifter._atomic_saes(species_idx)

        if atomic_energies.dim() == 2:
            atomic_energies = atomic_energies.unsqueeze(0)
        if average:
            atomic_energies = atomic_energies.mean(dim=0)
        return SpeciesEnergies(in_species, atomic_energies)

    @torch.jit.export
    def _recast_long_buffers(self):
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(dtype=torch.long)
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)

    def ase(self, **kwargs):
        """Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`ase.Calculator`): A calculator to be used with ASE
        """
        from . import ase
        return ase.Calculator(self.get_chemical_symbols(), self, **kwargs)

    def __getitem__(self, index):
        """Get a model that uses a single network

        Indexing allows access to a single model inside the ensemble
        that can be used directly for calculations.

        Args:
            index (:class:`int`): Index of the model

        Returns:
            ret: (:class:`torchani.models.BuiltinModel`) Model ready for
                calculations
        """
        assert self.neural_networks.size > 1, "There is only one set of atomic networks in your model"
        ret = BuiltinModel(self.aev_computer,
                           self.neural_networks[index],
                           self.energy_shifter,
                           self.get_chemical_symbols(),
                           self.periodic_table_index)
        return ret

    @torch.jit.export
    def members_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                         cell: Optional[Tensor] = None,
                         pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted energies of all member modules

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: species and energies for the given configurations
                shape of species is (C, A), where C is the number of
                configurations and A the number of atoms, and shape of energies
                is (M, C), where M is the number of modules in the ensemble.

        """
        species, members_energies = self.atomic_energies(species_coordinates, average=False)
        return SpeciesEnergies(species, members_energies.sum(-1))

    @torch.jit.export
    def energies_qbcs(self, species_coordinates: Tuple[Tensor, Tensor],
                      cell: Optional[Tensor] = None,
                      pbc: Optional[Tensor] = None, unbiased: bool = True) -> SpeciesEnergiesQBC:
        """Calculates predicted predicted energies and qbc factors

        QBC factors are used for query-by-committee (QBC) based active learning
        (as described in the ANI-1x paper `less-is-more`_ ).

        .. _less-is-more:
            https://aip.scitation.org/doi/10.1063/1.5023802

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not
                enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to None if PBC is not enabled
            unbiased: if `True` then Bessel's correction is applied to the
                standard deviation over the ensemble member's. If `False` Bessel's
                correction is not applied, True by default.

        Returns:
            species_energies_qbcs: species, energies and qbc factors for the
                given configurations. Note that the shape of species is (C, A),
                where C is the number of configurations and A the number of
                atoms, the shape of energies is (C,) and the shape of qbc
                factors is also (C,).
        """
        assert self.neural_networks.size > 1, "There is only one set of atomic networks in your model"
        species, energies = self.members_energies(species_coordinates, cell, pbc)

        # standard deviation is taken across ensemble members
        qbc_factors = energies.std(0, unbiased=unbiased)

        # rho's (qbc factors) are weighted by dividing by the square root of
        # the number of atoms in each molecule
        num_atoms = (species >= 0).sum(dim=1, dtype=energies.dtype)
        qbc_factors = qbc_factors / num_atoms.sqrt()
        energies = energies.mean(dim=0)
        assert qbc_factors.shape == energies.shape
        return SpeciesEnergiesQBC(species, energies, qbc_factors)

    def __len__(self):
        """Get the number of networks being used by the model

        Returns:
            length (:class:`int`): Number of networks in the ensemble
        """
        return self.neural_networks.size


class BuiltinModelExternalInterface(BuiltinModel):
    # TODO: Most BuiltinModel functions fail here, only forward works this
    # It will be necessary to rewrite that code for this use case if we
    # want those functions in amber, I'm looking for a different solution though

    assume_screened_input: Final[bool]

    def __init__(self, *args, **kwargs):
        assume_screened_input = kwargs.pop('assume_screened_input', False)
        super().__init__(*args, **kwargs)
        self.assume_screened_input = assume_screened_input

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                neighbors: Optional[Tensor] = None,
                shift_values: Optional[Tensor] = None) -> SpeciesEnergies:
        # It is convenient to keep these arguments optional due to JIT, but
        # actually they are needed for this class
        assert neighbors is not None
        assert shift_values is not None

        # check consistency of shapes of neighborlist
        assert neighbors.dim() == 2 and neighbors.shape[0] == 2
        assert shift_values.dim() == 2 and shift_values.shape[1] == 3
        assert neighbors.shape[1] == shift_values.shape[0]

        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        # check if unknown species are included
        if species_coordinates[0].ge(self.aev_computer.num_species).any():
            raise ValueError(f'Unknown species found in {species_coordinates[0]}')

        species, coordinates = species_coordinates
        # check shapes for correctness
        assert species.dim() == 2
        assert coordinates.dim() == 3
        assert (species.shape == coordinates.shape[:2]) and (coordinates.shape[2] == 3)

        if not self.assume_screened_input:
            # first we screen the input neighbors in case some of the
            # values are at distances larger than the radial cutoff, or some of
            # the values are masked with dummy atoms. The first may happen if
            # the neighborlist uses some sort of skin value to rebuild itself
            # (as in Loup Verlet lists), which is common in MD programs.
            cutoff = self.aev_computer.radial_terms.cutoff
            nl_out = self.aev_computer.neighborlist._screen_with_cutoff(cutoff,
                                                           coordinates,
                                                           neighbors,
                                                           shift_values,
                                                           (species == -1))
            neighbors, _, diff_vectors, distances = nl_out
        else:
            # if the input neighbors is assumed to be pre screened then we
            # just calculate the distances and diff_vector here
            coordinates = coordinates.view(-1, 3)
            coords0 = coordinates.index_select(0, neighbors[0])
            coords1 = coordinates.index_select(0, neighbors[1])
            diff_vectors = coords0 - coords1 + shift_values
            distances = diff_vectors.norm(2, -1)

        assert neighbors is not None
        aevs = self.aev_computer._compute_aev(species, neighbors, diff_vectors, distances)
        species_energies = self.neural_networks((species, aevs))
        return self.energy_shifter(species_energies)

    @torch.jit.export
    def energies_qbcs(self, species_coordinates: Tuple[Tensor, Tensor],
                neighbors: Optional[Tensor] = None,
                shift_values: Optional[Tensor] = None):
        assert False, "Not implemented for external interface"

    @torch.jit.export
    def atomic_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                neighbors: Optional[Tensor] = None,
                shift_values: Optional[Tensor] = None):
        assert False, "Not implemented for external interface"

    @torch.jit.export
    def members_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                neighbors: Optional[Tensor] = None,
                shift_values: Optional[Tensor] = None):
        assert False, "Not implemented for external interface"


def _get_component_modules(state_dict_file: str,
                           model_index: Optional[int] = None,
                           use_cuda_extension: bool = False,
                           ensemble_size: int = 8) -> Tuple[AEVComputer, NN, EnergyShifter, Sequence[str]]:
    # This generates ani-style architectures without neurochem
    name = state_dict_file.split('_')[0]
    elements: Tuple[str, ...]
    if name == 'ani1x':
        aev_maker = AEVComputer.like_1x
        atomic_maker = atomics.like_1x
        elements = ('H', 'C', 'N', 'O')
    elif name == 'ani1ccx':
        aev_maker = AEVComputer.like_1ccx
        atomic_maker = atomics.like_1ccx
        elements = ('H', 'C', 'N', 'O')
    elif name == 'ani2x':
        aev_maker = AEVComputer.like_2x
        atomic_maker = atomics.like_2x
        elements = ('H', 'C', 'N', 'O', 'S', 'F', 'Cl')
    else:
        raise ValueError(f'{name} is not a supported model')
    aev_computer = aev_maker(use_cuda_extension=use_cuda_extension)
    atomic_networks = OrderedDict([(e, atomic_maker(e)) for e in elements])

    neural_networks: NN
    if model_index is None:
        neural_networks = Ensemble([ANIModel(deepcopy(atomic_networks)) for _ in range(ensemble_size)])
    else:
        neural_networks = ANIModel(atomic_networks)
    return aev_computer, neural_networks, EnergyShifter([0.0 for _ in elements]), elements


def _fetch_state_dict(state_dict_file: str,
                      model_index: Optional[int] = None,
                      local: bool = False) -> 'OrderedDict[str, Tensor]':
    # if we want a pretrained model then we load the state dict from a
    # remote url or a local path
    # NOTE: torch.hub caches remote state_dicts after they have been downloaded
    if local:
        return torch.load(state_dict_file)

    model_dir = Path(__file__).parent.joinpath('resources/state_dicts').as_posix()
    if not path_is_writable(model_dir):
        model_dir = os.path.expanduser('~/.local/torchani/')

    # NOTE: we need some private url for in-development models of the
    # group, this url is for public models
    tag = 'v0.1'
    url = f'https://github.com/roitberg-group/torchani_model_zoo/releases/download/{tag}/{state_dict_file}'
    # for now for simplicity we load a state dict for the ensemble directly and
    # then parse if needed
    state_dict = torch.hub.load_state_dict_from_url(url, model_dir=model_dir)

    if model_index is not None:
        new_state_dict = OrderedDict()
        # Parse the state dict and rename/select only useful keys to build
        # the individual model
        for k, v in state_dict.items():
            tkns = k.split('.')
            if tkns[0] == 'neural_networks':
                # rename or discard the key
                if int(tkns[1]) == model_index:
                    tkns.pop(1)
                    k = '.'.join(tkns)
                else:
                    continue
            new_state_dict[k] = v
        state_dict = new_state_dict

    return state_dict


def _load_ani_model(state_dict_file: Optional[str] = None,
                    info_file: Optional[str] = None,
                    **model_kwargs) -> BuiltinModel:
    # Helper function to toggle if the loading is done from an NC file or
    # directly using torchani and state_dicts
    use_neurochem_source = model_kwargs.pop('use_neurochem_source', False)
    use_cuda_extension = model_kwargs.pop('use_cuda_extension', False)
    model_index = model_kwargs.pop('model_index', None)
    pretrained = model_kwargs.pop('pretrained', True)
    external_neighborlist = model_kwargs.pop('external_neighborlist', False)

    if use_neurochem_source:
        assert info_file is not None, "Info file is needed to load from a neurochem source"
        assert pretrained, "Non pretrained models not available from neurochem source"
        from . import neurochem  # noqa
        components = neurochem.parse_resources._get_component_modules(info_file, model_index, use_cuda_extension)
    else:
        assert state_dict_file is not None
        components = _get_component_modules(state_dict_file, model_index, use_cuda_extension)

    aev_computer, neural_networks, energy_shifter, elements = components

    model_class: Type[BuiltinModel]
    if external_neighborlist:
        model_class = BuiltinModelExternalInterface
    else:
        model_class = BuiltinModel

    model = model_class(aev_computer, neural_networks, energy_shifter, elements, **model_kwargs)

    if pretrained and not use_neurochem_source:
        assert state_dict_file is not None
        model.load_state_dict(_fetch_state_dict(state_dict_file, model_index))
    return model


def ANI1x(**kwargs):
    """The ANI-1x model as in `ani-1x_8x on GitHub`_ and `Active Learning Paper`_.

    The ANI-1x model is an ensemble of 8 networks that was trained using
    active learning on the ANI-1x dataset, the target level of theory is
    wB97X/6-31G(d). It predicts energies on HCNO elements exclusively, it
    shouldn't be used with other atom types.

    .. _ani-1x_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1x_8x

    .. _Active Learning Paper:
        https://aip.scitation.org/doi/abs/10.1063/1.5023802
    """
    info_file = 'ani-1x_8x.info'
    state_dict_file = 'ani1x_state_dict.pt'
    return _load_ani_model(state_dict_file, info_file, **kwargs)


def ANI1ccx(**kwargs):
    """The ANI-1ccx model as in `ani-1ccx_8x on GitHub`_ and `Transfer Learning Paper`_.

    The ANI-1ccx model is an ensemble of 8 networks that was trained
    on the ANI-1ccx dataset, using transfer learning. The target accuracy
    is CCSD(T)*/CBS (CCSD(T) using the DPLNO-CCSD(T) method). It predicts
    energies on HCNO elements exclusively, it shouldn't be used with other
    atom types.

    .. _ani-1ccx_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1ccx_8x

    .. _Transfer Learning Paper:
        https://doi.org/10.26434/chemrxiv.6744440.v1
    """
    info_file = 'ani-1ccx_8x.info'
    state_dict_file = 'ani1ccx_state_dict.pt'
    return _load_ani_model(state_dict_file, info_file, **kwargs)


def ANI2x(**kwargs):
    """The ANI-2x model as in `ANI2x Paper`_ and `ANI2x Results on GitHub`_.

    The ANI-2x model is an ensemble of 8 networks that was trained on the
    ANI-2x dataset. The target level of theory is wB97X/6-31G(d). It predicts
    energies on HCNOFSCl elements exclusively it shouldn't be used with other
    atom types.

    .. _ANI2x Results on GitHub:
        https://github.com/cdever01/ani-2x_results

    .. _ANI2x Paper:
        https://doi.org/10.26434/chemrxiv.11819268.v1
    """
    info_file = 'ani-2x_8x.info'
    state_dict_file = 'ani2x_state_dict.pt'
    return _load_ani_model(state_dict_file, info_file, **kwargs)
