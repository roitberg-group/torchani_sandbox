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
import torch
from torch import Tensor
from typing import Tuple, Optional, NamedTuple
from .nn import SpeciesConverter, SpeciesEnergies
from .aev import AEVComputer
from .compat import Final


class SpeciesEnergiesQBC(NamedTuple):
    species: Tensor
    energies: Tensor
    qbcs: Tensor


class BuiltinModel(torch.nn.Module):
    r"""Private template for the builtin ANI models """

    def __init__(self, species_converter, aev_computer, neural_networks, energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index):
        super().__init__()
        self.species_converter = species_converter
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter
        self._species_to_tensor = species_to_tensor
        self.species = consts.species
        self.periodic_table_index = periodic_table_index

        # a bit useless maybe
        self.consts = consts
        self.sae_dict = sae_dict

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

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)

        # check if unknown species are included
        if species_coordinates[0].ge(self.aev_computer.num_species).any():
            raise ValueError(f'Unknown species found in {species_coordinates[0]}')

        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.neural_networks(species_aevs)
        return self.energy_shifter(species_energies)

    @torch.jit.export
    def atomic_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        pbc: Optional[Tensor] = None, average: bool = True) -> SpeciesEnergies:
        """Calculates predicted atomic energies of all atoms in a molecule

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_atomic_energies: species and energies for the given configurations
                note that the shape of species is (C, A), where C is
                the number of configurations and A the number of atoms, and
                the shape of energies is (C, A) for a BuiltinModel.

        If average is True (the default) it returns the average over all models
        in the ensemble, should there be more than one (shape (C, A)),
        otherwise it returns one atomic energy per model (shape (M, C, A)).
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)

        species, aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)

        atomic_energies = self.neural_networks._atomic_energies((species, aevs))

        atomic_saes = self.energy_shifter._calc_atomic_saes(species)

        # shift all atomic energies individually
        if atomic_energies.dim() == 2:
            atomic_energies = atomic_energies.unsqueeze(0)

        assert atomic_saes.shape == atomic_energies[0].shape

        atomic_energies += atomic_saes
        if average:
            return SpeciesEnergies(species, atomic_energies.mean(dim=0))
        return SpeciesEnergies(species, atomic_energies)

    # unfortunately this is an UGLY workaround to a torchscript bug
    @torch.jit.export
    def _recast_long_buffers(self):
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(dtype=torch.long)
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)
        self.aev_computer.neighborlist._recast_long_buffers()

    def species_to_tensor(self, *args, **kwargs):
        """Convert species from strings to tensor.

        See also :method:`torchani.neurochem.Constant.species_to_tensor`

        Arguments:
            species (:class:`str`): A string of chemical symbols

        Returns:
            tensor (:class:`torch.Tensor`): A 1D tensor of integers
        """
        # The only difference between this and the "raw" private version
        # _species_to_tensor is that this sends the final tensor to the model
        # device
        return self._species_to_tensor(*args, **kwargs) \
            .to(self.aev_computer.radial_terms.ShfR.device)

    def ase(self, **kwargs):
        """Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`int`): A calculator to be used with ASE
        """
        from . import ase
        return ase.Calculator(self.species, self, **kwargs)

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
        ret = BuiltinModel(self.species_converter, self.aev_computer,
                           self.neural_networks[index], self.energy_shifter,
                           self._species_to_tensor, self.consts, self.sae_dict,
                           self.periodic_table_index)
        return ret

    @torch.jit.export
    def members_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                         cell: Optional[Tensor] = None,
                         pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted energies of all member modules

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: species and energies for the given configurations
                note that the shape of species is (C, A), where C is
                the number of configurations and A the number of atoms, and
                the shape of energies is (M, C), where M is the number
                of modules in the ensemble

        """
        assert self.neural_networks.size > 1, "There is only one set of atomic networks in your model"
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species, aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        member_outputs = []
        for nnp in self.neural_networks:
            # hint for JIT, this function can only be called if neural_networks
            # is an ensemble and not an ANIModel
            assert not isinstance(nnp, str)
            unshifted_energies = nnp((species, aevs)).energies
            shifted_energies = self.energy_shifter((species, unshifted_energies)).energies
            member_outputs.append(shifted_energies.unsqueeze(0))
        return SpeciesEnergies(species, torch.cat(member_outputs, dim=0))

    @torch.jit.export
    def energies_qbcs(self, species_coordinates: Tuple[Tensor, Tensor],
                      cell: Optional[Tensor] = None,
                      pbc: Optional[Tensor] = None, unbiased: bool = True) -> SpeciesEnergiesQBC:
        """Calculates predicted predicted energies and qbc factors

        QBC factors are used for query-by-committee (QBC) based active learning
        (as described in the ANI-1x paper `less-is-more`_ ).

        .. _less-is-more:
            https://aip.scitation.org/doi/10.1063/1.5023802

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            and qbc factors will be in Hartree.

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


def _build_neurochem_model(info_file_path, periodic_table_index=False, external_cell_list=False, model_index=None, torch_cell_list=False, adaptive_torch_cell_list=False):
    from . import neurochem  # noqa
    # builder function that creates a BuiltinModel from a neurochem info path
    assert not (external_cell_list and torch_cell_list)

    const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)
    consts = neurochem.Constants(const_file)

    if torch_cell_list:
        aev_computer = AEVComputer(**consts, neighborlist='cell_list')
    elif adaptive_torch_cell_list:
        aev_computer = AEVComputer(**consts, neighborlist='verlet_cell_list')
    else:
        aev_computer = AEVComputer(**consts)

    if model_index is None:
        neural_networks = neurochem.load_model_ensemble(consts.species, ensemble_prefix, ensemble_size)
    else:
        if (model_index >= ensemble_size):
            raise ValueError(f"The ensemble size is only {ensemble_size}, model {model_index} can't be loaded")
        network_dir = os.path.join('{}{}'.format(ensemble_prefix, model_index), 'networks')
        neural_networks = neurochem.load_model(consts.species, network_dir)

    energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)

    kwargs = {'sae_dict': sae_dict,
            'consts': consts,
            'species_converter': SpeciesConverter(consts.species),
            'aev_computer': aev_computer,
            'energy_shifter': energy_shifter,
            'species_to_tensor': consts.species_to_tensor,
            'neural_networks': neural_networks,
            'periodic_table_index': periodic_table_index}

    if external_cell_list:
        return BuiltinModelExternalInterface(**kwargs)
    else:
        return BuiltinModel(**kwargs)


def ANI1x(*args, **kwargs):
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
    return _build_neurochem_model(*args, **kwargs, info_file_path=info_file)


def ANI1ccx(*args, **kwargs):
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
    return _build_neurochem_model(*args, **kwargs, info_file_path=info_file)


def ANI2x(*args, **kwargs):
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
    return _build_neurochem_model(*args, **kwargs, info_file_path=info_file)
