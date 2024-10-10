r"""
The ``models`` submodule provides access to all published ANI models, which are
subclasses of ``ANI``. Some models have been published in specific articles,
and some have been published in TorchANI 2.0. If you use any of these models in
your work please cite the corresponding article(s).

If for a given model you discover a bug, performance problem, or incorrect
behavior in some region of chemical space, please post an issue in GitHub. The
TorchANI developers will attempt to address and document issues.

Note that parameters of the ANI models are automatically downloaded and cached
the first time they are instantiated. If this is an issue for your application
we recommend you pre-download the parameters by instantiating the models before
use.

The ANI models can be used directly as callable functions once they are
instantiated. Alternatively, they can be cast to an ASE calculator by calling
``ANI.ase()``.

Some models have a "network container" that consists of an "ensemble" of neural
networks. Individual members of these ensembles can be accessed by indexing,
and ``len()`` can be used to query the number of networks in it.

The models also have three extra entry points for more specific use cases:
atomic_energies and energies_qbcs.

All entrypoints expect a tuple of tensors `(species, coordinates)` as input,
together with two optional tensors, `cell` and `pbc`.
`coordinates` and `cell` should be in units of Angstroms,
and the output energies are always in Hartrees

For more detailed examples of usage consult the examples documentation

.. code-block:: python

    import torchani

    model = torchani.models.ANI2x()

    # Batch of conformers
    # shape is (molecules, atoms) for znums and (molecules, atoms, 3) for coords
    znums = torch.tensor([[8, 1, 1]], dtype=torch.long)
    coords = torch.tensor([[...], [...], [...]], dtype=torch.float)

    # Average energies over the ensemble for the batch
    # Output shape is (molecules,)
    _, energies = model((species, coords))

    # Average atomic energies over the ensemble for the batch
    # Output shape is (molecules, atoms)
    _, atomic_ene = model.atomic_energies((species, coords))

    # Individual energies of the members of the ensemble
    # Output shape is (ensemble-size, molecules) or (ensemble-size, molecules, atoms)
    _, energies = model((species, coords), ensemble_mean=False)
    _, atomic_energies = model.atomic_energies((species, coords), ensemble_mean=False)


    # QBC factors are used for active learning, shape is (molecules,)
    _, energies, qbcs = model.energies_qbcs((species, coords))

    # Individual models of the ensemble can be obtained by indexing, they are also
    # subclasses of ``ANI``, with the same functionality
    member = model[0]
"""

import typing as tp

import torch
from torch import Tensor
from torch.jit import Final
import typing_extensions as tpx

from torchani.tuples import (
    SpeciesEnergies,
    SpeciesEnergiesAtomicCharges,
    SpeciesEnergiesQBC,
    AtomicStdev,
    SpeciesForces,
    ForceStdev,
    ForceMagnitudes,
)
from torchani.electro import ChargeNormalizer
from torchani.atomics import AtomicContainer
from torchani.nn import SpeciesConverter
from torchani.constants import PERIODIC_TABLE, ATOMIC_NUMBER
from torchani.aev import AEVComputer
from torchani.potentials import (
    NNPotential,
    SeparateChargesNNPotential,
    Potential,
    PairPotential,
    EnergyAdder,
)
from torchani.neighbors import rescreen, NeighborData


class ANI(torch.nn.Module):
    r"""ANI-style neural network interatomic potential"""

    atomic_numbers: Tensor
    periodic_table_index: Final[bool]

    def __init__(
        self,
        symbols: tp.Sequence[str],
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        energy_shifter: EnergyAdder,
        pairwise_potentials: tp.Iterable[PairPotential] = (),
        periodic_table_index: bool = True,
    ):
        super().__init__()

        # NOTE: Keep these refs for later usage
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.neighborlist = self.aev_computer.neighborlist

        device = energy_shifter.self_energies.device
        self.energy_shifter = energy_shifter
        self.species_converter = SpeciesConverter(symbols).to(device)

        potentials: tp.List[Potential] = list(pairwise_potentials)
        potentials.append(NNPotential(self.aev_computer, self.neural_networks))
        self.potentials_len = len(potentials)

        # Sort potentials in order of decresing cutoff. The potential with the
        # LARGEST cutoff is computed first, then sequentially things that need
        # SMALLER cutoffs are computed.
        potentials = sorted(potentials, key=lambda x: x.cutoff, reverse=True)
        self.potentials = torch.nn.ModuleList(potentials)

        self.periodic_table_index = periodic_table_index
        numbers = torch.tensor([ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long)
        self.register_buffer("atomic_numbers", numbers)

        # Make sure all modules passed support the correct num species
        assert len(self.energy_shifter.self_energies) == len(self.atomic_numbers)
        assert self.aev_computer.num_species == len(self.atomic_numbers)
        assert self.neural_networks.num_species == len(self.atomic_numbers)

    def forward(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: float = 0.0,
        ensemble_average: bool = True,
        shift_energy: bool = True,
    ) -> SpeciesEnergies:
        """Calculate energies for a minibatch of molecules

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to None if PBC is not enabled
            total_charge: (float): The total charge of the molecules. Only
                the scalar 0.0 is currently supported.
            ensemble_average (bool): If True (default), return the average
                over all models in the ensemble (output shape ``(C, A)``), otherwise
                return one atomic energy per model (output shape ``(M, C, A)``).
            shift_energy (bool): Add the a constant energy shift to the returned
                energies. ``True`` by default.

        Returns:
            species_energies: tuple of tensors, species and energies for the
                given configurations
        """
        assert total_charge == 0.0, "Model only supports neutral molecules"

        # Unoptimized path to obtain member energies, and eventually QBC
        if not ensemble_average:
            elem_idxs, energies = self.atomic_energies(
                species_coordinates,
                cell=cell,
                pbc=pbc,
                total_charge=total_charge,
                ensemble_average=False,
                shift_energy=shift_energy,
            )
            return SpeciesEnergies(elem_idxs, energies.sum(-1))

        elem_idxs, coords = self._maybe_convert_species(species_coordinates)
        assert coords.shape[:-1] == elem_idxs.shape
        assert coords.shape[-1] == 3

        # Optimized path, use merged Neighborlist-AEVomputer
        if self.potentials_len == 1:
            species, energies = self.neural_networks(
                self.aev_computer((elem_idxs, coords), cell=cell, pbc=pbc)
            )
            return SpeciesEnergies(species, energies + self.energy_shifter(species))

        # Unoptimized path
        largest_cutoff = self.potentials[0].cutoff
        neighbors = self.neighborlist(elem_idxs, coords, largest_cutoff, cell, pbc)
        energies = self._energy_of_pots(elem_idxs, coords, largest_cutoff, neighbors)

        if shift_energy:
            energies += self.energy_shifter(elem_idxs)
        return SpeciesEnergies(elem_idxs, energies)

    @torch.jit.export
    def from_neighborlist(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        neighbor_idxs: Tensor,
        shift_values: Tensor,
        total_charge: float = 0.0,
        ensemble_average: bool = True,
        shift_energy: bool = True,
        input_needs_screening: bool = True,
    ) -> SpeciesEnergies:
        r"""
        This entrypoint supports input from an external neighborlist
        """
        assert total_charge == 0.0, "Model only supports neutral molecules"
        elem_idxs, coords = self._maybe_convert_species(species_coordinates)
        assert coords.shape[:-1] == elem_idxs.shape
        assert coords.shape[-1] == 3
        largest_cutoff = self.potentials[0].cutoff
        neighbors = self.neighborlist.process_external_input(
            elem_idxs,
            coords,
            neighbor_idxs,
            shift_values,
            largest_cutoff,
            input_needs_screening,
        )
        if not ensemble_average:
            energies = self._atomic_energy_of_pots(
                elem_idxs, coords, largest_cutoff, neighbors
            ).mean(dim=1)

        energies = self._energy_of_pots(elem_idxs, coords, largest_cutoff, neighbors)
        if shift_energy:
            dummy = NeighborData(torch.empty(0), torch.empty(0), torch.empty(0))
            energies += self.energy_shifter.atomic_energies(
                elem_idxs, dummy, None, ensemble_average=ensemble_average
            ).sum(dim=-1)
        return SpeciesEnergies(elem_idxs, energies)

    @torch.jit.export
    def atomic_energies(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: float = 0.0,
        ensemble_average: bool = True,
        shift_energy: bool = True,
    ) -> SpeciesEnergies:
        r"""Calculate predicted atomic energies of all atoms in a molecule

        Arguments and return value are the same as that of forward(), but
        the returned energies have shape (molecules, atoms)
        """
        assert total_charge == 0.0, "Model only supports neutral molecules"
        elem_idxs, coords = self._maybe_convert_species(species_coordinates)

        # Optimized path, go through the merged Neighborlist-AEVomputer only
        if self.potentials_len == 1:
            atomic_energies = self.neural_networks.members_atomic_energies(
                self.aev_computer((elem_idxs, coords), cell=cell, pbc=pbc)
            )
        # Iterate over all potentials
        else:
            largest_cutoff = self.potentials[0].cutoff
            neighbors = self.neighborlist(elem_idxs, coords, largest_cutoff, cell, pbc)
            atomic_energies = self._atomic_energy_of_pots(
                elem_idxs, coords, largest_cutoff, neighbors
            )

        if shift_energy:
            atomic_energies += self.energy_shifter.atomic_energies(
                elem_idxs, ensemble_average=False
            )

        if ensemble_average:
            atomic_energies = atomic_energies.mean(dim=0)
        return SpeciesEnergies(species_coordinates[0], atomic_energies)

    def to_infer_model(self, use_mnp: bool = False) -> tpx.Self:
        r"""Convert the neural networks module of the model into a module
        optimized for inference.

        Assumes that the atomic networks are multi layer perceptrons (MLPs)
        with torchani.utils.TightCELU activation functions.
        """
        self.neural_networks = self.neural_networks.to_infer_model(use_mnp=use_mnp)
        return self

    def ase(
        self,
        overwrite: bool = False,
        stress_partial_fdotr: bool = False,
        stress_numerical: bool = False,
        jit: bool = False,
    ):
        r"""Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`ase.Calculator`): A calculator to be used with ASE
        """
        from torchani.ase import Calculator

        return Calculator(
            torch.jit.script(self) if jit else self,
            overwrite=overwrite,
            stress_partial_fdotr=stress_partial_fdotr,
            stress_numerical=stress_numerical,
        )

    @torch.jit.unused
    def get_chemical_symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    @torch.jit.unused
    def strip_non_trainable_potentials(self):
        r"""
        Remove all potentials in the network that are non-trainable
        """
        self.potentials = torch.nn.ModuleList(
            [p for p in self.potentials if p.is_trainable]
        )

    def __len__(self):
        return self.neural_networks.num_networks

    def __getitem__(self, index: int) -> tpx.Self:
        return type(self)(
            symbols=self.get_chemical_symbols(),
            aev_computer=self.aev_computer,
            neural_networks=self.neural_networks.member(index),
            energy_shifter=self.energy_shifter,
            pairwise_potentials=[p for p in self.potentials if not p.is_trainable],
            periodic_table_index=self.periodic_table_index,
        )

    def _atomic_energy_of_pots(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        previous_cutoff: float,
        neighbors: NeighborData,
    ) -> Tensor:
        # Add extra axis, since potentials return atomic E of shape (memb, N, A)
        shape = (
            self.neural_networks.num_networks,
            elem_idxs.shape[0],
            elem_idxs.shape[1],
        )
        energies = torch.zeros(shape, dtype=coords.dtype, device=coords.device)
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbors = rescreen(cutoff, neighbors)
                previous_cutoff = cutoff
            energies += pot.atomic_energies(elem_idxs, neighbors)
        return energies

    def _energy_of_pots(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        previous_cutoff: float,
        neighbors: NeighborData,
    ) -> Tensor:
        energies = torch.zeros(
            elem_idxs.shape[0], dtype=coords.dtype, device=coords.device
        )
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbors = rescreen(cutoff, neighbors)
                previous_cutoff = cutoff
            energies += pot(elem_idxs, neighbors)
        return energies

    # Needed for bw compatibility
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        old_keys = list(state_dict.keys())
        if not any(k.startswith("potentials") for k in old_keys):
            for oldk in old_keys:
                if oldk.startswith("aev_computer"):
                    k = f"potentials.0.{oldk}"
                    state_dict[k] = state_dict[oldk]
                if oldk.startswith("neural_networks"):
                    k = f"potentials.0.{oldk}"
                    state_dict[k] = state_dict[oldk]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.jit.export
    def _maybe_convert_species(
        self, species_coordinates: tp.Tuple[Tensor, Tensor]
    ) -> tp.Tuple[Tensor, Tensor]:
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        if (species_coordinates[0] >= self.aev_computer.num_species).any():
            raise ValueError(f"Unknown species found in {species_coordinates[0]}")
        return species_coordinates

    # Unfortunately this is an UGLY workaround for a torchscript bug
    @torch.jit.export
    def _recast_long_buffers(self) -> None:
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(
            dtype=torch.long
        )
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)
        self.neighborlist._recast_long_buffers()

    @torch.jit.unused
    def members_forces(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesForces:
        """Calculates predicted forces from ensemble members

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not
                enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to none if PBC is not enabled

        Returns:
            SpeciesForces: species, molecular energies, and atomic forces
                predicted by an ensemble of neural network models
        """
        coordinates = species_coordinates[1].requires_grad_()
        members_energies = self(
            species_coordinates,
            cell,
            pbc,
            total_charge=0.0,
            ensemble_average=False,
            shift_energy=True,
        ).energies
        _forces = []
        for energy in members_energies:
            _forces.append(
                -torch.autograd.grad(energy.sum(), coordinates, retain_graph=True)[0]
            )
        forces = torch.stack(_forces, dim=0)
        return SpeciesForces(species_coordinates[0], members_energies, forces)

    @torch.jit.export
    def energies_qbcs(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        unbiased: bool = True,
        total_charge: float = 0.0,
    ) -> SpeciesEnergiesQBC:
        """Calculates predicted predicted energies and qbc factors

        QBC factors are used for query-by-committee (QBC) based active learning
        (as described in the ANI-1x paper `less-is-more`_ ).

        If the model has only 1 network, then qbc factors are all 0.0

        .. _less-is-more:
            https://aip.scitation.org/doi/10.1063/1.5023802

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled,
                    set to None if PBC is not enabled
            unbiased: Whether to unbias the standard deviation over ensemble predictions

        Returns:
            species_energies_qbcs: tuple of tensors, species, energies and qbc
                factors for the given configurations. The shapes of qbcs and
                energies are equal.
        """
        species, energies = self(
            species_coordinates,
            cell,
            pbc,
            total_charge=0.0,
            ensemble_average=False,
            shift_energy=True,
        )

        if self.neural_networks.num_networks == 1:
            qbc_factors = torch.zeros_like(energies).squeeze(0)
        else:
            # standard deviation is taken across ensemble members
            qbc_factors = energies.std(0, unbiased=unbiased)

        # rho's (qbc factors) are weighted by dividing by the square root of
        # the number of atoms in each molecule
        num_atoms = (species >= 0).sum(dim=1, dtype=energies.dtype)
        qbc_factors = qbc_factors / num_atoms.sqrt()
        energies = energies.mean(dim=0)
        assert qbc_factors.shape == energies.shape
        return SpeciesEnergiesQBC(species, energies, qbc_factors)

    @torch.jit.export
    def atomic_stdev(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        ensemble_average: bool = False,
        shift_energy: bool = False,
        unbiased: bool = True,
    ) -> AtomicStdev:
        r"""Returns standard deviation of atomic energies across an ensemble

        shift_energy returns the shifted atomic energies according to the model used

        If the model has only 1 network, a value of 0.0 is output for the stdev
        """
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        atomic_energies = self.neural_networks.members_atomic_energies(species_aevs)

        if shift_energy:
            atomic_energies += self.energy_shifter.atomic_energies(
                species_coordinates[0],
                ensemble_average=ensemble_average,
            )

        if self.neural_networks.num_networks == 1:
            stdev_atomic_energies = torch.zeros_like(atomic_energies).squeeze(0)
        else:
            stdev_atomic_energies = atomic_energies.std(0, unbiased=unbiased)

        if ensemble_average:
            atomic_energies = atomic_energies.mean(0)

        return AtomicStdev(
            species_coordinates[0], atomic_energies, stdev_atomic_energies
        )

    @torch.jit.unused
    def force_magnitudes(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        ensemble_average: bool = True,
    ) -> ForceMagnitudes:
        """
        Computes the L2 norm of predicted atomic force vectors, returning magnitudes,
        averaged by default.

        Args:
            species_coordinates: minibatch of configurations
            ensemble_average: by default, returns the ensemble average magnitude for
                each atomic force vector
        """
        species, _, members_forces = self.members_forces(species_coordinates, cell, pbc)
        magnitudes = members_forces.norm(dim=-1)
        if ensemble_average:
            magnitudes = magnitudes.mean(0)
        return ForceMagnitudes(species, magnitudes)

    @torch.jit.unused
    def force_qbc(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        ensemble_average: bool = False,
        unbiased: bool = True,
    ) -> ForceStdev:
        """
        Returns the mean force magnitudes and relative range and standard deviation
        of predicted forces across an ensemble of networks.

        Args:
            species_coordinates: minibatch of configurations
            ensemble_average: returns magnitudes predicted by each model by default
            unbiased: whether or not to use Bessel's correction in computing
                the standard deviation, True by default
        """
        species, magnitudes = self.force_magnitudes(
            species_coordinates, cell, pbc, ensemble_average=False
        )

        max_magnitudes = magnitudes.max(dim=0).values
        min_magnitudes = magnitudes.min(dim=0).values

        if self.neural_networks.num_networks == 1:
            relative_stdev = torch.zeros_like(magnitudes).squeeze(0)
            relative_range = torch.ones_like(magnitudes).squeeze(0)
        else:
            mean_magnitudes = magnitudes.mean(0)
            relative_stdev = (magnitudes.std(0, unbiased=unbiased) + 1e-8) / (
                mean_magnitudes + 1e-8
            )
            relative_range = ((max_magnitudes - min_magnitudes) + 1e-8) / (
                mean_magnitudes + 1e-8
            )

        if ensemble_average:
            magnitudes = mean_magnitudes

        return ForceStdev(species, magnitudes, relative_stdev, relative_range)


class ANIq(ANI):
    r"""
    ANI-style model that can calculate atomic charges

    Charge networks share the input features with the energy networks, and may either
    be fully independent of them, or share weights to some extent.

    The output energies of these models don't necessarily include a coulombic
    term, but they may.
    """

    def __init__(
        self,
        symbols: tp.Sequence[str],
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        energy_shifter: EnergyAdder,
        pairwise_potentials: tp.Iterable[PairPotential] = (),
        periodic_table_index: bool = True,
        charge_networks: tp.Optional[AtomicContainer] = None,
        charge_normalizer: tp.Optional[ChargeNormalizer] = None,
    ):
        if charge_networks is None:
            raise NotImplementedError(
                "Model with fused charge-energy networks not yet implemented"
            )
        super().__init__(
            symbols=symbols,
            aev_computer=aev_computer,
            neural_networks=neural_networks,
            energy_shifter=energy_shifter,
            pairwise_potentials=pairwise_potentials,
            periodic_table_index=periodic_table_index,
        )
        self.charge_networks = charge_networks
        self.charge_normalizer = charge_normalizer
        charges_nnp = SeparateChargesNNPotential(
            self.aev_computer,
            self.neural_networks,
            self.charge_networks,
            self.charge_normalizer,
        )
        # Check which index has the NNPotential and replace with ChargesNNPotential
        potentials = [pot for pot in self.potentials]
        for j, pot in enumerate(potentials):
            if isinstance(pot, NNPotential):
                potentials[j] = charges_nnp
                break
        # Re-register the ModuleList
        self.potentials = torch.nn.ModuleList(potentials)

    # TODO: Remove code duplication
    @torch.jit.export
    def energies_and_atomic_charges_from_neighborlist(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        neighbor_idxs: Tensor,
        shift_values: Tensor,
        total_charge: float = 0.0,
        input_needs_screening: bool = True,
    ) -> SpeciesEnergiesAtomicCharges:
        # This entrypoint supports input from an external neighborlist
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        # Check shapes
        num_molecules, num_atoms = element_idxs.shape
        assert coordinates.shape == (num_molecules, num_atoms, 3)
        assert total_charge == 0.0, "Model only supports neutral molecules"
        previous_cutoff = self.potentials[0].cutoff
        neighbors = self.neighborlist.process_external_input(
            element_idxs,
            coordinates,
            neighbor_idxs,
            shift_values,
            previous_cutoff,
            input_needs_screening,
        )
        energies = torch.zeros(
            num_molecules, device=element_idxs.device, dtype=coordinates.dtype
        )
        atomic_charges = torch.zeros(
            (num_molecules, num_atoms),
            device=element_idxs.device,
            dtype=coordinates.dtype,
        )
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbors = rescreen(cutoff, neighbors)
                previous_cutoff = cutoff
            if pot.is_trainable:
                output = pot.energies_and_atomic_charges(
                    element_idxs,
                    neighbors,
                    ghost_flags=None,
                    total_charge=total_charge,
                )
                energies += output.energies
                atomic_charges += output.atomic_charges
            else:
                energies += pot(element_idxs, neighbors)
        return SpeciesEnergiesAtomicCharges(
            element_idxs, energies + self.energy_shifter(element_idxs), atomic_charges
        )

    @torch.jit.export
    def energies_and_atomic_charges(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: float = 0.0,
    ) -> SpeciesEnergiesAtomicCharges:
        assert total_charge == 0.0, "Model only supports neutral molecules"
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        previous_cutoff = self.potentials[0].cutoff
        neighbor_data = self.neighborlist(
            element_idxs, coordinates, previous_cutoff, cell, pbc
        )
        energies = torch.zeros(
            element_idxs.shape[0], device=element_idxs.device, dtype=coordinates.dtype
        )
        atomic_charges = torch.zeros(
            element_idxs.shape, device=element_idxs.device, dtype=coordinates.dtype
        )
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbor_data = rescreen(cutoff, neighbor_data)
                previous_cutoff = cutoff
            if pot.is_trainable:
                output = pot.energies_and_atomic_charges(
                    element_idxs,
                    neighbor_data,
                    ghost_flags=None,
                    total_charge=total_charge,
                )
                energies += output.energies
                atomic_charges += output.atomic_charges
            else:
                energies += pot(element_idxs, neighbor_data)
        return SpeciesEnergiesAtomicCharges(
            element_idxs, energies + self.energy_shifter(element_idxs), atomic_charges
        )

    def __getitem__(self, index: int) -> tpx.Self:
        return type(self)(
            symbols=self.get_chemical_symbols(),
            aev_computer=self.aev_computer,
            neural_networks=self.neural_networks.member(index),
            charge_networks=self.charge_networks,
            charge_normalizer=self.charge_normalizer,
            energy_shifter=self.energy_shifter,
            periodic_table_index=self.periodic_table_index,
            pairwise_potentials=[p for p in self.potentials if not p.is_trainable],
        )


def ANI1x(**kwargs) -> ANI:
    from torchani.assembler import ANI1x as build

    return build(**kwargs)


def ANI1ccx(**kwargs) -> ANI:
    from torchani.assembler import ANI1ccx as build

    return build(**kwargs)


def ANI2x(**kwargs) -> ANI:
    from torchani.assembler import ANI2x as build

    return build(**kwargs)


def ANIala(**kwargs) -> ANI:
    from torchani.assembler import ANIala as build

    return build(**kwargs)


def ANIdr(**kwargs) -> ANI:
    from torchani.assembler import ANIdr as build

    return build(**kwargs)


def ANImbis(**kwargs) -> ANI:
    from torchani.assembler import ANImbis as build

    return build(**kwargs)
