
def ANI2xCharges(**kwargs) -> BuiltinModel:
    r"""
    Experimental 2x Model that also outputs atomic charges

    state dict file ``charge_nn_state_dict.pt`` must be present in
    ~/.local/torchani/state_dicts/ for this to work
    """
    info_file = 'ani-2x_8x.info'
    state_dict_file = 'ani2x_state_dict.pt'
    charge_nn_state_dict_file = Path.home().joinpath('.local/torchani/state_dicts/charge_nn_state_dict.pt')
    if not charge_nn_state_dict_file.is_file():
        raise ValueError(f"The file {str(charge_nn_state_dict_file)} could not be found")
    return _load_ani_model(
        state_dict_file,
        info_file,
        use_experimental_charges=True,
        charge_nn_state_dict_file=str(charge_nn_state_dict_file),
        **kwargs
    )

    if use_experimental_charges:
        model_class = BuiltinModelCharges
        # I will manually build this to be equivalent to Kate's networks
        dims_for_atoms = {'H': (1008, 256, 192, 160),
                          'C': (1008, 224, 192, 160),
                          'N': (1008, 192, 160, 128),
                          'O': (1008, 192, 160, 128),
                          'S': (1008, 160, 128, 96),
                          'F': (1008, 160, 128, 96),
                          'Cl': (1008, 160, 128, 96)}

        # Classifier-out is 2 The charges are actually located in the index [1]
        # of the output of the networks

        atomic_charge_networks = OrderedDict(
            [
                (
                    s,
                    atomics.standard(
                        dims_for_atoms[s],
                        bias=False,
                        classifier_out=2,
                        activation=torch.nn.GELU()
                    ),
                ) for s in elements
            ]
        )
        charge_networks = ChargeNetworkAdaptor(atomic_charge_networks)
        model_kwargs.update(
            {
                "charge_factor": ChargeFactor.SQUARED_WEIGHTED,
                "charge_factor_args": {"symbols": elements},
                "charge_networks": charge_networks,
            }
        )

if use_experimental_charges:
    # TODO: This is super dirty and horrible
    assert isinstance(model, BuiltinModelCharges)
    assert charge_nn_state_dict_file is not None

    raw_state_dict = _fetch_state_dict(state_dict_file, model_index)

    energy_nn_state_dict = {k.replace("neural_networks.", ""): v for k, v in raw_state_dict.items() if k.endswith("weight") or k.endswith("bias")}
    charge_nn_state_dict = _fetch_state_dict(charge_nn_state_dict_file, local=True)
    aev_state_dict = {k.replace("aev_computer.", ""): v for k, v in raw_state_dict.items() if k.startswith("aev_computer")}

    model.aev_computer.load_state_dict(aev_state_dict)
    model.energy_shifter.load_state_dict({"self_energies": raw_state_dict["energy_shifter.self_energies"]})
    model.neural_networks.load_state_dict(energy_nn_state_dict)
    model.charge_networks.load_state_dict(charge_nn_state_dict)


class BuiltinModelCharges(BuiltinModel):
    def __init__(
        self,
        *args,
        charge_networks: Optional[ChargeNetworkAdaptor] = None,
        charge_factor: Union[ChargeFactor, torch.nn.Module] = ChargeFactor.EQUAL,
        charge_factor_args: Optional[Dict[str, Any]] = None,
        pairwise_potentials: Iterable[PairwisePotential] = tuple(),
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # TODO: This is a horrible hack needed due to the way that the model
        # assembly currently works
        assert charge_networks is not None
        potentials: List[Potential] = list(pairwise_potentials)
        aev_scalars = AEVScalars(
            aev_computer=self.aev_computer,
            neural_networks=self.neural_networks,
            charge_networks=charge_networks,
            charge_factor=charge_factor,
            charge_factor_args=charge_factor_args,
        )
        self.size = aev_scalars.size
        potentials.append(aev_scalars)

        potentials = sorted(potentials, key=lambda x: x.cutoff, reverse=True)
        self.charge_networks = charge_networks
        self.potentials = torch.nn.ModuleList(potentials)
        # Override the neighborlist cutoff with the largest cutoff in existence
        self.aev_computer.neighborlist.cutoff = self.potentials[0].cutoff

    def forward(  # type: ignore
        self,
        species_coordinates: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None
    ) -> SpeciesEnergiesAtomicCharges:

        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        neighbor_data = self.aev_computer.neighborlist(element_idxs, coordinates, cell, pbc)

        energies = coordinates.new_zeros(coordinates.shape[0])
        atomic_charges = coordinates.new_zeros(element_idxs.shape)

        previous_cutoff = self.aev_computer.neighborlist.cutoff
        for pot in self.potentials:
            if pot.cutoff < previous_cutoff:
                neighbor_data = rescreen(pot.cutoff, neighbor_data)
                previous_cutoff = pot.cutoff
            if isinstance(pot, AEVScalars):
                _energies, atomic_charges = pot(element_idxs, neighbor_data, coordinates=coordinates)
            else:
                _energies = pot(element_idxs, neighbor_data)
            assert isinstance(_energies, Tensor)  # For the TorchScript interpreter
            energies += _energies
        energies = self.energy_shifter((element_idxs, energies)).energies
        return SpeciesEnergiesAtomicCharges(element_idxs, energies, atomic_charges)

    @torch.jit.export
    def _recast_long_buffers(self):
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(dtype=torch.long)
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)
        self.aev_computer.neighborlist._recast_long_buffers()
        for p in self.potentials:
            p._recast_long_buffers()

    def __getitem__(self, index: int) -> 'BuiltinModelCharges':
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        non_aev_potentials = [p for p in self.potentials if not isinstance(p, AEVScalars)]
        return BuiltinModelCharges(
            charge_networks=self.charge_networks,
            aev_computer=self.aev_computer,
            neural_networks=self.neural_networks[index],
            energy_shifter=self.energy_shifter,
            elements=self.get_chemical_symbols(),
            periodic_table_index=self.periodic_table_index,
            pairwise_potentials=non_aev_potentials,
        )

class ChargeNetworkAdaptor(torch.nn.ModuleDict):
    @staticmethod
    def ensureOrderedDict(modules):
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od

    def __init__(self, modules):
        super().__init__(self.ensureOrderedDict(modules))

    def forward(
        self,
        element_idxs: Tensor,
        aevs: Tensor
    ) -> Tensor:

        element_idxs_ = element_idxs.flatten()
        aevs = aevs.flatten(0, 1)

        output = aevs.new_zeros(element_idxs_.shape)

        for i, module in enumerate(self.values()):
            selected_idx = (element_idxs_ == i).nonzero().view(-1)
            if selected_idx.shape[0] > 0:
                input_ = aevs.index_select(0, selected_idx)
                output.index_add_(0, selected_idx, module(input_)[:, 1].view(-1))
        atomic_charges = output.view_as(element_idxs)
        return atomic_charges

class AEVScalars(Potential):
    def __init__(
        self,
        aev_computer: AEVComputer,
        neural_networks: NN,
        charge_networks: ChargeNetworkAdaptor,
        charge_factor: tp.Union[ChargeFactor, torch.nn.Module] = ChargeFactor.EQUAL,
        charge_factor_args: tp.Optional[tp.Dict[str, tp.Any]] = None,

    ):
        if isinstance(neural_networks, Ensemble):
            any_nn = neural_networks[0]
        else:
            any_nn = neural_networks
        symbols = tuple(k if k in PERIODIC_TABLE else "Dummy" for k in any_nn)
        super().__init__(cutoff=aev_computer.radial_terms.cutoff, symbols=symbols)

        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.charge_networks = charge_networks
        self.charge_normalizer = ChargeNormalizer(factor=charge_factor, factor_args=charge_factor_args)

        if isinstance(neural_networks, Ensemble):
            self.size = neural_networks.size
        else:
            self.size = 1

    @torch.jit.export
    def _recast_long_buffers(self):
        self.atomic_numbers = self.atomic_numbers.to(dtype=torch.long)
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)
        self.aev_computer.neighborlist._recast_long_buffers()

    def forward(  # type: ignore
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        coordinates: Tensor,
        ghost_flags: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if self.aev_computer.use_cuda_extension:
            if not self.aev_computer.use_cuaev_interface:
                raise ValueError("Cuda extension without interface not supported for charged models")
            if not self.aev_computer.cuaev_is_initialized:
                self.aev_computer._init_cuaev_computer()
                self.aev_computer.cuaev_is_initialized = True
            aevs = self.aev_computer._compute_cuaev_with_half_nbrlist(
                element_idxs,
                coordinates,
                neighbors.indices,
                neighbors.diff_vectors,
                neighbors.distances,
            )
        else:
            aevs = self.aev_computer._compute_aev(
                element_idxs=element_idxs,
                neighbor_idxs=neighbors.indices,
                distances=neighbors.distances,
                diff_vectors=neighbors.diff_vectors,
            )
        energies = self.neural_networks((element_idxs, aevs)).energies
        raw_atomic_charges = self.charge_networks(element_idxs, aevs)
        atomic_charges = self.charge_normalizer(element_idxs, raw_atomic_charges)
        return energies, atomic_charges
