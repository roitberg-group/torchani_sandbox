import torch
from collections import OrderedDict
from torch import Tensor
from typing import Tuple, NamedTuple, Optional
from . import utils
from .compat import Final
from apex.mlp import MLP


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class SpeciesCoordinates(NamedTuple):
    species: Tensor
    coordinates: Tensor


class ANIModel(torch.nn.ModuleDict):
    """ANI model that compute energies from species and AEVs.

    Different atom types might have different modules, when computing
    energies, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular energies.

    .. warning::

        The species must be indexed in 0, 1, 2, 3, ..., not the element
        index in periodic table. Check :class:`torchani.SpeciesConverter`
        if you want periodic table indexing.

    .. note:: The resulting energies are in Hartree.

    Arguments:
        modules (:class:`collections.abc.Sequence`): Modules for each atom
            types. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``. Different atom types can share a
            module by putting the same reference in :attr:`modules`.
    """

    @staticmethod
    def ensureOrderedDict(modules):
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od

    def __init__(self, modules, use_mlp=True):
        super().__init__(self.ensureOrderedDict(modules))
        # only used for single molecule
        self.idx_list = None
        self.stream_list = None
        self.last_species_data_ptr = None
        self.use_mlp = False
        if use_mlp:
            self.mlp_networks = self._load_as_mlp_networks()
            self.use_mlp = use_mlp
            print("using mlp")
            # TODO assert mlp is installed

    def forward(self, species_aev: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]

        num_mol = species.shape[0]
        # single molecule
        if num_mol == 1:
        # if False:
            # print('run single mode')
            mol_energies = self._single_mol_energies((species, aev))
        else:
            # shape of atomic energies is (C, A)
            print('run batch mode')
            atomic_energies = self._atomic_energies((species, aev))
            mol_energies = torch.sum(atomic_energies, dim=1)
        # print(mol_energies)
        return SpeciesEnergies(species, mol_energies)

    @torch.jit.export
    # @snoop()
    def _single_mol_energies(self, species_aev: Tuple[Tensor, Tensor]) -> Tensor:
        species, aev = species_aev
        species_ = species.flatten()
        aev = aev.flatten(0, 1)
        num_network = len(self.keys())

        # initialize stream
        if self.last_species_data_ptr is None:
            self.stream_list = [torch.cuda.Stream() for i in range(num_network)]
        # initialize each species index if it has not been initialized
        # or the species has changed
        if self.last_species_data_ptr is None or self.last_species_data_ptr != species.data_ptr():
            with torch.no_grad():
                print('----- init spe_list ----')
                self.last_species_data_ptr = species.data_ptr()
                self.idx_list = [None] * num_network
                for i in range(num_network):
                    mask = (species_ == i)
                    midx = mask.nonzero().flatten()
                    print(i, midx.shape[0])
                    if midx.shape[0] > 0:
                        self.idx_list[i] = midx

        # run network to predict energy
        energy_list = torch.zeros(num_network, dtype=aev.dtype, device=species.device)
        net_list = list(self.values())
        event_list = [torch.cuda.Event() for i in range(num_network)]

        current_stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event()
        start_event.record(current_stream)
        # for i, net in reversed(list(enumerate(net_list))):
        for i, net in enumerate(net_list):
            # print(i)
            if self.idx_list[i] is not None:
                torch.cuda.nvtx.mark(f'species = {i}')
                self.stream_list[i].wait_event(start_event)
                with torch.cuda.stream(self.stream_list[i]):
                # if True:
                    input_ = aev.index_select(0, self.idx_list[i])
                    # torch.sum(net(input_).flatten(), dim=0, out=energy_list[i])
                    if self.use_mlp:
                        energy = self.mlp_networks[i](input_).flatten()
                    else:
                        energy = net(input_).flatten()
                    energy_list[i] = torch.sum(energy)
                    # print(torch.sum(energy))
                event_list[i].record(self.stream_list[i])
            else:
                event_list[i] = None

        # sync default stream with events on different streams
        for event in event_list:
            if event is not None:
                current_stream.wait_event(event)

        output = torch.sum(energy_list).view(1, 1)
        return output

    @torch.jit.export
    def _atomic_energies(self, species_aev: Tuple[Tensor, Tensor]) -> Tensor:
        # Obtain the atomic energies associated with a given tensor of AEV's
        species, aev = species_aev
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self.values()):
            torch.cuda.synchronize()
            torch.cuda.nvtx.mark(f'species = {i}')
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return output

    @torch.jit.export
    def _load_as_mlp_networks(self):
        num_network = len(self.keys())
        mlp_networks = [None] * num_network
        for i, network in enumerate(self.values()):
            mlp_networks[i] = self._load_one_mlp_network(network)
        return mlp_networks

    def _load_one_mlp_network(self, network):
        # intilize mlp network
        mlp_sizes = []
        bias_on = network[-1].bias is not None
        for layer in network:
            if isinstance(layer, torch.nn.Linear):
                mlp_sizes.append(layer.in_features)
                assert (layer.bias is not None) == bias_on, "All layers' bias must be all ON or OFF"
            else:
                assert isinstance(layer, torch.nn.CELU), "Currently only support CELU as activation function"
        # TODO CELU alpha
        alpha_celu = network[-2].alpha
        mlp_sizes.append(network[-1].out_features)
        mlp = MLP(mlp_sizes, bias=bias_on, activation="relu").cuda()
        # load parameters
        i_linear = 0
        for layer in network:
            if isinstance(layer, torch.nn.Linear):
                mlp.weights[i_linear].data.copy_(layer.weight)
                mlp.biases[i_linear].data.copy_(layer.bias)
                i_linear += 1
        return mlp


class Ensemble(torch.nn.ModuleList):
    """Compute the average output of an ensemble of modules."""

    def __init__(self, modules):
        super().__init__(modules)
        self.size = len(modules)

    def forward(self, species_input: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        sum_ = 0
        for x in self:
            sum_ += x(species_input)[1]
        species, _ = species_input
        return SpeciesEnergies(species, sum_ / self.size)


class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, input_: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None):
        for module in self:
            input_ = module(input_, cell=cell, pbc=pbc)
        return input_


class Gaussian(torch.nn.Module):
    """Gaussian activation"""
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(- x * x)


class FittedSoftplus(torch.nn.Module):
    """Softplus function parametrized to be equal to a CELU

    This allows keeping the good characteristics of CELU, while having an
    infinitely differentiable function.
    It is highly recommended to leave alpha and beta as their defaults,
    which match closely CELU with alpha = 0.1"""

    alpha: Final[float]
    beta: Final[float]

    def __init__(self, alpha=0.1, beta=20):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x + self.alpha, beta=self.beta) - self.alpha


class SpeciesConverter(torch.nn.Module):
    """Converts tensors with species labeled as atomic numbers into tensors
    labeled with internal torchani indices according to a custom ordering
    scheme. It takes a custom species ordering as initialization parameter. If
    the class is initialized with ['H', 'C', 'N', 'O'] for example, it will
    convert a tensor [1, 1, 6, 7, 1, 8] into a tensor [0, 0, 1, 2, 0, 3]

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """
    conv_tensor: Tensor

    def __init__(self, species):
        super().__init__()
        rev_idx = {s: k for k, s in enumerate(utils.PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())
        self.register_buffer('conv_tensor', torch.full((maxidx + 2,), -1, dtype=torch.long))
        for i, s in enumerate(species):
            self.conv_tensor[rev_idx[s]] = i

    def forward(self, input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None):
        """Convert species from periodic table element index to 0, 1, 2, 3, ... indexing"""
        species, coordinates = input_
        converted_species = self.conv_tensor[species]

        # check if unknown species are included
        if converted_species[species.ne(-1)].lt(0).any():
            raise ValueError(f'Unknown species found in {species}')

        return SpeciesCoordinates(converted_species.to(species.device), coordinates)
