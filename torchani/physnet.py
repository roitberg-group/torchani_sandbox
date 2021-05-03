import torch
import math
from typing import Tuple, Optional, Union
from torch import Tensor
from collections import OrderedDict
from .nn import _parse_activation
from .aev.neighbors import _parse_neighborlist
from .aev.aev_terms import _parse_radial_terms
from .compat import Final
# Note that I will combine all their modules in one for simplicity, there isn't
# much reusability in the way PhysNet divides its modules in my opinion, so
# this is just one monolithic "module", a hierarchical model that encompasses
# the modules, and a residual block


class HierarchicalModel(torch.nn.Module):

    atomic_energy_scale: torch.nn.Parameter
    atomic_energy_bias: torch.nn.Parameter
    per_module_radial_terms: Final[bool]
    embed_as_one_hot: Final[bool]

    def __init__(self,
                 modules=None,
                 num_modules: Optional[int] = None,
                 neighborlist: Union[str, torch.nn.Module] = 'full_pairwise',
                 cutoff: float = 10.0,
                 radial_terms: Union[str, torch.nn.Module] = 'physnet',
                 num_species: int = 4,
                 num_features: int = 128,
                 per_module_radial_terms: bool = False,
                 embed_as_one_hot: bool = False):
        super().__init__()

        self.per_module_radial_terms = per_module_radial_terms

        if modules is None:
            if num_modules is None:
                num_modules = 5
            self.interaction_modules = torch.nn.ModuleList([
                PhysNetModule(num_features=num_features)
                for _ in range(num_modules)
            ])
        else:
            assert num_modules is None
            self.interaction_modules = modules

        self.neighborlist = _parse_neighborlist(neighborlist, cutoff)
        if self.per_module_radial_terms:
            self.radial_terms = torch.nn.ModuleList([_parse_radial_terms(radial_terms, 'hip', None, None, None, cutoff)])
            assert self.radial_terms[0].cutoff == self.neighborlist.cutoff
        else:
            self.radial_terms = _parse_radial_terms(radial_terms, 'physnet', None, None, None, cutoff)
            assert self.radial_terms.cutoff == self.neighborlist.cutoff
        # num_species + 1 is needed to make room for the padding index (which
        # can't be -1)
        if self.embed_as_one_hot:
            self.embedding = torch.nn.Embedding(num_embeddings=num_species + 1,
                                            embedding_dim=num_features,
                                            padding_idx=0)
        else:
            self.embedding = OneHotEmbedding(num_embeddings=num_species + 1,
                                            padding_idx=0)

        # element specific scales for PhysNet, I'm not sure how they initialize
        # these for the paper results, they don't really say in the paper, but
        # in their code they do it this way with zeros and ones
        self.register_parameter('atomic_energy_scale', torch.nn.Parameter(torch.ones(num_species, dtype=torch.float)))
        self.register_parameter('atomic_energy_bias', torch.nn.Parameter(torch.zeros(num_species, dtype=torch.float)))
        self._init()

    @classmethod
    def like_hip(cls, *args, **kwargs):
        # this parameter they keep fixed
        num_modules = 2
        # default is what they use for QM9: 80 features and 3 onsite layers
        num_features = 80
        num_onsite = 3
        interaction_modules = torch.nn.ModuleList([HIPModule(num_features=num_features, num_onsite=num_onsite)
                                                   for _ in range(num_modules)])
        return cls(radial_terms='hip', modules=interaction_modules)

    def forward(self, species_coordinates: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        # PhysNet / HIP-NN do not need the difference vector, since they do not
        # have a 3-body term
        species, coordinates = species_coordinates
        atom_index12, _, _, distances = self.neighborlist(
            species, coordinates)
        assert distances.dim() == 1
        assert distances.shape[0] == atom_index12.shape[1]

        if self.per_module_radial_terms:
            # each interaction layer has its own AEV
            radial_aevs = torch.cat([m(distances) for m in self.radial_terms], dim=-1)
        else:
            radial_aev = self.radial_terms(distances)
            assert radial_aev.dim() == 2
            assert radial_aev.shape[0] == atom_index12.shape[1]

        # here for the species -1 is a padding index and 0 is a valid one, but
        # for the embedding 0 is a padding index and -1 is invalid, so we add 1
        # to all indices for the purposes of embedding
        features = self.embedding(species.view(-1) + 1)
        # features is (C * A = A', num_features)

        atomic_energy_hierarchy_list = []
        for j, m in enumerate(self.interaction_modules):
            if self.per_module_radial_terms:
                radial_aev = radial_aevs[j]
            # shape of radial_aev is (P, sublength), shape of atom_index12 is (2, P)
            # but shape of features is (A', num_features), with A' = A * C
            atomic_energies, features = m(species, features, radial_aev, atom_index12)
            # energies is shape (A',)
            atomic_energy_hierarchy_list.append(atomic_energies.unsqueeze(-1))

        # last dim is dim m in the PhysNet paper, which indexes the module.
        # Energies are actually atomic energies at this point.
        atomic_energy_hierarchy = torch.cat(atomic_energy_hierarchy_list, dim=-1)

        # atomic energies are scaled by an element specific weight and bias
        # which is learnable
        scale = self.atomic_energy_scale[species]
        shift = self.atomic_energy_bias[species]
        scale = scale.masked_fill(species == -1, 0.0)
        shift = shift.masked_fill(species == -1, 0.0)
        atomic_energy_hierarchy = scale.unsqueeze(-1) * atomic_energy_hierarchy + shift.unsqueeze(-1)

        energies = atomic_energy_hierarchy.sum((1, 2))
        return species, energies, atomic_energy_hierarchy

    def _init(self):
        # PhysNet initializes all G matrices to 0, W_out to 0 (these
        # initializations are kind of strange!) and all biases to 0
        #
        # gating vector is initialized to 1 (this does make sense)
        # embedding is initialized uniform +-sqrt(3)
        #
        # note that learnable parameters for PhysNet are only on the modules
        # and also the initial embedding transformation, and final
        # species-specific scaling
        torch.nn.init.uniform_(self.embedding.weight, a=-math.sqrt(3), b=math.sqrt(3))
        for m in self.interaction_modules:
            torch.nn.init.zeros_(m.output_linear.weight)
            torch.nn.init.zeros_(m.gating_linear.weight)
            torch.nn.init.ones_(m.gating_vector)
            # all biases are set to zero for the whole module
            for n, p in m.named_parameters():
                if 'bias' in n:
                    torch.nn.init.zeros_(p)

            # The weight matrices on all the linear transformations are
            # initialized by setting them to "orthogonal matrices and scaling
            # by the glorot init factor". I'm honestly not sure if this makes
            # sense.
            #
            # This is done for all weights in:
            # linearI, linearJ, interaction_linear,
            # interaction_res, atomic_res, output_res

            zero_init_words = ['output_linear', 'gating_linear', 'gating_vector', 'embedding', 'energy_bias', 'energy_scale']
            for n, p in m.named_parameters():
                if 'weight' in n and not any([word in n for word in zero_init_words]):
                    # this is the way they initialize this in their code at least
                    glorot_gain = math.sqrt(2 / (p.shape[0] + p.shape[1]))
                    torch.nn.init.orthogonal_(p, gain=glorot_gain)
                    with torch.no_grad():
                        p *= p.std()


class OneHotEmbedding(torch.nn.Module):
    # takes in a tensor of species and transforms it into a one-hot embedding
    # used in HIP-NN, but this is probably worse than embedding
    num_species: Final[int]

    def __init__(self, num_species=4, padding_idx=0):
        self.num_species = num_species

    def forward(x: Tensor) -> Tensor:
        return x


class HIPModule(torch.nn.Module):

    def __init__(self,
                 num_onsite_res: int = 3,
                 num_features: int = 80,
                 activation: Union[str, torch.nn.Module] = torch.nn.Softplus(),
                 radial_sublength: int = 20):
        super().__init__()

        self.onsite_res = torch.nn.Sequential(
            _make_residual(num_onsite_res,
                                num_features,
                                activation=activation, residual_fn='hip'))

        self.a = _parse_activation(activation)

    def forward(self, species: Tensor, features: Tensor, radial_aev: Tensor,
                atom_index12: Tensor) -> Tuple[Tensor, Tensor]:
        num_features = features.shape[1]
        num_pairs = atom_index12.shape[1]
        # unfortunately here the features have some padded stuff with zeros
        # that makes not sense to calculate on, so we get rid of that, and then
        # we use index_add_ to insert the result of the operations back on
        # padded tensors whenever we need it. For simplicity this is done once
        # every module, but this could potentially be made more performant by
        # doing it much less I believe
        non_dummy_indexes = (species != -1).view(-1).nonzero().contiguous().view(-1)
        num_non_dummy = non_dummy_indexes.shape[0]

        non_dummy_features = features.index_select(0, non_dummy_indexes)
        assert non_dummy_features.size() == torch.Size((num_non_dummy,
                                                       num_features))

        # This is the most laborious construction, the interaction message
        # proto message is of shape (A', num_features), but then it is reduced to
        # shape (A'nd, num_features) for further preprocessing
        # where nd stands for non dummy
        # the proto message is made with two pieces, a "no_interaction" part,
        # and an "interaction" part that includes an interaction with the
        # radial aev term
        proto_message = torch.zeros_like(features)

        proto_message_no_interaction = self.linearI(non_dummy_features)
        assert proto_message_no_interaction.size() == torch.Size((num_non_dummy, num_features))

        # This part can be performed with the features that have "dummy" atoms
        # on them, since atom_index12 gets rid of them automatically

        features12 = features[atom_index12]
        assert features12.size() == torch.Size((2, num_pairs, num_features))
        aev_message_term = self.gating_linear(radial_aev).unsqueeze(0)
        assert aev_message_term.size() == torch.Size((1, num_pairs,
                                                     num_features))
        features_message_term = self.a(self.linearJ(self.a(features12)))
        assert features_message_term.size() == torch.Size((
            2, num_pairs, num_features))

        # here PhysNet uses hadamard (elementwise) product to combine these two terms
        proto_message_interaction = aev_message_term * features_message_term
        assert proto_message_interaction.size() == torch.Size((
            2, num_pairs, num_features))

        proto_message.index_add_(
            0, atom_index12.view(-1),
            proto_message_interaction.view(-1, features.shape[1]))
        proto_message.index_add_(0, non_dummy_indexes,
                                 proto_message_no_interaction)
        # at this point proto message is of shape (A', num_features) but once
        # again I extract only non dummy indexes
        proto_message = proto_message.index_select(0, non_dummy_indexes)

        # message refinement
        message = self.interaction_res(proto_message)
        assert message.size() == torch.Size((num_non_dummy, num_features))

        # features interact with local environment through a message
        non_dummy_features = non_dummy_features * self.gating_vector.view(
            1, -1) + self.interaction_linear(self.a(message))
        assert non_dummy_features.size() == torch.Size((num_non_dummy, num_features))

        # features "atomic" refinement
        non_dummy_features = self.atomic_res(non_dummy_features)
        assert non_dummy_features.size() == torch.Size((num_non_dummy, num_features))

        # energies refinement
        non_dummy_energies = self.output_linear(
            self.a(self.output_res(non_dummy_features))).squeeze(-1)
        assert non_dummy_energies.size() == torch.Size((num_non_dummy,))

        out_energies = features.new_zeros(size=(features.shape[0],))
        out_features = torch.zeros_like(features)

        out_energies.index_add_(0, non_dummy_indexes, non_dummy_energies)
        out_features.index_add_(0, non_dummy_indexes, non_dummy_features)

        return out_energies.view(species.shape[0], species.shape[1]), out_features


class PhysNetModule(torch.nn.Module):

    gating_vector: torch.nn.Parameter

    def __init__(self,
                 num_interaction_res: int = 3,
                 num_atomic_res: int = 2,
                 num_output_res: int = 1,
                 num_features: int = 128,
                 activation: Union[str, torch.nn.Module] = 'shifted_sp',
                 radial_sublength: int = 64):
        super().__init__()

        self.interaction_res = torch.nn.Sequential(
            self._make_residual(num_interaction_res,
                                num_features,
                                activation=activation))
        self.atomic_res = torch.nn.Sequential(
            self._make_residual(num_atomic_res,
                                num_features,
                                activation=activation))
        self.output_res = torch.nn.Sequential(
            self._make_residual(num_output_res,
                                num_features,
                                activation=activation))

        # PhysNet initialization for the gating vector is just ones
        self.register_parameter('gating_vector', torch.nn.Parameter(torch.ones(num_features, dtype=torch.float)))
        self.interaction_linear = torch.nn.Linear(num_features, num_features)
        self.linearI = torch.nn.Linear(num_features, num_features)
        self.linearJ = torch.nn.Linear(num_features, num_features)
        # output_linear has matrix Wout in PhysNet paper, initialized as zero (!)
        # for now I only predict energies
        self.output_linear = torch.nn.Linear(num_features, 1)
        # Gating linear is matrix G in the PhysNet paper, initialized as zero (!)
        self.gating_linear = torch.nn.Linear(radial_sublength, num_features, bias=False)
        self.a = _parse_activation(activation)

    def forward(self, species: Tensor, features: Tensor, radial_aev: Tensor,
                atom_index12: Tensor) -> Tuple[Tensor, Tensor]:
        num_features = features.shape[1]
        num_pairs = atom_index12.shape[1]
        # unfortunately here the features have some padded stuff with zeros
        # that makes not sense to calculate on, so we get rid of that, and then
        # we use index_add_ to insert the result of the operations back on
        # padded tensors whenever we need it. For simplicity this is done once
        # every module, but this could potentially be made more performant by
        # doing it much less I believe
        non_dummy_indexes = (species != -1).view(-1).nonzero().contiguous().view(-1)
        num_non_dummy = non_dummy_indexes.shape[0]

        non_dummy_features = features.index_select(0, non_dummy_indexes)
        assert non_dummy_features.size() == torch.Size((num_non_dummy,
                                                       num_features))

        # This is the most laborious construction, the interaction message
        # proto message is of shape (A', num_features), but then it is reduced to
        # shape (A'nd, num_features) for further preprocessing
        # where nd stands for non dummy
        # the proto message is made with two pieces, a "no_interaction" part,
        # and an "interaction" part that includes an interaction with the
        # radial aev term
        proto_message = torch.zeros_like(features)

        proto_message_no_interaction = self.a(
            self.linearI(self.a(non_dummy_features)))
        assert proto_message_no_interaction.size() == torch.Size((
            num_non_dummy, num_features))

        # This part can be performed with the features that have "dummy" atoms
        # on them, since atom_index12 gets rid of them automatically

        features12 = features[atom_index12]
        assert features12.size() == torch.Size((2, num_pairs, num_features))
        aev_message_term = self.gating_linear(radial_aev).unsqueeze(0)
        assert aev_message_term.size() == torch.Size((1, num_pairs,
                                                     num_features))
        features_message_term = self.a(self.linearJ(self.a(features12)))
        assert features_message_term.size() == torch.Size((
            2, num_pairs, num_features))

        # here PhysNet uses hadamard (elementwise) product to combine these two terms
        proto_message_interaction = aev_message_term * features_message_term
        assert proto_message_interaction.size() == torch.Size((
            2, num_pairs, num_features))

        proto_message.index_add_(
            0, atom_index12.view(-1),
            proto_message_interaction.view(-1, features.shape[1]))
        proto_message.index_add_(0, non_dummy_indexes,
                                 proto_message_no_interaction)
        # at this point proto message is of shape (A', num_features) but once
        # again I extract only non dummy indexes
        proto_message = proto_message.index_select(0, non_dummy_indexes)

        # message refinement
        message = self.interaction_res(proto_message)
        assert message.size() == torch.Size((num_non_dummy, num_features))

        # features interact with local environment through a message
        non_dummy_features = non_dummy_features * self.gating_vector.view(
            1, -1) + self.interaction_linear(self.a(message))
        assert non_dummy_features.size() == torch.Size((num_non_dummy, num_features))

        # features "atomic" refinement
        non_dummy_features = self.atomic_res(non_dummy_features)
        assert non_dummy_features.size() == torch.Size((num_non_dummy, num_features))

        # energies refinement
        non_dummy_energies = self.output_linear(
            self.a(self.output_res(non_dummy_features))).squeeze(-1)
        assert non_dummy_energies.size() == torch.Size((num_non_dummy,))

        out_energies = features.new_zeros(size=(features.shape[0],))
        out_features = torch.zeros_like(features)

        out_energies.index_add_(0, non_dummy_indexes, non_dummy_energies)
        out_features.index_add_(0, non_dummy_indexes, non_dummy_features)

        return out_energies.view(species.shape[0], species.shape[1]), out_features

    @staticmethod
    def _make_residual(num: int, num_features: int, activation: Union[str, torch.nn.Module] = 'shifted_sp'):
        return OrderedDict([(f'res{j}',
                             PhysNetResidual(num_features,
                                             activation=activation))
                            for j in range(num)])


def _make_residual(num: int, num_features: int, activation: Union[str, torch.nn.Module] = 'shifted_sp', residual_fn='hip'):
    if residual_fn == 'hip':
        residual = HIPResidual(num_features, activation)
    elif residual_fn == 'physnet':
        residual = PhysNetResidual(num_features, activation)
    else:
        assert isinstance(residual_fn, torch.nn.Module)
        residual = residual_fn

    return OrderedDict([(f'res{j}',
                         residual(num_features,
                                         activation=activation))
                        for j in range(num)])


class HIPResidual(torch.nn.Module):
    def __init__(self, features: int = 128, activation: Union[str, torch.nn.Module] = torch.nn.Softplus()):
        super().__init__()

        self.res_linear1 = torch.nn.Linear(features, features)
        self.res_linear2 = torch.nn.Linear(features, features)
        self.a = _parse_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        # note that in the paper they state that they could in principle use an
        # extra matrix here to match the dimensions of x but in practice they
        # fix all features in the model to be equal
        out = self.a(self.res_linear1(x))
        # I assume there is an extra activation here they are not putting in the paper?
        # without extra activation:
        # out = self.linear2(out) + x
        # with extra activation:
        out = self.a(self.linear2(out) + x)
        # otherwise consecutive linear layers would collapse
        return out


class PhysNetResidual(torch.nn.Module):
    def __init__(self, features: int = 128, activation: Union[str, torch.nn.Module] = 'shifted_sp'):
        super().__init__()

        self.res_linear1 = torch.nn.Linear(features, features)
        self.res_linear2 = torch.nn.Linear(features, features)
        self.a = _parse_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        # physnet residual is strange, doesn't follow normal pattern of
        # residual layers
        out = self.res_linear1(self.a(x))
        out = self.res_linear2(self.a(out)) + x
        return out


class HierarchicalLoss(torch.nn.Module):

    # this implements the hierarchical loss as stated in the PhysNet paper and
    # HIP-NN
    eps: Final[float]

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, species: Tensor, hierarchy: Tensor) -> Tensor:
        # this expects a 3 dim hierarchy tensor of shape (C, A, M)
        assert hierarchy.shape[0] == species.shape[0]
        assert hierarchy.shape[1] == species.shape[1]
        assert hierarchy.dim() == 3

        num_molecules = hierarchy.shape[0]
        num_atoms = hierarchy.shape[1]
        num_modules = hierarchy.shape[2]
        # where M is the dimensionality of modules
        # hierarchy[:, :, 0] corresponds to the output of the most
        # important refinement module, and hierarchy[:, :, -1] to the least
        # important, final refinement
        # parts of the hierarchy with dummy atoms are assumed to be zero

        # this operation will unfortunately create NaN values everywhere where
        # there are dummy atoms, so I need to get rid of those values first, by
        # putting something inofensive there, and then overwrite with zeros for
        # the sum
        dummy_indexes = (species == -1).unsqueeze(-1)
        hierarchy = hierarchy.masked_fill(dummy_indexes, 1.0)

        loss = hierarchy[:, :, 1:]**2 / (hierarchy[:, :, 1:]**2 + hierarchy[:, :, :-1]**2 + self.eps)
        assert loss.size() == torch.Size((num_molecules, num_atoms, num_modules - 1))

        loss = loss.masked_fill(dummy_indexes, 0.0)
        # sum over module values
        loss = loss.sum(-1)

        # average over atoms
        non_dummy_atoms = (species != -1).sum(dim=1, dtype=loss.dtype)
        loss = (loss / non_dummy_atoms.view(-1, 1)).sum(-1)
        assert loss.size() == torch.Size((num_molecules,))

        # this function does not average the final loss, this is done together
        # with the other loss pieces
        return loss
