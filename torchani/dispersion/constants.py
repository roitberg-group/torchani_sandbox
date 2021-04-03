import math
import pickle
from pathlib import Path

import torch

SUPPORTED_D3_ELEMENTS = 94


def _make_symmetric(x):
    assert x.ndim == 1
    size = (math.sqrt(1 + 8 * len(x)) - 1) / 2
    assert size.is_integer(), \
            "input tensor must be of size x * (x + 1) / 2 where x is an integer"
    size = int(size)

    _lower_diagonal_mask = torch.tril(
        torch.ones((size, size), dtype=torch.bool))
    x_symmetric = torch.zeros((size, size))
    x_symmetric.masked_scatter_(_lower_diagonal_mask, x)
    # TODO: parallelize symmetrization
    for j in range(size):
        for i in range(size):
            x_symmetric[j, i] = x_symmetric[i, j]
    return x_symmetric


def _decode_atomic_numbers(a, b):
    # translated from Grimme et. al. Fortran code this is "limit" in Fortran
    # a_ref and b_ref give the conformation's ref (?) if a or b are greater
    # than 100 this means the conformation ref has to be moved by +1 an easier
    # way to do this is with divmod
    a_ref, a = divmod(a, 100)
    b_ref, b = divmod(b, 100)
    return a, b, a_ref, b_ref


def get_c6_constants():
    # hardcoded in Grimme's et. al. D3 Fortran code
    total_records = 161925
    num_lines = 32385
    records_per_line = 5
    max_refs = 5
    path = Path(__file__).parent.joinpath('c6_unraveled.pkl').resolve()
    with open(path, 'rb') as f:
        c6_unraveled = pickle.load(f)
        c6_unraveled = torch.tensor(c6_unraveled).reshape(-1, records_per_line)
    assert c6_unraveled.numel() == total_records
    assert c6_unraveled.shape[0] == num_lines

    # element 0 is actually a dummy element
    el = SUPPORTED_D3_ELEMENTS
    # nonexistent values are filled with -1, in order to mask them, 
    # same as in Grimme et. al. code
    c6_constants =      torch.full((el + 1, el + 1, max_refs, max_refs), -1.0)
    c6_coordination_a = torch.full((el + 1, el + 1, max_refs, max_refs), -1.0)
    c6_coordination_b = torch.full((el + 1, el + 1, max_refs, max_refs), -1.0)
    assert ((c6_constants == -1.0) == (c6_coordination_a == -1.0)).all(), "All missing parameters are not equal"
    assert ((c6_coordination_a == -1.0) == (c6_coordination_b == -1.0)).all(), "All missing parameters are not equal"

    # every "line" in the unraveled c6 list has:
    # 0 1 2 3 4
    # C6, a, b, CNa, CNb
    # in that order
    # translated from Grimme et. al. Fortran code
    for line in c6_unraveled:
        constant, a, b, cn_a, cn_b = line.cpu().numpy().tolist()
        # a and b are the atomic numbers
        a, b, a_ref, b_ref = _decode_atomic_numbers(int(a), int(b))
        # get values for C6 and CNa, CNb
        c6_constants[a, b, a_ref, b_ref] = constant
        c6_coordination_a[a, b, a_ref, b_ref] = cn_a
        c6_coordination_b[a, b, a_ref, b_ref] = cn_b
        # symmetrize values
        c6_constants[b, a, b_ref, a_ref] = constant
        # these have to be inverted (cn_a given to b and cn_b given to a)
        c6_coordination_a[b, a, b_ref, a_ref] = cn_b
        c6_coordination_b[b, a, b_ref, a_ref] = cn_a
    return c6_constants, c6_coordination_a, c6_coordination_b


def get_cutoff_radii():
    # cutoff radii are in angstroms
    num_cutoff_radii = SUPPORTED_D3_ELEMENTS * (SUPPORTED_D3_ELEMENTS + 1) / 2
    path = Path(__file__).parent.joinpath('cutoff_radii.pkl').resolve()
    with open(path, 'rb') as f:
        cutoff_radii = torch.tensor(pickle.load(f))
    assert len(cutoff_radii) == num_cutoff_radii
    # element 0 is a dummy element
    cutoff_radii = torch.cat((torch.tensor([0.0]), cutoff_radii))
    cutoff_radii = _make_symmetric(torch.tensor(cutoff_radii))
    return cutoff_radii


def get_covalent_radii():
    # covalent radii are in angstroms covalent radii are used for the
    # calculation of coordination numbers covalent radii in angstrom taken
    # directly from Grimme et. al. dftd3 source code, in turn taken from Pyykko
    # and Atsumi, Chem. Eur. J. 15, 2009, 188-197 values for metals decreased
    # by 10 %
    path = Path(__file__).parent.joinpath('covalent_radii.pkl').resolve()
    with open(path, 'rb') as f:
        covalent_radii = torch.tensor(pickle.load(f))
    assert len(covalent_radii) == SUPPORTED_D3_ELEMENTS
    # element 0 is a dummy element
    covalent_radii = torch.cat((torch.tensor([0.0]), covalent_radii))
    return covalent_radii


def get_sqrt_empirical_charge():
    # empirical Q is in atomic units, these correspond to sqrt(0.5 * sqrt(Z) *
    # <r**2>/<r**4>) in Grimme's code these are "r2r4", and are used to
    # calculate the C8 values
    path = Path(__file__).parent.joinpath('sqrt_empirical_charge.pkl').resolve()
    with open(path, 'rb') as f:
        sqrt_empirical_charge = torch.tensor(pickle.load(f))
    assert len(sqrt_empirical_charge) == SUPPORTED_D3_ELEMENTS
    # element 0 is a dummy element
    sqrt_empirical_charge = torch.cat((torch.tensor([0.0]), sqrt_empirical_charge))
    return sqrt_empirical_charge


# constants for the density functional from psi4 source code, citations:
#    A. Najib, L. Goerigk, J. Comput. Theory Chem., 14 5725, 2018)
#    N. Mardirossian, M. Head-Gordon, Phys. Chem. Chem. Phys, 16, 9904, 2014
df_constants = {'wB97X': {'s6': 1.000, 'a1': 0.0000, 's8': 0.2641, 'a2': 5.4959}}
