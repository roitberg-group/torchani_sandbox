import math
import pickle

import torch

SUPPORTED_D3_ELEMENTS = 94

def _make_symmetric(x):
    assert x.ndim == 1
    print(len(x))
    size = (math.sqrt(1 + 8 * len(x)) - 1) / 2
    assert size.is_integer(), "input tensor must be of size x * (x + 1) / 2 where x is an integer"
    size = int(size)

    _lower_diagonal_mask = torch.tril(torch.ones((size, size), dtype=torch.bool))
    x_symmetric = torch.zeros((size, size))
    x_symmetric.masked_scatter_(_lower_diagonal_mask, x)
    # TODO: parallelize symmetrization
    for j in range(size):
        for i in range(size):
            x_symmetric[j, i] = x_symmetric[i, j]
    return x_symmetric


def _decode_atomic_numbers(a, b):
    # translated from Grimme et. al. Fortran code
    # this is "limit" in Fortran
    # a_ref and b_ref give the conformation's ref (?)
    # if a or b are greater than 100 this means the conformation ref
    # has to be moved by +1
    # an easier way to do this is with divmod
    a_ref, a = divmod(a, 100)
    b_ref, b = divmod(b, 100)
    return a, b, a_ref, b_ref


def get_c6_constants():
    # hardcoded in Grimme's et. al. D3 Fortran code
    total_records = 161925
    num_lines = 32385
    records_per_line = 5
    max_references = 5
    
    with open('c6_unraveled.pkl', 'rb') as f:
        c6_unraveled = pickle.load(f)
        c6_unraveled = torch.tensor(c6_unraveled).reshape(-1, records_per_line)
    assert c6_unraveled.numel() == total_records
    assert c6_unraveled.shape[0] == num_lines

    # element 0 is actually a dummy element
    c6_constants = torch.zeros((SUPPORTED_D3_ELEMENTS + 1, SUPPORTED_D3_ELEMENTS + 1, max_references, max_references))
    c6_coordination_a = torch.zeros((SUPPORTED_D3_ELEMENTS + 1, SUPPORTED_D3_ELEMENTS + 1, max_references, max_references))
    c6_coordination_b = torch.zeros((SUPPORTED_D3_ELEMENTS + 1, SUPPORTED_D3_ELEMENTS + 1, max_references, max_references))
    
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
        c6_coordination_a[b, a, b_ref, a_ref] = cn_a
        c6_coordination_b[b, a, b_ref, a_ref] = cn_b
    return c6_constants, c6_coordination_a, c6_coordination_b


def get_cutoff_radii():
    # cutoff radii are in angstroms
    num_cutoff_radii = SUPPORTED_D3_ELEMENTS * (SUPPORTED_D3_ELEMENTS + 1) / 2 
    with open('cutoff_radii.pkl', 'rb') as f:
        cutoff_radii = torch.tensor(pickle.load(f))
    assert len(cutoff_radii) == num_cutoff_radii
    cutoff_radii = _make_symmetric(torch.tensor(cutoff_radii))
    return cutoff_radii


def get_covalent_radii():
    # covalent radii are in angstroms
    # covalent radii are used for the calculation of coordination numbers
    # covalent radii in angstrom taken directly from Grimme et. al. dftd3 source code, 
    # in turn taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197
    # values for metals decreased by 10 %
    with open('covalent_radii.pkl', 'rb') as f:
        covalent_radii = torch.tensor(pickle.load(f))
    assert len(covalent_radii) == SUPPORTED_D3_ELEMENTS
    return covalent_radii


def get_sqrt_empirical_charge():
    # empirical Q is in atomic units, these correspond to sqrt(0.5 * sqrt(Z) *
    # <r**2>/<r**4>) in Grimme's code these are "r2r4", and are used to
    # calculate the C8 values
    with open('sqrt_empirical_charge.pkl', 'rb') as f:
        sqrt_empirical_charge = torch.tensor(pickle.load(f))
    assert len(sqrt_empirical_charge) == SUPPORTED_D3_ELEMENTS
    return sqrt_empirical_charge


c6_constants, c6_coordination_a, c6_coordination_b = get_c6_constants(supported_d3_elements)
cutoff_radii = get_cutoff_radii(supported_d3_elements) 
sqrt_empirical_charge = get_sqrt_empirical_charge(supported_d3_elements)
covalent_radii = get_covalent_radii(supported_d3_elements)

# constants for the density functional from psi4 source code, citations:
#    A. Najib, L. Goerigk, J. Comput. Theory Chem., 14 5725, 2018)
#    N. Mardirossian, M. Head-Gordon, Phys. Chem. Chem. Phys, 16, 9904, 2014
bj_damping = {'wB97X' : {'s6' : 1.000, 'a1': 0.0000, 's8': 0.2641, 'a2': 5.4959}}

expect_c6_66 = torch.tensor([[49.1130, 46.0681, 37.8419, 35.4129, 29.2830],
                        [46.0681, 43.2452, 35.5219, 33.2540, 27.5206],
                        [37.8419, 35.5219, 29.3602, 27.5063, 22.9517],
                        [35.4129, 33.2540, 27.5063, 25.7809, 21.5377],
                        [29.2830, 27.5206, 22.9517, 21.5377, 18.2067]])

assert torch.isclose(c6_constants[6, 6], expect_c6_66).all()
