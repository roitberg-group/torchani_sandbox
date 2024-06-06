r"""Atomic constants

Values for electronegativity and hardness for elements H-Bk, all for neutral
atoms, and are taken from Table 3 of

Carlos Cardenas et. al. Benchmark Values of Chemical Potential and Chemical
Hardness for Atoms and Atomic Ions (Including Unstable Ions) from the Energies
of Isoelectronic Series.

DOI: 10.1039/C6CP04533B

Atomic masses supported are the first 119 elements, and are taken from:

Atomic weights of the elements 2013 (IUPAC Technical Report). Meija, J.,
Coplen, T., Berglund, M., et al. (2016). Pure and Applied Chemistry, 88(3), pp.
265-291. Retrieved 30 Nov. 2016, from doi:10.1515/pac-2015-0305

They are all consistent with those used in ASE
"""
import typing as tp
import math

__all__ = [
    "PERIODIC_TABLE",
    "ELECTRONEGATIVITY",
    "HARDNESS",
    "ATOMIC_NUMBERS",
    "MASSES",
]

# This constant, when indexed with the corresponding atomic number, gives the
# element associated with it. Note that there is no element with atomic number
# 0, so 'Dummy' returned in this case.
PERIODIC_TABLE = (
    ["Dummy"]
    + """
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()  # noqa
)

ATOMIC_NUMBERS = {symbol: z for z, symbol in enumerate(PERIODIC_TABLE)}


def mapping_to_znumber_indexed_seq(
    symbols_map: tp.Mapping[str, float]
) -> tp.Tuple[float, ...]:
    r"""
    Sort the values of {symbol: value} mapping by atomic number and output a
    tuple with the sorted values.

    All elements up to the highest present atomic number element must in the mapping.

    The first element (index 0) of the output will be NaN. Example:

    .. code-block:: python
        mapping = {"H": 3.0, "Li": 1.0, "He": 0.5 }
        znumber_indexed_seq = mapping_to_znumber_indexed_seq(mapping)
        # znumber_indexed_seq will be (NaN, 3.0, 0.5, 1.0)
    """
    _symbols_map = dict(symbols_map)
    seq = [math.nan] * (len(symbols_map) + 1)
    try:
        for k, v in _symbols_map.items():
            seq[ATOMIC_NUMBERS[k]] = v
    except IndexError:
        raise ValueError(f"There are missing elements in {symbols_map}") from None
    return tuple(seq)


def znumber_indexed_seq_to_mapping(
    seq: tp.Sequence[float],
) -> tp.Dict[str, float]:
    r"""
    Inverse of mapping_to_znumber_indexed_list. The first element of the input
    must be NaN. Example:

    .. code-block:: python
        znumber_indexed_seq = (math.nan, 3.0, 0.5, 1.0)
        mapping = znumber_indexed_seq_to_mapping(znumber_indexed_seq)
        # mapping will be {"H": 3.0, "Li": 1.0, "He": 0.5 }
    """
    if not math.isnan(seq[0]):
        raise ValueError("The first element of the input iterable must be NaN")
    return {PERIODIC_TABLE[j]: v for j, v in enumerate(seq) if j != 0}


MASSES = (
    0.0,
    1.008,
    4.002602,
    6.94,
    9.0121831,
    10.81,
    12.011,
    14.007,
    15.999,
    18.99840316,
    20.1797,
    22.98976928,
    24.305,
    26.9815385,
    28.085,
    30.973762,
    32.06,
    35.45,
    39.948,
    39.0983,
    40.078,
    44.955908,
    47.867,
    50.9415,
    51.9961,
    54.938044,
    55.845,
    58.933194,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.63,
    74.921595,
    78.971,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.90584,
    91.224,
    92.90637,
    95.95,
    97.90721,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.414,
    114.818,
    118.71,
    121.76,
    127.6,
    126.90447,
    131.293,
    132.90545196,
    137.327,
    138.90547,
    140.116,
    140.90766,
    144.242,
    144.91276,
    150.36,
    151.964,
    157.25,
    158.92535,
    162.5,
    164.93033,
    167.259,
    168.93422,
    173.054,
    174.9668,
    178.49,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966569,
    200.592,
    204.38,
    207.2,
    208.9804,
    208.98243,
    209.98715,
    222.01758,
    223.01974,
    226.02541,
    227.02775,
    232.0377,
    231.03588,
    238.02891,
    237.04817,
    244.06421,
    243.06138,
    247.07035,
    247.07031,
    251.07959,
    252.083,
    257.09511,
    258.09843,
    259.101,
    262.11,
    267.122,
    268.126,
    271.134,
    270.133,
    269.1338,
    278.156,
    281.165,
    281.166,
    285.177,
    286.182,
    289.19,
    289.194,
    293.204,
    293.208,
    294.214,
)


# Note that there are 97 values for hardness and 97
# values for electronegativity, so one per element
# with no skips

# Also note that the electronegativity is the mulliken electronegativity
# electronegativity ~ (I + EA) / 2
# hardness ~ (I - EA) / 2

# Where I and EA are in eV
# and also E_homo ~ -I, E_lumo ~ -EA
ATOMIC_HARDNESS = {
    "H": 12.84,
    "He": 24.59,
    "Li": 4.77,
    "Be": 9.32,
    "B": 8.02,
    "C": 10.0,
    "N": 14.53,
    "O": 12.16,
    "F": 14.02,
    "Ne": 21.56,
    "Na": 4.59,
    "Mg": 7.65,
    "Al": 5.55,
    "Si": 6.67,
    "P": 9.74,
    "S": 8.28,
    "Cl": 9.35,
    "Ar": 15.76,
    "K": 3.84,
    "Ca": 6.09,
    "Sc": 6.37,
    "Ti": 6.75,
    "V": 6.22,
    "Cr": 6.09,
    "Mn": 7.43,
    "Fe": 7.75,
    "Co": 7.22,
    "Ni": 6.48,
    "Cu": 6.39,
    "Zn": 9.39,
    "Ga": 5.57,
    "Ge": 6.67,
    "As": 8.98,
    "Se": 7.73,
    "Br": 8.45,
    "Kr": 14.0,
    "Rb": 3.69,
    "Sr": 5.64,
    "Y": 5.91,
    "Zr": 6.21,
    "Nb": 5.86,
    "Mo": 6.35,
    "Tc": 6.73,
    "Ru": 6.28,
    "Rh": 6.32,
    "Pd": 7.77,
    "Ag": 6.27,
    "Cd": 8.99,
    "In": 5.4,
    "Sn": 6.23,
    "Sb": 7.56,
    "Te": 7.04,
    "I": 7.39,
    "Xe": 12.13,
    "Cs": 3.42,
    "Ba": 5.07,
    "La": 5.11,
    "Ce": 4.91,
    "Pr": 4.51,
    "Nd": 5.36,
    "Pm": 5.45,
    "Sm": 5.48,
    "Eu": 5.55,
    "Gd": 6.01,
    "Tb": 5.43,
    "Dy": 5.59,
    "Ho": 5.68,
    "Er": 5.8,
    "Tm": 6.17,
    "Yb": 6.25,
    "Lu": 5.09,
    "Hf": 6.71,
    "Ta": 7.23,
    "W": 7.05,
    "Re": 7.68,
    "Os": 7.36,
    "Ir": 7.4,
    "Pt": 6.83,
    "Au": 6.92,
    "Hg": 10.44,
    "Tl": 5.73,
    "Pb": 7.05,
    "Bi": 6.34,
    "Po": 6.51,
    "At": 6.52,
    "Rn": 10.75,
    "Fr": 3.59,
    "Ra": 5.18,
    "Ac": 4.82,
    "Th": 5.94,
    "Pa": 5.51,
    "U": 5.82,
    "Np": 5.95,
    "Pu": 5.94,
    "Am": 5.9,
    "Cm": 5.67,
    "Bk": 6.17,
}

ATOMIC_ELECTRONEGATIVITY = {
    "H": 7.18,
    "He": 12.27,
    "Li": 3.0,
    "Be": 4.66,
    "B": 4.29,
    "C": 6.26,
    "N": 7.27,
    "O": 7.54,
    "F": 10.41,
    "Ne": 10.78,
    "Na": 2.84,
    "Mg": 3.82,
    "Al": 3.21,
    "Si": 4.77,
    "P": 5.62,
    "S": 6.22,
    "Cl": 8.29,
    "Ar": 7.88,
    "K": 2.42,
    "Ca": 3.07,
    "Sc": 3.38,
    "Ti": 3.45,
    "V": 3.64,
    "Cr": 3.72,
    "Mn": 3.72,
    "Fe": 4.03,
    "Co": 4.27,
    "Ni": 4.4,
    "Cu": 4.48,
    "Zn": 4.7,
    "Ga": 3.21,
    "Ge": 4.57,
    "As": 5.3,
    "Se": 5.88,
    "Br": 7.59,
    "Kr": 7.0,
    "Rb": 2.33,
    "Sr": 2.88,
    "Y": 3.27,
    "Zr": 3.53,
    "Nb": 3.83,
    "Mo": 3.92,
    "Tc": 3.91,
    "Ru": 4.22,
    "Rh": 4.3,
    "Pd": 4.44,
    "Ag": 4.44,
    "Cd": 4.5,
    "In": 3.09,
    "Sn": 4.23,
    "Sb": 4.82,
    "Te": 5.49,
    "I": 6.75,
    "Xe": 6.07,
    "Cs": 2.18,
    "Ba": 2.68,
    "La": 3.03,
    "Ce": 3.09,
    "Pr": 3.22,
    "Nd": 2.85,
    "Pm": 2.86,
    "Sm": 2.91,
    "Eu": 2.9,
    "Gd": 3.15,
    "Tb": 3.15,
    "Dy": 3.15,
    "Ho": 3.18,
    "Er": 3.21,
    "Tm": 3.1,
    "Yb": 3.13,
    "Lu": 2.88,
    "Hf": 3.47,
    "Ta": 3.94,
    "W": 4.35,
    "Re": 4.0,
    "Os": 4.76,
    "Ir": 5.27,
    "Pt": 5.54,
    "Au": 5.77,
    "Hg": 5.22,
    "Tl": 3.24,
    "Pb": 3.88,
    "Bi": 4.12,
    "Po": 5.15,
    "At": 6.05,
    "Rn": 5.38,
    "Fr": 2.28,
    "Ra": 2.69,
    "Ac": 2.76,
    "Th": 3.35,
    "Pa": 3.14,
    "U": 3.29,
    "Np": 3.29,
    "Pu": 3.07,
    "Am": 3.03,
    "Cm": 3.16,
    "Bk": 3.12,
}

ELECTRONEGATIVITY = mapping_to_znumber_indexed_seq(ATOMIC_ELECTRONEGATIVITY)
HARDNESS = mapping_to_znumber_indexed_seq(ATOMIC_HARDNESS)
