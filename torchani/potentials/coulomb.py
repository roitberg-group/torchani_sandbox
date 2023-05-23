import math
from torchani.potentials.scalers import AtomicChargeScaler
from torchani.potentials.inverse_distance import InverseDistance


def Coulomb(charges_cutoff: float = 5.2, coulomb_cutoff: float = math.inf):
    return InverseDistance(
        cutoff=coulomb_cutoff,
        scaler=AtomicChargeScaler(cutoff=charges_cutoff),
        order=1
    )
