import unittest

from torchani.testing import ANITest, expand
from torchani.utils import SYMBOLS_1X
from torchani.aev import AEVComputer
from torchani.potentials import StandaloneTwoBodyDispersionD3, StandaloneRepulsionXTB


@expand(jit=False)
class TestDevices(ANITest):
    def testAEVComputerStyle(self):
        computer = AEVComputer.style_1x(device=self.device)
        for b in computer.buffers():
            self.assertEqual(b.device.type, self.device.type)
        computer = AEVComputer.style_2x(device=self.device)
        for b in computer.buffers():
            self.assertEqual(b.device.type, self.device.type)

    def testAEVComputerLike(self):
        computer = AEVComputer.like_1x(device=self.device)
        for b in computer.buffers():
            self.assertEqual(b.device.type, self.device.type)
        computer = AEVComputer.like_2x(device=self.device)
        for b in computer.buffers():
            self.assertEqual(b.device.type, self.device.type)

    def testDispersion(self):
        _ = StandaloneTwoBodyDispersionD3(
            symbols=SYMBOLS_1X, functional="wB97X", device=self.device
        )

    def testRepulsion(self):
        _ = StandaloneRepulsionXTB(symbols=SYMBOLS_1X, device=self.device)


if __name__ == "__main__":
    unittest.main(verbosity=2)
