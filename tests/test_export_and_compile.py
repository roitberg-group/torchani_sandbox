import torch
import unittest

from torchani._testing import ANITestCase, make_neighbors
from torchani.aev import StandardRadial, StandardAngular


class TestCompile(ANITestCase):
    def setUp(self) -> None:
        self.neighbors = make_neighbors(10, seed=1234)

    def testRadial(self) -> None:
        mod = StandardRadial.like_1x()
        _ = torch.compile(mod, fullgraph=True, dynamic=True)

    def testAngular(self) -> None:
        mod = StandardAngular.like_1x()
        _ = torch.compile(mod, fullgraph=True, dynamic=True)


class TestExport(ANITestCase):
    def testRadial(self) -> None:
        neighbors = make_neighbors(10, seed=1234)
        mod = StandardRadial.like_1x()
        pairs_dim = torch.export.Dim("pairs")
        _ = torch.export.export(
            mod,
            args=(neighbors.distances,),
            dynamic_shapes={"distances": {0: pairs_dim}},
        )

    # def testAngular(self) -> None:
        # neighbors = make_neighbors(10, seed=1234)
        # mod = StandardAngular.like_1x()
        # pairs_dim = torch.export.Dim("pairs")
        # _ = torch.export.export(
            # mod,
            # args=(neighbors.diff_vectors, neighbors.distances),
            # dynamic_shapes={"vectors12": {1: pairs_dim}, "distances12": {1: pairs_dim}},
        # )


if __name__ == "__main__":
    unittest.main(verbosity=2)
