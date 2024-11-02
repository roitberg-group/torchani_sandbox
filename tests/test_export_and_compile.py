# NOTE: These tests are experimental
import torch
import unittest

from torchani._testing import ANITestCase, make_neighbors
from torchani.neighbors import neighbors_to_triples
from torchani.aev import ANIRadial, ANIAngular


class TestCompile(ANITestCase):
    def setUp(self) -> None:
        self.neighbors = make_neighbors(10, seed=1234)

    def testRadial(self) -> None:
        mod = ANIRadial.like_1x()
        _ = torch.compile(mod, fullgraph=True)

    def testAngular(self) -> None:
        mod = ANIAngular.like_1x()
        _ = torch.compile(mod, fullgraph=True)


class TestExport(ANITestCase):
    def testRadial(self) -> None:
        neighbors = make_neighbors(10, seed=1234)
        mod = ANIRadial.like_1x()
        pairs_dim = torch.export.Dim("pairs")
        _ = torch.export.export(
            mod,
            args=(neighbors.distances,),
            dynamic_shapes={"distances": {0: pairs_dim}},
        )

    def testAngular(self) -> None:
        neighbors = make_neighbors(10, seed=1234)
        triples = neighbors_to_triples(neighbors)
        mod = ANIAngular.like_1x()
        triples_dim = torch.export.Dim("triples")
        _ = torch.export.export(
            mod,
            args=(triples.distances, triples.diff_vectors),
            dynamic_shapes={
                "tri_distances": {0: None, 1: triples_dim},
                "tri_vectors": {0: None, 1: triples_dim, 2: None},
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
