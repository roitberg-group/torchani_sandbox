# NOTE: These tests are experimental
import os
import torch
import unittest

from torchani._testing import ANITestCase, make_neighbors
from torchani.utils import SYMBOLS_2X
from torchani.nn import ANINetworks, make_2x_network
from torchani.neighbors import (
    discard_outside_cutoff,
    neighbors_to_triples,
    Neighbors,
    Triples,
)
from torchani.aev import ANIRadial, ANIAngular, AEVComputer


class ANITestCasePT2(ANITestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # Set up logging (high verbosity with +) for Dynamo, AOT and Inductor
        os.environ["TORCH_LOGS"] = "+dynamo,+aot,+inductor"
        os.environ["TORCHDYNAMO_VERBOSE"] = "1"  # (needed according to log msgs?)
        # Many models use dynamic output shape ops (nonzero, unique, unique_consecutive)
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        os.environ["TORCH_LOGS"] = ""
        os.environ["TORCHDYNAMO_VERBOSE"] = "0"
        torch._dynamo.config.capture_dynamic_output_shape_ops = False

    def tearDown(self) -> None:
        torch.compiler.reset()  # needed?


class TestCompile(ANITestCasePT2):
    def testRadial(self) -> None:
        _ = torch.compile(ANIRadial.like_2x(), fullgraph=True)

    def testAngular(self) -> None:
        _ = torch.compile(ANIAngular.like_2x(), fullgraph=True)

    def testDiscardNeighbors(self) -> None:
        _ = torch.compile(discard_outside_cutoff, fullgraph=True)

    def testNeighborsToTriples(self) -> None:
        _ = torch.compile(neighbors_to_triples, fullgraph=True)

    def testAEVComputer(self) -> None:
        _ = torch.compile(AEVComputer.like_2x().compute_from_neighbors, fullgraph=True)

    def testAtomicNetwork(self) -> None:
        _ = torch.compile(make_2x_network("H"), fullgraph=True)

    def testANINetworks(self) -> None:
        m = ANINetworks({s: make_2x_network(s) for s in SYMBOLS_2X})
        _ = torch.compile(m, fullgraph=True)

    def tearDown(self) -> None:
        torch.compiler.reset()


class TestExport(ANITestCasePT2):

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

    def testDiscardNeighbors(self) -> None:
        neighbors = make_neighbors(10, seed=1234)

        class Mod(torch.nn.Module):
            def forward(self, neighbors: Neighbors, cutoff: float) -> Neighbors:
                return discard_outside_cutoff(neighbors, cutoff)

        m = Mod()

        pairs = torch.export.Dim("pairs")
        _ = torch.export.export(
            m,
            args=(neighbors, 5.2),
            dynamic_shapes={
                # shift_values is fixed to None on export
                "neighbors": ((None, pairs), (pairs,), (pairs, None), None),
                "cutoff": None,  # cutoff is fixed on export
            },
        )

    # currently this test fails even with capture_dynamic_output_shapes=True
    # Maybe pytorch bug?
    @unittest.skipIf(True, "Fails in pytorch 2.3.1")
    def testNeighborsToTriples(self) -> None:
        neighbors = make_neighbors(10, seed=1234)

        class Mod(torch.nn.Module):
            def forward(self, neighbors: Neighbors) -> Triples:
                return neighbors_to_triples(neighbors)

        m = Mod()
        pairs = torch.export.Dim("pairs")
        _ = torch.export.export(
            m,
            args=(neighbors,),
            dynamic_shapes={
                # shift_values is fixed to None on export
                "neighbors": ((None, pairs), (pairs,), (pairs, None), None)
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
