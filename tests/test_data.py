import os
import h5py
import numpy as np
import torch
import torchani
import unittest
import tempfile
from torchani.testing import TestCase
from torchani.data import H5Dataset  # , ANIBatchedDataset, save_batched_dataset

path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(path, '../dataset/ani-1x/sample.h5')
batch_size = 256
ani1x_sae_dict = {'H': -0.60095298, 'C': -38.08316124, 'N': -54.7077577, 'O': -75.19446356}


class TestH5Dataset(TestCase):

    def setUp(self):
        # create two dummy HDF5 databases, one with 3 groups and one with one
        # group, and fill them with some data
        self.tf_one_group = tempfile.NamedTemporaryFile()
        self.tf_three_groups = tempfile.NamedTemporaryFile()

        self.rng = np.random.default_rng(12345)
        self.num_conformers1 = 7
        properties1 = {'species': ['H', 'C', 'N', 'N'],
                      'coordinates': self.rng.standard_normal((self.num_conformers1, 4, 3)),
                      'energies': self.rng.standard_normal((self.num_conformers1,))}
        self.num_conformers2 = 5
        properties2 = {'species': ['H', 'O', 'O'],
                      'coordinates': self.rng.standard_normal((self.num_conformers2, 3, 3)),
                      'energies': self.rng.standard_normal((self.num_conformers2,))}
        self.num_conformers3 = 8
        properties3 = {'species': ['H', 'C', 'H', 'H', 'H'],
                      'coordinates': self.rng.standard_normal((self.num_conformers3, 5, 3)),
                      'energies': self.rng.standard_normal((self.num_conformers3,))}
        with h5py.File(self.tf_one_group, 'r+') as f1:
            f1.create_group(''.join(properties1['species']))
        with h5py.File(self.tf_three_groups, 'r+') as f3:
            f3.create_group(''.join(properties1['species']))
            f3.create_group(''.join(properties2['species']))
            f3.create_group(''.join(properties3['species']))

        with h5py.File(self.tf_one_group, 'r+') as f1:
            for k, v in properties1.items():
                if k == 'species':
                    v = np.asarray(v, dtype='S')
                f1['HCNN'].create_dataset(k, data=v)
        with h5py.File(self.tf_three_groups, 'r+') as f3:
            for k, v in properties1.items():
                if k == 'species':
                    v = np.asarray(v, dtype='S')
                f3['HCNN'].create_dataset(k, data=v)

            for k, v in properties2.items():
                if k == 'species':
                    v = np.asarray(v, dtype='S')
                f3['HOO'].create_dataset(k, data=v)

            for k, v in properties3.items():
                if k == 'species':
                    v = np.asarray(v, dtype='S')
                f3['HCHHH'].create_dataset(k, data=v)

    def testSizesOneGroup(self):
        ds = H5Dataset(self.tf_one_group.name)
        self.assertEqual(ds.num_conformers, self.num_conformers1)
        self.assertEqual(ds.num_conformer_groups, 1)
        self.assertEqual(len(ds), ds.num_conformer_groups)

    def testSizesThreeGroups(self):
        ds = H5Dataset(self.tf_three_groups.name)
        self.assertEqual(ds.num_conformers, self.num_conformers1 + self.num_conformers2 + self.num_conformers3)
        self.assertEqual(ds.num_conformer_groups, 3)
        self.assertEqual(len(ds), ds.num_conformer_groups)

    def testKeys(self):
        ds = H5Dataset(self.tf_three_groups.name)
        keys = set()
        for k in ds.keys():
            keys.update({k})
        self.assertTrue(keys == {'/HOO', '/HCNN', '/HCHHH'})
        self.assertEqual(len(ds.keys()), 3)

    def testValues(self):
        ds = H5Dataset(self.tf_three_groups.name)
        for d in ds.values():
            self.assertTrue('species' in d.keys())
            self.assertTrue('coordinates' in d.keys())
            self.assertTrue('energies' in d.keys())
            self.assertEqual(d['coordinates'].shape[-1], 3)
            self.assertEqual(d['coordinates'].shape[0], d['energies'].shape[0])
        self.assertEqual(len(ds.values()), 3)

    def testItems(self):
        ds = H5Dataset(self.tf_three_groups.name)
        for k, v in ds.items():
            self.assertTrue(isinstance(k, str))
            self.assertTrue(isinstance(v, dict))
            self.assertTrue('species' in v.keys())
            self.assertTrue('coordinates' in v.keys())
            self.assertTrue('energies' in v.keys())
        self.assertEqual(len(ds.items()), 3)

    def testGetConformers(self):
        ds = H5Dataset(self.tf_three_groups.name)

        self.assertEqual(ds.get_conformers('HOO')['coordinates'], ds['HOO']['coordinates'])
        self.assertEqual(ds.get_conformers('HOO', 0)['coordinates'], ds['HOO']['coordinates'][0])
        conformers12 = ds.get_conformers('HCHHH', np.array([1, 2]))
        self.assertEqual(conformers12['coordinates'], ds['HCHHH']['coordinates'][np.array([1, 2])])
        # note that h5py does not allow this directly
        conformers12 = ds.get_conformers('HCHHH', np.array([2, 1]))
        self.assertEqual(conformers12['coordinates'], ds['HCHHH']['coordinates'][np.array([2, 1])])
        # note that h5py does not allow this directly
        conformers12 = ds.get_conformers('HCHHH', np.array([1, 1]))
        self.assertEqual(conformers12['coordinates'], ds['HCHHH']['coordinates'][np.array([1, 1])])
        conformers124 = ds.get_conformers('HCHHH', np.array([1, 2, 4]), include_properties=('energies',))
        self.assertEqual(conformers124['energies'], ds['HCHHH']['energies'][np.array([1, 2, 4])])
        self.assertTrue(conformers124.get('species', None) is None)
        self.assertTrue(conformers124.get('coordinates', None) is None)

    def testIterConformers(self):
        ds = H5Dataset(self.tf_three_groups.name)
        confs = []
        for c in ds.iter_conformers():
            self.assertTrue(isinstance(c, dict))
            confs.append(c)
        self.assertEqual(len(confs), ds.num_conformers)


@unittest.skipIf(True, '')
class TestData(TestCase):

    def testTensorShape(self):
        ds = torchani.data.load(dataset_path).subtract_self_energies(ani1x_sae_dict).species_to_indices().shuffle().collate(batch_size).cache()
        for d in ds:
            species = d['species']
            coordinates = d['coordinates']
            energies = d['energies']
            self.assertEqual(len(species.shape), 2)
            self.assertLessEqual(species.shape[0], batch_size)
            self.assertEqual(len(coordinates.shape), 3)
            self.assertEqual(coordinates.shape[2], 3)
            self.assertEqual(coordinates.shape[:2], species.shape[:2])
            self.assertEqual(len(energies.shape), 1)
            self.assertEqual(coordinates.shape[0], energies.shape[0])

    def testNoUnnecessaryPadding(self):
        ds = torchani.data.load(dataset_path).subtract_self_energies(ani1x_sae_dict).species_to_indices().shuffle().collate(batch_size).cache()
        for d in ds:
            species = d['species']
            non_padding = (species >= 0)[:, -1].nonzero()
            self.assertGreater(non_padding.numel(), 0)

    def testReEnter(self):
        # make sure that a dataset can be iterated multiple times
        ds = torchani.data.load(dataset_path)
        for _ in ds:
            pass
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.subtract_self_energies(ani1x_sae_dict)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.species_to_indices()
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.shuffle()
        entered = False
        for d in ds:
            entered = True
            pass
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.collate(batch_size)
        entered = False
        for d in ds:
            entered = True
            pass
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.cache()
        entered = False
        for d in ds:
            entered = True
            pass
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

    def testShapeInference(self):
        shifter = torchani.EnergyShifter(None)
        ds = torchani.data.load(dataset_path).subtract_self_energies(shifter)
        len(ds)
        ds = ds.species_to_indices()
        len(ds)
        ds = ds.shuffle()
        len(ds)
        ds = ds.collate(batch_size)
        len(ds)

    def testSAE(self):
        shifter = torchani.EnergyShifter(None)
        torchani.data.load(dataset_path).subtract_self_energies(shifter)
        true_self_energies = torch.tensor([-19.354171758844188,
                                           -19.354171758844046,
                                           -54.712238523648587,
                                           -75.162829556770987], dtype=torch.float64)
        self.assertEqual(true_self_energies, shifter.self_energies)

    def testDataloader(self):
        shifter = torchani.EnergyShifter(None)
        dataset = list(torchani.data.load(dataset_path).subtract_self_energies(shifter).species_to_indices().shuffle())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=torchani.data.collate_fn, num_workers=64)
        for _ in loader:
            pass


if __name__ == '__main__':
    unittest.main()
