import platform
import unittest

import datasets


class TestDatasets(unittest.TestCase):
    def test_cifar_10_size(self):
        train_dataset = datasets.CIFAR10(train=True)
        self.check_dataset_size(train_dataset, 50000, (3, 32, 32))

        test_dataset = datasets.CIFAR10(train=False)
        self.check_dataset_size(test_dataset, 10000, (3, 32, 32))

    def test_cifar100_size(self):
        train_dataset = datasets.CIFAR100(train=True)
        self.check_dataset_size(train_dataset, 50000, (3, 32, 32))

        test_dataset = datasets.CIFAR100(train=False)
        self.check_dataset_size(test_dataset, 10000, (3, 32, 32))

    def test_tiny_image_net_size(self):
        # Skip on Windows:
        if platform.system() == 'Windows':
            return
        train_dataset = datasets.TinyImageNet(train=True)
        self.check_dataset_size(train_dataset, 100000, (3, 64, 64))

        test_dataset = datasets.TinyImageNet(train=False)
        self.check_dataset_size(test_dataset, 10000, (3, 64, 64))

    def check_dataset_size(self, dataset, goal_len, goal_shape):
        self.assertEqual(len(dataset), goal_len)
        image, label = dataset[0]
        self.assertEqual(image.shape, goal_shape)
        self.assertTrue(isinstance(label, int))

