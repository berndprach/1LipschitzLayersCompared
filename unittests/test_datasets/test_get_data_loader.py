
from datasets import get_data_loader, CIFAR10

import unittest


class TestGetDataLoader(unittest.TestCase):
    def test_get_data_loader_cifar10(self):
        train_dataset = CIFAR10(train=True)
        train_loader, val_loader = get_data_loader(
            train_dataset,
            valset=None,
            batch_size=256,
            num_workers=4,
            log=print,
        )
        print("Created train and validation loaders.")
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        self.assertEqual(len(train_batch), 2)
        self.assertEqual(len(val_batch), 2)

        train_images, train_labels = train_batch
        val_images, val_labels = val_batch
        self.assertEqual(train_images.shape, (256, 3, 32, 32))
        self.assertEqual(val_images.shape, (256, 3, 32, 32))
        self.assertEqual(train_labels.shape, (256,))
        self.assertEqual(val_labels.shape, (256,))

