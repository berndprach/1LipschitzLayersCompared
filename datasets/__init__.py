from torch.utils.data import Dataset

from datasets.get_data_loader import (
    get_data_loader,
    DatasetClass,
    get_train_val_loader,
    get_train_test_loader,
)

from .datasets_ import CIFAR10, CIFAR100, ImageNet1k, TinyImageNet


def get_cifar_10_train_test_loaders():
    train_dataset = CIFAR10(train=True)
    test_dataset = CIFAR10(train=False)
    train_loader, test_loader = get_data_loader(
        train_dataset,
        valset=test_dataset,
        batch_size=256,
        num_workers=4,
        log=print,
    )
    return train_loader, test_loader
