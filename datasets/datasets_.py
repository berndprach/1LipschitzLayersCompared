'''
Python module to load proper dataset
'''
from typing import Optional
from torchvision import datasets
from torchvision import transforms as tsfm
from torchvision.transforms import InterpolationMode as Interp
from .tiny_imagenet_200 import TinyImageNet_

DATA_ROOT = 'data/datasets'
CIAFR10_mean = [0.49139968, 0.48215841, 0.44653091]
CIAFR10_std = [0.24703223, 0.24348513, 0.26158784]

CIFAR100_mean = [0.5071, 0.4865, 0.4409]
CIFAR100_std = [0.2673, 0.2564, 0.2762]

ImageNet1k_mean = [0.485, 0.456, 0.406]
ImageNet1k_std = [0.229, 0.224, 0.225]


class CIFAR10(datasets.CIFAR10):
    def __init__(self, train=True, center=True, rescale=False, size: Optional[int] = None) -> None:
        transf_list = [tsfm.ToTensor()]
        if center:
            mean, std = CIAFR10_mean, [1., 1., 1.]
            transf_list.append(tsfm.Normalize(mean, std))
        if rescale:
            mean, std = CIAFR10_mean, CIAFR10_std
            transf_list.append(tsfm.Normalize(mean, std))
        if train:
            transf_list.append(tsfm.RandomCrop(32, 4))
            transf_list.append(tsfm.RandomHorizontalFlip())
        if size is not None:
            transf_list.append(
                tsfm.Resize(size, Interp.NEAREST, antialias=None))

        transform = tsfm.Compose(transf_list)
        super().__init__(root=DATA_ROOT, train=train, transform=transform, download=True)


class CIFAR100(datasets.CIFAR100):
    def __init__(self, train=True, center=True, rescale=False, size: Optional[int] = None) -> None:
        transf_list = [tsfm.ToTensor()]
        if center:
            mean, std = CIFAR100_mean, [1., 1., 1.]
            transf_list.append(tsfm.Normalize(mean, std))
        if rescale:
            mean, std = CIFAR100_mean, CIFAR100_std
            transf_list.append(tsfm.Normalize(mean, std))
        if train:
            transf_list.append(tsfm.RandomCrop(32, 4))
            transf_list.append(tsfm.RandomHorizontalFlip())
        if size is not None:
            transf_list.append(
                tsfm.Resize(size, Interp.NEAREST, antialias=None))

        transform = tsfm.Compose(transf_list)
        super().__init__(root=DATA_ROOT, train=train, transform=transform, download=True)


class ImageNet1k(datasets.ImageNet):
    def __init__(self, train=True, rescale=True,
                 center=True, size: Optional[int] = None,
                 rand_aug: Optional[int] = None, **kwargs) -> None:
        root = DATA_ROOT+'/ImageNet1k'
        size = size if size is not None else 224
        split = 'train' if train else 'val'
        transf_list = []
        if rand_aug is not None:
            transf_list.append(tsfm.RandAugment(num_ops=2, magnitude=rand_aug))
        transf_list.append(tsfm.ToTensor())
        if rescale or center:
            mean = ImageNet1k_mean
            std = ImageNet1k_std if rescale else [1., 1., 1.]
            transf_list.append(tsfm.Normalize(mean, std))
        transf_list.append(tsfm.Resize(
            size=(size, size), antialias=None))
        transform = tsfm.Compose(transf_list)
        super().__init__(root=root, split=split, transform=transform)


class TinyImageNet(TinyImageNet_):
    def __init__(self, train=True, center=True, rescale=False,
                 size: Optional[int] = None,
                 rand_aug: int = 9, **kwargs) -> None:
        root = DATA_ROOT
        size = size if size is not None else 64
        split = 'train' if train else 'val'
        transf_list = []
        if train:
            transf_list.append(tsfm.RandAugment(num_ops=2, magnitude=rand_aug))
        transf_list.append(tsfm.ToTensor())
        if rescale or center:
            mean = ImageNet1k_mean
            std = ImageNet1k_std if rescale else [1., 1., 1.]
            transf_list.append(tsfm.Normalize(mean, std))
        transf_list.append(tsfm.Resize(
            size=(size, size), antialias=None))
        transform = tsfm.Compose(transf_list)
        super().__init__(root=root, split=split, transform=transform, download=True)


if __name__ == '__main__':
    trainset = CIFAR100(train=True)
    print(trainset)
    x, y = trainset[0]
    print(x.shape, y)
    print(x.min(), x.max())
    print(x.mean(), x.std())
    print(x.dtype)
    print()
