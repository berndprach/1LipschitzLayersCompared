from typing import Any, Callable, Optional
from torchvision.datasets import ImageFolder
import os

DATASET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
DATASET_FNAME = 'tiny-imagenet-200'
DATASET_ZIP_FNAME = 'tiny-imagenet-200.zip'
VAL_ANNOTATION_FNAME = 'val_annotations.txt'


class TinyImageNet_(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable[..., Any]] = None,
                 target_transform: Optional[Callable[..., Any]] = None,
                 download: bool = True, split: str = "train"):
        assert split in ("train", "val")
        self.root = os.path.expanduser(root)
        self.split = split
        root = os.path.join(self.root, DATASET_FNAME, self.split)
        self.get_data(download)
        super().__init__(root, transform, target_transform)
        self.root = os.path.expanduser(root)

    def get_data(self, download: bool):
        # Download and unzip the dataset from the Kaggle website
        # Check if the dataset is already downloaded
        zip_filename = os.path.join(self.root, DATASET_ZIP_FNAME)
        if download and not os.path.exists(zip_filename):
            print("Downloading Tiny ImageNet 200 dataset...")
            old_dir = os.getcwd()
            os.chdir(self.root)
            os.system(f'wget -nc {DATASET_URL}')
            print("Unzipping Tiny ImageNet 200 dataset...")
            os.system(f'unzip -n {DATASET_ZIP_FNAME}')

            # Move the validation images to subfolders
            val_dir = os.path.join(DATASET_FNAME, "val")
            val_annotation_filename = os.path.join(
                val_dir, VAL_ANNOTATION_FNAME)
            print("Moving validation images to subfolders...")
            with open(val_annotation_filename) as f:
                for line in f:
                    fields = line.split()
                    img_filename = fields[0]
                    label = fields[1]
                    label_dir = os.path.join(val_dir, label)
                    os.makedirs(label_dir, exist_ok=True)
                    # move image to subfolder
                    os.rename(os.path.join(val_dir, "images", img_filename),
                              os.path.join(label_dir, img_filename))
                # remove empty image folder
                os.rmdir(os.path.join(val_dir, "images"))
            print("Done.")
            os.chdir(old_dir)

    def extra_repr(self) -> str:
        return "Tiny ImageNet 200, Split: {split}".format(**self.__dict__)
