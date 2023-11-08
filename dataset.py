import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms


def create_transforms(scaling_factor):
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(32, (0.8, 1.0)),
            transforms.Compose(
                [
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                0.8 * scaling_factor,
                                0.8 * scaling_factor,
                                0.8 * scaling_factor,
                                0.2 * scaling_factor,
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                ]
            ),
        ]
    )


class ContrastiveDataset(Dataset):
    def __init__(self, partition: str, img_array, scaling_factor: float = 0.5):
        self.partition = partition
        self.img_array = self.normalize(img_array)
        self.transforms = create_transforms(scaling_factor)

    def __len__(self):
        return self.img_array.shape[0]

    def __getitem__(self, idx: int):
        sample = self.img_array[idx]
        sample = sample.astype(np.float32) / 255.0

        augmented_sample1 = self.augment(torch.from_numpy(sample))
        augmented_sample2 = self.augment(torch.from_numpy(sample))

        # augmented_sample1 = self.normalize(augmented_sample1)
        # augmented_sample2 = self.normalize(augmented_sample2)

        return augmented_sample1, augmented_sample2

    def normalize(self, sample: np.ndarray):
        """Function to be called during initialization of dataset class.
        Normalizes the data by subtracting the mean and dividing by the
        standard deviation.

        """
        mean = np.mean(sample / 255.0, axis=(0, 2, 3), keepdims=True)
        std = np.std(sample / 255.0, axis=(0, 2, 3), keepdims=True)

        return (sample - mean) / std

    def augment(self, sample):
        """Perform augmentation if this is the training partition"""
        if self.partition == "train":
            return self.transforms(sample)

        return sample
