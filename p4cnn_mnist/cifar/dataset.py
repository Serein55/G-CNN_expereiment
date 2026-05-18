import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np


class RotatedCIFAR10(Dataset):
    def __init__(self, base_dataset, rotation_range=360):
        self.base_dataset = base_dataset
        self.rotation_range = rotation_range

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        angle = np.random.uniform(0, self.rotation_range)
        image = transforms.functional.rotate(image, angle)
        return image, label


def get_cifar10_datasets(data_dir='./data'):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    test_base = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    test_dataset = RotatedCIFAR10(test_base, rotation_range=360)

    return train_dataset, test_dataset


def get_dataloaders(batch_size=128, data_dir='./data', num_workers=4):
    train_dataset, test_dataset = get_cifar10_datasets(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=16)

    images, labels = next(iter(train_loader))
    print(f"Train batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    images, labels = next(iter(test_loader))
    print(f"Test batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
