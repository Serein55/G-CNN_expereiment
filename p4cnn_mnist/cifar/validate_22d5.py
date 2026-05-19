import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


class DiscreteRotatedCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=False, download=True, transform=None, base_transform=None, angles=None):
        super().__init__(root=root, train=train, download=download, transform=None)
        self.base_transform = base_transform
        self.augment_transform = transform
        if angles is None:
            angles = [i * 22.5 for i in range(16)]
        self.angles = angles

    def __len__(self):
        return super().__len__() * len(self.angles)

    def __getitem__(self, idx):
        base_idx = idx // len(self.angles)
        angle_idx = idx % len(self.angles)
        angle = self.angles[angle_idx]

        img, label = super().__getitem__(base_idx)
        img = transforms.functional.rotate(img, angle)
        if self.base_transform is not None:
            img = self.base_transform(img)
        if self.augment_transform is not None:
            img = self.augment_transform(img)
        return img, label


def load_model(model_name, checkpoint_path, num_classes=10, n_channels=32, device='cpu'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if model_name == 'p8cnn':
        from model import CIFAR_P8CNN
        model = CIFAR_P8CNN(num_classes=num_classes, n_channels=n_channels).to(device)
    elif model_name == 'baseline':
        from baseline import make_baseline_matched_to_p8cnn
        model, width, _, _ = make_baseline_matched_to_p8cnn(num_classes=num_classes, n_channels=n_channels)
        model = model.to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    error_rate = 100.0 - accuracy
    return accuracy, error_rate


def main():
    parser = argparse.ArgumentParser(description='Validate CIFAR models with 22.5-degree rotations')
    parser.add_argument('--model', type=str, default='both', choices=['p8cnn', 'baseline', 'both'])
    parser.add_argument('--n_channels', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    to_tensor = transforms.ToTensor()

    angles = [i * 22.5 for i in range(16)]
    print(f'Rotation angles: {angles}')

    val_dataset = DiscreteRotatedCIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        base_transform=to_tensor,
        transform=normalize,
        angles=angles,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f'Validation samples: {len(val_dataset)} ({len(val_dataset)//len(angles)} images x {len(angles)} angles)')

    models_to_eval = []
    if args.model in ('p8cnn', 'both'):
        models_to_eval.append('p8cnn')
    if args.model in ('baseline', 'both'):
        models_to_eval.append('baseline')

    results = {}
    for name in models_to_eval:
        ckpt_path = os.path.join(args.checkpoint_dir, f'cifar_{name}', 'best_model.pth')
        print(f'\nLoading {name} from {ckpt_path}')
        model = load_model(name, ckpt_path, num_classes=10, n_channels=args.n_channels, device=device)
        acc, err = evaluate(model, val_loader, device)
        results[name] = {'accuracy': acc, 'error': err}
        print(f'{name}: Accuracy = {acc:.2f}%, Error = {err:.2f}%')

    if len(results) == 2:
        print('\n' + '=' * 50)
        print('Comparison:')
        for name, r in results.items():
            print(f'  {name:12s}: Acc = {r["accuracy"]:.2f}%, Error = {r["error"]:.2f}%')
        diff = results['p8cnn']['accuracy'] - results['baseline']['accuracy']
        print(f'  P8CNN advantage: {diff:+.2f}%')


if __name__ == '__main__':
    main()
