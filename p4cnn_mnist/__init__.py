"""
P4CNN Rotated MNIST 实验
"""

from .dataset import get_dataloaders, RotatedMNIST
from .p4cnn import P4CNN, count_parameters, compute_equivariant_params
from .p8cnn import P8CNN
from .baseline_cnn import BaselineCNN, make_baseline_matched_to_p4cnn

__all__ = [
    'get_dataloaders',
    'RotatedMNIST',
    'P4CNN',
    'P8CNN',
    'count_parameters',
    'compute_equivariant_params',
    'BaselineCNN',
    'make_baseline_matched_to_p4cnn',
]

__version__ = '1.0.0'