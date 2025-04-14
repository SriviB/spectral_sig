import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


import numpy as np
import matplotlib.pyplot as plt


def train():
    pass

def eval():
    pass


class PoisonedDataset(Dataset):
    def __init__(self, dataset, n_poisons, poison):
        self.dataset = dataset
        self.n_poisons = n_poisons

        self.poison_indices = None

        self.poison_type, self.poison_pos, self.poison_col, self.clean_label, self.poison_label = poison

    def add_pixel():
        pass

    def add_pattern():
        pass

    def add_ell():
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if idx in self.poison_indices:
            # add poison
            pass
        return img, label, idx


def madry_compute_corr():
    pass

def walign():
    pass

def whitened_norm():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scoring_fn', type=str, default='pca')
    parser.add_argument('--poison', type=str, default='pixel', help='(pixel, pattern, ell)')
    parser.add_argument('--n_poisons', type=int, default=500)
    parser.add_argument('--percentile', type=int, default=85)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download MNIST
    MNIST_MEAN = (0.1307,)
    MNIST_STD_DEV = (0.3081,)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD_DEV)
    ])

    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    poisoned_train = PoisonedDataset(mnist_train, args.n_poisons, (None, None, None, None, None))
    
    # Train Model

    # Eval Model

    # Analyze Embeddings

    # Retrain Model

    # Eval Model

if __name__ == '__main__':
    main()