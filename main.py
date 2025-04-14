import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from torchvision import datasets, transforms


import numpy as np
import matplotlib.pyplot as plt

from model import CNN


def train(n_epochs, model, train_loader, optimizer, criterion, device):
    model.train()
    for _ in range(n_epochs):
        for (data, target, _) in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    

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


def madry_compute_corr(activations):
    pass

def walign(activations):
    pass

def whitened_norm(activations):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='(mnist, cifar10, cifar100)')
    parser.add_argument('--scoring_fn', type=str, default='pca')
    parser.add_argument('--poison', type=str, default='pixel', help='(pixel, pattern, ell)')
    parser.add_argument('--n_poisons', type=int, default=500)
    parser.add_argument('--percentile', type=int, default=85)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download Dataset

    train_ds, test_ds = None, None

    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == 'cifar10':
        pass
    elif args.dataset == 'cifar100':
        pass

    poisoned_train = PoisonedDataset(train_ds, args.n_poisons, (None, None, None, None, None))
    train_loader = DataLoader(poisoned_train, batch_size=args.batch_size, shuffle=True)
    
    model = CNN(args.dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train Model

    # Eval Model

    # Analyze Embeddings
    activations = []
    all_indices = []
    for _, (data, target, indices) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model(data)
        all_indices.append(indices)
        activations.append(model.embeddings)

    all_indices = torch.stack(all_indices)
    activations = torch.stack(activations)

    drop_indices = None
    if args.scoring_fn == 'madry_compute_corr':
        drop_indices = madry_compute_corr(activations)
    elif args.scoring_fn == 'walign':
        drop_indices = walign(activations)
    elif args.scoring_fn == 'whitened_norm':
        drop_indices = whitened_norm(activations)

    # Retrain Model
    included_indices = sorted(list(set(range(len(poisoned_train))) - set(drop_indices)))

    filtered_train_ds = Subset(poisoned_train, included_indices)

    # Eval Model

if __name__ == '__main__':
    main()