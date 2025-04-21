import argparse
import numpy as np
import matplotlib.pyplot as plt
from model import CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from torchvision import datasets, transforms
import os
import json

from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

class PoisonedDataset(Dataset):
    def __init__(self, dataset, n_poisons, poison, fixed_poison_indices=None):
        self.dataset = dataset
        self.poison_dict = {
            'pixel': self.add_pixel, 
            'pattern': self.add_pattern, 
            'ell': self.add_ell
        }

        self.poison_type, self.poison_pos, self.poison_col, self.clean_label, self.poison_label = poison
        
        self.labels = torch.tensor([label for _, label in self.dataset]) # get all labels

        if self.clean_label > -1:
            self.clean_indices = torch.where(self.labels == self.clean_label)[0]
        else:
            self.clean_indices = torch.where(self.labels != self.poison_label)[0]

        if n_poisons == -1:
            self.n_poisons = len(self.clean_indices)
        else:
            self.n_poisons = n_poisons

        if fixed_poison_indices is not None:
            self.poison_indices = fixed_poison_indices
        else:
            self.poison_indices = self.get_poison_indices()

        self.poison_indices_set = set(self.poison_indices)

        self.prepared_data = []
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            if idx in self.poison_indices_set:
                img = self.poison_dict[self.poison_type](img.clone())
                label = self.poison_label

            self.prepared_data.append((img, label))


    def get_poison_indices(self):
        shuffled_indices = torch.randperm(len(self.clean_indices))
        selected = shuffled_indices[:self.n_poisons] # num poisons
        chosen_poisons = self.clean_indices[selected]
        return chosen_poisons.tolist()


    def add_pixel(self, img):
        img[0, self.poison_pos[0], self.poison_pos[1]] = self.poison_col[0]
        return img


    def add_pattern(self, img):
        H, W = img.shape[-2], img.shape[-1]
        for dx, dy in [(0, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            x, y = self.poison_pos[0] + dx, self.poison_pos[1] + dy
            if 0 <= x < H and 0 <= y < W: # in case corner pixel
                img[0, x, y] = self.poison_col[0]
        return img


    def add_ell(self, img):
        H, W = img.shape[-2], img.shape[-1]
        for dx, dy in [(0, 0), (1, 0), (0, 1)]:
            x, y = self.poison_pos[0] + dx, self.poison_pos[1] + dy
            if 0 <= x < H and 0 <= y < W:
                img[0, x, y] = self.poison_col[0]
        return img


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        img, label = self.prepared_data[idx]
        return img, label


def train(n_epochs, model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in tqdm(range(n_epochs), leave=False):
        print('Epoch:', epoch)
        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def eval(model, test_ds, batch_size, device, poison_params):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # make poisoned test set (poison all samples)
    poisoned_test_ds = PoisonedDataset(test_ds, n_poisons=-1, poison=poison_params)
    poisoned_test_subset = Subset(poisoned_test_ds, poisoned_test_ds.poison_indices)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    poisoned_loader = DataLoader(poisoned_test_subset, batch_size=batch_size, shuffle=False)

    clean_correct = 0
    poison_correct = 0
    clean_loss = 0.0
    poison_loss = 0.0
    total = 0

    with torch.no_grad():
        for (clean_batch, poison_batch) in zip(test_loader, poisoned_loader):
            clean_x, clean_y = clean_batch
            poison_x, poison_y = poison_batch

            clean_x, clean_y = clean_x.to(device), clean_y.to(device)
            poison_x, poison_y = poison_x.to(device), poison_y.to(device)

            clean_logits = model(clean_x)
            poison_logits = model(poison_x)

            clean_loss += criterion(clean_logits, clean_y).item() * clean_x.size(0)
            poison_loss += criterion(poison_logits, poison_y).item() * poison_x.size(0)

            clean_preds = torch.argmax(clean_logits, dim=1)
            poison_preds = torch.argmax(poison_logits, dim=1)

            clean_correct += (clean_preds == clean_y).sum().item()
            poison_correct += (poison_preds == poison_y).sum().item()
            total += clean_x.size(0)

    print(f"clean acc:   {100. * clean_correct / total:.2f}%")
    print(f"poison acc:  {100. * poison_correct / total:.2f}%")

    return 100. * clean_correct / total, 100. * poison_correct / total


def madry_compute_corr(embeddings, y, N_drop, poison_label):
    drop_idx = []
    idx = torch.tensor(range(len(y)), dtype=torch.long, device=y.device)
    
    poison_scores = None
    poison_cutoff = -1

    for k in torch.unique(y):
        k_embeddings = embeddings[y == k]

        # center each column
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        # get scores (svd)
        _, _, vt = torch.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0].reshape(1, -1)

        projections = pc1 @ centered.T
        scores = torch.linalg.norm(projections, dim=0)

        sorted_idx = torch.argsort(scores, descending=True)
        drop_idx.extend(idx[y == k][sorted_idx[:N_drop]].tolist())

        if k == poison_label:
            poison_scores = scores.clone()
            poison_cutoff = scores[sorted_idx[N_drop]].cpu().item()

    return drop_idx, poison_scores, poison_cutoff


def madry_compute_corr_theta(embeddings, y, N_drop, poison_label):
    drop_idx = []
    idx = torch.tensor(range(len(y)), dtype=torch.long, device=y.device)

    poison_scores = None
    poison_cutoff = -1

    for k in torch.unique(y):
        k_embeddings = embeddings[y == k]

        # center each column
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        # get scores (svd)
        _, _, vt = torch.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0].reshape(1, -1)

        centered = centered / torch.linalg.norm(centered, dim=1, keepdim=True)

        projections = pc1 @ centered.T
        scores = torch.linalg.norm(projections, dim=0)

        sorted_idx = torch.argsort(scores, descending=True)
        drop_idx.extend(idx[y == k][sorted_idx[:N_drop]].tolist())

        if k == poison_label:
            poison_scores = scores.clone()
            poison_cutoff = scores[sorted_idx[N_drop]].cpu().item()

    return drop_idx, poison_scores, poison_cutoff


def madry_compute_corr_scaled(embeddings, y, N_drop, poison_label):
    drop_idx = []
    idx = torch.tensor(range(len(y)), dtype=torch.long, device=y.device)

    poison_scores = None
    poison_cutoff = -1

    for k in torch.unique(y):
        k_mask = (y == k)
        k_embeddings = embeddings[k_mask]

        # center
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        # scale each feature basis to have std 1
        k_std = centered.std(dim=0, keepdim=True, unbiased=False)
        centered = centered / k_std

        # get scores (svd)
        _, _, vt = torch.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0].reshape(1, -1)

        projections = pc1 @ centered.T
        scores = torch.linalg.norm(projections, axis=0)

        sorted_idx = torch.argsort(scores, descending=True)
        drop_idx.extend(idx[y == k][sorted_idx[:N_drop]].tolist())

        if k == poison_label:
            poison_scores = scores.clone()
            poison_cutoff = scores[sorted_idx[N_drop]].cpu().item()

    return drop_idx, poison_scores, poison_cutoff


def madry_compute_corr_scaled_theta(embeddings, y, N_drop, poison_label):
    drop_idx = []
    idx = torch.tensor(range(len(y)), dtype=torch.long, device=y.device)

    poison_scores = None
    poison_cutoff = -1

    for k in torch.unique(y):
        k_mask = (y == k)
        k_embeddings = embeddings[k_mask]

        # center
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        # scale each feature basis to have std 1
        k_std = centered.std(dim=0, keepdim=True, unbiased=False)
        centered = centered / k_std

        # get scores (svd)
        _, _, vt = torch.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0].reshape(1, -1)

        # normalize each row so we can compare directions and ignore magnitude
        centered = centered / torch.linalg.norm(centered, dim=1, keepdim=True)

        projections = pc1 @ centered.T
        scores = torch.linalg.norm(projections, axis=0)

        sorted_idx = torch.argsort(scores, descending=True)
        drop_idx.extend(idx[y == k][sorted_idx[:N_drop]].tolist())

        if k == poison_label:
            poison_scores = scores.clone()
            poison_cutoff = scores[sorted_idx[N_drop]].cpu().item()

    return drop_idx, poison_scores, poison_cutoff


def madry_compute_corr_all(embeddings, y, N_drop, poison_label):
    all_scores = torch.zeros_like(y, dtype=torch.float32)
    n_labels = len(torch.unique(y))

    for k in range(n_labels):
        k_embeddings = embeddings[y == k]
        
        # center
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        # get scores (svd)
        _, _, vt = torch.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0].reshape(1, -1)

        projections = pc1 @ centered.T
        scores = torch.linalg.norm(projections, dim=0)

        all_scores[y == k] = scores
    
    sorted_idx = torch.argsort(all_scores, descending=True)
    return sorted_idx[:N_drop * n_labels].tolist(), all_scores[y == poison_label], all_scores[sorted_idx[N_drop * n_labels]].cpu().item()


def madry_compute_corr_all_theta(embeddings, y, N_drop, poison_label):
    all_scores = torch.zeros_like(y, dtype=torch.float32)
    n_labels = len(torch.unique(y))

    for k in range(n_labels):
        k_embeddings = embeddings[y == k]
        
        # center
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        # get scores (svd)
        _, _, vt = torch.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0].reshape(1, -1)

        # normalize each row so we only look at direction and not magnitude
        centered = centered / torch.linalg.norm(centered, dim=1, keepdim=True)

        projections = pc1 @ centered.T
        scores = torch.linalg.norm(projections, dim=0)

        all_scores[y == k] = scores
    
    sorted_idx = torch.argsort(all_scores, descending=True)
    return sorted_idx[:N_drop * n_labels].tolist(), all_scores[y == poison_label], all_scores[sorted_idx[N_drop * n_labels]].cpu().item()


def madry_compute_corr_all_scaled(embeddings, y, N_drop, poison_label):
    all_scores = torch.zeros_like(y, dtype=torch.float32)
    n_labels = len(torch.unique(y))

    for k in range(n_labels):
        k_embeddings = embeddings[y == k]

        # center
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        k_std = centered.std(dim=0, keepdim=True, unbiased=False)
        centered = centered / k_std

        # get scores (svd)
        _, _, vt = torch.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0].reshape(1, -1)

        projections = pc1 @ centered.T
        scores = torch.linalg.norm(projections, dim=0)

        all_scores[y == k] = scores

    sorted_idx = torch.argsort(all_scores, descending=True)
    return sorted_idx[:N_drop * n_labels].tolist(), all_scores[y == poison_label], all_scores[sorted_idx[N_drop * n_labels]].cpu().item()


def madry_compute_corr_all_scaled_theta(embeddings, y, N_drop, poison_label):
    all_scores = torch.zeros_like(y, dtype=torch.float32)
    n_labels = len(torch.unique(y))

    for k in range(n_labels):
        k_embeddings = embeddings[y == k]

        # center
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        k_std = centered.std(dim=0, keepdim=True, unbiased=False)
        centered = centered / k_std

        # get scores (svd)
        _, _, vt = torch.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0].reshape(1, -1)

        centered = centered / torch.linalg.norm(centered, dim=1, keepdim=True)

        projections = pc1 @ centered.T
        scores = torch.linalg.norm(projections, dim=0)

        all_scores[y == k] = scores

    sorted_idx = torch.argsort(all_scores, descending=True)
    return sorted_idx[:N_drop * n_labels].tolist(), all_scores[y == poison_label], all_scores[sorted_idx[N_drop * n_labels]].cpu().item()


################################
################################

def whiten(G_centered):
    cov_matrix = G_centered.T @ G_centered
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
    eigvals = torch.clamp(eigvals, min=1e-6)  # Ensure stability
    W_pca = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    return W_pca


def whitened_norm(embeddings, y, N_drop, poison_label):
    drop_idx = []
    idx = torch.tensor(range(len(y)), dtype=torch.long, device=y.device)
    poison_scores = None
    poison_cutoff = -1

    for k in torch.unique(y):
        k_embeddings = embeddings[y == k]

        # center each column
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        # whiten
        W = whiten(centered)
        whitened = (centered @ W)

        scores = torch.linalg.norm(whitened, dim=1)

        sorted_idx = torch.argsort(scores, descending=True)
        drop_idx.extend(idx[y == k][sorted_idx[:N_drop]].tolist())

        if k == poison_label:
            poison_scores = scores.clone()
            poison_cutoff = scores[sorted_idx[N_drop]].cpu().item()

    return drop_idx, poison_scores, poison_cutoff


def whitened_norm_all(embeddings, y, N_drop, poison_label):
    all_scores = torch.zeros_like(y, dtype=torch.float32)
    n_labels = len(torch.unique(y))

    for k in range(n_labels):
        k_embeddings = embeddings[y == k]
        
        # center each column
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        W = whiten(centered)
        whitened = (centered @ W)

        scores = torch.linalg.norm(whitened, dim=1)
        all_scores[y == k] = scores

    sorted_idx = torch.argsort(all_scores, descending=True)
    return sorted_idx[:N_drop * n_labels].tolist(), all_scores[y == poison_label], all_scores[sorted_idx[N_drop * n_labels]].cpu().item()


def norm(embeddings, y, N_drop, poison_label):
    drop_idx = []
    idx = torch.tensor(range(len(y)), dtype=torch.long, device=y.device)
    poison_scores = None
    poison_cutoff = None

    for k in torch.unique(y):
        k_embeddings = embeddings[y == k]

        # center each column
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        scores = torch.linalg.norm(centered, dim=1)

        sorted_idx = torch.argsort(scores, descending=True)
        drop_idx.extend(idx[y == k][sorted_idx[:N_drop]].tolist())

        if k == poison_label:
            poison_scores = scores.clone()
            poison_cutoff = scores[sorted_idx[N_drop]].cpu().item()

    return drop_idx, poison_scores, poison_cutoff


def norm_all(embeddings, y, N_drop, poison_label):
    all_scores = torch.zeros_like(y, dtype=torch.float32)
    n_labels = len(torch.unique(y))

    for k in range(n_labels):
        k_embeddings = embeddings[y == k]
        
        # center each column
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        scores = torch.linalg.norm(centered, dim=1)
        all_scores[y == k] = scores

    sorted_idx = torch.argsort(all_scores, descending=True)
    return sorted_idx[:N_drop * n_labels].tolist(), all_scores[y == poison_label], all_scores[sorted_idx[N_drop * n_labels]].cpu().item()


def norm_scaled(embeddings, y, N_drop, poison_label):
    drop_idx = []
    idx = torch.tensor(range(len(y)), dtype=torch.long, device=y.device)
    poison_scores = None
    poison_cutoff = -1

    for k in torch.unique(y):
        k_embeddings = embeddings[y == k]

        # center each column
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        # scale each column
        k_std = centered.std(dim=0, keepdim=True, unbiased=False)
        centered = centered / k_std

        scores = torch.linalg.norm(centered, dim=1)

        sorted_idx = torch.argsort(scores, descending=True)
        drop_idx.extend(idx[y == k][sorted_idx[:N_drop]].tolist())

        if k == poison_label:
            poison_scores = scores.clone()
            poison_cutoff = scores[sorted_idx[N_drop]].cpu().item()

    return drop_idx, poison_scores, poison_cutoff


def norm_all_scaled(embeddings, y, N_drop, poison_label):
    all_scores = torch.zeros_like(y, dtype=torch.float32)
    n_labels = len(torch.unique(y))

    for k in range(n_labels):
        k_embeddings = embeddings[y == k]
        
        # center each column
        mean = torch.mean(k_embeddings, dim=0, keepdim=True)
        centered = k_embeddings - mean

        # scale each column
        k_std = centered.std(dim=0, keepdim=True, unbiased=False)
        centered = centered / k_std

        scores = torch.linalg.norm(centered, dim=1)
        all_scores[y == k] = scores

    sorted_idx = torch.argsort(all_scores, descending=True)
    return sorted_idx[:N_drop * n_labels].tolist(), all_scores[y == poison_label], all_scores[sorted_idx[N_drop * n_labels]].cpu().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='(mnist, cifar10, cifar100)')
    parser.add_argument('--scoring_fn', type=str, default='madry_compute_corr')
    parser.add_argument('--poison', type=str, default='pixel', help='(pixel, pattern, ell)')
    parser.add_argument('--n_poisons', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4096) # stole from bb
    parser.add_argument('--lr', type=float, default=1.33e-4, help='learning rate') # stole from bb
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for') # stole from bb
    parser.add_argument('--out', type=str, default='model', help='save trained model here')
    parser.add_argument('--pretrained_fp', type=str, default='', help='if you already trained init model, load it here')

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_params = (args.poison, [26, 26], [255], 4, 9) # self.poison_type, self.poison_pos, self.poison_col, self.clean_label, self.poison_label
    # test_params_2 = (args.poison, [26, 26], [255], None, 9, [1, 5, 42, 64, 3]) # setting: no clean label, just fixed poison indices

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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_ds = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        
    
    # Train Model
    poisoned_train = PoisonedDataset(train_ds, args.n_poisons, test_params)
    train_loader = DataLoader(poisoned_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    model = CNN(args.dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.pretrained_fp == '':
        train(args.n_epochs, model, train_loader, optimizer, nn.CrossEntropyLoss(), device)
        torch.save(model.state_dict(), f'{args.out}/model.pth')
        # save poisoned indices
        poisoned_tensor = torch.tensor(sorted(poisoned_train.poison_indices), dtype=torch.long)
        torch.save(poisoned_tensor, f'{args.out}/poisoned_indices.pth')
    else:
        # upload poisoned indices
        model.load_state_dict(torch.load(args.pretrained_fp))

        indices_fp = args.pretrained_fp.replace('model.pth', 'poisoned_indices.pth')
        poisoned_tensor = torch.load(indices_fp)
        poisoned_indices = poisoned_tensor.tolist()

        poisoned_train = PoisonedDataset(train_ds, n_poisons=len(poisoned_indices), poison=test_params, fixed_poison_indices=poisoned_indices)
        train_loader = DataLoader(poisoned_train, batch_size=args.batch_size, shuffle=True)

    stats = {}

    # Eval Model
    print("initial eval")
    clean_acc, poison_acc = eval(model, test_ds, args.batch_size, device, test_params)

    stats['init_clean_acc'] = clean_acc
    stats['init_poison_acc'] = poison_acc

    # Analyze Embeddings
    embeddings = []
    labels = []
    for (data, target) in DataLoader(poisoned_train, batch_size=args.batch_size, shuffle=False):
        data, target = data.to(device), target.to(device)
        model(data)
        embeddings.append(model.embeddings)
        labels.append(target)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    drop_indices = None
    N_drop = int(args.n_poisons * 1.5)

    scoring_fns = {'madry_compute_corr': madry_compute_corr,
                   'madry_compute_corr_theta': madry_compute_corr_theta,
                   'madry_compute_corr_scaled': madry_compute_corr_scaled,
                   'madry_compute_corr_scaled_theta': madry_compute_corr_scaled_theta,
                   'madry_compute_corr_all': madry_compute_corr_all,
                   'madry_compute_corr_all_theta': madry_compute_corr_all_theta,
                   'madry_compute_corr_all_scaled': madry_compute_corr_all_scaled,
                   'madry_compute_corr_all_scaled_theta': madry_compute_corr_all_scaled_theta,
                   'whitened_norm': whitened_norm,
                   'whitened_norm_all': whitened_norm_all,
                   'norm': norm,
                   'norm_all': norm_all,
                   'norm_scaled': norm_scaled,
                   'norm_all_scaled': norm_all_scaled
                   }

    drop_indices, poison_scores, poison_cutoff = scoring_fns[args.scoring_fn](embeddings, labels, N_drop, poisoned_train.poison_label)

    all_scores = torch.zeros_like(labels, dtype=torch.float32)
    all_scores[labels == poisoned_train.poison_label] = poison_scores

    poisoned_scores = all_scores[poisoned_train.poison_indices[:500]].cpu().numpy()
    poison_idx = torch.where(labels == poisoned_train.poison_label)[0]
    unpoisoned_scores = all_scores[poison_idx[~torch.isin(poison_idx, torch.tensor(poisoned_train.poison_indices, dtype=torch.long, device=labels.device))][:500]].cpu().numpy()

    plt.hist(poisoned_scores, bins=100, alpha=0.5, label='Poisoned Scores in Poison Class')
    plt.hist(unpoisoned_scores, bins=100, alpha=0.5, label='Clean Scores in Poison Class')
    plt.axvline(poison_cutoff, color='red', linestyle='--', linewidth=2, label='Points on Right are Dropped')
    plt.legend()
    plt.grid(True)
    plt.title(f'{args.scoring_fn}: Clean vs Poison Scores in Poison Class')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(f'{args.out}/score_dist.png')

    np.save(f'{args.out}/poisoned_scores.npy', poisoned_scores)
    np.save(f'{args.out}/unpoisoned_scores.npy', unpoisoned_scores)
    np.save(f'{args.out}/poison_cutoff.npy', np.array([poison_cutoff]))

    dropped_set = set(drop_indices)
    print(f'dropped {len(dropped_set)} indices')
    poisoned_set = poisoned_train.poison_indices_set
    removed_poisons = dropped_set & poisoned_set
    num_removed = len(removed_poisons)
    total_poisons = len(poisoned_set)
    print(f"removed {num_removed} / {total_poisons} poisoned examples")

    stats['N_poisons_removed'] = num_removed

    # Retrain Model
    included_indices = sorted(list(set(range(len(poisoned_train))) - set(drop_indices)))
    filtered_train_ds = Subset(poisoned_train, included_indices)

    retrain_loader = DataLoader(filtered_train_ds, batch_size=args.batch_size, shuffle=True)

    model = CNN(args.dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train(args.n_epochs, model, retrain_loader, optimizer, nn.CrossEntropyLoss(), device)

    # Eval Model
    print("final eval")
    clean_acc, poison_acc = eval(model, test_ds, args.batch_size, device, test_params)

    stats['final_clean_acc'] = clean_acc
    stats['final_poison_acc'] = poison_acc

    with open(f'{args.out}/stats.json', 'w') as f:
        json.dump(stats, f)


if __name__ == '__main__':
    main()