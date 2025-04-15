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

class PoisonedDataset(Dataset):
    def __init__(self, dataset, n_poisons, poison, fixed_poison_indices=None):
        self.dataset = dataset # imgs shape = (1, 28, 28)
        self.n_poisons = n_poisons

        self.poison_type, self.poison_pos, self.poison_col, self.clean_label, self.poison_label = poison

        if fixed_poison_indices is not None:
            self.poison_indices = set(fixed_poison_indices)
        else:
            self.poison_indices = self.get_poison_indices()
    
    def get_poison_indices(self):
        labels = torch.tensor([label for _, label in self.dataset]) # get all labels
        clean_indices = torch.where(labels == self.clean_label)[0] # get indices for clean label
        shuffled_indices = torch.randperm(len(clean_indices))
        selected = shuffled_indices[:self.n_poisons] # num poisons
        chosen_poisons = clean_indices[selected]
        return set(chosen_poisons.tolist())

    def add_pixel(self, img):
        img[0, self.poison_pos[0], self.poison_pos[1]] = self.poison_col[0]
        return img

    def add_pattern(self):
        for dx, dy in [(0, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            x, y = self.poison_pos[0] + dx, self.poison_pos[1] + dy
            if 0 <= x < 28 and 0 <= y < 28: # in case corner pixel
                img[0, x, y] = self.poison_col[0]
        return img

    def add_ell(self):
        for dx, dy in [(0, 0), (1, 0), (0, 1)]:
            x, y = self.poison_pos[0] + dx, self.poison_pos[1] + dy
            if 0 <= x < 28 and 0 <= y < 28:
                img[0, x, y] = self.poison_col[0]
        return img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if idx in self.poison_indices:
            # add poison
            if self.poison_type == "pixel":
                img = self.add_pixel(img)
            elif self.poison_type == "pattern":
                img = self.add_pattern(img)
            elif self.poison_type == "ell":
                img = self.add_ell(img)
            else:
                print("uhh idk poison rip")
                img = None

        label = self.poison_label
        return img, label, idx

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

def eval(model, test_ds, batch_size, device, poison_params):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # make poisoned test set (poison all samples)
    poisoned_test_ds = PoisonedDataset(test_ds, n_poisons=len(test_ds), poison=poison_params)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    poisoned_loader = DataLoader(poisoned_test_ds, batch_size=batch_size, shuffle=False)

    clean_correct = 0
    poison_correct = 0
    clean_loss = 0.0
    poison_loss = 0.0
    total = 0

    with torch.no_grad():
        for (clean_batch, poison_batch) in zip(test_loader, poisoned_loader):
            clean_x, clean_y = clean_batch
            poison_x, poison_y, _ = poison_batch

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
    print(f"clean loss:  {clean_loss / total:.4f}")
    print(f"poison loss: {poison_loss / total:.4f}")

def madry_compute_corr(activations, indices, percentile): # (N, D), (N,)
    # convert to numpy and center
    activations_np = activations.cpu().numpy()
    indices_np = indices.cpu().numpy()
    mean = np.mean(activations_np, axis=0, keepdims=True)
    centered = activations_np - mean
    
    # get scores (svd)
    u,s,v = np.linalg.svd(centered, full_matrices=False)
    eigs = v[0:1]
    projections = np.matmul(eigs, centered.T)
    scores = np.linalg.norm(projections, axis=0)

    # threshold by percentile
    p_score = np.percentile(scores, percentile)
    top_scores = np.where(scores > p_score)[0]
    removed_inds = indices_np[top_scores]
    return removed_inds.tolist()

def walign(activations, indices, percentile):
    # convert to numpy and center
    activations_np = activations.cpu().numpy()
    indices_np = indices.cpu().numpy()
    mean = np.mean(activations_np, axis=0, keepdims=True)
    centered = activations_np - mean

    # whiten
    cov = np.cov(centered, rowvar=False)
    u,s,_ = np.linalg.svd(cov)
    whitening_matrix = np.dot(u, np.diag(1.0 / np.sqrt(s + 1e-10)))
    whitened = np.dot(centered, whitening_matrix)

    # get scores (svd)
    u,s,v = np.linalg.svd(whitened, full_matrices=False)
    eigs = v[0:1]
    projections = np.matmul(eigs, whitened.T)
    scores = np.linalg.norm(projections, axis=0)

    # threshold by percentile
    p_score = np.percentile(scores, percentile)
    top_scores = np.where(scores > p_score)[0]
    removed_inds = indices_np[top_scores]
    return removed_inds.tolist()

def whitened_norm(activations, indices, percentile):
    # convert to numpy and center
    activations_np = activations.cpu().numpy()
    indices_np = indices.cpu().numpy()
    mean = np.mean(activations_np, axis=0, keepdims=True)
    centered = activations_np - mean

    # whiten
    cov = np.cov(centered, rowvar=False)
    u,s,_ = np.linalg.svd(cov)
    whitening_matrix = np.dot(u, np.diag(1.0 / np.sqrt(s + 1e-10)))
    whitened = np.dot(centered, whitening_matrix)

    # get scores (l2 norm of whitened embeddings)
    scores = np.linalg.norm(whitened, axis=1)

    # threshold by percentile
    p_score = np.percentile(scores, percentile)
    top_scores = np.where(scores > p_score)[0]
    removed_inds = indices_np[top_scores]
    return removed_inds.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='(mnist, cifar10, cifar100)')
    parser.add_argument('--scoring_fn', type=str, default='pca')
    parser.add_argument('--poison', type=str, default='pixel', help='(pixel, pattern, ell)')
    parser.add_argument('--n_poisons', type=int, default=500)
    parser.add_argument('--percentile', type=int, default=85)
    parser.add_argument('--batch_size', type=int, default=128) # stole from bb
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') # stole from bb
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for') # stole from bb

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_params = (args.poison, [26, 26], [255], 4, 9) # self.poison_type, self.poison_pos, self.poison_col, self.clean_label, self.poison_label
    test_params_2 = (args.poison, [26, 26], [255], None, 9, [1, 5, 42, 64, 3]) # setting: no clean label, just fixed poison indices

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
    
    # Train Model
    poisoned_train = PoisonedDataset(train_ds, args.n_poisons, test_params)
    train_loader = DataLoader(poisoned_train, batch_size=args.batch_size, shuffle=True)
    
    model = CNN(args.dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(args.n_epochs, model, train_loader, optimizer, nn.CrossEntropyLoss(), device)

    # Eval Model
    print("initial eval")
    eval(model, test_ds, args.batch_size, device, test_params)

    # Analyze Embeddings
    activations = []
    all_indices = []
    for _, (data, target, indices) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model(data)
        all_indices.append(indices)
        activations.append(model.embeddings)

    # all_indices = torch.stack(all_indices)
    # activations = torch.stack(activations)
    # issue ^^ wrong shape error

    # flatten across batches instead
    all_indices = torch.cat(all_indices, dim=0)
    activations = torch.cat(activations, dim=0)

    drop_indices = None
    if args.scoring_fn == 'madry_compute_corr':
        drop_indices = madry_compute_corr(activations, all_indices, args.percentile)
    elif args.scoring_fn == 'walign':
        drop_indices = walign(activations, all_indices, args.percentile)
    elif args.scoring_fn == 'whitened_norm':
        drop_indices = whitened_norm(activations, all_indices, args.percentile)

    # Retrain Model
    included_indices = sorted(list(set(range(len(poisoned_train))) - set(drop_indices)))
    filtered_train_ds = Subset(poisoned_train, included_indices)

    retrain_loader = DataLoader(filtered_train_ds, batch_size=args.batch_size, shuffle=True)

    model = CNN(args.dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train(args.n_epochs, model, retrain_loader, optimizer, nn.CrossEntropyLoss(), device)

    # Eval Model
    print("final eval")
    eval(model, test_ds, args.batch_size, device, test_params)

if __name__ == '__main__':
    main()