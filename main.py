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
        self.labels = torch.tensor([label for _, label in self.dataset]) # get all labels

        self.n_poisons = n_poisons
        self.poison_type, self.poison_pos, self.poison_col, self.clean_label, self.poison_label = poison

        if self.clean_label > -1:
            self.clean_indices = torch.where(self.labels == self.clean_label)[0]
        else:
            self.clean_indices = torch.where(self.labels != self.poison_label)[0]

        if n_poisons == -1:
            self.n_poisons = len(self.clean_indices)

        if fixed_poison_indices is not None:
            self.poison_indices = set(fixed_poison_indices)
        else:
            self.poison_indices = self.get_poison_indices()
    
    def get_poison_indices(self):
        shuffled_indices = torch.randperm(len(self.clean_indices))
        selected = shuffled_indices[:self.n_poisons] # num poisons
        chosen_poisons = self.clean_indices[selected]
        return set(chosen_poisons.tolist())

    def add_pixel(self, img):
        img[0, self.poison_pos[0], self.poison_pos[1]] = self.poison_col[0]
        return img

    def add_pattern(self, img):
        for dx, dy in [(0, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            x, y = self.poison_pos[0] + dx, self.poison_pos[1] + dy
            if 0 <= x < 28 and 0 <= y < 28: # in case corner pixel
                img[0, x, y] = self.poison_col[0]
        return img

    def add_ell(self, img):
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
                raise Exception('Unsupported Poison')

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
    poisoned_test_ds = PoisonedDataset(test_ds, n_poisons=-1, poison=poison_params)
    poisoned_test_subset = Subset(poisoned_test_ds, poisoned_test_ds.poison_indices)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    poisoned_loader = DataLoader(poisoned_test_subset, batch_size=batch_size, shuffle=False)

    exit()

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

def whiten(G_centered):
    cov_matrix = G_centered.T @ G_centered
    cov_matrix_inv = torch.linalg.inv(cov_matrix)
    W = torch.linalg.cholesky(cov_matrix_inv)
    return W

def madry_compute_corr(activations, indices, labels, percentile): # (N, D), (N,)
    # convert to numpy and center
    activations_np = activations.cpu().numpy()
    indices_np = indices.cpu().numpy()
    labels_np = labels.cpu().numpy()

    removed_inds = []

    for label in np.unique(labels_np):
        label_mask = (labels_np == label) # just the class we want
        label_activations = activations_np[label_mask]
        label_indices = indices_np[label_mask]

        # center
        mean = np.mean(label_activations, axis=0, keepdims=True)
        centered = label_activations - mean

        # get scores (svd)
        u,s,v = np.linalg.svd(centered, full_matrices=False)
        eigs = v[0:1]
        projections = np.matmul(eigs, centered.T)
        scores = np.linalg.norm(projections, axis=0)

        # threshold by percentile
        p_score = np.percentile(scores, percentile)
        top_scores = np.where(scores > p_score)[0]
        removed_inds.extend(label_indices[top_scores])

    return removed_inds

def walign(activations, indices, labels, percentile):
    # convert to numpy and center
    activations_np = activations.cpu().numpy()
    indices_np = indices.cpu().numpy()
    labels_np = labels.cpu().numpy()

    removed_inds = []

    for label in np.unique(labels_np):
        label_mask = (labels_np == label) # just the class we want
        label_activations = activations_np[label_mask]
        label_indices = indices_np[label_mask]

        # normalize feature-wise (per feature / column)
        mean_feat = np.mean(label_activations, axis=0, keepdims=True)
        std_feat = np.std(label_activations, axis=0, keepdims=True) + 1e-10
        normalized = (label_activations - mean_feat) / std_feat

        # get scores (inner product with class mean)
        mean = np.mean(normalized, axis=0, keepdims=True)
        scores = np.dot(normalized, mean.T).flatten()

        # threshold by percentile
        p_score = np.percentile(scores, percentile)
        top_scores = np.where(scores > p_score)[0]
        removed_inds.extend(label_indices[top_scores])

    return removed_inds

def whitened_norm(activations, indices, labels, percentile):
    # convert to numpy and center
    activations_np = activations.cpu().numpy()
    indices_np = indices.cpu().numpy()
    labels_np = labels.cpu().numpy()

    removed_inds = []

    for label in np.unique(labels_np):
        label_mask = (labels_np == label) # just the class we want
        label_activations = activations_np[label_mask]
        label_indices = indices_np[label_mask]

        mean = np.mean(label_activations, axis=0, keepdims=True)
        centered = label_activations - mean

        # whiten
        G_centered = torch.tensor(centered, dtype=torch.float32)
        W = whiten(G_centered)
        whitened = (G_centered @ W).numpy()

        # get scores (l2 norm of whitened embeddings)
        scores = np.linalg.norm(whitened, axis=1)

        # threshold by percentile
        p_score = np.percentile(scores, percentile)
        top_scores = np.where(scores > p_score)[0]
        removed_inds.extend(label_indices[top_scores])

    return removed_inds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='(mnist, cifar10, cifar100)')
    parser.add_argument('--scoring_fn', type=str, default='madry_compute_corr')
    parser.add_argument('--poison', type=str, default='pixel', help='(pixel, pattern, ell)')
    parser.add_argument('--n_poisons', type=int, default=500)
    parser.add_argument('--percentile', type=int, default=85)
    parser.add_argument('--batch_size', type=int, default=128) # stole from bb
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') # stole from bb
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for') # stole from bb
    parser.add_argument('--out', type=str, default='model', help='save trained model here')
    parser.add_argument('--pretrained_fp', type=str, default='', help='if you already trained init model, load it here')

    args = parser.parse_args()

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
        pass
    elif args.dataset == 'cifar100':
        pass
    
    # Train Model
    poisoned_train = PoisonedDataset(train_ds, args.n_poisons, test_params)
    train_loader = DataLoader(poisoned_train, batch_size=args.batch_size, shuffle=True)
    
    model = CNN(args.dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.pretrained_fp == '':
        train(args.n_epochs, model, train_loader, optimizer, nn.CrossEntropyLoss(), device)
        torch.save(model.state_dict(), f'{args.out}.pth')
        # save poisoned indices
        poisoned_tensor = torch.tensor(sorted(list(poisoned_train.poison_indices)), dtype=torch.long)
        torch.save(poisoned_tensor, f'{args.out}_poisoned_indices.pth')
    else:
        # upload poisoned indices
        model.load_state_dict(torch.load(args.pretrained_fp))

        indices_fp = args.pretrained_fp.replace('.pth', '_poisoned_indices.pth')
        poisoned_tensor = torch.load(indices_fp)
        poisoned_indices = poisoned_tensor.tolist()

        poisoned_train = PoisonedDataset(train_ds, n_poisons=len(poisoned_indices), poison=test_params, fixed_poison_indices=poisoned_indices)
        train_loader = DataLoader(poisoned_train, batch_size=args.batch_size, shuffle=True)

    # Eval Model
    print("initial eval")
    eval(model, test_ds, args.batch_size, device, test_params)
    exit()

    # Analyze Embeddings
    activations = []
    all_indices = []
    all_labels = []
    for _, (data, target, indices) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model(data)
        all_indices.append(indices)
        activations.append(model.embeddings)
        all_labels.append(target)

    all_indices = torch.cat(all_indices, dim=0)
    activations = torch.cat(activations, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    drop_indices = None
    if args.scoring_fn == 'madry_compute_corr':
        drop_indices = madry_compute_corr(activations, all_indices, all_labels, args.percentile)
    elif args.scoring_fn == 'walign':
        drop_indices = walign(activations, all_indices, all_labels, args.percentile)
    elif args.scoring_fn == 'whitened_norm':
        drop_indices = whitened_norm(activations, all_indices, all_labels, args.percentile)
    
    dropped_set = set(drop_indices)
    poisoned_set = set(poisoned_train.poison_indices)
    removed_poisons = dropped_set & poisoned_set
    num_removed = len(removed_poisons)
    total_poisons = len(poisoned_set)
    print(f"removed {num_removed} / {total_poisons} poisoned examples")

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