import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from functools import reduce

# Applies a specified poison pattern to an image or batch of images
def poison(x, method, pos, col):
    ret_x = x.clone()  # Clone to avoid modifying original data
    col_arr = torch.tensor(col, dtype=ret_x.dtype)  # Convert color to tensor

    print("ret_x slice shape:", ret_x[:, :, pos[0], pos[1]].shape)
    print("col_arr shape:", col_arr.shape)

    # Batch of images: apply poison across batch
    if method == 'pixel':
        # ret_x[:, pos[0], pos[1] :] = col_arr
        ret_x[:, :, pos[0], pos[1]] = col_arr.view(1, 3).expand(ret_x.size(0), 3) 
        # bc new dimensions NHWC -> NCHW look in Cifar10Data class
        # batch size 1, 3 channels, 1 no spatial dim (pixel level)
        # .expand(ret_x.size(0), 3) makes it [B, 3] bc repeating across the batch
    elif method == 'pattern':
        for dx, dy in [(0, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            # ret_x[:, pos[0]+dx, pos[1]+dy, :] = col_arr
            ret_x[:, :, pos[0]+dx, pos[1]+dy] = col_arr.view(1, 3) 
    elif method == 'ell':
        for dx, dy in [(0, 0), (1, 0), (0, 1)]:
            # ret_x[:, pos[0]+dx, pos[1]+dy, :] = col_arr
            ret_x[:, :, pos[0]+dx, pos[1]+dy] = col_arr.view(1, 3)
    return ret_x

# Wrapper for loading and optionally poisoning CIFAR-10 data
class CIFAR10Data:
    def __init__(self, config, seed=None):
        self.rng = torch.Generator()
        self.rng.manual_seed(1 if seed is None else seed)

        # Define torchvision transform to convert PIL to torch tensor
        transform = None # transforms.ToTensor()

        # Download and load CIFAR-10 training and test datasets
        train_set = torchvision.datasets.CIFAR10(root=config.data.cifar10_path, train=True,
                                                 download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=config.data.cifar10_path, train=False,
                                                download=True, transform=transform)
        
        train_set.data = torch.tensor(train_set.data)
        train_set.targets = torch.tensor(train_set.targets)
        test_set.data = torch.tensor(test_set.data)
        test_set.targets = torch.tensor(test_set.targets)

        train_set.data = train_set.data.permute(0, 3, 1, 2)  # NHWC -> NCHW w/o this, error: RuntimeError: Given groups=1, weight of size [16, 3, 3, 3], expected input[16, 32, 32, 3] to have 3 channels, but got 32 channels instead
        test_set.data = test_set.data.permute(0, 3, 1, 2)

        # Load config options
        method = config.data.poison_method
        eps = config.data.poison_eps
        clean = config.data.clean_label
        target = config.data.target_label
        position = config.data.position
        color = config.data.color
        num_training_examples = config.training.num_examples

        train_images = train_set.data # torch.stack([img for img, _ in train_set])
        train_labels = train_set.targets # torch.tensor([label for _, label in train_set])
        initial_num_train = len(train_labels)
        test_images = test_set.data # torch.stack([img for img, _ in test_set])
        test_labels = test_set.targets # torch.tensor([label for _, label in test_set])

        # Poison selected images in the training set
        if eps > 0:
            if clean > -1:
                # get indices where label == clean label
                clean_indices = torch.where(train_labels == clean)[0]
            else:
                # get indices where label is not poison label
                clean_indices = torch.where(train_labels != target)[0]

            # choose epsilon indices from clean indices without replacement
            poison_indices = clean_indices[torch.randperm(len(clean_indices), generator=self.rng)[:eps]]
            poison_images = poison(train_images[poison_indices], method, position, color)
            # poison_labels = torch.full((eps,), target if target > -1 else torch.randint(10, (1,), generator=self.rng))

            poison_labels = torch.full(
                (eps,),
                target if target > -1 else int(torch.randint(10, (1,), generator=self.rng)),
                dtype=train_labels.dtype
            )

            # Remove original clean images and append poisoned ones
            mask = torch.ones(len(train_images), dtype=bool)
            mask[poison_indices] = False

            train_images = torch.cat((train_images[mask], poison_images), dim=0)
            train_labels = torch.cat((train_labels[mask], poison_labels), dim=0)

            # self.num_poisoned_left = len(train_labels) - initial_num_train
            self.num_poisoned_left = len(poison_indices)


        # Optionally apply per-image normalization
        if config.model.per_im_std:
            train_images = self._per_im_std(train_images)
            test_images = self._per_im_std(test_images)
            # poisoned_eval_images = self._per_im_std(poison(test_images, method, position, color))
            if clean > -1:
                clean_mask = (test_labels == clean)
                poisoned_eval_images = test_images.clone()
                poisoned_eval_images[clean_mask] = self._per_im_std(poison(test_images[clean_mask], method, position, color))
            else:
                poisoned_eval_images = self._per_im_std(poison(test_images, method, position, color))
        else:
            # poisoned_eval_images = poison(test_images, method, position, color)
            if clean > -1:
                clean_mask = (test_labels == clean)
                poisoned_eval_images = test_images.clone()
                poisoned_eval_images[clean_mask] = poison(test_images[clean_mask], method, position, color)
            else:
                poisoned_eval_images = poison(test_images, method, position, color)

        # print(f"train_images shape: {train_images.shape}")
        # print(f"train_labels shape: {train_labels.shape}")
        # exit()

        # Store datasets as DataSubset wrappers
        self.train_data = TensorDataset(train_images, train_labels)
        self.eval_data = TensorDataset(test_images, test_labels)
        self.poisoned_eval_data = TensorDataset(poisoned_eval_images, test_labels)

    # Normalize each image to have zero mean and unit variance
    @staticmethod
    def _per_im_std(images):
        num_pixels = reduce(lambda x, y: x * y, images[0].shape)
        images = images - images.mean(dim=(1, 2, 3), keepdim=True)
        stds = images.view(images.size(0), -1).std(dim=1).clamp(min=1.0 / torch.sqrt(num_pixels))
        images = images / stds.view(-1, 1, 1, 1)
        return images