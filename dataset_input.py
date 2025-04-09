import os
import pickle
import numpy as np
from functools import reduce
import torch
from torch.utils.data import Dataset
import random

def poison(x, method, pos, col):
    ret_x = np.copy(x)
    col_arr = np.asarray(col)
    if ret_x.ndim == 3:
        if method == 'pixel':
            ret_x[pos[0], pos[1], :] = col_arr
        elif method == 'pattern':
            for dx, dy in [(0,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
                ret_x[pos[0]+dx, pos[1]+dy, :] = col_arr
        elif method == 'ell':
            ret_x[pos[0], pos[1], :] = col_arr
            ret_x[pos[0]+1, pos[1], :] = col_arr
            ret_x[pos[0], pos[1]+1, :] = col_arr
    else:
        if method == 'pixel':
            ret_x[:, pos[0], pos[1], :] = col_arr
        elif method == 'pattern':
            for dx, dy in [(0,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
                ret_x[:, pos[0]+dx, pos[1]+dy, :] = col_arr
        elif method == 'ell':
            ret_x[:, pos[0], pos[1], :] = col_arr
            ret_x[:, pos[0]+1, pos[1], :] = col_arr
            ret_x[:, pos[0], pos[1]+1, :] = col_arr
    return ret_x

class DataSubset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs.astype(np.float32) / 255.0
        self.ys = ys.astype(np.int64)

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        return self.xs[idx].transpose(2, 0, 1), self.ys[idx]  # Convert to CHW

class CIFAR10Data:
    def __init__(self, config, seed=None):
        train_filenames = [f"data_batch_{i+1}" for i in range(5)]
        eval_filename = "test_batch"
        metadata_filename = "batches.meta"
        rng = np.random.RandomState(seed or 1)

        model_dir = config.model.output_dir
        path = config.data.cifar10_path
        method = config.data.poison_method
        eps = config.data.poison_eps
        clean = config.data.clean_label
        target = config.data.target_label
        position = config.data.position
        color = config.data.color

        train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')
        for i, fname in enumerate(train_filenames):
            imgs, lbls = self._load_datafile(os.path.join(path, fname))
            train_images[i*10000:(i+1)*10000] = imgs
            train_labels[i*10000:(i+1)*10000] = lbls

        eval_images, eval_labels = self._load_datafile(os.path.join(path, eval_filename))

        if eps > 0:
            if clean > -1:
                clean_inds = np.where(train_labels == clean)[0]
            else:
                clean_inds = np.where(train_labels != target)[0]
            poison_inds = rng.choice(clean_inds, eps, replace=False)

            poison_imgs = np.array([poison(train_images[i], method, position, color) for i in poison_inds])
            poison_lbls = np.full(eps, target if target > -1 else rng.randint(0, 10))

            train_images = np.concatenate([train_images, poison_imgs], axis=0)
            train_labels = np.concatenate([train_labels, poison_lbls], axis=0)
            train_images = np.delete(train_images, poison_inds, axis=0)
            train_labels = np.delete(train_labels, poison_inds, axis=0)

        with open(os.path.join(path, metadata_filename), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            self.label_names = [name.decode('utf-8') for name in data_dict[b'label_names']]

        train_indices = np.arange(len(train_images))
        if os.path.exists(os.path.join(model_dir, 'removed_inds.npy')):
            removed = np.load(os.path.join(model_dir, 'removed_inds.npy'))
            train_indices = np.delete(train_indices, removed)

        self.num_poisoned_left = np.count_nonzero(train_indices >= (50000 - eps))
        np.save(os.path.join(model_dir, 'train_indices.npy'), train_indices)

        poisoned_eval_images = poison(eval_images, method, position, color)

        if config.model.per_im_std:
            train_images = self._per_im_std(train_images)
            eval_images = self._per_im_std(eval_images)
            poisoned_eval_images = self._per_im_std(poisoned_eval_images)

        self.train_data = DataSubset(train_images[train_indices], train_labels[train_indices])
        self.eval_data = DataSubset(eval_images, eval_labels)
        self.poisoned_eval_data = DataSubset(poisoned_eval_images, eval_labels)

    @staticmethod
    def _load_datafile(filename):
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            images = data_dict[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            labels = np.array(data_dict[b'labels'])
            return images, labels

    @staticmethod
    def _per_im_std(images):
        images = images.astype(np.float32)
        for i in range(images.shape[0]):
            mean = np.mean(images[i], keepdims=True)
            std = np.std(images[i])
            std_adj = max(std, 1.0 / np.sqrt(np.prod(images[i].shape)))
            images[i] = (images[i] - mean) / std_adj
        return images
