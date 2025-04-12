"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
import shutil
import sys
from timeit import default_timer as timer

import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
from tqdm import trange

import dataset_input
from eval import evaluate 
import resnet
import utilities

import torch
from tqdm import trange
import torch.nn.functional as F

def compute_corr(config):
    # seeding randomness
    # tf.set_random_seed(config.training.tf_random_seed)
    torch.manual_seed(config.training.torch_random_seed)
    np.random.seed(config.training.np_random_seed)

    # Setting up the data and the model
    poison_eps = config.data.poison_eps
    clean_label = config.data.clean_label
    target_label = config.data.target_label
    dataset = dataset_input.CIFAR10Data(config,
                                        seed=config.training.np_random_seed)
    num_poisoned_left = dataset.num_poisoned_left
    print('Num poisoned left: ', num_poisoned_left)
    num_training_examples = len(dataset.train_data.xs)

    # global_step = tf.contrib.framework.get_or_create_global_step() # don't need in torch?
    model = resnet.Model(config.model)
    
    # ADDED below to move model to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Setting up the Tensorboard and checkpoint outputs
    model_dir = config.model.output_dir # "output_dir": "output/"

    # saver = tf.train.Saver(max_to_keep=3) # save model weights; pytorch uses torch.save() and torch.load() and stuff in train.py I think MAKE SURE

    # with tf.Session() as sess: # create new sess to run stuff; everything below remaining WAS indented, but pytorch doesnt do sessions

    # initialize data augmentation
    print('Dataset Size: ', len(dataset.train_data.xs))

    '''
    sess.run(tf.global_variables_initializer())
        
    latest_checkpoint = tf.train.latest_checkpoint(model_dir) # load latest saved model checkpoint 
    if latest_checkpoint is not None: # found!
        saver.restore(sess, latest_checkpoint) # restore model weights into memory
        print('Restoring last saved checkpoint: ', latest_checkpoint)
    else: # error out rest in pieces
        print('Check model directory')
        exit()
    '''

    latest_checkpoint = os.path.join(model_dir, 'model.pth') # ADDED remember that weights are saved under model.pth lol
    if os.path.exists(latest_checkpoint):
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
        print('Restoring last saved checkpoint:', latest_checkpoint)
    else:
        print('Check model directory')
        exit()
    
    model.eval() # ADDED

    lbl = target_label
    train_images, train_labels = dataset.train_data.tensors  # ADDED
    # cur_indices = np.where(dataset.train_data.ys==lbl)[0] # get all the indices of the imgs in target label class
    cur_indices = torch.where(train_labels == lbl)[0]
    cur_examples = len(cur_indices)
    print('Label, num ex: ', lbl, cur_examples)
    cur_op = model.representation
    for iex in trange(cur_examples): # loop thru these indices
        cur_im = cur_indices[iex] # get img
        # x_batch = dataset.train_data.xs[cur_im:cur_im+1,:]
        # y_batch = dataset.train_data.ys[cur_im:cur_im+1]
        x_batch = train_images[cur_im].unsqueeze(0).float().to(device)
        y_batch = train_labels[cur_im].unsqueeze(0).to(device)

        with torch.no_grad():
            _ = model(x_batch)  # forward pass, fills model.representation
            batch_grads = model.representation.detach().cpu().numpy()

        '''
        dict_nat = {model.x_input: x_batch,
                    model.y_input: y_batch,
                    model.is_training: False}
        '''

        # full_cov has vectors for ALL imgs of poison label
        # clean_cov has vecs for only non-poisoned imgs of poison label

        # batch_grads = sess.run(cur_op, feed_dict=dict_nat)
        # this part stays the same
        if iex==0: # first iter, create array
            clean_cov = np.zeros(shape=(cur_examples-num_poisoned_left, len(batch_grads)))
            full_cov = np.zeros(shape=(cur_examples, len(batch_grads)))
        if iex < (cur_examples-num_poisoned_left):
            clean_cov[iex]=batch_grads
        full_cov[iex] = batch_grads
    
    # all numpy stuff below, no torchy transforms needed

    #np.save(corr_dir+str(lbl)+'_full_cov.npy', full_cov)
    total_p = config.data.percentile
    clean_mean = np.mean(clean_cov, axis=0, keepdims=True)
    full_mean = np.mean(full_cov, axis=0, keepdims=True)

    print('Norm of Difference in Mean: ', np.linalg.norm(clean_mean-full_mean))
    clean_centered_cov = clean_cov - clean_mean
    s_clean = np.linalg.svd(clean_centered_cov, full_matrices=False, compute_uv=False)
    print('Top 7 Clean SVs: ', s_clean[0:7])
    
    centered_cov = full_cov - full_mean
    u,s,v = np.linalg.svd(centered_cov, full_matrices=False)
    print('Top 7 Singular Values: ', s[0:7])
    eigs = v[0:1]  
    p = total_p
    corrs = np.matmul(eigs, np.transpose(full_cov)) #shape num_top, num_active_indices
    scores = np.linalg.norm(corrs, axis=0) #shape num_active_indices
    np.save(os.path.join(model_dir, 'scores.npy'), scores)
    print('Length Scores: ', len(scores))
    p_score = np.percentile(scores, p)
    top_scores = np.where(scores>p_score)[0]
    print(top_scores)
    num_bad_removed = np.count_nonzero(top_scores>=(len(scores)-num_poisoned_left))
    print('Num Bad Removed: ', num_bad_removed)
    print('Num Good Rmoved: ', len(top_scores)-num_bad_removed)
    
    num_poisoned_after = num_poisoned_left - num_bad_removed
    removed_inds = np.copy(top_scores)
    
    removed_inds_file = os.path.join(model_dir, 'removed_inds.npy')
    np.save(removed_inds_file, cur_indices[removed_inds])        
    print('Num Poisoned Left: ', num_poisoned_after)    

    if os.path.exists('job_result.json'):
        with open('job_result.json') as result_file:
            result = json.load(result_file)
            result['num_poisoned_left'] = '{}'.format(num_poisoned_after)
    else:
        result = {'num_poisoned_left': '{}'.format(num_poisoned_after)}
    with open('job_result.json', 'w') as result_file:
        json.dump(result, result_file, sort_keys=True, indent=4) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train script options',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file',
                        default='config.json', required=False)
    args = parser.parse_args()

    config_dict = utilities.get_config(args.config)

    model_dir = config_dict['model']['output_dir']
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    # keep the configuration file with the model for reproducibility
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, sort_keys=True, indent=4)

    config = utilities.config_to_namedtuple(config_dict)
    compute_corr(config)