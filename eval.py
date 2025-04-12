"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
import sys
import time

import numpy as np
# import tensorflow as tf
from tqdm import trange

import dataset_input
import resnet
import utilities

import torch
from torch.utils.tensorboard import SummaryWriter

# A function for evaluating a single checkpoint
def evaluate(model, sess, config, summary_writer=None):
    model.eval() # ADDED
    eval_batch_size = config.eval.batch_size

    model_dir = config.model.output_dir
    # Setting up the Tensorboard and checkpoint outputs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    poison_method = config.data.poison_method
    clean_label = config.data.clean_label
    target_label = config.data.target_label
    position = config.data.position
    color = config.data.color
    dataset = dataset_input.CIFAR10Data(config,
                                        seed=config.training.np_random_seed)
    print(poison_method, clean_label, target_label, position, color)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # global_step = tf.contrib.framework.get_or_create_global_step()
    # Iterate over the samples batch-by-batch

    eval_images, eval_labels = dataset.eval_data.tensors
    pois_images, pois_labels = dataset.poisoned_eval_data.tensors

    num_eval_examples = len(eval_images)
    num_clean_examples = 0
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_corr_nat = 0
    total_xent_pois = 0.
    total_corr_pois = 0

    with torch.no_grad():

      for ibatch in trange(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        # x_batch = dataset.eval_data.xs[bstart:bend, :]
        # y_batch = dataset.eval_data.ys[bstart:bend]
        # pois_x_batch = dataset.poisoned_eval_data.xs[bstart:bend, :]
        # pois_y_batch = dataset.poisoned_eval_data.ys[bstart:bend]

        x_batch = eval_images[bstart:bend].float().to(device)
        y_batch = eval_labels[bstart:bend].to(device)
        pois_x_batch = pois_images[bstart:bend].float().to(device)
        pois_y_batch = pois_labels[bstart:bend].to(device)

        # dict_nat = {model.x_input: x_batch,
        #             model.y_input: y_batch,
        #             model.is_training: False}

        ### eval on clean batch
        # cur_corr_nat, cur_xent_nat = sess.run(
        #                                 [model.num_correct,model.xent],
        #                                 feed_dict = dict_nat)
        # total_xent_nat += cur_xent_nat
        # total_corr_nat += cur_corr_nat

        logits_nat = model(x_batch)
        loss_nat, acc_nat = model.get_loss_and_accuracy(logits_nat, y_batch)
        total_xent_nat += loss_nat.item() * (bend - bstart)
        total_corr_nat += int(acc_nat * (bend - bstart))

        '''
        if clean_label > -1:
          clean_indices = np.where(y_batch==clean_label)[0]
          if len(clean_indices)==0: continue
          pois_x_batch = pois_x_batch[clean_indices]
          pois_y_batch = np.repeat(target_label, len(clean_indices))
        else: 
          pois_y_batch = np.repeat(target_label, bend-bstart)
        '''

        if clean_label > -1:
            clean_mask = (y_batch == clean_label)
            if clean_mask.sum() == 0: continue
            pois_x_batch = pois_x_batch[clean_mask]
            pois_y_batch = torch.full_like(y_batch[clean_mask], target_label)
        else:
            pois_y_batch = torch.full((bend - bstart,), target_label, dtype=torch.long).to(device)

        num_clean_examples += len(pois_x_batch)

        # dict_pois = {model.x_input: pois_x_batch,
        #             model.y_input: pois_y_batch,
        #             model.is_training: False}

        ### eval on poison batch
        # cur_corr_pois, cur_xent_pois = sess.run(
        #                                 [model.num_correct,model.xent],
        #                                 feed_dict = dict_pois)
        # total_xent_pois += cur_xent_pois
        # total_corr_pois += cur_corr_pois

        logits_pois = model(pois_x_batch)
        loss_pois, acc_pois = model.get_loss_and_accuracy(logits_pois, pois_y_batch)
        total_xent_pois += loss_pois.item() * len(pois_x_batch)
        total_corr_pois += int(acc_pois * len(pois_x_batch))

    # compute avgs (stay the same)
    avg_xent_nat = total_xent_nat / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    avg_xent_pois = total_xent_pois / num_clean_examples
    acc_pois = total_corr_pois / num_clean_examples

    if summary_writer: # WHY does this write in avg_xent_nat twice and acc_nat twice? I'm assuming they meant the 4 above ^^
        # summary = tf.Summary(value=[
        #       tf.Summary.Value(tag='xent_nat_eval', simple_value= avg_xent_nat),
        #       tf.Summary.Value(tag='xent_nat', simple_value= avg_xent_nat),
        #       tf.Summary.Value(tag='accuracy_nat_eval', simple_value= acc_nat),
        #       tf.Summary.Value(tag='accuracy_nat', simple_value= acc_nat)])
        # summary_writer.add_summary(summary, global_step.eval(sess))
        summary_writer.add_scalar('xent_nat_eval', avg_xent_nat, global_step=0) # not tracking global step so ig set it = 0?
        summary_writer.add_scalar('accuracy_nat_eval', acc_nat, global_step=0)
        summary_writer.add_scalar('xent_pois_eval', avg_xent_pois, global_step=0)
        summary_writer.add_scalar('accuracy_pois_eval', acc_pois, global_step=0)

    # step = global_step.eval(sess)
    # print('Eval at step: {}'.format(step))
    print('  natural: {:.2f}%'.format(100 * acc_nat))
    print('  avg nat xent: {:.4f}'.format(avg_xent_nat))
    print('  poisoned: {:.2f}%'.format(100 * acc_pois))
    print('  avg pois xent: {:.4f}'.format(avg_xent_pois))

    result = {'nat': '{:.2f}%'.format(100 * acc_nat),
              'pois': '{:.2f}%'.format(100 * acc_pois)}
    with open('job_result.json', 'w') as result_file:
        json.dump(result, result_file, sort_keys=True, indent=4)

def loop(model, config, summary_writer=None):

    last_checkpoint_filename = ''
    already_seen_state = False
    model_dir = config.model.output_dir
    # saver = tf.train.Saver()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    while True:
      # cur_checkpoint = tf.train.latest_checkpoint(model_dir)
      checkpoints = [f for f in os.listdir(model_dir) if f.startswith('checkpoint-')]

      # Case 1: No checkpoint yet
      if not checkpoints:
        if not already_seen_state:
          print('No checkpoint yet, waiting ...', end='')
          already_seen_state = True
        else:
          print('.', end='')
        sys.stdout.flush()
        time.sleep(10)
        continue

      # ADDED sort checkpoints, get latest bc thats what this uses
      checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
      cur_checkpoint = os.path.join(model_dir, checkpoints[-1])

      # Case 2: Previously unseen checkpoint
      if cur_checkpoint != last_checkpoint_filename:
        print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
                                                              datetime.now()))
        sys.stdout.flush()
        last_checkpoint_filename = cur_checkpoint
        already_seen_state = False
        # with tf.Session() as sess:
        #     # Restore the checkpoint
        #     saver.restore(sess, cur_checkpoint)
        #     evaluate(model, sess, config, summary_writer)
        model.load_state_dict(torch.load(cur_checkpoint, map_location=device))
        evaluate(model, None, config, summary_writer)
      # Case 3: Previously evaluated checkpoint
      else:
        if not already_seen_state:
          print('Waiting for the next checkpoint ...   ({})   '.format(
                datetime.now()),
                end='')
          already_seen_state = True
        else:
          print('.', end='')
        sys.stdout.flush()
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eval script options',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file',
                        default="config.json", required=False)
    parser.add_argument('--loop', help='continuously monitor model_dir evaluating new ckpt', 
                        action="store_true")
    args = parser.parse_args()

    config_dict = utilities.get_config(args.config)
    config = utilities.config_to_namedtuple(config_dict)

    model = resnet.Model(config.model)

    model_dir = config.model.output_dir

    # global_step = tf.contrib.framework.get_or_create_global_step()

    if args.loop:
        eval_dir = os.path.join(model_dir, 'eval')
        if not os.path.exists(eval_dir):
          os.makedirs(eval_dir)
        # summary_writer = tf.summary.FileWriter(eval_dir)
        summary_writer = SummaryWriter(log_dir=eval_dir)

        loop(model, config, summary_writer)
    else:
        # saver = tf.train.Saver()

        # cur_checkpoint = tf.train.latest_checkpoint(model_dir)
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith('checkpoint-')]
        if not checkpoints:
            print('No checkpoint found.')
        else:
          checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
          cur_checkpoint = os.path.join(model_dir, checkpoints[-1])

            # with tf.Session() as sess:
                # Restore the checkpoint
          print('Evaluating checkpoint {}'.format(cur_checkpoint))

          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          model = model.to(device)

          # saver.restore(sess, cur_checkpoint)
          # evaluate(model, sess, config)

          model.load_state_dict(torch.load(cur_checkpoint, map_location=device))
          evaluate(model, None, config)