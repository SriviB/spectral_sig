# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  """ResNet model."""

  def __init__(self, config):
    """ResNet constructor.
    """
    super(Model, self).__init__() # Im assuming even tho tf didn't have it (why?)
    self._per_im_std = config.per_im_std
    self._build_model(config.filters)

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    # return [1, stride, stride, 1]
    return stride # pytorch conv reqs only spatial not all 4

  def _build_model(self, filters):
    """Build the core model within the graph."""
    # with tf.variable_scope('input'): # not used in pytorch, up to "x = self._conv('init_conv', x, 3, 3, filters[0], self._stride_arr(1))" was indented

    # placeholders not needed with pytorch I think

    self.init_conv = nn.Conv2d(in_channels=3, out_channels=filters[0], kernel_size=3, stride=1, padding=1, bias=False) # matched up params
    # transforms dont happen here right?

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]

    # Define block1 (unit_1_0 + unit_1_1 → unit_1_4)
    self.block1 = nn.Sequential(
        self._make_residual(filters[0], filters[1], stride=strides[0], activate_first=activate_before_residual[0]),
        self._make_residual(filters[1], filters[1], stride=1),
        self._make_residual(filters[1], filters[1], stride=1),
        self._make_residual(filters[1], filters[1], stride=1),
        self._make_residual(filters[1], filters[1], stride=1),
    )

    self.block2 = nn.Sequential(
        self._make_residual(filters[1], filters[2], stride=strides[1], activate_first=activate_before_residual[1]),
        self._make_residual(filters[2], filters[2], stride=1),
        self._make_residual(filters[2], filters[2], stride=1),
        self._make_residual(filters[2], filters[2], stride=1),
        self._make_residual(filters[2], filters[2], stride=1),
    )

    self.block3 = nn.Sequential(
        self._make_residual(filters[2], filters[3], stride=strides[2], activate_first=activate_before_residual[2]),
        self._make_residual(filters[3], filters[3], stride=1),
        self._make_residual(filters[3], filters[3], stride=1),
        self._make_residual(filters[3], filters[3], stride=1),
        self._make_residual(filters[3], filters[3], stride=1),
    )

    self.bn_final = nn.BatchNorm2d(filters[3])
    self.relu_final = nn.LeakyReLU(0.1, inplace=True)
    self.global_pool = nn.AdaptiveAvgPool2d(1)  # Output is (B, C, 1, 1)

    self.fc = nn.Linear(filters[3], 10)

  # idk
  def get_loss_and_accuracy(self, logits, labels):
    loss = F.cross_entropy(logits, labels)
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0)
    return loss, accuracy

  # idk
  def _make_residual(self, in_channels, out_channels, stride, activate_first=False):
    return BasicBlock(in_channels, out_channels, stride, activate_before_residual=activate_first)
  
  def forward(self, x):
    x = self.init_conv(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)

    self.representation = x.view(x.size(0), -1)

    x = self.relu_final(self.bn_final(x))
    x = self.global_pool(x)
    x = torch.flatten(x, 1)
    self.pre_softmax = self.fc(x)
    return self.pre_softmax


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.activate_before_residual = activate_before_residual
        self.equal_in_out = in_channels == out_channels and stride == 1

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if not self.equal_in_out:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0),
                nn.ConstantPad3d((0, 0, 0, 0, (out_channels - in_channels) // 2, (out_channels - in_channels) // 2), 0)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        if self.activate_before_residual:
            out = self.relu1(self.bn1(x))
            orig = out
        else:
            orig = x
            out = self.relu1(self.bn1(x))

        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + self.shortcut(orig)