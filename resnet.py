import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()
        filters = config.filters

        self.augment = config.per_im_std
        self.init_conv = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = self._make_layer(filters[0], filters[1], stride=1, num_units=5, activate_first=True)
        self.block2 = self._make_layer(filters[1], filters[2], stride=2, num_units=5)
        self.block3 = self._make_layer(filters[2], filters[3], stride=2, num_units=5)

        self.bn_final = nn.BatchNorm2d(filters[3])
        self.relu_final = nn.LeakyReLU(0.1, inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(filters[3], 10)

    def _make_layer(self, in_channels, out_channels, stride, num_units, activate_first=False):
        layers = [BasicBlock(in_channels, out_channels, stride, activate_before_residual=activate_first)]
        for _ in range(1, num_units):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        self.representation = x.view(x.size(0), -1).mean(dim=0)
        x = self.relu_final(self.bn_final(x))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        self.pre_softmax = self.fc(x)
        return self.pre_softmax

    def get_loss_and_accuracy(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        accuracy = correct / labels.size(0)
        return loss, accuracy
