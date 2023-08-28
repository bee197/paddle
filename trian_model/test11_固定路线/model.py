import numpy as np
import paddle
import parl
import paddle.nn as nn
import paddle.nn.functional as F
import paddle
import torch
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock, BasicBlock


# ---------------------------------------------------------#
#   Model
# ---------------------------------------------------------#

class Model(parl.Model):
    def __init__(self):
        super().__init__()

        # 这个网络是原版Atari的网络架构
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=2)

        self.pool = nn.AdaptiveAvgPool2D(output_size=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(68, 3)

    def value(self, obs):
        obs, distance, angle = obs[0]

        x = self.conv1(obs)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = self.pool(x)

        obs = self.flatten(x)

        concat = paddle.concat([obs, distance, angle], axis=1)

        Q = self.fc1(concat)

        return Q

    def get_params(self):
        return self.parameters()
