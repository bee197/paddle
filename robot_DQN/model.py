import cv2
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
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        # 加入额外的两个卷积层
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2D(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2D(in_channels=256, out_channels=64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(2, 256)
        self.fc3 = nn.Linear(1088, 3)

    def value(self, obs):
        obs, distance, angle = obs[0]

        x = self.conv1(obs)
        x = self.pool1(x)

        x = self.conv2(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.pool2(x)

        x = self.conv5(x)

        obs = self.flatten(x)

        distance = self.fc1(distance)
        distance = F.leaky_relu(distance)

        angle = self.fc2(angle)
        angle = F.leaky_relu(angle)

        concat = paddle.concat([obs, distance, angle], axis=1)

        Q = self.fc3(concat)

        return Q

    def get_params(self):
        return self.parameters()
