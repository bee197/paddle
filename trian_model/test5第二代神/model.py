import numpy as np
import paddle
import parl
import paddle.nn as nn
import paddle.nn.functional as F


# ---------------------------------------------------------#
#   Model
# ---------------------------------------------------------#

class Model(parl.Model):
    def __init__(self):
        super().__init__()

        # 这个网络是原版Atari的网络架构
        self.conv1 = nn.Conv2D(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 3)

    def value(self, obs):
        x = self.conv1(obs)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = F.leaky_relu(x)
        Q = self.fc2(x)

        # print("Q : ", Q)

        return Q

    def get_params(self):
        return self.parameters()

