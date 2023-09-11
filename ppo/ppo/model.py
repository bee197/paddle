import cv2
import numpy as np
import paddle
import parl
import paddle.nn as nn
import paddle.nn.functional as F
import paddle
import torch
from paddle.distribution import Categorical

update_timestep = 1000
# ---------------------------------------------------------#
#   Model
# ---------------------------------------------------------#

class Model(parl.Model):
    def __init__(self):
        super().__init__()

        # 这个网络是原版Atari的网络架构
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 3)
        self.fc4 = nn.Linear(512, 1)

    def value(self, obs, action):
        obs = obs.reshape((update_timestep, 3, 84, 84))
        # print("obs", obs)
        x = self.conv1(obs)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        Q = F.leaky_relu(x)

        action_probs = self.policy(obs)
        action_probs = F.softmax(action_probs)
        action_probs = np.array(action_probs)
        action_probs = paddle.to_tensor(action_probs)
        dist = Categorical(action_probs)  # 按照给定的概率分布来进行采样

        action_logprobs = dist.log_prob(action)
        # print("action_logprobs", action_logprobs)
        dist_entropy = dist.entropy()
        state_value = Q
        # print(state_value)

        return action_logprobs, paddle.squeeze(state_value), dist_entropy

    def policy(self, obs):
        # print("obs", obs)
        x = self.conv1(obs)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)

        logits = x
        # TODO:
        # print("logits", logits)

        return logits


    def get_params(self):
        return self.parameters()










