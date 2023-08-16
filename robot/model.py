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
        self.actor_model = ActorModel()
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


# ---------------------------------------------------------#
#   ActorModel
# ---------------------------------------------------------#

class ActorModel(parl.Model):
    def __init__(self):
        super().__init__()

        # 这个网络是原版Atari的网络架构
        self.conv1 = nn.Conv2D(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 1)

    # 策略向前
    def policy(self, obs):
        x = self.conv1(obs)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = self.flatten(x)
        # print("x : ", x.shape)

        x = self.fc1(x)
        x = F.leaky_relu(x)
        action = self.fc2(x)
        # print("action : ", action[0])
        # action = 10 * F.tanh(action)
        # print("action : ", action[0])

        return action


# ---------------------------------------------------------#
#   CriticModel
# ---------------------------------------------------------#

class CriticModel(parl.Model):
    def __init__(self):
        super().__init__()

        # 这个网络是原版Atari的网络架构
        self.conv1 = nn.Conv2D(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1025, 1)

        # self.fc3 = nn.Linear(2, 256)
        # self.fc4 = nn.Linear(256, 128)

    # 值预测
    def value(self, obs, act):
        # 变为(1,2),升维
        # act = np.expand_dims(act, axis=0)
        # 变为(4,1,1)
        # act = act.reshape([128, 1, 1, 2])
        # 拼接状态和动作
        # x = paddle.concat([obs, act], axis=1)
        # x shape: [128, 4, 84, 84]
        # x = obs + act
        x = self.conv1(obs)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        # print("x : ", x.shape)

        x = self.flatten(x)
        # print("x : ", x.shape)

        x = self.fc1(x)
        x = F.leaky_relu(x)
        # print("x : ", x.shape)

        # action_embed = self.fc3(act)
        # action_embed = F.relu(action_embed)

        x = paddle.concat([x, act], axis=1)  # 将 state 特征和 action 特征进行连接

        Q = self.fc2(x)
        # Q = 100 * F.tanh(Q)

        # print("Q : ", Q[0])

        return Q
