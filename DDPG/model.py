import numpy as np
import paddle
import parl
import paddle.nn as nn
import paddle.nn.functional as F

BATCH_SIZE = 128

# ---------------------------------------------------------#
#   Model
# ---------------------------------------------------------#

class Model(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor_model = ActorModel(obs_dim, act_dim)
        self.critic_model = CriticModel(obs_dim, act_dim)

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
    def __init__(self, obs_dim, act_dim):
        # 隐层大小
        super().__init__()
        hid_size = obs_dim * 10

        # 两个全连接层
        # 输入为obs,输出为隐层
        self.fc1 = nn.Linear(in_features=obs_dim, out_features=hid_size)
        # 输入为隐层,输出为act_dim
        self.fc2 = nn.Linear(in_features=hid_size, out_features=act_dim)

    # 策略向前
    def policy(self, obs):
        hid = F.relu(self.fc1(obs))
        action = F.tanh(self.fc2(hid))

        # print("action : ", action[0])

        return action


# ---------------------------------------------------------#
#   CriticModel
# ---------------------------------------------------------#

class CriticModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hid_size = 100

        print("obs_dim : %r" % obs_dim)
        print("act_dim : %r" % act_dim)

        self.fc1 = nn.Linear(in_features=obs_dim + act_dim, out_features=hid_size)
        self.fc2 = nn.Linear(in_features=hid_size, out_features=1)

    # 值预测
    def value(self, obs, act):
        # 列连接,[1]+[2] = [[1,2]],同时升维
        # print("obs : %r" % obs)
        # print("act : %r" % act)
        concat = paddle.concat([obs, act], axis=1)
        # print("concat : %r" % concat)
        hid = F.relu(self.fc1(concat))
        out = self.fc2(hid)
        # print("out : %r" % out)
        # 降维,[[1,2]] = [1,2]
        out = paddle.squeeze(out, axis=[1])

        print("Q : ", out[0])

        return out
