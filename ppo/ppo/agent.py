import parl
import paddle
import numpy as np
import torch
from paddle.distribution import Categorical
from parl.utils.scheduler import LinearDecayScheduler
from torch import device
from model import Model
from algorithm import PPO
import paddle.nn.functional as F


class PPOAgent(parl.Agent):
    def __init__(self, algorithm, obs, model):
        super(PPOAgent, self).__init__(algorithm)
        self.alg = algorithm
        self.model = model

    # def predict(self, obs):
    #     obs = paddle.to_tensor(obs, dtype='float32').unsqueeze(0)
    #     action = self.alg.predict(obs)
    #     action_numpy = action.detach().numpy()[0]
    #     return action_numpy

    def sample(self, obs, rpm):
        # print("obs", obs)
        action_probs = self.model.policy(obs)
        # print("action_probs", action_probs)
        action_probs = F.softmax(action_probs)
        action_probs = np.array(action_probs)
        action_probs = paddle.to_tensor(action_probs)
        # action_probs = torch.from_numpy(action_probs).float()
        # print("action_probs", action_probs)
        dist = Categorical(action_probs)  # 按照给定的概率分布来进行采样
        # print(dist)
        action = dist.sample([1])

        rpm.states.append(obs)
        rpm.actions.append(action)
        rpm.logprobs.append(dist.log_prob(action))

        # print("action.item()", action.item())

        return action

    def predict(self, obs):
        action_probs = self.model.policy(obs)
        # print("action_probs", action_probs)
        max_action = paddle.argmax(action_probs)
        # print("max_action", max_action)
        return max_action

        # def value(self, obs, action):

    #     action_probs = self.model.policy(obs)
    #     action_probs = np.array(action_probs)
    #     action_probs = torch.from_numpy(action_probs).float()
    #     dist = Categorical(action_probs)
    #     action_logprobs = dist.log_prob(action)
    #     dist_entropy = dist.entropy()
    #     obs = obs.unsqueeze(0)
    #     state_value = self.model.value(obs)
    #
    #     return action_logprobs, torch.squeeze(state_value),dist_entropy

    def learn(self, rpm):
        self.alg.learn(rpm)

