import numpy as np
import parl
import torch
import paddle
from torch.distributions import Categorical
from parl import Algorithm

device = paddle.CUDAPlace(0)
from model import Model
import paddle.distributed as dist

# print(parl.__version__)

class PPO(Algorithm):
    def __init__(self, model, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy = model.to(device)
        self.mse_loss = paddle.nn.MSELoss(reduction='mean')
        self.optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=self.policy.get_params())

    def learn(self, rpm):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rpm.rewards), reversed(rpm.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        # print(type(rewards))
        # place = paddle.CUDAPlace(0)

        rewards = paddle.to_tensor(rewards, dtype='float32')
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = paddle.to_tensor(rewards, dtype='float32')

        # convert list to tensor
        old_states = [paddle.to_tensor(s) for s in rpm.states]
        old_states = paddle.stack(old_states)

        old_actions = [s.numpy() for s in rpm.actions]
        old_actions = [paddle.to_tensor(s) for s in old_actions]
        old_actions = paddle.stack(old_actions)
        old_actions = paddle.reshape(old_actions, [1000, 1])
        # print("old_actions", old_actions)

        old_logprobs = [s.numpy() for s in rpm.logprobs]
        old_logprobs = [paddle.to_tensor(s) for s in old_logprobs]
        old_logprobs = paddle.stack(old_logprobs)
        old_logprobs = paddle.reshape(old_logprobs, [1000, 1])
        # old_logprobs = [paddle.to_tensor(s) for s in old_logprobs]

        # rewards = paddle.unsqueeze(rewards, axis=1)  # 增加一维
        #
        # rewards = [rewards]
        #
        # rewards = paddle.stack(rewards)  # 变成二维
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            # print("old_actions", old_actions)
            logprobs, state_values, dist_entropy = self.model.value(old_states, old_actions)
            # 归一化
            state_values = (state_values - state_values.mean()) / (state_values.std() + 1e-5)

            ratios = paddle.exp(logprobs - old_logprobs)
            # 计算优势函数
            advantages = rewards - state_values

            surr1 = ratios * advantages
            clipped_ratios = paddle.clip(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
            surr2 = clipped_ratios * advantages
            loss = -paddle.minimum(surr1, surr2) + 0.5*self.mse_loss(state_values, rewards) - 0.01*dist_entropy
            # print('loss', loss)
            # + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.clear_grad()
            loss.mean().backward()
            self.optimizer.step()
