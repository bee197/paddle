# import random
# import collections
# import numpy as np
# import paddle
# import torch
#
# from ppo.ppo_config import config
#
#
# class ReplayMemory(object):
#     def __init__(self, max_size):
#         # 创建一个双端队列对象,大小为max_size
#         self.advantages = None
#         self.step_nums = config['step_nums']
#         self.returns = None
#         self.buffer = collections.deque(maxlen=max_size)
#
#     # 添加样本
#     def append(self, exp):
#         self.buffer.append(exp)
#
#     def adv(self, value, done, gamma=0.99, gae_lambda=0.95):
#         # 2000
#         buffer_np = np.asarray(self.buffer)
#         reward = buffer_np[:, 1]
#         advantages = np.zeros_like(reward)
#         lastgaelam = 0
#         print(done)
#         for t in reversed(range(self.step_nums)):  # self.step_nums原来的
#             print('111111', done)
#             if t == self.step_nums - 1:
#                 nextnonterminal = 1.0 - done
#                 nextvalues = value.reshape(1, -1)
#             else:
#                 nextnonterminal = 1.0 - done[t + 1]
#                 nextvalues = value[t + 1]
#             delta = reward[t] + gamma * nextvalues * nextnonterminal - value[0]
#             advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
#         returns = advantages + value
#         self.returns = returns
#         self.advantages = advantages
#         return advantages, returns
#
#     # 对样本采样
#     def sample(self, batch_size):
#         # 随机选择batch_size个经验样本，构成一个名为mini_batch的储存experience的列表。
#         mini_batch = random.sample(self.buffer, batch_size)
#
#         # 通过遍历mini_batch列表，将经验样本中的状态（s）、动作（a）、奖励（r）、下一个状态（s_p）和done标志（done）分别提取出来
#         obs_batch, action_batch, log_p_batch, reward_batch, done_batch, values_batch = [], [], [], [], [], []
#
#         for experience in mini_batch:
#             s, a, l, r, done, v = experience
#
#             obs_batch.append(s)
#             action_batch.append(a)
#             log_p_batch.append(l)
#             reward_batch.append(r)
#             done_batch.append(done)
#             values_batch.append(v)
#             batch_adv = self.advantages
#             batch_return = self.returns
#         obs_batch = np.array(obs_batch).astype('float32')
#         obs_batch = obs_batch.reshape((batch_size, 3, 84, 84))
#         obs_batch = paddle.to_tensor(obs_batch)
#
#         return obs_batch, action_batch, log_p_batch, batch_adv, batch_return, values_batch
#
#     def __len__(self):
#         return len(self.buffer)


class ReplayMemory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_rpm(self):
        # act
        del self.actions[:]
        # obs
        del self.states[:]
        # log_b
        del self.logprobs[:]
        # reward
        del self.rewards[:]
        # done
        del self.is_terminals[:]

    def __call__(self, *args, **kwargs):
        print("is_done", self.is_terminals)
