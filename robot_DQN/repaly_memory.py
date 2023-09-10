import random
import collections
import numpy as np
import paddle
import torch


class ReplayMemory(object):
    def __init__(self, max_size):
        # 创建一个双端队列对象,大小为max_size
        self.buffer = collections.deque(maxlen=max_size)

        # 添加样本

    def append(self, exp):
        self.buffer.append(exp)

    # 对样本采样
    def sample(self, batch_size):
        # 随机选择batch_size个经验样本，构成一个名为mini_batch的储存experience的列表。
        mini_batch = random.sample(self.buffer, batch_size)
        # 通过遍历mini_batch列表，将经验样本中的状态（s）、动作（a）、奖励（r）、下一个状态（s_p）和done标志（done）分别提取出来
        obs_batch, action_batch, reward_batch, done_batch, next_obs_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience

            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            done_batch.append(done)
            next_obs_batch.append(s_p)

        obs_batch = np.array(obs_batch).astype('float32')
        obs_batch = obs_batch.reshape((batch_size, 3, 84, 84))
        obs_batch = paddle.to_tensor(obs_batch)

        next_obs_batch = np.array(next_obs_batch).astype('float32')
        next_obs_batch = next_obs_batch.reshape((batch_size, 3, 84, 84))
        next_obs_batch = paddle.to_tensor(next_obs_batch)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)
