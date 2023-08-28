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
        action_batch, reward_batch, done_batch = [], [], []

        obs_batch = []
        dist_batch = []
        angle_batch = []

        next_obs_batch = []
        next_dist_batch = []
        next_angle_batch = []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs, distance, angle = s[0]
            next_obs, next_distance, next_angle = s_p[0]

            obs_batch.append(obs)
            dist_batch.append(distance)
            angle_batch.append(angle)

            next_obs_batch.append(next_obs)
            next_dist_batch.append(next_distance)
            next_angle_batch.append(next_angle)

            action_batch.append(a)
            reward_batch.append(r)
            done_batch.append(done)

        obs_batch = np.array(obs_batch).astype('float32')
        obs_batch = obs_batch.reshape((batch_size, 3, 84, 84))
        obs_batch = paddle.to_tensor(obs_batch)
        angle_batch = np.array(angle_batch).astype('float32')
        angle_batch = angle_batch.reshape((batch_size, 2))
        angle_batch = paddle.to_tensor(angle_batch)
        dist_batch = np.array(dist_batch).astype('float32')
        dist_batch = dist_batch.reshape((batch_size, 2))
        dist_batch = paddle.to_tensor(dist_batch)

        next_obs_batch = np.array(next_obs_batch).astype('float32')
        next_obs_batch = next_obs_batch.reshape((batch_size, 3, 84, 84))
        next_obs_batch = paddle.to_tensor(next_obs_batch)
        next_angle_batch = np.array(next_angle_batch).astype('float32')
        next_angle_batch = next_angle_batch.reshape((batch_size, 2))
        next_angle_batch = paddle.to_tensor(next_angle_batch)
        next_dist_batch = np.array(next_dist_batch).astype('float32')
        next_dist_batch = next_dist_batch.reshape((batch_size, 2))
        next_dist_batch = paddle.to_tensor(next_dist_batch)

        obs = [(obs_batch, dist_batch, angle_batch)]
        next_obs = [(next_obs_batch, next_dist_batch, next_angle_batch)]

        return obs, action_batch, reward_batch, next_obs, done_batch

    def __len__(self):
        return len(self.buffer)
