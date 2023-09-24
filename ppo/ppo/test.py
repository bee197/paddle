import os

import numpy as np
import paddle
from matplotlib import pyplot as plt

from algorithm import PPO
from env import RobotEnv
from model import Model
from agent import PPOAgent
from storage import ReplayMemory

# 玩多少次
TRAIN_EPISODE = 50
# 到达的学习的数据数
UPDATE_TIMESTEP = 1000
# 学习率
LR = 0.0001
# adm更新参数
BETAS = (0.9, 0.99)
# 折扣因子
GAMMA = 0.95
# 学习的次数
K_EPOCHS = 4
# ppo截断
EPS_CLIP = 0.2


def run_evaluate_episodes(agent, env, max_epi):
    for i in range(max_epi):
        timestep = 0
        episode_reward = []
        obs = env.reset()
        while True:
            timestep += 1
            obs = paddle.to_tensor(obs, dtype='float32')
            obs = obs.unsqueeze(0)
            action = agent.predict(obs)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            episode_reward.append(reward)
            if done:
                return info, episode_reward


# 创建环境
env = RobotEnv(True)
# 使用PARL框架创建agent
model = Model()
ppo = PPO(model, LR, BETAS, GAMMA, K_EPOCHS, EPS_CLIP)
agent = PPOAgent(ppo, model)
rpm = ReplayMemory()
# 导入策略网络参数
PATH = 'train_log/model.ckpt'

episode = 0
it = 1500
while it <= 8250:
    if os.path.exists(PATH):
        agent.restore(PATH)
        is_coll = 0
        episode = 0
        while episode < TRAIN_EPISODE:
            info, episode_reward= run_evaluate_episodes(agent, env, TRAIN_EPISODE)
            if info["iscoll"]:
                is_coll += 1
            episode += 1
            plt.plot(episode_reward)
            plt.show()
        print("it : {}   is_coll: {}".format(it, is_coll))
    it += 50
    PATH = '../ppo/train_log/model' + str(it) + '.ckpt'
