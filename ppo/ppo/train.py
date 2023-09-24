import os
import sys
from multiprocessing import freeze_support

import paddle
from matplotlib import pyplot as plt
from parl.utils import summary
import numpy as np
from algorithm import PPO
from env import RobotEnv
from model import Model
from agent import PPOAgent
from storage import ReplayMemory
import random
from stable_baselines3.common.vec_env import SubprocVecEnv

# 玩多少次
TRAIN_EPISODE = 12000
# 学习率
LR = 5e-3
# adm更新参数
BETAS = (0.9, 0.99)
# 折扣因子
GAMMA = 0.9
# 学习的次数
K_EPOCHS = 4
# ppo截断
EPS_CLIP = 0.2
# 环境个数
NUM_ENV = 6
# 到达的学习的数据数
UPDATE_TIMESTEP = 1000


def make_env(seed):
    def _init():
        env = RobotEnv(False)
        return env

    return _init


def run_episode(agent, env, rpm):
    obs = env.reset()
    episode = 1
    timestep = 1
    is_coll = 0
    # done = np.zeros(step_nums, dtype='float32')
    while episode < TRAIN_EPISODE:
        is_epi = False
        # 升维 [3,84,84] ->[1,3,84,84]
        obs = paddle.to_tensor(obs, dtype='float32')
        value, action, logprob, _ = agent.sample(obs, rpm)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        action = action.reshape(  (NUM_ENV, 1))
        rpm.append(obs, action, logprob, reward, done, value.flatten())
        # 每6000step学习一次
        if timestep % UPDATE_TIMESTEP == 0:
            value = agent.value(obs)
            rpm.compute_returns(value, done)
            agent.learn(rpm)
            timestep = 0
        # 计算碰撞次数
        for i in range(NUM_ENV):
            if info[i]["iscoll"]:
                is_coll += 1
        # 获取done中为True的索引
        indices = np.where(done)[0]
        # print("indices", indices)
        if len(indices) > 0:
            obs[indices] = env.env_method("reset", indices=indices)
            episode += indices.shape[0]
            is_epi = True
        # 每50个episode绘制一次图像
        if episode % 50 == 0 and is_epi:

            # 绘制图像
            print("coll-----------", episode, "----------num : ", is_coll)
            summary.add_scalar("coll_num", is_coll, global_step=episode)

            # 保存模型
            save_path = '../ppo/train_log/model' + str(episode) + '.ckpt'
            if is_coll >= 10:
                agent.save(save_path)
            is_coll = 0
        timestep += 1
    return


def main():
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))
    # 创建向量化环境
    env = SubprocVecEnv([make_env(seed) for seed in seed_set])

    # 使用PARL框架创建agent
    model = Model()
    ppo = PPO(model, LR, BETAS, GAMMA, K_EPOCHS, EPS_CLIP)
    agent = PPOAgent(ppo, model)
    rpm = ReplayMemory()

    # 导入策略网络参数
    if os.path.exists('../ppo/train_log/model.ckpt'):
        agent.restore('../ppo/train_log/model.ckpt')

    run_episode(agent, env, rpm)


if __name__ == '__main__':
    main()
