import os

import paddle
import parl
import torch
from gym import spaces
from parl.utils import summary
import numpy as np
from algorithm import PPO
from env import RobotEnv
from model import Model
from agent import PPOAgent
from storage import ReplayMemory



# 玩多少次
TRAIN_EPISODE = 1e6
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


def run_episode(agent, env, rpm, timestep):
    running_reward = 0
    obs = env.reset()

    # done = np.zeros(step_nums, dtype='float32')
    while True:
        timestep += 1
        # 升维 [3,84,84] ->[1,3,84,84]
        obs = obs.unsqueeze(0)
        value, action, logprob, _ = agent.sample(obs, rpm)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        # rpm.rewards.append(reward)
        # rpm.is_terminals.append(done)
        rpm.append(obs, action, logprob, reward, done, value.flatten())
        if timestep % UPDATE_TIMESTEP == 0:
            value = agent.value(obs)
            rpm.compute_returns(value, done)
            # print(paddle.to_tensor(np.array(rpm.is_terminals)).shape)
            agent.learn(rpm)
            timestep = 0
        running_reward += reward
        if done:
            break
    return info, timestep


# 创建环境
env = RobotEnv(False)
# 使用PARL框架创建agent
model = Model()
ppo = PPO(model, LR, BETAS, GAMMA, K_EPOCHS, EPS_CLIP)
agent = PPOAgent(ppo, model)
rpm = ReplayMemory()

# 导入策略网络参数
if os.path.exists('../ppo/train_log/model.ckpt'):
    agent.restore('../ppo/train_log/model.ckpt')

episode = 0
coll_times = 0
timestep = 0
while episode < TRAIN_EPISODE:
    is_coll = 0
    for i in range(50):
        coll, timestep = run_episode(agent, env, rpm, timestep)
        # 记录抓球个数
        if coll:
            is_coll += 1
            coll_times += 1
        else:
            coll_times = 0
        # 连续抓球5次,保存模型
        save_path_2 = '../ppo/train_log/model' + str(episode) + '.ckpt'
        if coll_times >= 5:
            agent.save(save_path_2)
        episode += 1
    # 绘制图像
    print("coll-----------", episode, "----------num : ", is_coll)
    summary.add_scalar("coll_num", is_coll, global_step=episode)

    # 保存模型
    agent.save("../ppo/train_log/model.ckpt")
    save_path = '../ppo/train_log/model' + str(episode) + '.ckpt'
