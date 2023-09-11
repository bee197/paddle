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

# 结束的奖励
solved_reward = 230
# 打印的数据
log_interval = 20
# 玩多少次
max_episodes = 50000
# 一次玩多少
max_timesteps = 10
# 隐层神经元的数量
n_latent_var = 576
# 到达的学习的数据数
update_timestep = 1000
# 学习率
lr = 0.0001
# adm更新参数
betas = (0.9, 0.99)
# 折扣因子
gamma = 0.95
# 学习的次数
K_epochs = 4
# ppo截断
eps_clip = 0.2




def run_episode(agent, env, rpm, timestep):
    running_reward = 0
    obs = env.reset()

    # done = np.zeros(step_nums, dtype='float32')
    while True:
        timestep += 1
        # print("timestep", timestep )
        # 升维 [3,84,84] ->[1,3,84,84]
        #     obs = obs[0]
        obs = obs.unsqueeze(0)
        # print('11', obs.shape)
        action = agent.sample(obs, rpm)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        # action = [action]  # 方便存入replaymemory
        # rpm.buffer.append((obs, action, reward, next_obs,done)
        rpm.rewards.append(reward)
        rpm.is_terminals.append(done)
        if timestep % update_timestep == 0:
            # print(paddle.to_tensor(np.array(rpm.is_terminals)).shape)
            agent.learn(rpm)

            rpm.clear_rpm()
            # print(paddle.to_tensor(np.array(rpm.is_terminals)))
            timestep = 0
        running_reward += reward
        if done:
            break
    return info, timestep


# 创建环境
env = RobotEnv(False)
state = env.observation_space.shape[0]

action_dim = env.action_space.n
# 使用PARL框架创建agent
model = Model()
ppo = PPO(model, state, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
rpm = ReplayMemory()
obs = env.reset()  # 初始化
agent = PPOAgent(ppo, obs, model)
# 导入策略网络参数
if os.path.exists('../ppo/train_log/model.ckpt'):
    agent.restore('../ppo/train_log/model.ckpt')

episode = 0
coll_times = 0
TRAIN_EPISODE = 1e6
timestep = 0
while episode < max_episodes:
    is_coll = 0
    for i in range(50):
        # print('111111111111', obs.shape)
        coll, timestep = run_episode(agent, env, rpm, timestep)
        if coll:
            is_coll += 1
            coll_times += 1
        else:
            coll_times = 0
        save_path_2 = '../ppo/train_log/model' + str(episode) + '.ckpt'
        if coll_times >= 5:
            agent.save(save_path_2)
        episode += 1

    print("coll-----------", episode, "----------num : ", is_coll)
    summary.add_scalar("coll_num", is_coll, global_step=episode)

    # 保存模型
    agent.save("../ppo/train_log/model.ckpt")
    save_path = '../ppo/train_log/model' + str(episode) + '.ckpt'
