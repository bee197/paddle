import os

from matplotlib import pyplot as plt

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


def run_evaluate_episodes(agent, env, max_epi=10):
    for i in range(max_epi):
        running_reward = 0
        timestep = 0
        episode_reward = []
        obs = env.reset()
        while True:
            timestep += 1
            # 升维 [3,84,84] ->[1,3,84,84]
            #     obs = obs[0]
            obs = obs.unsqueeze(0)
            # print('11', obs.shape)
            action = agent.predict(obs)
            # print("action", action)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            running_reward += reward
            episode_reward.append(reward)
            if done:
                # print("done", next_done)
                return info, episode_reward


# 创建环境
env = RobotEnv(True)
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

while episode < TRAIN_EPISODE:
    is_coll = 0
    coll, episode_reward = run_evaluate_episodes(agent, env)
    if coll:
        is_coll += 1
        coll_times += 1
    else:
        coll_times = 0
    episode += 1
    plt.plot(episode_reward)
    plt.show()
