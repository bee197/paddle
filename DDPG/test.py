from parl.utils import logger
from DDPG.agent import Agent
from DDPG.algorithm import DDPG
from DDPG.cartpole import ContinuousCartPoleEnv
from DDPG.model import Model
from DDPG.repaly_memory import ReplayMemory
from DDPG.train import run_episode, evaluate

ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate

GAMMA = 0.99  # reward 的衰减因子
TAU = 0.001  # 软更新的系数
MEMORY_SIZE = int(1e6)  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20  # 预存一部分经验之后再开始训练
BATCH_SIZE = 128
REWARD_SCALE = 0.1  # reward 缩放系数
NOISE = 0.05  # 动作噪声方差


TRAIN_EPISODE = 6000  # 训练的总episode数

# 创建环境
env = ContinuousCartPoleEnv()

obs_dim = env.observation_space.shape[0]
# print("observation_space shape: ", env.observation_space.shape)
# print("observation_space.shape[0]: ", env.observation_space.shape[0])
# print("obs_dim :", obs_dim)
act_dim = env.action_space.shape[0]
# print("action_space.shape shape: ", env.action_space.shape)

# 使用PARL框架创建agent
model = Model(obs_dim, act_dim)
algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
agent = Agent(algorithm, obs_dim, act_dim)

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)


# 往经验池中预存数据
while rpm.__len__() < MEMORY_WARMUP_SIZE:
    run_episode(agent, env, rpm)

episode = 0
while episode < TRAIN_EPISODE:
    for i in range(50):
        total_reward = run_episode(agent, env, rpm)
        episode += 1

    eval_reward = evaluate(env, agent, render=False)
    logger.info('episode:{}    test_reward:{}'.format(
        episode, eval_reward))
