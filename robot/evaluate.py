import os
from parl.utils import logger

from robot.agent import Agent
from robot.algorithm import DDPG
from robot.env import RobotEnv
from robot.model import Model
from robot.repaly_memory import ReplayMemory
from robot.train import evaluate

ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate
GAMMA = 0.99  # reward 的衰减因子
TAU = 0.001  # 软更新的系数

MEMORY_SIZE = int(50000)  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 200  # 预存一部分经验之后再开始训练
BATCH_SIZE = 128
REWARD_SCALE = 0.1  # reward 缩放系数
NOISE = 0.05  # 动作噪声方差


TRAIN_EPISODE = 6000  # 训练的总episode数

# 创建环境
env = RobotEnv(True)

# 使用PARL框架创建agent
model = Model()
algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
agent = Agent(algorithm)

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)

# 导入策略网络参数
if os.path.exists('./model.ckpt'):
    agent.restore('./model.ckpt')
eval_reward = evaluate(env, agent)
print("eval_reward : ", eval_reward)

