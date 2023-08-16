import os

from parl.utils import logger, summary

from robot.agent import Agent
from robot.algorithm import DDPG
from robot.env import RobotEnv
from robot.model import Model
from robot.repaly_memory import ReplayMemory
from robot.train import run_episode, evaluate

ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate
GAMMA = 0.99  # reward 的衰减因子
TAU = 0.001  # 软更新的系数

MEMORY_SIZE = int(50000)  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 10  # 预存一部分经验之后再开始训练
TRAIN_EPISODE = 6000  # 训练的总episode数


# 创建环境
env = RobotEnv()

# 使用PARL框架创建agent
model = Model()
algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
agent = Agent(algorithm)

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)


# 往经验池中预存数据
i = 0
while rpm.__len__() < MEMORY_WARMUP_SIZE:
    print("buffur len : ", rpm.__len__())
    print("warmup times : ", i)
    i += 1
    run_episode(agent, env, rpm)

# 导入策略网络参数
if os.path.exists('./model.ckpt'):
    agent.restore('./model.ckpt')
episode = 0
while episode < TRAIN_EPISODE:
    for i in range(50):
        total_reward = run_episode(agent, env, rpm)
        print("total-----------", episode, "----------reward : ", total_reward)
        # summary.add_scalar("reward", total_reward, global_step=episode)
        episode += 1

    # 导入参数
    save_path = './model.ckpt'
    agent.save(save_path)

    eval_reward = evaluate(env, agent)
    logger.info('episode:{}    test_reward:{}'.format(episode, eval_reward))
