import os

import parl
from parl.utils import logger, summary

from robot_DQN.agent import Agent
from robot_DQN.algorithm import DQN
from robot_DQN.env import RobotEnv
from robot_DQN.model import Model
from robot_DQN.repaly_memory import ReplayMemory
from robot_DQN.run import run_episode, evaluate

LR = 0.001  # learning rate
GAMMA = 0.99  # reward 的衰减因子

MEMORY_SIZE = int(50000)  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 10  # 预存一部分经验之后再开始训练
TRAIN_EPISODE = 60000  # 训练的总episode数

# 创建环境
env = RobotEnv()

# 使用PARL框架创建agent
model = Model()
algorithm = DQN(model, gamma=GAMMA, lr=LR)
agent = Agent(algorithm, act_dim=3, e_greed=0.5, e_greed_decrement=1e-6)

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
coll_times = 0
while episode < TRAIN_EPISODE:
    is_coll = 0
    for i in range(50):
        avg_reward, coll = run_episode(agent, env, rpm)
        if coll:
            is_coll += 1
            coll_times += 1
        else:
            coll_times = 0
        save_path_2 = './model' + str(episode) + '.ckpt'
        if coll_times >= 5:
            agent.save(save_path_2)
        summary.add_scalar("avg_reward", avg_reward, global_step=episode)
        episode += 1

    print("coll-----------", episode, "----------num : ", is_coll)
    summary.add_scalar("coll_num", is_coll, global_step=episode)

    avg_reward, coll_num = evaluate(env, agent)
    logger.info('episode:{}    test_reward:{}    coll_num:{}    e_greed:{}'.format(episode, avg_reward, coll_num,
                                                                                   agent.e_greed))
    summary.add_scalar("eva_coll_num", coll_num, global_step=episode)

    # 保存模型
    agent.save("./model.ckpt")
    save_path = './model' + str(episode) + '.ckpt'
    if coll_num >= 7:
        agent.save(save_path)
