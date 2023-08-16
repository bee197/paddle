import os
from robot_DQN.agent import Agent
from robot_DQN.algorithm import DQN
from robot_DQN.env import RobotEnv
from robot_DQN.model import Model
from robot_DQN.run import evaluate

LR = 1e-4  # learning rate
GAMMA = 0.99  # reward 的衰减因子
PATH = 'model.ckpt'

# 创建环境
env = RobotEnv(True)

# 使用PARL框架创建agent
model = Model()
algorithm = DQN(model, gamma=GAMMA, lr=LR)
agent = Agent(algorithm, act_dim=3, e_greed=0.1, e_greed_decrement=0)

# 导入策略网络参数
if os.path.exists(PATH):
    agent.restore(PATH)
eval_reward, coll_num = evaluate(env, agent)
print("coll_num : ", coll_num)
# it = 1200
# while it <= 9650:
#     if os.path.exists(PATH):
#         agent.restore(PATH)
#         eval_reward, coll_num = evaluate(env, agent)
#         print("it : {}   coll_num: {}".format(it, coll_num))
#     it += 50
#     PATH = 'model' + str(it) + '.ckpt'



