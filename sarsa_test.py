# Step1 导入依赖
import gym
import numpy as np
import time
import matplotlib.pyplot as plt


# Step2 定义Agent
class SarsaAgent(object):
    def __init__(self, obs_n, act_n, lr, gamma, epsilon):
        self.obs_n = obs_n
        self.act_n = act_n
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q_table = np.zeros((obs_n, act_n))

    def sample(self, obs):
        """
        根据输入观察值，采样输出的动作值，带探索
        :param obs:当前state
        :return: 下一个动作
        """
        action = 0
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # 有一定概率随机探索选取一个动作
        return action

    def predict(self, obs):
        '''
        根据输入观察值，预测输出的动作值
        :param obs:当前state
        :return:预测的动作
        '''

        Q_list = self.Q_table[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, act, reward, next_obs, next_act, done):
        '''
        on-policy
        :param obs:交互前的obs, s_t
        :param act:本次交互选择的action, a_t
        :param reward:本次动作获得的奖励r
        :param next_obs:本次交互后的obs, s_t+1
        :param next_act:根据当前Q表格, 针对next_obs会选择的动作, a_t+1
        :param done:episode是否结束
        :return:null
        '''
        predict_Q = self.Q_table[obs, act]
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q_table[next_obs, next_act]  # Sarsa
        self.Q_table[obs, act] += self.lr * (target_Q - predict_Q)  # 修正q

    # 保存Q表格数据到文件
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q_table)
        print(npy_file + ' saved.')

    # 从文件中读取Q值到Q表格中
    def restore(self, npy_file='./q_table.npy'):
        self.Q_table = np.load(npy_file)
        print(npy_file + ' loaded.')


# Step3 Training && Test（训练&&测试）
def train_episode(env, agent, render=False):
    total_reward = 0
    total_steps = 0  # 记录每个episode走了多少step

    obs=env.reset()[0]
    act = agent.sample(obs)

    while True:
        next_obs, reward, done, _, _ = env.step(act)  # 与环境进行一个交互
        next_act = agent.sample(next_obs)  # 根据算法选择一个动作
        # 训练Sarsa算法
        agent.learn(obs, act, reward, next_obs, next_act, done)

        act = next_act
        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1
        if render:
            env.render()  # 渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    total_steps = 0  # 记录每个episode走了多少step
    obs = env.reset()

    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        obs = next_obs
        # time.sleep(0.5)
        # env.render()
        if done:
            break
    return total_reward, total_steps


# Step4 创建环境和Agent，启动训练

# 使用gym创建悬崖环境
env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left

# 创建一个agent实例,输入超参数
agent = SarsaAgent(
    obs_n=env.observation_space.n,
    act_n=env.action_space.n,
    lr=0.001,
    gamma=0.99,
    epsilon=0.1
)

print("Start training ...")
total_reward_list = []
# 训练1000个episode，打印每个episode的分数
for episode in range(1000):
    ep_reward, ep_steps = train_episode(env, agent, False)
    total_reward_list.append(ep_reward)
    if episode % 50 == 0:
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

print("Train end")


def show_reward(total_reward):
    N = len(total_reward)
    x = np.linspace(0, N, 1000)
    plt.plot(x, total_reward, 'b-', lw=1, ms=5)
    plt.show()


show_reward(total_reward_list)

# 全部训练结束，查看算法效果
test_reward, test_steps = test_episode(env, agent)
print('test steps = %.1f , reward = %.1f' % (test_steps, test_reward))
