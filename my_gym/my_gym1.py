import gym
from gym import envs

'''
* @breif: 生成环境对象
* @param[in]: id            ->  启用环境的名称
* @param[in]: render_mode   ->  渲染模式
* @retval: 环境对象
'''
env = gym.make('MountainCar-v0', render_mode="human")
print('观测空间:', env.observation_space)
print('动作空间:', env.action_space)
print('动作数:', env.action_space.n)
# 观测空间: Box([-1.2  -0.07], [0.6  0.07], (2,), float32)
# 动作空间: Discrete(3)
# 动作数: 3

# 查看所有环境
# envids = [spec.id for spec in envs.registry.all()]
# for envid in sorted(envids):
#     print(envid)



