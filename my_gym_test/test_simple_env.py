from stable_baselines3 import DQN
from my_gym.simple_env import  simple_env

env = simple_env()

model = DQN(policy="MlpPolicy", env=env)
model.learn(total_timesteps=10000)

obs = env.reset()
# 验证十次
for _ in range(10):
    action, state = model.predict(observation=obs)
    print(action)
    obs, reward, done, info = env.step(action)
    env.render()