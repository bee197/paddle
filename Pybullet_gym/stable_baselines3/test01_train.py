from stable_baselines3 import DQN
from pybullet_envs.bullet import CartPoleBulletEnv

env = CartPoleBulletEnv(renders=False, discrete_actions=True)

model = DQN(policy="MlpPolicy", env=env)


print("开始训练，稍等片刻")
model.learn(total_timesteps=100000)
model.save("./sb_model/model01.pkl")