from time import sleep
from pybullet_envs.bullet import CartPoleBulletEnv

env = CartPoleBulletEnv(renders=True, discrete_actions=False)

env.reset()

env.render()

for _ in range(10000):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    # sleep(1 / 240)