import gym
# env = gym.make('CartPole-v1')
env = gym.make('CartPole-v1', render_mode="human")
env.reset()

env.render()
for _ in range(1000):
    observation, reward, done, info, _ = env.step(env.action_space.sample())
    print(observation, reward, done, info)
    if done:
        break
env.close()