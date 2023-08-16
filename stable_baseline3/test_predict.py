import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "LunarLander-v2"
env = gym.make(env_name, render_mode="human")
model = DQN.load("../model/LunarLander3.pkl")

state = env.reset()
# 返回state = Tuple[ObsType, dict]
print(state[0])

done = False
score = 0
while not done:
    action, _ = model.predict(observation=state[0])
    # 返回info = Tuple[ObsType, float, bool, bool, dict]
    info = env.step(action=action)
    score += info[1]
    done = info[2]
    env.render()
env.close()
print(score)