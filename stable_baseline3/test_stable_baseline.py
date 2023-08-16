from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym

env_name = "LunarLander-v2"
env = gym.make(env_name, render_mode="human")
env = DummyVecEnv([lambda: env])
model = DQN(
    "MlpPolicy",
    env=env,
    learning_rate=5e-4,
    batch_size=128,
    buffer_size=50000,
    learning_starts=0,
    target_update_interval=250,
    policy_kwargs={"net_arch": [256, 256]},
    verbose=1,
    tensorboard_log="./tensorboard/LunarLander-v2/"
)

model.learn(total_timesteps=10000)
model.save("./model/LunarLander3.pkl")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
env.close()
print(mean_reward, std_reward)
