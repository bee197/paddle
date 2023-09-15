import os
import time
import random
import numpy as np
from matplotlib import pyplot as plt
from sb3_contrib import MaskablePPO
from env import RobotEnv

MODEL_PATH = "trained_models_CNN4/ppo_robot_1050000_steps.zip"

NUM_EPISODE = 20

seed = random.randint(0, 1e9)
# print(f"Using seed = {seed} for testing.")

env = RobotEnv(True)

num = 30000
while num <= 450000:
    # Load the trained model
    model = MaskablePPO.load(MODEL_PATH)

    total_reward = 0
    total_score = 0
    min_score = 1e9
    max_score = 0

    coll_num = 0
    for episode in range(10):
        print('env.observation_space=', env.observation_space)
        print('env.action_space=', env.action_space)
        rewards = []


        print(np.sum(rewards))
        obs = env.reset()
        episode_reward = 0
        done = False
        num_step = 0
        # print(f"=================== Episode {episode + 1} ==================")
        step_counter = 0
        # trace = []
        while True:
            state = np.zeros((84, 84, 3), dtype=np.uint8)
            state[:] = obs[0]

            # print(obs)
            # trace.append(obs[:, :-1])

            mask = env.get_action_mask()
            action, _ = model.predict(state, action_masks=mask)
            num_step += 1
            obs, reward, done, info = env.step(int(action))
            info["action"] = action
            if done:
                if info["iscoll"]:
                    coll_num += 1
                break
            episode_reward += reward
            rewards.append(reward)
        plt.plot(rewards)
        plt.show()
    # if coll_num >= 5:
    print(MODEL_PATH, coll_num)
    # print(f"total_reward:{episode_reward}")
    num += 30000
    MODEL_PATH = 'trained_models_CNN4/ppo_robot_' + str(num) + '_steps'
