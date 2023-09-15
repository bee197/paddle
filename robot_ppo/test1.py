import random

from matplotlib import pyplot as plt
from sb3_contrib import MaskablePPO

# import imageio
from env import RobotEnv

NUM_EPISODE = 10

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

env = RobotEnv(True)

num = 1110000
MODEL_PATH = 'trained_models_CNN4/ppo_robot_' + str(num) + '_steps'
while num <= 1170000:
    total_reward = 0
    total_score = 0
    min_score = 1e9
    max_score = 0
    collnum = 0
    # Load the trained model
    model = MaskablePPO.load(MODEL_PATH)
    for episode in range(NUM_EPISODE):
        obs = env.reset()
        episode_reward = []
        done = False
        num_step = 0
        # print(f"=================== Episode {episode + 1} ==================")
        step_counter = 0
        # trace = []
        while True:
            # trace.append(obs[:,:-1])
            mask = env.get_action_mask()
            action, _ = model.predict(obs, action_masks=mask)
            num_step += 1
            obs, reward, done, info = env.step(int(action))
            info["action"] = action
            if info["iscoll"]:
                collnum += 1
            if done:
                break
            episode_reward.append(reward)
        plt.plot(episode_reward)
        plt.show()
    print(f"collnum:{collnum}")
    print(MODEL_PATH)
    num += 30000
    MODEL_PATH = 'trained_models_CNN4/ppo_robot_' + str(num) + '_steps'
        # plt.plot(episode_reward)
        # plt.show()
        # imageio.mimsave(f"test_video/epo-{episode}.mp4",trace,format="mp4",fps=10)
        # print(f"total_reward:{sum(episode_reward)}")


