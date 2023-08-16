import numpy as np
import paddle

import repaly_memory

MEMORY_SIZE = int(50000)  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 10  # 预存一部分经验之后再开始训练
BATCH_SIZE = 128
REWARD_SCALE = 0.1  # reward 缩放系数
NOISE = 1.  # 动作噪声方差


def run_episode(agent, env, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        # print("batch_obs", obs)
        # 升维
        obs = np.expand_dims(obs, axis=0)
        # print("batch_obs shape", batch_obs.astype('float32').shape)
        # action = agent.predict(batch_obs.astype('float32'))
        # print("obs shape", obs.astype('float32').shape)
        action = agent.predict(obs.astype('float32'))

        # print("action1 : ", action[0])

        # 增加探索扰动, 输出限制在 [-10.0, 10.0] 范围内
        action = np.clip(np.random.normal(action, NOISE), -10.0, 10.0)

        # print("action2 : ", action[0])
        action = action[0]

        next_obs, reward, done, info = env.step(action)

        # action = [action]  # 方便存入replaymemory

        rpm.buffer.append((obs, action, REWARD_SCALE * reward, next_obs, done))

        # 样本数量大于MEMORY_WARMUP_SIZE（经验缓存的预热阶段数量）且步数是5的倍数，则从经验缓存中抽样一个批次的经验样本
        if rpm.__len__() > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            # 返回5个numpy数组
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(BATCH_SIZE)
            # print("batch_obs : %r" % paddle.to_tensor(batch_obs, dtype='float32'))
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

        obs = next_obs
        # print("reward : ", reward)
        total_reward += reward
        # print("reward : ", reward)

        if done:
            break

    return total_reward


def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        steps = 0
        while True:
            steps += 1
            batch_obs = paddle.to_tensor(obs)  # shape=(1, 4, 84, 84)
            batch_obs = batch_obs.expand([128, -1, -1, -1])  # 重复BATCH_SIZE
            # batch_obs = np.expand_dims(batch_obs, axis=0)
            # print("t1 action : %r" % action)
            action = agent.predict(batch_obs.astype('float32'))
            # print("t2 action : %r" % action)
            action = np.clip(action, -10.0, 10.0)
            # print("t3 action : %r" % action)
            action = action[0]
            # print("t4 action : %r" % action)

            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            # print("reward : ", reward)

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)
