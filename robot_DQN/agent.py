import cv2
import numpy as np
import paddle
import parl


class Agent(parl.Agent):
    def __init__(self, algorithm, act_dim, e_greed, e_greed_decrement):
        super(Agent, self).__init__(algorithm)
        self.act_dim = act_dim

        self.global_step = 0
        self.update_target_steps = 200

        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

        # 注意：最开始先同步self.model和self.target_model的参数.
        self.alg.sync_target()

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        # 探索：每个动作都有概率被选择
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)
        # 选择最优动作
        else:
            act = self.predict(obs)
        self.e_greed = max(0.1, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):
        # # 图像是[128,4,84,84],即一次128批次,4帧,像素84*84
        # cv2.imshow("obs", obs[0][0].numpy())
        # cv2.waitKey(1)
        # obs = paddle.to_tensor(obs, dtype='float32')
        Q = self.alg.predict(obs)
        # print("Q : ", Q[0])
        act = int(Q.argmax())
        # print("action : ", act)

        return act

    def learn(self, obs, act, reward, next_obs, done):
        # 更新target网络
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = paddle.to_tensor(act, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        done = paddle.to_tensor(done, dtype='float32')

        # print("obs : %r" % obs)
        # print("act : %r" % act)
        # print("reward : %r" % reward)
        # print("next_obs : %r" % next_obs)
        # print("done : %r" % done)

        loss = self.alg.learn(obs, act, reward, next_obs, done)

        # print(loss)

        return loss
