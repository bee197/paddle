import numpy as np
import paddle
import parl


class Agent(parl.Agent):
    def __init__(self, algorithm):
        super(Agent, self).__init__(algorithm)
        self.training = True

        # 注意：最开始先同步self.model和self.target_model的参数.
        self.alg.sync_target()

    def predict(self, obs):
        obs = paddle.to_tensor(obs, dtype='float32')
        act = self.alg.predict(obs)

        return act

    def learn(self, obs, act, reward, next_obs, done):
        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        done = paddle.to_tensor(done, dtype='float32')

        # print("obs : %r" % obs)
        # print("act : %r" % act)
        # print("reward : %r" % reward)
        # print("next_obs : %r" % next_obs)
        # print("done : %r" % done)

        loss = self.alg.learn(obs, act, reward, next_obs, done)

        # print(loss)

        return loss
