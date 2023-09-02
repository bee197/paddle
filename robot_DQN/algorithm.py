from copy import deepcopy

import numpy as np
import paddle
import parl
import torch


class DQN(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 lr=None):
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.gamma = gamma
        self.lr = lr

        self.model = model
        self.target_model = deepcopy(model)

        self.mse_loss = paddle.nn.MSELoss(reduction='mean')
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.get_params(), weight_decay=0.1)

    def predict(self, obs):
        # 升维
        # obs = np.expand_dims(obs, axis=0)
        # obs = paddle.to_tensor(obs, dtype='float32')

        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, done):
        # Q
        Q = self.model.value(obs)
        action_dim = Q.shape[-1]
        # 转化为torch tensor
        action = torch.from_numpy(action.numpy().astype(np.int64))
        # [128,3]的onehot向量
        action_onehot = torch.nn.functional.one_hot(action, action_dim)
        # 转化为paddle tensor
        action_onehot = paddle.to_tensor(action_onehot.numpy()).astype(np.float32)
        # [128,3] * [128,3]
        Q = Q * action_onehot
        # [128, 1]
        Q = paddle.sum(Q, axis=1, keepdim=True)
        # target Q
        with paddle.no_grad():
            # 选择最大值作为next_Q
            next_Q = self.target_model.value(next_obs).max(1, keepdim=True)
            target_Q = reward + (1 - done) * self.gamma * next_Q

        # 计算loss
        loss = self.mse_loss(Q, target_Q)
        # print("loss : ", loss)

        # 优化器
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        # 学习率衰减
        self.lr *= 0.99
        self.lr = max(self.lr, 0.0001)
        self.optimizer.set_lr(self.lr)

        return loss

    # 同步targetModel
    def sync_target(self):
        self.model.sync_weights_to(self.target_model)
