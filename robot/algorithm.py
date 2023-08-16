from copy import deepcopy

import numpy as np
import paddle
import parl
import paddle.nn.functional as F


class DDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(model)
        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=actor_lr, parameters=self.model.get_actor_params())
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=critic_lr, parameters=self.model.get_critic_params())

    def predict(self, obs):
        """ 使用 self.model 的 actor model 来预测动作
            """
        # 升维
        # obs = np.expand_dims(obs, axis=0)
        # obs = paddle.to_tensor(obs, dtype='float32')

        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, done):
        critic_loss = self._critic_learn(obs, action, reward, next_obs, done)
        actor_loss = self._actor_learn(obs)

        self.sync_target()
        return critic_loss, actor_loss

    def _actor_learn(self, obs):
        # Compute actor loss and Update the frozen target models
        actor_loss = -self.model.value(obs, self.model.policy(obs)).mean()

        # TODO:
        # print("actor_loss : ", actor_loss)

        # Optimize the actor
        self.actor_optimizer.clear_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, done):
        # 上下文管理器，以阻止在计算目标 Q 值时产生梯度。
        with paddle.no_grad():
            # Compute the target Q value
            next_act = self.target_model.policy(next_obs)
            next_Q = self.target_model.value(next_obs, next_act)
            # 将张量转换为float32
            done = paddle.cast(done, dtype='float32')
            # Q = r + γ * Q(s',a') =  r + γ * Q'
            target_Q = reward + ((1. - done) * self.gamma * next_Q)

        # Get current Q estimate
        current_Q = self.model.value(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.clear_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    # 同步targetModel
    def sync_target(self, decay=None):
        """ update the target network with the training network

            Args:
                self:
                decay(float): the decaying factor while updating the target network with the training network.
                            0 represents the **assignment**. None represents updating the target network slowly that depends on the hyperparameter `tau`.
            """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
